from __future__ import annotations

import asyncio
import math
import typing
import uuid

import pytest

DT = 0.25


@pytest.fixture
def shim():
    from arena_runtime.sim import _mechanism_shim

    return _mechanism_shim


@pytest.fixture
def on_dummy_sim():

    def run(body: typing.Callable[..., typing.Awaitable[None]], *, jackal: bool = False) -> None:
        from arena_rclpy_mixins import ArenaMixinNode
        from arena_rclpy_mixins.shared import Namespace
        from arena_runtime.env.realizer import Realizer

        from arena_runtime.sim.dummy_simulator import DummySimulator

        async def main() -> None:
            suffix = uuid.uuid4().hex[:8]
            node = ArenaMixinNode(f"shim_tick_{suffix}", namespace=f"/shim_tick_{suffix}")
            sim = DummySimulator(node=node, namespace=Namespace(f"/shim_tick_{suffix}"), realizer=Realizer())
            try:
                if jackal:
                    await body(sim, _register_jackal(sim))
                else:
                    await body(sim)
            finally:
                await sim.stop_mechanisms()
                node.destroy_node()

        asyncio.run(main())

    return run


def _register_jackal(sim) -> str:
    from arena_robots.Robot import RobotIdentifier
    from arena_runtime.shared import Pose, Robot

    model = RobotIdentifier.parse("jackal")
    robot = Robot(name="jackal", pose=Pose(), model=model, adapters={}, extra={})
    params = model.resolve_sync().model_params
    sim._register_agent_robot(robot, params)
    return robot.frame.tf(params.base_frame)


def _place(sim, frame: str, x: float, y: float, yaw: float = 0.0) -> None:
    import geometry_msgs.msg

    t = geometry_msgs.msg.TransformStamped()
    t.header.frame_id = "map"
    t.header.stamp.sec = 1
    t.child_frame_id = frame
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.rotation.z = math.sin(yaw / 2.0)
    t.transform.rotation.w = math.cos(yaw / 2.0)
    sim._mechanism_tf_buffer.set_transform(t, "test")


def _door(shim, sim, name: str = "d"):
    from arena_runtime.shared import Door, Position

    door = Door(name=name, start=Position(0.0, 0.0, 0.0), end=Position(1.0, 0.0, 0.0), activation_distance=(1.5, 1.5), transition_time=1.0, hold_time=2.0)
    _, closed = shim._door_geometry(door)
    runtime = shim._DoorRuntime(door=door, closed_pose=closed, open_pose=shim._door_open_pose(door, closed, "sliding"), effective_kind="sliding")
    sim._door_runtime[name] = runtime
    return runtime


def _elevator(shim, sim, name: str, x: float, destination: str):
    from arena_runtime.shared import Elevator, Position

    elevator = Elevator(name=name, position=Position(x, 0.0, 0.0), size=[2.0, 2.0, 2.5], door_side="+x", destinations=[destination], travel_time=0.0, transition_time=1.0, hold_time=2.0, activation_distance=1.0)
    door = shim._elevator_synthesized_door(elevator)
    _, closed = shim._door_geometry(door)
    door_runtime = shim._DoorRuntime(door=door, closed_pose=closed, open_pose=shim._door_open_pose(door, closed, "sliding"), effective_kind="sliding")
    runtime = shim._ElevatorRuntime(elevator=elevator, door_name=door.name, destination_name=elevator.destination)
    sim._door_runtime[door.name] = door_runtime
    sim._elevator_doors[name] = door.name
    sim._elevator_runtime[name] = runtime
    return runtime, door_runtime


def _lamp():
    from arena_runtime.shared import Pose, Position

    from arena_simulation_setup.shared import Light

    return Light(name="lamp", fixture="tube", pose=Pose(position=Position(1.0, 2.0, 2.5)), lumens=300.0)


def _spawn_lamp(sim) -> None:
    sim._lights["lamp"] = _lamp()
    assert sim._semantics.attach("light", "lamp", _lamp().semantics)


def test_idle_door_is_placed_closed_once_on_the_first_tick(shim, on_dummy_sim):
    async def body(sim):
        runtime = _door(shim, sim)
        await shim._tick(sim, DT)
        assert (runtime.state, runtime.progress, runtime.last_applied_progress) == (shim._DoorState.CLOSED, 0.0, 0.0)

    on_dummy_sim(body)


def test_robot_near_a_door_opens_it_although_the_sim_cannot_move_boxes(shim, on_dummy_sim):
    async def body(sim, jackal):
        runtime = _door(shim, sim)
        _place(sim, jackal, 0.5, 0.8)
        await shim._tick(sim, DT)
        assert runtime.last_trigger_sim_time == 0.0
        assert runtime.state == shim._DoorState.OPENING
        assert runtime.progress == pytest.approx(DT)
        assert runtime.last_applied_progress == pytest.approx(DT)

    on_dummy_sim(body, jackal=True)


def test_robot_without_a_transform_does_not_trigger_a_door(shim, on_dummy_sim):
    async def body(sim, jackal):
        runtime = _door(shim, sim)
        await shim._tick(sim, DT)
        assert runtime.last_trigger_sim_time == -math.inf
        assert runtime.state == shim._DoorState.CLOSED

    on_dummy_sim(body, jackal=True)


def test_elevator_without_a_cabin_door_is_left_alone(shim, on_dummy_sim):
    from arena_runtime.shared import Elevator, Position

    async def body(sim):
        elevator = Elevator(name="e", position=Position(0.0, 0.0, 0.0), size=[2.0, 2.0, 2.5], destinations=["f"])
        runtime = shim._ElevatorRuntime(elevator=elevator, door_name="e/door", destination_name="f", car=shim._CarState.DEPARTING, next_stop="f")
        sim._elevator_runtime["e"] = runtime
        await shim._tick(sim, DT)
        assert runtime.departing

    on_dummy_sim(body)


def test_recalled_elevator_holds_its_door_open_and_cancels_departure(shim, on_dummy_sim):
    from arena_simulation_setup.shared.semantics import SemanticCfg

    async def body(sim):
        runtime, door = _elevator(shim, sim, "a", 0.0, "b")
        runtime.car = shim._CarState.DEPARTING
        runtime.next_stop = "b"
        cfgs = [SemanticCfg(role="state", name="state", params={"windows": [{"start": 0.0, "end": 100.0, "value": "on"}], "regime": "alarm"})]
        assert sim._semantics.attach("schedule", "fire", cfgs)
        sim._semantics.set_recall("a", "alarm")
        await shim._tick(sim, DT)
        assert not runtime.departing
        assert door.last_trigger_sim_time == 0.0
        assert door.state == shim._DoorState.OPENING

    on_dummy_sim(body)


def test_departure_to_an_unknown_cabin_reopens_the_door(shim, on_dummy_sim):
    async def body(sim):
        runtime, door = _elevator(shim, sim, "a", 0.0, "ghost")
        runtime.car = shim._CarState.DEPARTING
        runtime.next_stop = "ghost"
        await shim._tick(sim, DT)
        assert not runtime.departing
        assert door.last_trigger_sim_time == 0.0
        assert door.progress == pytest.approx(DT)

    on_dummy_sim(body)


def test_robot_ride_releases_its_rider_when_the_teleport_fails(shim, on_dummy_sim):
    async def body(sim, jackal):
        reasons: list[str] = []
        sim.set_incident_callback(reasons.append)
        a, a_door = _elevator(shim, sim, "a", 0.0, "b")
        b, b_door = _elevator(shim, sim, "b", 10.0, "a")
        _place(sim, jackal, 0.0, 0.0)
        await shim._tick(sim, DT)
        assert a.departing
        assert a_door.state == shim._DoorState.CLOSED
        await shim._tick(sim, DT)
        assert not a.departing
        assert a.dispatched == set()
        assert b.pending_occupants == ()
        assert b.just_arrived == {}
        assert b_door.last_trigger_sim_time == 0.0
        assert b_door.state == shim._DoorState.OPENING
        assert reasons == ["jackal: elevator teleport 'a' -> 'b' failed: NotImplementedError()"]

    on_dummy_sim(body, jackal=True)


def test_arrival_from_an_unknown_cabin_teleports_nobody_and_finishes_the_tick(shim, on_dummy_sim):
    async def body(sim):
        b, b_door = _elevator(shim, sim, "b", 10.0, "ghost")
        b.queue_arrival(shim._Arrival(eta=0.0, source="ghost", occupants=(("rider", (0.0, 0.0)),)))
        _spawn_lamp(sim)
        await shim._tick(sim, DT)
        assert b.just_arrived == {"rider": False}
        assert b_door.state == shim._DoorState.OPENING
        assert sim._light_outputs == {"lamp": (1.0, 0.0)}

    on_dummy_sim(body)


def test_failed_teleport_of_an_untracked_rider_does_not_abort_the_tick(shim, on_dummy_sim):
    async def body(sim):
        b, _ = _elevator(shim, sim, "b", 10.0, "a")
        _elevator(shim, sim, "a", 0.0, "b")
        b.queue_arrival(shim._Arrival(eta=0.0, source="a", occupants=(("rider", (0.5, 0.0)),)))
        _spawn_lamp(sim)
        await shim._tick(sim, DT)
        assert b.just_arrived == {}
        assert sim._light_outputs == {"lamp": (1.0, 0.0)}

    on_dummy_sim(body)


def test_failed_teleport_without_an_incident_callback_only_logs(shim, on_dummy_sim):
    async def body(sim, jackal):
        _elevator(shim, sim, "a", 0.0, "b")
        b, _ = _elevator(shim, sim, "b", 10.0, "a")
        _place(sim, jackal, 0.0, 0.0)
        await shim._tick(sim, DT)
        await shim._tick(sim, DT)
        assert b.just_arrived == {}

    on_dummy_sim(body, jackal=True)


def test_light_output_is_pushed_once_per_change(shim, on_dummy_sim):
    async def body(sim):
        await sim.spawn_lights([_lamp()])
        assert sim._semantics.attach("light", "lamp", _lamp().semantics)
        await shim._tick(sim, DT)
        first = sim._light_outputs["lamp"]
        assert first == (1.0, 0.0)
        await shim._tick(sim, DT)
        assert sim._light_outputs["lamp"] is first
        assert sim.set_semantic_value("lamp", "lit", "false")
        await shim._tick(sim, DT)
        assert sim._light_outputs["lamp"] == (0.0, 0.0)

    on_dummy_sim(body)


def test_light_semantics_without_a_spawned_light_push_nothing(shim, on_dummy_sim):
    async def body(sim):
        assert sim._semantics.attach("light", "lamp", _lamp().semantics)
        await shim._tick(sim, DT)
        assert sim._light_outputs == {}

    on_dummy_sim(body)


def test_frame_light_follows_its_frame_and_ignores_jitter(shim, on_dummy_sim):
    from arena_runtime.shared import Pose, Position

    from arena_simulation_setup.shared import Light

    async def body(sim, jackal):
        head = Light(name="head", fixture="spot", frame=jackal, offset=Pose(position=Position(0.15, 0.0, 0.25)), lumens=1500.0)
        await sim.spawn_lights([head])
        await shim._tick(sim, DT)
        assert sim._light_poses == {}
        _place(sim, jackal, 2.0, 1.0, math.pi / 2)
        await shim._tick(sim, DT)
        placed, yaw = sim._light_poses["head"]
        assert (placed.x, placed.y, placed.z) == pytest.approx((2.0, 1.15, 0.25))
        assert yaw == pytest.approx(math.pi / 2)
        _place(sim, jackal, 2.01, 1.0, math.pi / 2)
        await shim._tick(sim, DT)
        assert sim._light_poses["head"][0] is placed
        _place(sim, jackal, 3.0, 1.0, math.pi / 2)
        await shim._tick(sim, DT)
        assert sim._light_poses["head"][0].x == pytest.approx(3.0)

    on_dummy_sim(body, jackal=True)


def test_attaching_semantics_starts_one_loop_that_ticks_until_stopped(shim, on_dummy_sim):
    async def body(sim):
        await sim.spawn_lights([_lamp()])
        assert sim.attach_semantics("light", "lamp", _lamp().semantics)
        task = sim._mechanism_loop_task
        shim._ensure_loop(sim)
        assert sim._mechanism_loop_task is task
        for _ in range(200):
            if sim._light_outputs:
                break
            await asyncio.sleep(0.01)
        assert sim._light_outputs == {"lamp": (1.0, 0.0)}
        assert not task.done()
        await sim.stop_mechanisms()
        assert task.cancelled()
        assert sim._mechanism_loop_task is None

    on_dummy_sim(body)


def test_reset_restores_an_elevator_without_a_cabin_door(shim, on_dummy_sim):
    from arena_runtime.shared import Elevator, Position

    async def body(sim):
        elevator = Elevator(name="e", position=Position(0.0, 0.0, 0.0), size=[2.0, 2.0, 2.5], destinations=["f"])
        runtime = shim._ElevatorRuntime(elevator=elevator, door_name="e/door", destination_name="f", just_arrived={"r": True}, car=shim._CarState.DEPARTING, dispatched={"r"})
        runtime.queue_arrival(shim._Arrival(eta=3.0, source="f", occupants=(("r", (0.0, 0.0)),)))
        sim._elevator_runtime["e"] = runtime
        shim.reset_mechanisms(sim)
        assert (runtime.arriving_eta, runtime.pending_occupants, runtime.just_arrived, runtime.departing, runtime.dispatched) == (-math.inf, (), {}, False, set())

    on_dummy_sim(body)


def test_pedestrian_rides_to_the_same_offset_in_the_destination_cabin(shim, on_dummy_sim):
    from arena_people_msgs.msg import Pedestrian, Pedestrians
    from arena_runtime.env.constants import Constants
    from arena_runtime.acoustics import AcousticsSimulatorRegistry
    from arena_runtime.humans.noop import NoopHumanSimulator

    async def body(sim, jackal):
        reasons: list[str] = []
        sim.set_incident_callback(reasons.append)
        acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=sim.node, namespace=sim._namespace)
        human = NoopHumanSimulator(node=sim.node, namespace=sim._namespace, simulator=sim, realizer=sim._realizer, acoustics=acoustics)
        sim.attach_human_simulator(human)
        walker = Pedestrian(name="walker")
        walker.pose.position.x = 0.5
        walker.pose.position.y = 0.3
        walker.pose.orientation.w = 1.0
        human._on_arena_peds(Pedestrians(pedestrians=[walker]))
        _place(sim, jackal, -0.4, -0.2)
        a, _ = _elevator(shim, sim, "a", 0.0, "b")
        b, _ = _elevator(shim, sim, "b", 10.0, "a")
        await shim._tick(sim, DT)
        await shim._tick(sim, DT)
        assert a.dispatched == {"walker"}
        assert b.just_arrived == {"walker": False}
        assert {name: xy for name, xy, _ in human.pedestrian_discs()} == {"walker": pytest.approx((10.5, 0.3))}
        assert reasons == ["jackal: elevator teleport 'a' -> 'b' failed: NotImplementedError()"]

    on_dummy_sim(body, jackal=True)


def test_cabin_door_reports_to_the_human_simulator_only_when_it_turns_open_or_stops_being_open(shim, on_dummy_sim):
    from arena_people_msgs.msg import Pedestrian, Pedestrians
    from arena_runtime.env.constants import Constants
    from arena_runtime.acoustics import AcousticsSimulatorRegistry
    from arena_runtime.humans.noop import NoopHumanSimulator

    class DoorLog(NoopHumanSimulator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.door_calls: list[dict[str, bool]] = []

        async def _set_elevator_doors_impl(self, doors):
            self.door_calls.append(dict(doors))
            return True

    def walker_at(x: float) -> Pedestrians:
        walker = Pedestrian(name="walker")
        walker.pose.position.x = x
        walker.pose.orientation.w = 1.0
        return Pedestrians(pedestrians=[walker])

    async def body(sim):
        acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=sim.node, namespace=sim._namespace)
        human = DoorLog(node=sim.node, namespace=sim._namespace, simulator=sim, realizer=sim._realizer, acoustics=acoustics)
        _elevator(shim, sim, "a", 0.0, "b")
        human._on_arena_peds(walker_at(1.6))
        for _ in range(3):
            await shim._tick(sim, DT)
        assert human.door_calls == []
        for _ in range(3):
            await shim._tick(sim, DT)
        assert human.door_calls == [{"a": True}]
        human._on_arena_peds(walker_at(20.0))
        shim.reset_mechanisms(sim)
        await shim._tick(sim, DT)
        await shim._tick(sim, DT)
        assert human.door_calls == [{"a": True}, {"a": False}]

    on_dummy_sim(body)


@pytest.mark.parametrize(("bystander_x", "departs"), [(1.8, True), (1.2, False)])
def test_loaded_cabin_reopens_only_for_a_body_in_its_door_threshold(shim, on_dummy_sim, bystander_x: float, departs: bool):
    from arena_people_msgs.msg import Pedestrian, Pedestrians
    from arena_runtime.env.constants import Constants
    from arena_runtime.acoustics import AcousticsSimulatorRegistry
    from arena_runtime.humans.noop import NoopHumanSimulator

    def at(name: str, x: float, y: float) -> Pedestrian:
        ped = Pedestrian(name=name)
        ped.pose.position.x = x
        ped.pose.position.y = y
        ped.pose.orientation.w = 1.0
        return ped

    async def body(sim):
        acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=sim.node, namespace=sim._namespace)
        human = NoopHumanSimulator(node=sim.node, namespace=sim._namespace, simulator=sim, realizer=sim._realizer, acoustics=acoustics)
        sim.attach_human_simulator(human)
        human._on_arena_peds(Pedestrians(pedestrians=[at("rider", 0.5, 0.3), at("bystander", bystander_x, 0.0)]))
        a, _ = _elevator(shim, sim, "a", 0.0, "b")
        _elevator(shim, sim, "b", 10.0, "a")
        await shim._tick(sim, DT)
        await shim._tick(sim, DT)
        assert a.dispatched == ({"rider"} if departs else set())

    on_dummy_sim(body)


def test_dummy_spawns_stairs_as_a_no_op(on_dummy_sim):
    from arena_runtime.shared import Position

    from arena_simulation_setup.shared import Stairs

    async def body(sim):
        assert await sim.spawn_stairs([Stairs(name="stairs_low", position=Position(10.3, 2.0, 0.0), to="stairs_high")])
        assert sim._stair_runtime == {}
        assert sim._mechanism_loop_task is None
        assert await sim.remove_mechanisms()

    on_dummy_sim(body)
