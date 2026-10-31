from __future__ import annotations

import asyncio
import math
import typing
import uuid

import numpy as np
import pytest

CLEAR = (2.0, 2.0)
IN_MAP = (8.5, 8.5)
ON_WALL = (2.5, 5.5)
CRATE = (5.5, 2.5)
BY_PED = (2.0, 8.2)
HALF_WIDTH = 0.2078
PED_RADIUS = 0.3


async def _until(pred: typing.Callable[[], bool], what: str, timeout: float = 20.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        await asyncio.sleep(0.01)


class Stage:
    def __init__(self, node, env, probe) -> None:
        import std_msgs.msg

        self.node = node
        self.env = env
        self.probe = probe
        self.reset_end = probe.create_publisher(std_msgs.msg.Empty, "/reset_end", 1)
        self.reset_start = probe.create_publisher(std_msgs.msg.Empty, "/reset_start", 1)
        self.trackers: list = []
        self.armed_ns = 0
        self.listened: dict[int, tuple] = {}

    def robot_manager(self):
        from arena_rclpy_mixins.shared import Namespace
        from arena_robots.Robot import RobotIdentifier

        from task_generator.manager.robot_manager.robot_manager import RobotManager
        from task_generator.shared import Pose, Robot

        robot = Robot(name="jackal", pose=Pose(), model=RobotIdentifier.parse("jackal"), adapters={"mobile": "none"}, extra={})
        rm = RobotManager(node=self.node, namespace=Namespace(self.node.get_namespace())(self.node.get_name()), environment_manager=self.env, robot=robot)
        self.trackers.append(rm._adapter._collision)
        return rm, rm._adapter._collision

    def tracker(self, rm, polygons: dict, footprint: list[list[float]] | None):
        from task_generator.manager.robot_manager.collision_tracker import CollisionTrackerNode

        tracker = CollisionTrackerNode(rm, polygons, footprint=footprint)
        self.node.executor.add_node(tracker)
        self.trackers.append(tracker)
        return tracker

    def place(self, rm, x: float, y: float, yaw: float = 0.0, after: float = 1.0) -> None:
        import geometry_msgs.msg

        t = geometry_msgs.msg.TransformStamped()
        t.header.frame_id = "map"
        t.header.stamp.sec, t.header.stamp.nanosec = divmod(self.armed_ns + round(after * 1e9), 10**9)
        t.child_frame_id = rm.base_frame
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.rotation.z = math.sin(yaw / 2.0)
        t.transform.rotation.w = math.cos(yaw / 2.0)
        self.node.tf_buffer.set_transform(t, "test")

    async def listen(self, tracker, state: bool = False) -> list:
        import arena_robots_msgs.msg
        import nav2_msgs.msg

        received: list = []
        msg_type, topic, publisher = (nav2_msgs.msg.CollisionMonitorState, "collision_monitor_state", tracker._pub_state.publisher) if state else (arena_robots_msgs.msg.CollisionEvents, "collision_events", tracker._pub_events)
        self.probe.create_subscription(msg_type, f"{tracker.get_namespace()}/{topic}", received.append, 10)
        await _until(lambda: publisher.get_subscription_count() > 0, f"the {topic} subscription")
        self.listened[id(received)] = (msg_type, publisher)
        return received

    async def arm(self, tracker) -> None:
        import std_msgs.msg

        await _until(lambda: self.reset_end.get_subscription_count() >= len(self.trackers), "the reset_end subscriptions")
        self.reset_end.publish(std_msgs.msg.Empty())
        await _until(lambda: tracker._valid_after is not None, "reset_end to arm the tracker")
        self.armed_ns = tracker._valid_after.nanoseconds

    async def peds(self, tracker, *peds: tuple[str, float, float]) -> None:
        from arena_people_msgs.msg import Pedestrian, Pedestrians

        publisher = self.probe.create_publisher(Pedestrians, "/arena_peds", 10)
        msg = Pedestrians()
        for name, x, y in peds:
            ped = Pedestrian(name=name)
            ped.pose.position.x = x
            ped.pose.position.y = y
            ped.pose.orientation.w = 1.0
            msg.pedestrians.append(ped)
        await _until(lambda: publisher.get_subscription_count() >= len(self.trackers), "the arena_peds subscriptions")
        publisher.publish(msg)
        await _until(lambda: tracker._peds_msg is not None and len(tracker._peds_msg.pedestrians) == len(peds), "the pedestrians to arrive")

    async def tick(self, tracker, received: list) -> list:
        seen = len(received)
        tracker._tick()
        await _until(lambda: len(received) > seen, "a published message")
        return list(received[-1].events)

    async def silent_tick(self, tracker, received: list) -> None:
        msg_type, publisher = self.listened[id(received)]
        seen = len(received)
        tracker._tick()
        publisher.publish(msg_type())
        await _until(lambda: len(received) > seen, "the marker after a silent tick")
        assert received[seen:] == [msg_type()]
        received.pop()


@pytest.fixture
def world_dir(tmp_path):
    from arena_simulation_setup.tree import DynamicPaths

    crate = tmp_path / "assets" / "Common" / "Object" / "crate"
    crate.mkdir(parents=True)
    (crate / "annotation.yaml").write_text("bounding_box: [[-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0]]\n")
    (tmp_path / "assets" / "Common" / "Material" / "plaster").mkdir(parents=True)
    previous = DynamicPaths.WORLD.path
    DynamicPaths.WORLD.path = tmp_path
    try:
        yield tmp_path
    finally:
        DynamicPaths.WORLD.path = previous


@pytest.fixture
def on_stage(world_dir):
    def run(body: typing.Callable[[Stage], typing.Awaitable[None]]) -> None:
        import rclpy
        from arena_rclpy_mixins.spin import create_executor
        from arena_runtime.sim.dummy_simulator import DummySimulator

        from task_generator.constants import Constants
        from task_generator.manager.environment_manager import EnvironmentManager
        from task_generator.manager.realizer import Realizer
        from task_generator.node import TaskGenerator
        from task_generator.simulators.acoustics import AcousticsSimulatorRegistry
        from task_generator.simulators.human.noop import NoopHumanSimulator

        async def main() -> None:
            executor = create_executor()
            node = TaskGenerator()
            probe = rclpy.create_node(f"collision_probe_{uuid.uuid4().hex[:8]}")
            executor.add_node(node)
            executor.add_node(probe)
            spin = asyncio.get_running_loop().run_in_executor(None, executor.spin)
            stage = None
            try:
                realizer = Realizer(level_origins={"0": (0.0, 0.0)})
                sim = DummySimulator(node=node, namespace=node._namespace, realizer=realizer)
                acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=node._namespace)
                human = NoopHumanSimulator(node=node, namespace=node._namespace, simulator=sim, realizer=realizer, acoustics=acoustics)
                env = EnvironmentManager(node=node, simulator=sim, human_simulator=human, realizer=realizer)
                await env.spawn_world_obstacles(_level(), "", None, _world_map())
                stage = Stage(node, env, probe)
                await body(stage)
            finally:
                node._heartbeat_stop.set()
                executor.shutdown()
                await spin
                for tracker in stage.trackers if stage is not None else ():
                    tracker.shutdown()
                probe.destroy_node()
                node.destroy_node()

        asyncio.run(main())

    return run


def _level():
    from arena_simulation_setup.shared import Obstacle
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
    from arena_simulation_setup.tree.World import Level
    from arena_simulation_setup.tree.World.model import WallPath
    from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

    crate = Obstacle(name="crate", pose=Pose(Position(*CRATE, 0.0), Orientation.from_yaw(0.0)), model=ObjectIdentifier("crate"))
    return Level(walls=[WallPath(path=[Position(0.0, 5.5), Position(4.0, 5.5)], material="plaster")], objects=[crate])


def _world_map():
    from arena_rclpy_mixins.Time import Time

    from task_generator.manager.world_manager.utils import WorldLayers, WorldMap, WorldOccupancy
    from task_generator.shared import Position

    grid = np.full((10, 10), WorldOccupancy.EMPTY, dtype=np.uint8)
    grid[0:3, 7:10] = WorldOccupancy.FULL
    return WorldMap(occupancy=WorldLayers(walls=WorldOccupancy(grid)), origin=Position(x=0.0, y=0.0), resolution=1.0, time=Time())


def _event(event) -> tuple:
    return event.kind, event.obstacle_id, event.distance, (event.obstacle_position.x, event.obstacle_position.y)


def _spec(name: str, kind: str, *, points=None, radius=None, action=None):
    from arena_robots.caps import PolygonSpec

    return PolygonSpec(name=name, type=kind, points=points, radius=radius, action_type=action, polygon_pub_topic=None, min_points=None, visualize=None, enabled=None, slowdown_ratio=None)


def test_mobile_adapter_tracks_the_robot_footprint_and_cap_polygons(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        assert tracker._footprint_base.bounds == pytest.approx((-0.229, -HALF_WIDTH, 0.229, HALF_WIDTH))
        assert {name: (entry["type"], entry["action_code"]) for name, entry in tracker._poly_cache.items()} == {"StopPolygon": ("polygon", 1), "SlowPolygon": ("circle", 2)}
        assert tracker._poly_cache["SlowPolygon"]["R_c"] == 0.25
        assert tracker.get_namespace() == str(rm.namespace)

    on_stage(body)


def test_map_cell_under_the_footprint_is_a_static_collision_at_the_robot(on_stage):
    import arena_robots_msgs.msg

    from task_generator.manager.collision_grid import MAP_ID

    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        stage.place(rm, *IN_MAP)
        assert [_event(e) for e in await stage.tick(tracker, received)] == [(arena_robots_msgs.msg.CollisionEvent.KIND_STATIC, MAP_ID, 0.0, IN_MAP)]
        assert received[-1].header.frame_id == "map"

    on_stage(body)


def test_authored_wall_under_the_footprint_is_a_wall_collision(on_stage):
    import arena_robots_msgs.msg

    from task_generator.manager.collision_grid import WALL_ID

    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        stage.place(rm, *ON_WALL)
        assert [_event(e) for e in await stage.tick(tracker, received)] == [(arena_robots_msgs.msg.CollisionEvent.KIND_WALL, WALL_ID, 0.0, ON_WALL)]

    on_stage(body)


def test_static_obstacle_collision_reports_the_obstacle_centroid(on_stage):
    import arena_robots_msgs.msg

    async def body(stage: Stage):
        assert set(stage.env.static_polygons) == {"crate"}
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        stage.place(rm, CRATE[0] + 0.1, CRATE[1], math.pi / 2)
        assert [_event(e) for e in await stage.tick(tracker, received)] == [(arena_robots_msgs.msg.CollisionEvent.KIND_STATIC, "crate", 0.0, CRATE)]

    on_stage(body)


def test_free_space_publishes_an_empty_event_list(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        stage.place(rm, *CLEAR)
        assert await stage.tick(tracker, received) == []

    on_stage(body)


def test_pedestrian_within_its_radius_collides_and_nan_pedestrians_are_skipped(on_stage):
    import arena_robots_msgs.msg

    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        await stage.peds(tracker, ("walker", 2.0, 8.6), ("far", 6.0, 6.0), ("lost", math.nan, 8.2))
        stage.place(rm, *BY_PED)
        assert [_event(e) for e in await stage.tick(tracker, received)] == [(arena_robots_msgs.msg.CollisionEvent.KIND_PEDESTRIAN, "walker", 0.0, (2.0, 8.6))]

    on_stage(body)


def test_pedestrian_just_beyond_its_radius_does_not_collide(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        await stage.peds(tracker, ("walker", BY_PED[0], BY_PED[1] + HALF_WIDTH + PED_RADIUS + 0.01))
        stage.place(rm, *BY_PED)
        assert await stage.tick(tracker, received) == []

    on_stage(body)


def test_wall_and_pedestrian_together_yield_both_events_wall_first(on_stage):
    import arena_robots_msgs.msg

    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        await stage.peds(tracker, ("walker", ON_WALL[0] + 0.3, ON_WALL[1]))
        stage.place(rm, *ON_WALL)
        assert [e.kind for e in await stage.tick(tracker, received)] == [arena_robots_msgs.msg.CollisionEvent.KIND_WALL, arena_robots_msgs.msg.CollisionEvent.KIND_PEDESTRIAN]

    on_stage(body)


def test_tracker_ignores_poses_before_the_first_reset_end(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        stage.place(rm, *IN_MAP, after=1e6)
        await stage.silent_tick(tracker, received)
        await stage.arm(tracker)
        assert len(await stage.tick(tracker, received)) == 1

    on_stage(body)


def test_tracker_ignores_poses_stamped_up_to_the_reset_end(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        stage.place(rm, *IN_MAP, after=0.0)
        await stage.silent_tick(tracker, received)
        stage.place(rm, *IN_MAP, after=0.5)
        assert len(await stage.tick(tracker, received)) == 1

    on_stage(body)


def test_reset_start_disarms_the_tracker_until_the_next_reset_end(on_stage):
    import std_msgs.msg

    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        stage.reset_start.publish(std_msgs.msg.Empty())
        await _until(lambda: tracker._valid_after is None, "reset_start to disarm the tracker")
        stage.place(rm, *IN_MAP, after=1e6)
        await stage.silent_tick(tracker, received)
        await stage.arm(tracker)
        assert len(await stage.tick(tracker, received)) == 1

    on_stage(body)


def test_tracker_is_silent_without_a_robot_transform(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        await stage.silent_tick(tracker, received)
        stage.place(rm, *IN_MAP)
        assert len(await stage.tick(tracker, received)) == 1

    on_stage(body)


def test_reset_end_drops_the_pedestrians_of_the_previous_episode(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.peds(tracker, ("walker", 2.0, 8.6))
        await stage.arm(tracker)
        assert tracker._peds_msg is None
        stage.place(rm, *BY_PED)
        assert await stage.tick(tracker, received) == []

    on_stage(body)


def test_monitor_state_prefers_the_strictest_hit_action(on_stage):
    async def body(stage: Stage):
        _, tracker = stage.robot_manager()
        edge = tracker._monitor_state(IN_MAP[0] - 0.22, IN_MAP[1], 0.0)
        assert (edge.polygon_name, edge.action_type) == ("SlowPolygon", 2)
        inside = tracker._monitor_state(*IN_MAP, 0.0)
        assert (inside.polygon_name, inside.action_type) == ("StopPolygon", 1)
        clear = tracker._monitor_state(*CLEAR, 0.0)
        assert (clear.polygon_name, clear.action_type) == ("", 0)

    on_stage(body)


def test_monitor_state_is_published_on_tick(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        states = await stage.listen(tracker, state=True)
        await stage.arm(tracker)
        stage.place(rm, *IN_MAP)
        tracker._tick()
        await _until(lambda: bool(states), "a collision monitor state")
        assert (states[-1].polygon_name, states[-1].action_type) == ("StopPolygon", 1)

    on_stage(body)


def test_monitor_state_counts_pedestrians_and_ranks_unknown_actions_last(on_stage):
    async def body(stage: Stage):
        rm, _ = stage.robot_manager()
        polygons = {
            "wide": _spec("wide", "circle", radius=1.0),
            "near": _spec("near", "polygon", points=[[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]], action="limit"),
            "line": _spec("line", "polygon", points=[[0.0, 0.0], [1.0, 0.0]], action="stop"),
            "open": _spec("open", "polygon", action="stop"),
        }
        tracker = stage.tracker(rm, polygons, footprint=None)
        assert set(tracker._poly_cache) == {"wide", "near"}
        assert tracker._pub_events is None
        await stage.peds(tracker, ("walker", 3.0, 2.0))
        hit = tracker._monitor_state(*CLEAR, 0.0)
        assert (hit.polygon_name, hit.action_type) == ("near", 4)
        turned = tracker._monitor_state(*CLEAR, math.pi)
        assert (turned.polygon_name, turned.action_type) == ("wide", 0)

    on_stage(body)


def test_circle_without_a_radius_is_rejected(on_stage):
    async def body(stage: Stage):
        rm, _ = stage.robot_manager()
        with pytest.raises(ValueError, match="'dot': circle without a radius"):
            stage.tracker(rm, {"dot": _spec("dot", "circle", action="stop")}, footprint=None)

    on_stage(body)


def test_tracker_without_a_footprint_publishes_only_the_monitor_state(on_stage):
    async def body(stage: Stage):
        rm, _ = stage.robot_manager()
        tracker = stage.tracker(rm, {"wide": _spec("wide", "circle", radius=1.0, action="slowdown")}, footprint=[[0.0, 0.0], [1.0, 0.0]])
        assert tracker._footprint_base is None
        assert tracker._pub_events is None
        states = await stage.listen(tracker, state=True)
        await stage.arm(tracker)
        stage.place(rm, *IN_MAP)
        tracker._tick()
        await _until(lambda: any(state.polygon_name == "wide" for state in states), "the monitor state of the footprintless tracker")

    on_stage(body)


def test_collision_fails_the_episode_only_under_fail_on_collision(on_stage):
    from rclpy.parameter import Parameter

    from task_generator.manager.robot_manager import RobotsManager
    from task_generator.manager.world_manager.world_manager_ros import WorldManagerROS
    from task_generator.tasks.task import Task

    async def body(stage: Stage):
        robots = RobotsManager(node=stage.node, environment_manager=stage.env)
        world = WorldManagerROS(node=stage.node, environment_manager=stage.env)
        stage.node._task = Task(node=stage.node, environment_manager=stage.env, robots_manager=robots, world_manager=world)
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        stage.place(rm, *IN_MAP)
        assert len(await stage.tick(tracker, received)) == 1
        assert stage.node._task.abort_reason is None
        assert stage.node.set_parameters([Parameter("task.episode.fail_on_collision", value=True)])[0].successful
        stage.place(rm, *CLEAR, after=2.0)
        assert await stage.tick(tracker, received) == []
        assert stage.node._task.abort_reason is None
        stage.place(rm, *IN_MAP, after=3.0)
        assert len(await stage.tick(tracker, received)) == 1
        assert stage.node._task.abort_reason == "collision"

    on_stage(body)
