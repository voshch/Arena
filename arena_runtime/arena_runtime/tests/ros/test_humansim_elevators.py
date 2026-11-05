"""Elevator cabins and stair halves as teleport zones, door states as zone access and teleports the arena_humansim bridge hands to the engine."""

from __future__ import annotations

import asyncio
import math
import threading
import typing
import uuid

import pytest

pytest.importorskip("rclpy")

from arena_humansim_msgs.srv import NotifyStimulus, SetTeleportZones, SetZoneAccess, TeleportAgents
from arena_simulation_setup.shared import Elevator, Stairs
from arena_simulation_setup.shared.world import STAIRS_SEAM_INSET
from arena_simulation_setup.tree.World.model import Level, World
from arena_simulation_setup.tree.World.view import _ring_free_cabins
from arena_simulation_setup.utils.geometry import Pose, Position
from arena_runtime.env.realizer import Realizer
from arena_runtime.humans import stair_height, stair_running
from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator

ENV_OFFSET = (100.0, 50.0)
LEVEL_ORIGINS = {"0": (0.0, 0.0), "1": (30.0, 0.0), "2": (60.0, 0.0)}


def _realizer() -> Realizer:
    return Realizer(Realizer._Configuration(x=ENV_OFFSET[0], y=ENV_OFFSET[1], prefix="env_0"), level_origins=dict(LEVEL_ORIGINS))


def _adapter(realizer: Realizer) -> ArenaHumanSimulator:
    adapter = object.__new__(ArenaHumanSimulator)
    adapter._realizer = realizer
    return adapter


@pytest.mark.parametrize(
    ("door_side", "doorway", "facing_in"),
    [
        ("+x", (1.45, 0.0), math.pi),
        ("-x", (-1.45, 0.0), 0.0),
        ("+y", (0.0, 1.95), -math.pi / 2),
        ("-y", (0.0, -1.95), math.pi / 2),
    ],
)
def test_cabin_doorway_waits_outside_its_door_side_facing_the_cabin(door_side: str, doorway: tuple[float, float], facing_in: float) -> None:
    realizer = _realizer()
    elevator = realizer.realize(Elevator(name="lift", position=Position(4.0, 2.0), size=[2.0, 3.0, 2.5], door_side=door_side), "1")
    msg = _adapter(realizer)._zone_msg(elevator)

    assert (msg.entry.x, msg.entry.y) == pytest.approx((34.0 + doorway[0], 2.0 + doorway[1]))
    assert math.remainder(msg.entry.theta - facing_in, math.tau) == pytest.approx(0.0)
    assert math.remainder(msg.hold.theta - facing_in - math.pi, math.tau) == pytest.approx(0.0)


def test_cabin_goes_to_the_engine_in_the_laid_out_frame() -> None:
    realizer = _realizer()
    elevator = realizer.realize(Elevator(name="lift", position=Position(4.0, 2.0), size=[2.0, 3.0, 2.5], door_side="+y"), "1")
    msg = _adapter(realizer)._zone_msg(elevator)

    assert msg.name == realizer.prefix("lift", "1")
    assert (msg.hold.x, msg.hold.y) == pytest.approx((34.0, 2.0))
    assert [(p.x, p.y) for p in msg.footprint] == pytest.approx([(33.0, 0.5), (35.0, 0.5), (35.0, 3.125), (33.0, 3.125)])
    assert msg.open is False
    assert msg.delay < 0.0


def test_linked_levels_ring_names_each_next_realized_cabin_as_destination() -> None:
    world = World(levels={level_id: Level(id=level_id, source="floor", elevators=[Elevator(name=f"lift_{level_id}", position=Position(1.0, 1.0), door_side="+x")]) for level_id in LEVEL_ORIGINS})
    _ring_free_cabins(world)
    realizer = _realizer()
    realizer.reset_elevator_levels(world.elevator_levels())
    adapter = _adapter(realizer)
    msgs = [adapter._zone_msg(realizer.realize(elevator, level_id)) for level_id, level in world.levels.items() for elevator in level.elevators]

    assert [list(msg.destinations) for msg in msgs] == [[msgs[1].name], [msgs[2].name], [msgs[0].name]]
    assert [msg.hold.x for msg in msgs] == pytest.approx([1.0, 31.0, 61.0])


class _Engine:
    """In-process stand-in for the engine's zone, teleport and stimulus services, recording every request."""

    def __init__(self, node: typing.Any) -> None:
        self.zones: list[list[str]] = []
        self.requests: list[SetTeleportZones.Request] = []
        self.access: list[dict[str, bool]] = []
        self.teleports: list[TeleportAgents.Request] = []
        self.stimuli: list[tuple[int, str, float]] = []
        node.create_service(SetTeleportZones, str(node.service_namespace(ArenaHumanSimulator.SERVICE_SET_TELEPORT_ZONES)), self._set_zones)
        node.create_service(SetZoneAccess, str(node.service_namespace(ArenaHumanSimulator.SERVICE_SET_ZONE_ACCESS)), self._set_access)
        node.create_service(TeleportAgents, str(node.service_namespace(ArenaHumanSimulator.SERVICE_TELEPORT_AGENTS)), self._teleport)
        node.create_service(NotifyStimulus, str(node.service_namespace(ArenaHumanSimulator.SERVICE_NOTIFY_STIMULUS)), self._notify)

    def _set_zones(self, request: SetTeleportZones.Request, response: SetTeleportZones.Response) -> SetTeleportZones.Response:
        self.zones.append([zone.name for zone in request.zones])
        self.requests.append(request)
        response.success = True
        return response

    def _set_access(self, request: SetZoneAccess.Request, response: SetZoneAccess.Response) -> SetZoneAccess.Response:
        self.access.append(dict(zip(request.name, request.open, strict=True)))
        response.success = True
        return response

    def _teleport(self, request: TeleportAgents.Request, response: TeleportAgents.Response) -> TeleportAgents.Response:
        self.teleports.append(request)
        response.success = True
        return response

    def _notify(self, request: NotifyStimulus.Request, response: NotifyStimulus.Response) -> NotifyStimulus.Response:
        self.stimuli.append((request.agent_id, request.stimulus, request.intensity))
        response.success = True
        return response


def _on_bridge(body: typing.Callable[[ArenaHumanSimulator, _Engine], typing.Awaitable[None]]) -> None:
    import rclpy.executors
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.sim.dummy_simulator import DummySimulator
    from arena_runtime.env.constants import Constants
    from arena_runtime.acoustics import AcousticsSimulatorRegistry

    async def main() -> None:
        suffix = uuid.uuid4().hex[:8]
        namespace = Namespace(f"/humansim_elevators_{suffix}")
        node = ArenaMixinNode(f"humansim_elevators_{suffix}", namespace=str(namespace))
        realizer = _realizer()
        sim = DummySimulator(node=node, namespace=namespace, realizer=realizer)
        acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=namespace)
        bridge = ArenaHumanSimulator(node=node, namespace=namespace, simulator=sim, realizer=realizer, acoustics=acoustics)
        engine = _Engine(node)
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        try:
            for client in (bridge._set_teleport_zones_client, bridge._set_zone_access_client, bridge._teleport_agents_client, bridge._notify_stimulus_client):
                assert await client.ensure(timeout_sec=5.0)
            await body(bridge, engine)
        finally:
            await sim.stop_mechanisms()
            executor.shutdown()
            thread.join(timeout=2.0)
            node.destroy_node()

    asyncio.run(main())


def test_teleport_reaches_the_engine_by_agent_id_in_the_engine_frame() -> None:
    from arena_people_msgs.msg import Pedestrian, Pedestrians

    async def body(bridge: ArenaHumanSimulator, engine: _Engine) -> None:
        bridge._agent_names = {7: "/env_0/walker"}
        walker = Pedestrian(name="/env_0/walker")
        walker.pose.orientation.z = math.sin(math.pi / 4)
        walker.pose.orientation.w = math.cos(math.pi / 4)
        bridge._on_arena_peds(Pedestrians(pedestrians=[walker]))

        assert await bridge.pedestrian_teleport({"/env_0/walker": (131.0, 52.0), "12": (100.5, 49.0)})

        (request,) = engine.teleports
        assert list(request.agent_id) == [7, 12]
        assert [(pose.x, pose.y) for pose in request.pose] == pytest.approx([(31.0, 2.0), (0.5, -1.0)])
        assert [pose.theta for pose in request.pose] == pytest.approx([math.pi / 2, 0.0])
        assert set(bridge._teleport_stamps) == {7, 12}
        assert engine.stimuli == [(7, "teleported", 1.0), (12, "teleported", 1.0)]

    _on_bridge(body)


def test_teleport_of_an_unknown_pedestrian_fails_but_moves_the_known_ones() -> None:
    async def body(bridge: ArenaHumanSimulator, engine: _Engine) -> None:
        bridge._agent_names = {7: "/env_0/walker"}

        assert not await bridge.pedestrian_teleport({"/env_0/walker": (131.0, 52.0), "/env_0/ghost": (100.0, 50.0)})
        assert [list(request.agent_id) for request in engine.teleports] == [[7]]

    _on_bridge(body)


def test_world_cabins_reach_the_engine_and_leave_with_the_world() -> None:
    from arena_runtime.humans.utils import ObstacleLayer

    async def body(bridge: ArenaHumanSimulator, engine: _Engine) -> None:
        realizer = bridge._realizer
        cabins = [realizer.realize(Elevator(name=f"lift_{level_id}", position=Position(1.0, 1.0), door_side="+x"), level_id) for level_id in ("0", "1")]

        assert await bridge.spawn_elevators(cabins)
        await bridge.remove_obstacles(purge=ObstacleLayer.WORLD)
        await bridge.remove_obstacles(purge=ObstacleLayer.WORLD)
        assert await bridge.spawn_elevators([])

        assert engine.zones == [[cabin.name for cabin in cabins], []]

    _on_bridge(body)


def test_door_changes_reach_the_engine_by_cabin_name() -> None:
    async def body(bridge: ArenaHumanSimulator, engine: _Engine) -> None:
        await bridge.elevator_doors_changed({"env_0/0/lift_0": True, "env_0/1/lift_1": False})

        assert engine.access == [{"env_0/0/lift_0": True, "env_0/1/lift_1": False}]

    _on_bridge(body)


def _escalator(realizer: Realizer) -> tuple[Stairs, Stairs]:
    """An up escalator from level 0 to level 1, realized: the boarding half and its exit half."""
    realizer.reset_stair_levels({"esc_up_low": "0", "esc_up_high": "1"})
    low = Stairs(name="esc_up_low", position=Position(10.3, 5.0), kind="escalator", size=[2.6, 1.0], mouth_side="-x", to="esc_up_high")
    high = Stairs(name="esc_up_high", position=Position(3.7, 5.0), kind="escalator", size=[2.6, 1.0], mouth_side="+x", top=True)
    return realizer.realize(low, "0"), realizer.realize(high, "1")


def _stair_adapter(realizer: Realizer, stairs: tuple[Stairs, ...], running: dict[str, bool]) -> ArenaHumanSimulator:
    adapter = _adapter(realizer)
    adapter._known_stairs = stairs
    adapter._stair_running = running
    return adapter


def test_stair_half_goes_to_the_engine_walked_from_the_mouth_to_the_seam() -> None:
    realizer = _realizer()
    low, high = _escalator(realizer)
    adapter = _stair_adapter(realizer, (low, high), {})
    board, exit_half = adapter._stair_zone_msg(low), adapter._stair_zone_msg(high)

    assert (board.name, list(board.destinations), board.kind) == (realizer.prefix("esc_up_low", "0"), [realizer.prefix("esc_up_high", "1")], "escalator")
    assert (board.entry.x, board.entry.y, board.entry.theta) == pytest.approx((9.0, 5.0, 0.0))
    assert (board.hold.x, board.hold.y) == pytest.approx((9.0 + 2.6 - STAIRS_SEAM_INSET, 5.0))
    assert [c for p in board.footprint for c in (p.x, p.y)] == pytest.approx([9.0, 4.5, 11.6, 4.5, 11.6, 5.5, 9.0, 5.5])
    assert (board.open, board.belt) == (True, pytest.approx(0.5))
    assert (list(exit_half.destinations), exit_half.open, exit_half.belt) == ([], True, pytest.approx(-0.5))
    assert (exit_half.entry.x, exit_half.entry.y) == pytest.approx((35.0, 5.0))


def test_half_whose_partner_level_is_not_loaded_has_no_destination() -> None:
    realizer = _realizer()
    low, _high = _escalator(realizer)
    msg = _stair_adapter(realizer, (low,), {})._stair_zone_msg(low)

    assert (list(msg.destinations), msg.open) == ([], True)


@pytest.mark.parametrize("stopped", ["esc_up_low", "esc_up_high"])
def test_stopped_escalator_stills_both_halves_and_closes_its_boarding_half(stopped: str) -> None:
    realizer = _realizer()
    low, high = _escalator(realizer)
    adapter = _stair_adapter(realizer, (low, high), {realizer.prefix(stopped, "0" if stopped == "esc_up_low" else "1"): False})
    board, exit_half = adapter._stair_zone_msg(low), adapter._stair_zone_msg(high)

    assert (board.open, board.belt) == (False, 0.0)
    assert (exit_half.open, exit_half.belt) == (True, 0.0)


def test_stairs_are_open_and_still_whatever_is_reported() -> None:
    realizer = _realizer()
    realizer.reset_stair_levels({"up": "0", "down": "1"})
    up = realizer.realize(Stairs(name="up", position=Position(10.3, 2.0), mouth_side="-x", to="down"), "0")
    down = realizer.realize(Stairs(name="down", position=Position(3.7, 2.0), mouth_side="+x", to="up", top=True), "1")
    for running in ({}, {up.name: True}):
        msgs = [_stair_adapter(realizer, (up, down), running)._stair_zone_msg(stair) for stair in (up, down)]
        assert [(m.kind, m.open, m.belt, list(m.destinations)) for m in msgs] == [("stairs", True, 0.0, [down.name]), ("stairs", True, 0.0, [up.name])]


def test_stair_running_follows_both_halves_of_an_escalator() -> None:
    low, high = _escalator(_realizer())
    other = Stairs(name="other", position=Position(0.0, 0.0), kind="escalator")
    assert stair_running(low, (low, high, other), {}) is True
    assert stair_running(low, (low, high, other), {high.name: False}) is False
    assert stair_running(high, (low, high, other), {low.name: False}) is False
    assert stair_running(high, (low, high, other), {low.name: True, "other": False}) is True


def test_pedestrian_height_follows_the_half_it_stands_on() -> None:
    low, high = _escalator(_realizer())
    assert stair_height((low, high), 109.0 + 0.5, 55.0) == 0.0
    assert stair_height((low, high), 109.0 + 1.3, 55.0) == pytest.approx(1.5 * 0.5 / 1.8)
    assert stair_height((low, high), 109.0 + 2.6, 55.2) == pytest.approx(1.5)
    assert stair_height((low, high), 135.0 - 1.3, 55.0) == pytest.approx(-1.5 * 0.5 / 1.8)
    assert stair_height((low, high), 120.0, 55.0) == 0.0
    assert stair_height((low, high), 110.3, 56.0) == 0.0


def test_pedestrian_height_on_a_ramp_rises_from_its_base() -> None:
    realizer = _realizer()
    ramp = realizer.realize(Stairs(name="ramp", kind="ramp", position=Position(22.0, 5.0, -1.0), size=[24.0, 8.0], rise=1.0), "0")
    mouth = ramp.mouth
    assert stair_height((ramp,), mouth.x, mouth.y) == pytest.approx(-1.0)
    assert stair_height((ramp,), mouth.x + 12.0, mouth.y) == pytest.approx(-0.5)
    assert stair_height((ramp,), mouth.x - 1.0, mouth.y) == 0.0


def test_robots_take_the_height_of_the_ramp_under_their_env_local_pose() -> None:
    from arena_runtime.env.environment_manager import EnvironmentManager

    realizer = _realizer()
    ramp = realizer.realize(Stairs(name="ramp", kind="ramp", position=Position(22.0, 5.0, -1.0), size=[24.0, 8.0], rise=1.0), "0")
    environment = object.__new__(EnvironmentManager)
    environment._realizer = realizer
    environment._ramps = (ramp,)
    mouth = realizer.ezilear(Pose(position=ramp.mouth)).position
    assert environment.ramp_height(Position(mouth.x + 12.0, mouth.y)) == pytest.approx(-0.5)
    assert environment.ramp_height(Position(mouth.x - 1.0, mouth.y)) == 0.0


def test_ramps_send_no_teleport_zone() -> None:
    async def body(bridge: ArenaHumanSimulator, engine: _Engine) -> None:
        low, high = _escalator(bridge._realizer)
        ramp = bridge._realizer.realize(Stairs(name="ramp", kind="ramp", position=Position(22.0, 5.0), size=[24.0, 8.0], rise=1.0), "0")
        bridge.spawn_stairs([low, high, ramp])
        assert await bridge.spawn_elevators([])
        assert [zone.name for zone in engine.requests[0].zones] == [low.name, high.name]

    _on_bridge(body)


def test_escalator_running_changes_resend_every_zone() -> None:
    from arena_runtime.humans.utils import ObstacleLayer

    async def body(bridge: ArenaHumanSimulator, engine: _Engine) -> None:
        low, high = _escalator(bridge._realizer)
        bridge.spawn_stairs([low, high])
        assert await bridge.spawn_elevators([])
        await bridge.stairs_changed({low.name: False})
        await bridge.stairs_changed({low.name: True})
        await bridge.remove_obstacles(purge=ObstacleLayer.WORLD)

        assert [[(zone.name, zone.open, round(zone.belt, 3)) for zone in request.zones] for request in engine.requests] == [
            [(low.name, True, 0.5), (high.name, True, -0.5)],
            [(low.name, False, 0.0), (high.name, True, 0.0)],
            [(low.name, True, 0.5), (high.name, True, -0.5)],
            [],
        ]

    _on_bridge(body)
