"""Elevator rides and stair crossings for robots: WorldManager hops, `via` expansion, ride phases, start-level frame."""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

try:
    from arena_rclpy_mixins.Time import Time
    from arena_simulation_setup.shared import Elevator, Stairs
    from arena_simulation_setup.shared.judge import PhaseMonitor, Sample, zone_polygons
    from arena_simulation_setup.shared.task import TaskPhase
    from arena_simulation_setup.tree.World.Scenario import RobotGoal
    from arena_simulation_setup.tree.World import Level, World, Zone
    from arena_simulation_setup.utils.geometry import Orientation, Pose
    from arena_simulation_setup.utils.geometry import Position as GeoPosition

    from arena_runtime.env.robot_manager.elevator_routing import STAIRS_ALIGN_TOLERANCE, STAIRS_APPROACH_TOLERANCE, MIN_RIDE_TOLERANCE, parse_via, ride_phase, route_phases
    from arena_runtime.env.robot_manager.task_runner import TaskRunner
    from arena_runtime.env.world_manager.utils import WorldLayers, WorldMap, WorldOccupancy
    from arena_runtime.env.world_manager.world_manager import WorldManager
    from arena_runtime.shared import Position
    from arena_runtime.env.request import ElevatorLegPhase, ElevatorRidePhase, StairsLegPhase, StairsRidePhase, GoToPhase, PlayGesturePhase, TaskRequest
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

import numpy as np


def _zone(name: str, x0: float, y0: float, x1: float, y1: float, elevators: list | None = None) -> tuple[Zone, list[Elevator]]:
    corners = [GeoPosition(x0, y0), GeoPosition(x1, y0), GeoPosition(x1, y1), GeoPosition(x0, y1)]
    return Zone(name=name, polygon=corners), elevators or []


def _elev(name: str, x: float, y: float, *destinations: str) -> Elevator:
    return Elevator(name=name, position=GeoPosition(x, y, 0.0), destinations=list(destinations))


def _wm(levels: dict[str, list], origins: dict[str, tuple[float, float]] | None = None) -> WorldManager:
    wm = WorldManager.__new__(WorldManager)
    wm._world = World(levels={lid: Level(id=lid, zones=[zone for zone, _ in zones], elevators=[e for _, elevators in zones for e in elevators]) for lid, zones in levels.items()})
    wm._map = WorldMap(
        occupancy=WorldLayers(walls=WorldOccupancy(np.full((1, 1), WorldOccupancy.EMPTY, dtype=np.uint8))),
        origin=Position(x=0.0, y=0.0),
        resolution=0.05,
        time=Time(),
        level_origins=dict(origins or {}),
    )
    wm._multi_map = None
    return wm


def _open_world() -> WorldManager:
    """Single level with a same-level elevator pair (map_empty: elevator_ground <-> elevator_top)."""
    return _wm(
        {
            "0": [
                _zone(
                    "hall",
                    0,
                    0,
                    40,
                    25,
                    [_elev("elevator_ground", 31, 5, "elevator_top"), _elev("elevator_top", 31, 18, "elevator_ground")],
                )
            ]
        }
    )


def _two_storey(origin1: tuple[float, float] = (50.0, 0.0)) -> WorldManager:
    return _wm(
        {
            "0": [_zone("lobby", 0, 0, 20, 20, [_elev("e0", 5, 5, "e1")])],
            "1": [_zone("ward", 0, 0, 20, 20, [_elev("e1", 6, 6, "e0")])],
        },
        origins={"0": (0.0, 0.0), "1": origin1},
    )


_JACKAL = {"robot_radius": 0.3, "goal_tolerance": 1.0}


def _pose(x: float, y: float) -> Pose:
    return Pose(Position(x, y), Orientation.from_yaw(0.0))


# WorldManager hops


def test_same_level_pair_has_no_bfs_route_but_explicit_hop():
    wm = _open_world()
    assert wm.level_hops("0", "0") == []
    hop = wm.elevator_hop("elevator_ground")
    assert (hop.board_name, hop.dest_name) == ("elevator_ground", "elevator_top")
    assert (hop.board_position.x, hop.board_position.y) == (31.0, 5.0)
    assert (hop.dest_position.x, hop.dest_position.y) == (31.0, 18.0)
    assert hop.target == ""
    assert hop.radius == pytest.approx(1.0)


def test_cross_level_route_applies_level_origins():
    wm = _two_storey()
    hops = wm.level_hops("0", "1")
    assert len(hops) == 1
    assert (hops[0].board_position.x, hops[0].board_position.y) == (5.0, 5.0)
    assert (hops[0].dest_position.x, hops[0].dest_position.y) == (56.0, 6.0)


def test_multi_destination_hop_picks_target_token():
    wm = _wm(
        {
            "0": [
                _zone(
                    "hall",
                    0,
                    0,
                    40,
                    40,
                    [
                        _elev("hub", 5, 5, "east", "west"),
                        _elev("east", 35, 5, "hub"),
                        _elev("west", 5, 35, "hub"),
                    ],
                )
            ]
        }
    )
    assert [d[2] for d in wm.elevator_destinations(wm.find_elevator("hub")[1])] == ["east", "west"]
    by_to = wm.elevator_hop("hub", to="west")
    assert (by_to.dest_name, by_to.target) == ("west", "west")
    toward = wm.elevator_hop("hub", toward=Position(36.0, 6.0))
    assert (toward.dest_name, toward.target) == ("east", "east")
    with pytest.raises(ValueError):
        wm.elevator_hop("hub", to="nowhere")
    with pytest.raises(ValueError):
        wm.elevator_hop("no_such_elevator")


def test_multi_destination_bfs_sets_target_only_when_ambiguous():
    wm = _wm(
        {
            "0": [_zone("a", 0, 0, 10, 10, [_elev("e0", 5, 5, "e1", "e2")])],
            "1": [_zone("b", 0, 0, 10, 10, [_elev("e1", 5, 5, "e0")])],
            "2": [_zone("c", 0, 0, 10, 10, [_elev("e2", 5, 5, "e0")])],
        },
        origins={"0": (0.0, 0.0), "1": (20.0, 0.0), "2": (40.0, 0.0)},
    )
    (hop,) = wm.level_hops("0", "2")
    assert (hop.board_name, hop.dest_name, hop.target) == ("e0", "e2", "e2")
    (back,) = wm.level_hops("2", "0")
    assert back.target == ""


# level_of_point frame


def test_level_of_point_is_in_the_abstract_flattened_frame():
    wm = _two_storey(origin1=(50.0, 0.0))
    assert wm.level_of_point(10.0, 10.0) == "0"
    assert wm.level_of_point(60.0, 10.0) == "1"
    assert wm.level_of_point(10.0, 10.0) != "1"
    assert wm.level_of_point(10.0 + 500.0, 10.0) == ""


# via expansion / routing


def test_via_expands_into_ride_then_goal():
    wm = _open_world()
    phases = route_phases([GoToPhase(pose=_pose(20.0, 20.0), via=("elevator_ground",))], Position(2.0, 2.0), wm, **_JACKAL)
    assert [type(p) for p in phases] == [ElevatorLegPhase, ElevatorRidePhase, ElevatorLegPhase, GoToPhase]
    approach, ride, exit_leg, goal = phases
    assert (approach.pose.position.x, approach.pose.position.y) == (pytest.approx(33.2), 5.0)
    assert (exit_leg.pose.position.x, exit_leg.pose.position.y) == (pytest.approx(33.2), 18.0)
    assert approach.tolerance_radius == pytest.approx(1.6)
    assert exit_leg.tolerance_radius == pytest.approx(1.1)
    assert (ride.pose.position.x, ride.pose.position.y) == (pytest.approx(30.45), 5.0)
    assert (ride.target, ride.until, ride.destination) == ("elevator_ground", "robot in elevator_top", "")
    assert ride.tolerance_angle == pytest.approx(math.pi)
    assert ride.tolerance_radius == pytest.approx(0.625)
    assert approach.hold_time == ride.hold_time == exit_leg.hold_time == 0.0
    assert goal.via == () and (goal.pose.position.x, goal.pose.position.y) == (20.0, 20.0)


_DOOR_AXIS = {None: (1.0, 0.0), "+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}


def _sized_pair(size: float, door_side: str | None) -> WorldManager:
    cabins = [
        Elevator(name="lift_a", position=GeoPosition(12.0, 12.0, 0.0), destinations=["lift_b"], size=[size, size, 2.0], door_side=door_side),
        Elevator(name="lift_b", position=GeoPosition(12.0, 30.0, 0.0), destinations=["lift_a"], size=[size, size, 2.0], door_side=door_side),
    ]
    return _wm({"0": [_zone("hall", 0, 0, 40, 40, cabins)]})


def _along_and_across(point: Position, centre: Position, door_side: str | None) -> tuple[float, float]:
    ox, oy = _DOOR_AXIS[door_side]
    dx, dy = point.x - centre.x, point.y - centre.y
    return dx * ox + dy * oy, abs(dx * oy - dy * ox)


@pytest.mark.parametrize(("robot_radius", "size", "door_side"), [(0.2, 2.0, None), (0.7, 2.0, "+y"), (1.4, 4.0, "-x")])
def test_ride_geometry_fits_small_and_large_robots(robot_radius, size, door_side):
    wm = _sized_pair(size, door_side)
    phases = route_phases([GoToPhase(pose=_pose(30.0, 35.0), via=("lift_a",))], Position(2.0, 2.0), wm, robot_radius=robot_radius, goal_tolerance=1.0)
    approach, ride, exit_leg, _goal = phases
    board, dest = Position(12.0, 12.0), Position(12.0, 30.0)
    half_depth = size / 2.0
    along, across = _along_and_across(ride.pose.position, board, door_side)
    assert along <= 0.0 and across == pytest.approx(0.0)
    assert half_depth + along >= robot_radius
    boarding_radius = wm.find_elevator("lift_a")[1].boarding_radius(robot_radius)
    assert 0.0 < ride.tolerance_radius <= boarding_radius
    assert ride.tolerance_radius <= 1.0
    for leg, centre in ((approach, board), (exit_leg, dest)):
        along, across = _along_and_across(leg.pose.position, centre, door_side)
        assert across == pytest.approx(0.0)
        assert along - half_depth >= robot_radius


def test_ride_tolerance_is_clamped_for_a_robot_that_barely_fits():
    wm = _sized_pair(2.0, None)
    phases = route_phases([GoToPhase(pose=_pose(30.0, 35.0), via=("lift_a",))], Position(2.0, 2.0), wm, robot_radius=1.0, goal_tolerance=1.0)
    ride = phases[1]
    assert wm.find_elevator("lift_a")[1].boarding_radius(1.0) <= 0.0
    assert ride.tolerance_radius == pytest.approx(MIN_RIDE_TOLERANCE)
    assert (ride.pose.position.x, ride.pose.position.y) == (pytest.approx(12.0), pytest.approx(12.0))


def _three_storey_chain() -> WorldManager:
    def storey(level: str, dest: str) -> list:
        return [_zone(f"{level}_flat", 0, 0, 8, 4, [Elevator(name=f"{level}_elevator", position=GeoPosition(5.0, 0.0, 0.0), door_side="+y", destinations=[f"{dest}_elevator"])])]

    return _wm({"1": storey("1", "2"), "2": storey("2", "3"), "3": storey("3", "1")}, origins={"1": (0.0, 0.0), "2": (0.0, -11.0), "3": (0.0, -22.0)})


def test_chained_rides_leave_the_delivering_cabin_before_reboarding():
    wm = _three_storey_chain()
    phases = route_phases([GoToPhase(pose=_pose(7.0, -20.4))], Position(6.0, 2.4), wm, **_JACKAL)
    assert [type(p) for p in phases] == [ElevatorLegPhase, ElevatorRidePhase, ElevatorLegPhase, ElevatorLegPhase, ElevatorRidePhase, ElevatorLegPhase, GoToPhase]
    assert [(p.target, p.until) for p in phases if isinstance(p, ElevatorRidePhase)] == [("1_elevator", "robot in 2_elevator"), ("2_elevator", "robot in 3_elevator")]
    exit_leg = phases[2]
    assert (exit_leg.pose.position.x, exit_leg.pose.position.y) == (pytest.approx(5.0), pytest.approx(-8.8))
    assert exit_leg.pose.position.y - exit_leg.tolerance_radius > -11.0 + 1.0
    zones = zone_polygons(wm.world_compacted())
    monitor = PhaseMonitor(exit_leg, "jackal")
    assert monitor.step(_at(0.0, 5.32, -10.34), zones) is False
    assert monitor.step(_at(1.0, 5.0, -9.85), zones) is True


def test_cross_level_goal_via_single_destination_elevator_ends_on_goal_level():
    wm = _two_storey()
    goal = _pose(60.0, 10.0)
    phases = route_phases([GoToPhase(pose=goal)], Position(2.0, 2.0), wm, **_JACKAL)
    rides = [p for p in phases if isinstance(p, ElevatorRidePhase)]
    assert len(rides) == 1
    (ride,) = rides
    assert (ride.target, ride.until, ride.destination) == ("e0", "robot in e1", "")
    goal_level = wm.level_of_point(goal.position.x, goal.position.y)
    assert goal_level == "1"
    assert wm.find_elevator("e1")[0] == goal_level
    exit_leg = phases[phases.index(ride) + 1]
    assert isinstance(exit_leg, ElevatorLegPhase)
    assert wm.level_of_point(exit_leg.pose.position.x, exit_leg.pose.position.y) == goal_level
    assert phases[-1].pose == goal


def test_via_dict_destination_syntax_roundtrip():
    assert parse_via("hub>west") == ("hub", "west")
    assert parse_via("hub") == ("hub", "")


def test_no_via_same_level_is_unchanged():
    wm = _open_world()
    goal = GoToPhase(pose=_pose(20.0, 20.0))
    gesture = PlayGesturePhase(gesture="wave")
    assert route_phases([goal, gesture], Position(2.0, 2.0), wm, **_JACKAL) == [goal, gesture]


def test_cross_level_goal_gets_ride_phase():
    wm = _two_storey()
    phases = route_phases([GoToPhase(pose=_pose(60.0, 10.0))], Position(2.0, 2.0), wm, **_JACKAL)
    assert [type(p) for p in phases] == [ElevatorLegPhase, ElevatorRidePhase, ElevatorLegPhase, GoToPhase]
    assert (phases[1].target, phases[1].until) == ("e0", "robot in e1")


def test_via_on_other_level_is_reached_by_auto_route():
    wm = _wm(
        {
            "0": [_zone("lobby", 0, 0, 20, 20, [_elev("e0", 5, 5, "e1")])],
            "1": [_zone("ward", 0, 0, 40, 20, [_elev("e1", 6, 6, "e0"), _elev("shuttle_a", 10, 10, "shuttle_b"), _elev("shuttle_b", 30, 10, "shuttle_a")])],
        },
        origins={"0": (0.0, 0.0), "1": (50.0, 0.0)},
    )
    phases = route_phases([GoToPhase(pose=_pose(85.0, 15.0), via=("shuttle_a",))], Position(2.0, 2.0), wm, **_JACKAL)
    assert [p.target for p in phases if isinstance(p, ElevatorRidePhase)] == ["e0", "shuttle_a"]
    assert isinstance(phases[-1], GoToPhase) and not isinstance(phases[-1], ElevatorRidePhase)


def test_routing_uses_start_level_not_stale_pose():
    """Reset flows submit before the robot is moved: the start level must come from the request start."""
    wm = _two_storey()
    goal = [GoToPhase(pose=_pose(60.0, 10.0))]
    assert len(route_phases(goal, Position(55.0, 15.0), wm, **_JACKAL)) == 1
    assert len(route_phases(goal, Position(2.0, 2.0), wm, **_JACKAL)) == 4


def test_unknown_via_raises():
    with pytest.raises(ValueError):
        route_phases([GoToPhase(pose=_pose(1.0, 1.0), via=("ghost",))], Position(2.0, 2.0), _open_world(), **_JACKAL)


# ride phase completion


def _at(t: float, x: float, y: float) -> Sample:
    return Sample(t=t, robots={"jackal": (x, y, 0.0)})


def test_ride_phase_completes_only_after_boarding_and_arrival():
    wm = _open_world()
    zones = zone_polygons(wm.world_compacted())
    monitor = PhaseMonitor(ride_phase(wm.elevator_ride(wm.elevator_hop("elevator_ground"), 0.3), Orientation.from_yaw(0.0), 1.0), "jackal")
    assert monitor.step(_at(0.0, 31.0, 18.0), zones) is False
    assert monitor.held is False
    assert monitor.step(_at(1.0, 31.2, 5.1), zones) is False
    assert monitor.held is True
    assert monitor.step(Sample(t=2.0), zones) is False
    assert monitor.step(_at(3.0, 30.9, 18.2), zones) is True


def test_ride_phase_survives_the_recorded_form():
    wm = _open_world()
    zones = zone_polygons(wm.world_compacted())
    replayed = TaskPhase.parse(ride_phase(wm.elevator_ride(wm.elevator_hop("elevator_ground"), 0.3), Orientation.from_yaw(0.0), 1.0).serialize())
    monitor = PhaseMonitor(replayed, "jackal")
    assert monitor.step(_at(0.0, 31.0, 18.0), zones) is False
    assert monitor.step(_at(1.0, 31.2, 5.1), zones) is False
    assert monitor.step(_at(2.0, 30.9, 18.2), zones) is True


# RobotManager integration


class _FakeAdapter:
    signals: frozenset = frozenset()
    signal = None
    routes_levels = False

    def __init__(self):
        self.dispatched: list = []
        self.cabin_opens = 0

    async def dispatch_phase(self, phase, robot):
        self.dispatched.append(phase)

    async def on_cabin_open(self, robot):
        self.cabin_opens += 1

    def on_phase_tick(self, phase, robot):
        pass

    def client_for(self, kind):
        return SimpleNamespace(status=0, reason=None, is_done=lambda: True)


class _MeshAdapter(_FakeAdapter):
    routes_levels = True


class _Logger:
    def get_child(self, _name):
        return self

    def __getattr__(self, _name):
        return lambda *a, **kw: None


def _robot_manager(wm: WorldManager, env_offset: tuple[float, float] = (100.0, 0.0), climbs: tuple[str, ...] = (), adapter_cls: type[_FakeAdapter] = _FakeAdapter):
    from arena_runtime.env.robot_manager.robot_manager import RobotManager
    from arena_runtime.env.request import TaskKind

    writes: list[tuple[str, str, str]] = []
    robot_conf = SimpleNamespace(
        GOAL_TOLERANCE_RADIUS=SimpleNamespace(value=1.0),
        GOAL_TOLERANCE_ANGLE=SimpleNamespace(value=math.pi),
        GOAL_HOLD_TIME=SimpleNamespace(value=0.0),
        GOAL_SIGNAL=SimpleNamespace(value=""),
    )
    doors: dict[str, str] = {}
    node = SimpleNamespace(
        _world_manager=wm,
        _set_semantic=lambda entity, field, value: writes.append((entity, field, value)) or True,
        _semantic_value=lambda entity, field: doors.get(entity, "closed") if field == "cabin_door" else None,
        doors=doors,
        get_logger=lambda: _Logger(),
        on_task_submitted=lambda manager: None,
        ped_names=lambda: [],
        _known_subjects=set(),
        conf=SimpleNamespace(Robot=robot_conf),
    )

    class _RM(RobotManager):
        pose = property(lambda self: self._pose)

    rm = _RM.__new__(_RM)
    rm._NodeInterface__node = node
    rm._environment_manager = SimpleNamespace(
        realize=lambda pose: Pose(Position(pose.position.x + env_offset[0], pose.position.y + env_offset[1]), pose.orientation),
        ezilear=lambda pose: Pose(Position(pose.position.x - env_offset[0], pose.position.y - env_offset[1]), pose.orientation),
    )
    adapter = adapter_cls()
    rm._adapters = {TaskKind.GOTO_POSE: adapter}
    rm._adapter_instances = [adapter]
    rm._adapter = None
    rm._start_pos = _pose(2.0, 2.0)
    rm._runner = TaskRunner("jackal")
    rm._runner.begin_episode()
    rm._placed = True
    rm._unsupported_kinds_logged = set()
    rm._abort_episode = None
    rm._phase_failure = None
    rm._cabin_to_clear = ""
    rm._publish_goal_task = None
    rm._watchers = []
    rm._robot = SimpleNamespace(name="jackal")
    rm._robot_radius = 0.3
    rm._climbs = frozenset(climbs)
    rm._pose = None
    return rm, adapter, writes


def _set_pose(rm, x, y):
    rm._pose = None if x is None else _pose(x, y)


def _tick(rm, t, x, y):
    """Judge one tick with the robot at the abstract (x, y)."""
    zones = zone_polygons(rm.node._world_manager.world_compacted())
    return asyncio.run(rm.tick(_at(t, x, y), zones))


def test_submit_task_routes_before_realize_and_uses_the_given_start():
    wm = _two_storey()
    rm, adapter, _ = _robot_manager(wm)
    asyncio.run(rm.submit_task(TaskRequest(phases=[GoToPhase(pose=_pose(60.0, 10.0))]), _pose(55.0, 15.0)))
    assert len(rm._runner.phases) == 1
    assert rm._runner.phases[0].phase.pose.position.x == pytest.approx(60.0)
    assert adapter.dispatched[0].pose.position.x == pytest.approx(160.0)

    rm2, adapter2, _ = _robot_manager(wm)
    asyncio.run(rm2.submit_task(TaskRequest(phases=[GoToPhase(pose=_pose(60.0, 10.0))])))
    assert isinstance(rm2._runner.phases[0].phase, ElevatorLegPhase)
    ride = rm2._runner.phases[1].phase
    assert isinstance(ride, ElevatorRidePhase)
    assert (ride.target, ride.until) == ("e0", "robot in e1")
    assert ride.pose.position.x == pytest.approx(4.45)
    assert adapter2.dispatched[0].pose.position.x == pytest.approx(107.2)
    assert rm2.goal.position.x == pytest.approx(107.2)


def test_submit_task_without_start_routes_from_the_live_pose():
    wm = _two_storey()
    rm, _, _ = _robot_manager(wm)
    _set_pose(rm, 155.0, 15.0)
    asyncio.run(rm.submit_task(TaskRequest(phases=[GoToPhase(pose=_pose(60.0, 10.0))])))
    assert len(rm._runner.phases) == 1


def test_ride_sets_elevator_target_and_waits_for_teleport():
    wm = _wm(
        {"0": [_zone("hall", 0, 0, 40, 40, [_elev("hub", 5, 5, "east", "west"), _elev("east", 35, 5, "hub"), _elev("west", 5, 35, "hub")])]},
    )
    rm, adapter, writes = _robot_manager(wm, env_offset=(0.0, 0.0))
    _set_pose(rm, 2.0, 2.0)
    asyncio.run(rm.submit_task(TaskRequest(phases=[GoToPhase(pose=_pose(6.0, 30.0), via=("hub",))]), _pose(2.0, 2.0)))
    assert writes == []
    _tick(rm, 1.0, 7.2, 5.0)
    assert rm._runner.active == 1
    assert writes == [("hub", "target", "west")]
    _tick(rm, 2.0, 6.2, 5.0)
    assert rm._runner.active == 1 and len(writes) == 1
    _tick(rm, 3.0, 5.1, 5.0)
    assert rm._runner.active == 1
    assert writes[-1] == ("hub", "target", "west")
    assert len(writes) == 2
    _tick(rm, 4.0, 5.1, 5.0)
    assert len(writes) == 2
    _tick(rm, 5.0, 5.0, 35.2)
    assert writes[-1] == ("hub", "target", "")
    assert rm._runner.phases[1].outcome == "met"
    assert rm._runner.active == 2 and adapter.dispatched[-1].pose.position.x == pytest.approx(7.2)
    _tick(rm, 6.0, 7.2, 35.0)
    assert rm._runner.active == 3 and adapter.dispatched[-1].pose.position.y == pytest.approx(30.0)
    _tick(rm, 7.0, 6.0, 30.0)
    assert asyncio.run(rm.is_done) is True


def test_costmap_clear_runs_once_per_ride_after_the_delivering_door_opens():
    wm = _wm(
        {"0": [_zone("hall", 0, 0, 40, 40, [_elev("hub", 5, 5, "east", "west"), _elev("east", 35, 5, "hub"), _elev("west", 5, 35, "hub")])]},
    )
    rm, adapter, _ = _robot_manager(wm, env_offset=(0.0, 0.0))
    doors = rm.node.doors
    _set_pose(rm, 2.0, 2.0)
    asyncio.run(rm.submit_task(TaskRequest(phases=[GoToPhase(pose=_pose(6.0, 12.0), via=("hub>west", "west"))]), _pose(2.0, 2.0)))
    assert [type(state.phase) for state in rm._runner.phases] == [ElevatorLegPhase, ElevatorRidePhase, ElevatorLegPhase, ElevatorLegPhase, ElevatorRidePhase, ElevatorLegPhase, GoToPhase]
    doors["hub"] = "open"
    _tick(rm, 1.0, 7.2, 5.0)
    _tick(rm, 2.0, 6.2, 5.0)
    _tick(rm, 3.0, 5.1, 5.0)
    doors["hub"] = "closed"
    _tick(rm, 4.0, 5.0, 35.2)
    assert rm._runner.active == 2 and adapter.cabin_opens == 0
    doors["west"] = "opening"
    _tick(rm, 5.0, 5.0, 35.2)
    assert adapter.cabin_opens == 0
    doors["west"] = "open"
    _tick(rm, 6.0, 5.0, 35.2)
    assert adapter.cabin_opens == 1
    _tick(rm, 7.0, 7.2, 35.0)
    _tick(rm, 8.0, 7.2, 35.0)
    assert rm._runner.active == 4 and adapter.cabin_opens == 1
    _tick(rm, 9.0, 6.2, 35.0)
    _tick(rm, 10.0, 5.1, 35.0)
    doors["west"] = "closed"
    _tick(rm, 11.0, 5.0, 5.2)
    assert rm._runner.active == 5 and adapter.cabin_opens == 1
    _tick(rm, 12.0, 5.0, 5.2)
    assert adapter.cabin_opens == 1
    doors["hub"] = "open"
    _tick(rm, 13.0, 5.0, 5.2)
    _tick(rm, 14.0, 7.2, 5.0)
    assert adapter.cabin_opens == 2


def test_ride_outcome_ignores_the_adapter_status():
    wm = _open_world()
    rm, adapter, _ = _robot_manager(wm, env_offset=(0.0, 0.0))
    adapter.client_for = lambda kind: SimpleNamespace(status=6, reason="aborted", is_done=lambda: True)
    asyncio.run(rm.submit_task(TaskRequest(phases=[GoToPhase(pose=_pose(20.0, 20.0), via=("elevator_ground",))]), _pose(2.0, 2.0)))
    _tick(rm, 1.0, 33.2, 5.0)
    _tick(rm, 2.0, 31.0, 5.0)
    _tick(rm, 3.0, 31.0, 18.0)
    assert [state.outcome for state in rm._runner.phases[:2]] == ["failed", "met"]


def test_single_destination_ride_never_writes_target():
    wm = _open_world()
    rm, _, writes = _robot_manager(wm, env_offset=(0.0, 0.0))
    asyncio.run(rm.submit_task(TaskRequest(phases=[GoToPhase(pose=_pose(20.0, 20.0), via=("elevator_ground",))]), _pose(2.0, 2.0)))
    assert writes == []


# scenario schema + frame helper


def test_scenario_via_parsing():
    rg = RobotGoal.parse({"start": [1.0, 1.0], "phases": [{"goto": [20.0, 20.0], "via": ["elevator_ground"]}]})
    (phase,) = rg.phase_list()
    assert phase.via == ("elevator_ground",)
    goto = TaskPhase.parse({"goto": [1.0, 2.0], "via": [{"elevator": "hub", "to": "west"}, "x"]})
    assert goto.via == ("hub>west", "x")
    assert TaskPhase.parse(goto.serialize()).via == goto.via
    with pytest.raises(ValueError):
        TaskPhase.parse({"goto": [1.0, 2.0], "via": [{"lift": "hub"}]})


def test_hop_target_tokens_resolve_in_runtime_elevator():
    """Hop `target` tokens are accepted by the runtime elevator resolver."""
    from arena_runtime.sim._mechanism_shim import _ElevatorRuntime
    from arena_runtime.env.realizer import Realizer

    wm = _wm(
        {"0": [_zone("hall", 0, 0, 40, 40, [_elev("hub", 5, 5, "east", "west"), _elev("east", 35, 5, "hub"), _elev("west", 5, 35, "hub")])]},
    )
    realizer = Realizer(Realizer._Configuration(prefix="env3"), level_origins={"0": (0.0, 0.0)})
    realizer.reset_elevator_levels({"hub": "0", "east": "0", "west": "0"})
    hub = wm.find_elevator("hub")[1]
    runtime = _ElevatorRuntime(elevator=hub, door_name="", destination_name=",".join(realizer._realize_destination(cabin) for cabin in hub.destinations))
    for to, realized in (("east", runtime.destinations[0]), ("west", runtime.destinations[1])):
        token = wm.elevator_hop("hub", to=to).target
        assert runtime.resolve_target(token) == realized


def test_ride_goal_sits_inside_cabin_beyond_goal_tolerance():
    """A robot stopping one goal tolerance short of the ride goal is still inside the cabin."""
    import types

    from arena_runtime.env.world_manager.world_manager import WorldManager
    from arena_runtime.shared import Position

    centre = Position(x=3.0, y=11.0, z=0.0)
    for side, expect in [('-y', (3.0, 11.55)), ('+y', (3.0, 10.45)), ('+x', (2.45, 11.0)), ('-x', (3.55, 11.0))]:
        elevator = types.SimpleNamespace(door_side=side, size=(2.0, 2.0, 2.0))
        target = WorldManager._cabin_back_target(elevator, centre, 0.3)
        assert (round(target.x, 6), round(target.y, 6)) == expect
        ox, oy = {'+x': (1, 0), '-x': (-1, 0), '+y': (0, 1), '-y': (0, -1)}[side]
        stop = (target.x + ox * 1.0, target.y + oy * 1.0)
        assert abs(stop[0] - centre.x) < 1.0 and abs(stop[1] - centre.y) < 1.0


# stair crossings

_STAIRS = frozenset({"stairs"})
_GO1 = {"robot_radius": 0.32, "goal_tolerance": 1.0}


def _stair(name: str, x: float, y: float, mouth_side: str, to: str = "", kind: str = "stairs", top: bool = False) -> Stairs:
    return Stairs(name=name, position=GeoPosition(x, y, 0.0), kind=kind, size=[2.6, 1.4 if kind == "stairs" else 1.0], mouth_side=mouth_side, to=to, top=top)


def _stair_wm(levels: dict[str, list[Stairs]], elevators: dict[str, list[Elevator]] | None = None) -> WorldManager:
    wm = _wm({lid: [_zone(f"floor_{lid}", 0, 0, 14, 10, (elevators or {}).get(lid))] for lid in levels}, origins={lid: (20.0 * i, 0.0) for i, lid in enumerate(levels)})
    for lid, stairs in levels.items():
        wm._world.levels[lid].stairs = list(stairs)
    return wm


def _stairs_demo(elevators: dict[str, list[Elevator]] | None = None) -> WorldManager:
    """The stairs_demo layout: a stair pair, an up and a down escalator between two levels laid out 20 m apart."""
    return _stair_wm(
        {
            "0": [
                _stair("stairs_low", 10.3, 2.0, "-x", "stairs_high"),
                _stair("esc_up_low", 10.3, 5.0, "-x", "esc_up_high", kind="escalator"),
                _stair("esc_down_low", 10.3, 7.2, "-x", kind="escalator"),
            ],
            "1": [
                _stair("stairs_high", 3.7, 2.0, "+x", "stairs_low", top=True),
                _stair("esc_up_high", 3.7, 5.0, "+x", kind="escalator", top=True),
                _stair("esc_down_high", 3.7, 7.2, "+x", "esc_down_low", kind="escalator", top=True),
            ],
        },
        elevators,
    )


def _crossings(phases: list) -> list[str]:
    return [p.target for p in phases if isinstance(p, StairsRidePhase) and p.until is None]


def test_climber_crosses_the_stairs_to_a_goal_on_the_next_level():
    wm = _stairs_demo()
    goal = GoToPhase(pose=_pose(32.0, 8.0))
    phases = route_phases([goal], Position(2.0, 2.0), wm, climbs=("stairs",), **_GO1)
    assert [type(p) for p in phases] == [StairsLegPhase, StairsRidePhase, StairsLegPhase, GoToPhase]
    approach, ride, exit_leg, last = phases
    assert (approach.pose.position.x, approach.pose.position.y) == (pytest.approx(8.0), pytest.approx(2.0))
    assert approach.pose.orientation.to_yaw() == pytest.approx(0.0)
    assert approach.tolerance_radius == pytest.approx(0.25)
    assert (ride.target, ride.until) == ("stairs_high", None)
    assert (ride.pose.position.x, ride.pose.position.y) == (pytest.approx(9.0 + 2.6 - 0.32 - 0.05), pytest.approx(2.0))
    assert ride.tolerance_radius == pytest.approx((0.8 - 0.32 - 0.05) / 2.0)
    assert (exit_leg.pose.position.x, exit_leg.pose.position.y) == (pytest.approx(26.0), pytest.approx(2.0))
    assert math.cos(exit_leg.pose.orientation.to_yaw()) == pytest.approx(1.0)
    assert exit_leg.tolerance_radius == pytest.approx(0.9)
    assert approach.hold_time == ride.hold_time == exit_leg.hold_time == 0.0
    assert last == goal


def test_stairs_run_both_ways():
    wm = _stairs_demo()
    phases = route_phases([GoToPhase(pose=_pose(2.0, 2.0))], Position(30.0, 2.0), wm, climbs=("stairs",), **_GO1)
    approach, ride, exit_leg, _ = phases
    assert _crossings(phases) == ["stairs_low"]
    assert approach.pose.position.x == pytest.approx(26.0)
    assert math.cos(approach.pose.orientation.to_yaw()) == pytest.approx(-1.0)
    assert ride.pose.position.x == pytest.approx(25.0 - 2.6 + 0.32 + 0.05)
    assert exit_leg.pose.position.x == pytest.approx(8.0)
    assert math.cos(exit_leg.pose.orientation.to_yaw()) == pytest.approx(-1.0)


def test_escalators_board_only_on_their_boarding_half():
    wm = _stairs_demo()
    up = route_phases([GoToPhase(pose=_pose(32.0, 8.0))], Position(2.0, 2.0), wm, climbs=("escalator",), **_GO1)
    down = route_phases([GoToPhase(pose=_pose(2.0, 8.0))], Position(30.0, 8.0), wm, climbs=("escalator",), **_GO1)
    assert _crossings(up) == ["esc_up_high"]
    assert _crossings(down) == ["esc_down_low"]


def test_robot_without_climbs_never_routes_through_stairs():
    goal = GoToPhase(pose=_pose(32.0, 8.0))
    assert route_phases([goal], Position(2.0, 2.0), _stairs_demo(), **_GO1) == [goal]
    assert route_phases([goal], Position(2.0, 2.0), _stairs_demo(), climbs=("escalator",), **_GO1)[1].target == "esc_up_high"
    wm = _stairs_demo({"0": [_elev("e0", 2.0, 8.0, "e1")], "1": [_elev("e1", 2.0, 8.0, "e0")]})
    phases = route_phases([goal], Position(2.0, 2.0), wm, **_GO1)
    assert [type(p) for p in phases] == [ElevatorLegPhase, ElevatorRidePhase, ElevatorLegPhase, GoToPhase]


def test_climbed_stair_wins_a_tie_with_an_elevator():
    wm = _stairs_demo({"0": [_elev("e0", 2.0, 8.0, "e1")], "1": [_elev("e1", 2.0, 8.0, "e0")]})
    (stairs,) = wm.level_hops("0", "1", _STAIRS)
    (lift,) = wm.level_hops("0", "1")
    assert (stairs.board_name, stairs.dest_name, stairs.kind) == ("stairs_low", "stairs_high", "stairs")
    assert lift.board_name == "e0"


def test_fewest_hops_beats_a_longer_climb():
    wm = _stair_wm(
        {
            "0": [_stair("s01_low", 10.3, 2.0, "-x", "s01_high")],
            "1": [_stair("s01_high", 3.7, 2.0, "+x", "s01_low", top=True), _stair("s12_low", 10.3, 5.0, "-x", "s12_high")],
            "2": [_stair("s12_high", 3.7, 5.0, "+x", "s12_low", top=True)],
        },
        {"0": [_elev("e0", 2.0, 8.0, "e2")], "2": [_elev("e2", 2.0, 8.0, "e0")]},
    )
    assert [h.board_name for h in wm.level_hops("0", "2", _STAIRS)] == ["e0"]
    assert [h.board_name for h in wm.level_hops("0", "1", _STAIRS)] == ["s01_low"]
    assert [h.board_name for h in wm.level_hops("1", "0", _STAIRS)] == ["s01_high"]
    del wm._world.levels["0"].elevators[0]
    assert [h.board_name for h in wm.level_hops("0", "2", _STAIRS)] == ["s01_low", "s12_low"]
    assert wm.level_hops("0", "2") == []


def test_via_names_a_stair_half():
    wm = _stairs_demo()
    phases = route_phases([GoToPhase(pose=_pose(32.0, 8.0), via=("stairs_low",))], Position(2.0, 2.0), wm, climbs=("stairs",), **_GO1)
    assert [type(p) for p in phases] == [StairsLegPhase, StairsRidePhase, StairsLegPhase, GoToPhase]
    assert phases[-1].via == ()
    assert _crossings(route_phases([GoToPhase(pose=_pose(32.0, 8.0), via=("stairs_low>stairs_high",))], Position(2.0, 2.0), wm, climbs=("stairs",), **_GO1)) == ["stairs_high"]
    assert wm.via_hop("1.stairs_high", climbs=_STAIRS).dest_name == "stairs_low"
    assert (wm.via_position("stairs_low").x, wm.via_position("stairs_low").y) == (pytest.approx(9.0), pytest.approx(2.0))
    for via, climbs in (("stairs_low", ()), ("esc_up_high", ("escalator",)), ("stairs_low>nowhere", ("stairs",))):
        with pytest.raises(ValueError):
            route_phases([GoToPhase(pose=_pose(32.0, 8.0), via=(via,))], Position(2.0, 2.0), wm, climbs=climbs, **_GO1)


def test_stair_into_an_unloaded_level_is_no_route():
    wm = _stairs_demo({"0": [_elev("e0", 2.0, 8.0, "e1")], "1": [_elev("e1", 2.0, 8.0, "e0")]})
    del wm._world.levels["1"]
    assert wm.level_hops("0", "1", frozenset({"stairs", "escalator"})) == []
    with pytest.raises(ValueError):
        wm.stair_hop("stairs_low", climbs=_STAIRS)


def test_via_stair_on_another_level_is_reached_by_auto_route():
    wm = _stairs_demo()
    phases = route_phases([GoToPhase(pose=_pose(2.0, 8.0), via=("esc_down_high",))], Position(2.0, 2.0), wm, climbs=("stairs", "escalator"), **_GO1)
    assert _crossings(phases) == ["stairs_high", "esc_down_low"]


@pytest.mark.parametrize("robot_radius", [0.2, 0.32, 0.45, 0.7])
def test_crossing_goal_lies_past_the_trigger_and_clear_of_the_seam_wall(robot_radius):
    from arena_simulation_setup.shared.world import STAIRS_SEAM_INSET

    wm = _stairs_demo()
    ride = wm.stair_ride(wm.stair_hop("stairs_low", climbs=_STAIRS), robot_radius)
    stair = wm.find_stair("stairs_low")[1]
    along = stair.along(ride.board_target.x, ride.board_target.y)
    assert along >= stair.length - STAIRS_SEAM_INSET
    assert stair.crossed(ride.board_target.x, ride.board_target.y)
    if robot_radius <= 0.4:
        assert along + robot_radius <= stair.length + 1e-9
    phase = route_phases([GoToPhase(pose=_pose(32.0, 8.0))], Position(2.0, 2.0), wm, robot_radius=robot_radius, goal_tolerance=1.0, climbs=("stairs",))[1]
    assert phase.tolerance_radius > 0.0
    assert along - phase.tolerance_radius > stair.length - STAIRS_SEAM_INSET


def test_approach_leg_is_met_only_facing_the_seam():
    wm = _stairs_demo()
    zones = zone_polygons(wm.world_compacted())
    approach = route_phases([GoToPhase(pose=_pose(32.0, 8.0))], Position(2.0, 2.0), wm, climbs=("stairs",), **_GO1)[0]
    assert isinstance(approach, StairsLegPhase)
    assert approach.tolerance_angle == STAIRS_ALIGN_TOLERANCE
    assert approach.tolerance_radius == STAIRS_APPROACH_TOLERANCE
    x, y = approach.pose.position.x, approach.pose.position.y
    assert PhaseMonitor(approach, "jackal").step(Sample(t=0.0, robots={"jackal": (x, y, math.pi / 2)}), zones) is False
    assert PhaseMonitor(approach, "jackal").step(Sample(t=0.0, robots={"jackal": (x, y, 0.0)}), zones) is True


def test_crossing_completes_once_the_robot_lands_on_the_partner_half():
    wm = _stairs_demo()
    zones = zone_polygons(wm.world_compacted())
    ride = route_phases([GoToPhase(pose=_pose(32.0, 8.0))], Position(2.0, 2.0), wm, climbs=("stairs",), **_GO1)[1]
    monitor = PhaseMonitor(ride, "jackal")
    assert monitor.step(_at(0.0, 8.0, 2.0), zones) is False
    assert monitor.step(_at(1.0, 10.0, 2.0), zones) is False
    assert monitor.step(_at(2.0, 22.95, 2.0), zones) is True


def test_submit_task_takes_the_stairs_for_a_climbing_robot():
    wm = _stairs_demo()
    rm, adapter, writes = _robot_manager(wm, env_offset=(0.0, 0.0), climbs=("stairs",))
    asyncio.run(rm.submit_task(TaskRequest(phases=[GoToPhase(pose=_pose(32.0, 8.0))]), _pose(2.0, 2.0)))
    assert [type(state.phase) for state in rm._runner.phases] == [StairsLegPhase, StairsRidePhase, StairsLegPhase, GoToPhase]
    _tick(rm, 1.0, 8.0, 2.0)
    assert rm._runner.active == 1 and adapter.dispatched[-1].pose.position.x == pytest.approx(9.0 + 2.6 - 0.3 - 0.05)
    _tick(rm, 2.0, 10.0, 2.0)
    _tick(rm, 3.0, 22.95, 2.0)
    assert rm._runner.phases[1].outcome == "met"
    assert rm._runner.active == 2 and adapter.dispatched[-1].pose.position.x == pytest.approx(26.0)
    _tick(rm, 4.0, 26.0, 2.0)
    assert rm._runner.active == 3
    assert writes == []

    plain, _, _ = _robot_manager(wm, env_offset=(0.0, 0.0))
    asyncio.run(plain.submit_task(TaskRequest(phases=[GoToPhase(pose=_pose(32.0, 8.0))]), _pose(2.0, 2.0)))
    assert len(plain._runner.phases) == 1


# planners that cross stairs themselves


def test_walked_stairs_route_without_phases():
    goal = GoToPhase(pose=_pose(32.0, 8.0))
    assert route_phases([goal], Position(2.0, 2.0), _stairs_demo(), walks=("stairs",), **_GO1) == [goal]
    assert route_phases([GoToPhase(pose=_pose(32.0, 8.0), via=("stairs_low",))], Position(2.0, 2.0), _stairs_demo(), walks=("stairs",), **_GO1) == [goal]


def test_walking_stairs_still_rides_elevators_where_no_walked_stair_leads():
    wm = _stairs_demo({"0": [_elev("e0", 2.0, 8.0, "e1")], "1": [_elev("e1", 2.0, 8.0, "e0")]})
    goal = GoToPhase(pose=_pose(32.0, 8.0))
    assert route_phases([goal], Position(2.0, 2.0), wm, walks=("stairs",), **_GO1) == [goal]
    phases = route_phases([goal], Position(2.0, 2.0), wm, walks=("escalator",), **_GO1)
    assert phases == [goal]
    down = route_phases([GoToPhase(pose=_pose(2.0, 2.0))], Position(30.0, 2.0), wm, walks=("escalator",), **_GO1)
    assert [type(p) for p in down] == [GoToPhase]
    lift = route_phases([goal], Position(2.0, 2.0), wm, walks=(), **_GO1)
    assert [type(p) for p in lift] == [ElevatorLegPhase, ElevatorRidePhase, ElevatorLegPhase, GoToPhase]


def test_walked_stair_then_elevator_keeps_only_the_ride():
    wm = _stair_wm(
        {
            "0": [_stair("s01_low", 10.3, 2.0, "-x", "s01_high")],
            "1": [_stair("s01_high", 3.7, 2.0, "+x", "s01_low", top=True)],
            "2": [],
        },
        {"1": [_elev("e1", 2.0, 8.0, "e2")], "2": [_elev("e2", 2.0, 8.0, "e1")]},
    )
    phases = route_phases([GoToPhase(pose=_pose(45.0, 5.0))], Position(2.0, 2.0), wm, walks=("stairs",), **_GO1)
    assert [type(p) for p in phases] == [ElevatorLegPhase, ElevatorRidePhase, ElevatorLegPhase, GoToPhase]
    assert phases[1].target == "e1"


def test_submit_task_rides_elevators_but_leaves_stairs_to_a_level_routing_planner():
    wm = _stairs_demo({"0": [_elev("e0", 2.0, 8.0, "e1")], "1": [_elev("e1", 2.0, 8.0, "e0")]})
    climber, _, _ = _robot_manager(wm, env_offset=(0.0, 0.0), climbs=("stairs",), adapter_cls=_MeshAdapter)
    asyncio.run(climber.submit_task(TaskRequest(phases=[GoToPhase(pose=_pose(32.0, 8.0))]), _pose(2.0, 2.0)))
    assert [type(state.phase) for state in climber._runner.phases] == [GoToPhase]

    wheeled, adapter, _ = _robot_manager(wm, env_offset=(0.0, 0.0), adapter_cls=_MeshAdapter)
    asyncio.run(wheeled.submit_task(TaskRequest(phases=[GoToPhase(pose=_pose(32.0, 8.0))]), _pose(2.0, 2.0)))
    assert [type(state.phase) for state in wheeled._runner.phases] == [ElevatorLegPhase, ElevatorRidePhase, ElevatorLegPhase, GoToPhase]
    assert not any(isinstance(state.phase, (StairsLegPhase, StairsRidePhase)) for state in wheeled._runner.phases)
    assert adapter.dispatched[0].pose.position.x == pytest.approx(wheeled._runner.phases[0].phase.pose.position.x)


def test_an_adapter_fails_its_active_phase_on_the_next_tick():
    wm = _two_storey()
    rm, _, _ = _robot_manager(wm, env_offset=(0.0, 0.0))
    asyncio.run(rm.submit_task(TaskRequest(phases=[GoToPhase(pose=_pose(10.0, 10.0), on_failure="stop_task"), GoToPhase(pose=_pose(12.0, 12.0))]), _pose(2.0, 2.0)))
    rm.fail_phase("no mesh path")
    _tick(rm, 1.0, 2.0, 2.0)
    first = rm._runner.phases[0]
    assert (first.outcome, first.reason) == ("failed", "no mesh path")
    assert rm._runner.active is None


def test_a_phase_failure_applies_only_to_the_phase_it_was_raised_on():
    wm = _two_storey()
    rm, adapter, _ = _robot_manager(wm, env_offset=(0.0, 0.0))
    asyncio.run(rm.submit_task(TaskRequest(phases=[GoToPhase(pose=_pose(10.0, 10.0)), GoToPhase(pose=_pose(12.0, 12.0))]), _pose(2.0, 2.0)))
    rm.fail_phase("no mesh path")
    _tick(rm, 1.0, 2.0, 2.0)
    assert rm._runner.phases[0].outcome == "failed"
    assert rm._runner.active == 1 and adapter.dispatched[-1].pose.position.x == pytest.approx(12.0)
    _tick(rm, 2.0, 2.0, 2.0)
    assert rm._runner.phases[1].outcome == "pending"
    rm.fail_phase("stale")
    rm.begin_episode()
    rm._placed = True
    asyncio.run(rm.submit_task(TaskRequest(phases=[GoToPhase(pose=_pose(10.0, 10.0))]), _pose(2.0, 2.0)))
    _tick(rm, 3.0, 2.0, 2.0)
    assert rm._runner.phases[0].outcome == "pending"
