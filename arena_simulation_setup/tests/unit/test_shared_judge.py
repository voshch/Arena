from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import shapely
import yaml
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from arena_simulation_setup.shared.conditions import (
    AllNode,
    AnyNode,
    EntityAtom,
    EpisodeCondition,
    HoldNode,
    MembershipAtom,
    NotAtom,
    PlaysAtom,
    atom_subjects,
    node_children,
    parse_atom,
)
from arena_simulation_setup.shared.judge import (
    PARK_DRIFT_M,
    PARK_TURN_RAD,
    WALL_GRID_RESOLUTION_M,
    ConditionMonitor,
    ParkTimer,
    PhaseMonitor,
    Sample,
    FLOAT_TOLERANCE,
    NodeMonitor,
    UntilMonitor,
    WallGrid,
    arrived,
    atom_holds,
    compare,
    node_holds,
    operator_verdict,
    zone_polygons,
)
from arena_simulation_setup.shared.conditions import parse_node, parse_until
from arena_simulation_setup.shared.task import GoToPhase, PlayGesturePhase, TaskPhase
from arena_simulation_setup.shared import Door
from arena_simulation_setup.tree.World import Level, WallPath, Zone
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

SQUARE = shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)])
ZONES = {"kitchen": SQUARE, "hallway": shapely.Polygon([(10, 0), (10, 4), (14, 4), (14, 0)])}


def sample(t=0.0, robot=(1.0, 1.0, 0.0), peds=None, fields=None, robots=None):
    robots = dict(robots or {})
    if robot is not None:
        robots["robot_0"] = robot
    values = dict(fields or {})
    return Sample(t=t, robots=robots, peds=dict(peds or {}), field=lambda entity, name: values.get((entity, name)))


def test_membership_robot_and_ped():
    s = sample(robot=(1.0, 1.0, 0.0), peds={"alice": (12.0, 1.0)})
    assert atom_holds(parse_atom("robot in kitchen"), s, ZONES, "robot_0") is True
    assert atom_holds(parse_atom("robot in hallway"), s, ZONES, "robot_0") is False
    assert atom_holds(parse_atom("alice in hallway"), s, ZONES, "robot_0") is True
    assert atom_holds(parse_atom("not alice in kitchen"), s, ZONES, "robot_0") is True


def test_membership_boundary_inclusive():
    s = sample(robot=(4.0, 2.0, 0.0))
    assert atom_holds(parse_atom("robot in kitchen"), s, ZONES, "robot_0") is True


def test_membership_unknown_inputs():
    s = sample(robot=None)
    assert atom_holds(parse_atom("robot in kitchen"), s, ZONES, "robot_0") is None
    absent = Sample(t=0.0, robots={"robot_0": (1.0, 1.0, 0.0)}, known=frozenset({"alice"}))
    assert atom_holds(parse_atom("alice in kitchen"), absent, ZONES, "robot_0") is False
    assert atom_holds(parse_atom("not alice in kitchen"), absent, ZONES, "robot_0") is True
    assert atom_holds(parse_atom("robot within 2 of alice"), absent, ZONES, "robot_0") is False
    assert atom_holds(parse_atom("robot in attic"), sample(), ZONES, "robot_0") is None
    assert atom_holds(parse_atom("bob in kitchen"), sample(), ZONES, "robot_0") is None
    assert atom_holds(parse_atom("not bob in kitchen"), sample(), ZONES, "robot_0") is None


def test_within_between_robots_and_peds():
    s = sample(robot=(0.0, 0.0, 0.0), peds={"alice": (3.0, 4.0)}, robots={"robot_1": (1.0, 0.0, 0.0)})
    assert atom_holds(parse_atom("robot within 5 of alice"), s, ZONES, "robot_0") is True
    assert atom_holds(parse_atom("robot within 4.9 of alice"), s, ZONES, "robot_0") is False
    assert atom_holds(parse_atom("robot_1 within 1 of robot"), s, ZONES, "robot_0") is True
    assert atom_holds(parse_atom("robot within 1 of carol"), s, ZONES, "robot_0") is None


def test_within_resolves_point_literals_on_either_side():
    s = sample(robot=(12.0, 4.0, 0.0), peds={"alice": (0.0, 0.0)})
    assert atom_holds(parse_atom("robot within 4.2 of 12.1,4.0"), s, ZONES, "robot_0") is True
    assert atom_holds(parse_atom("robot within 4.2 of 20,4"), s, ZONES, "robot_0") is False
    assert atom_holds(parse_atom("not robot within 4.2 of 20,4"), s, ZONES, "robot_0") is True
    assert atom_holds(parse_atom("-3,4 within 5 of alice"), s, ZONES, "robot_0") is True
    assert atom_holds(parse_atom("-3,4 within 4.9 of alice"), s, ZONES, "robot_0") is False


def test_point_literal_needs_no_robot_or_ped():
    s = Sample(t=0.0)
    assert s.point("1.5,-2", "robot_0") == (1.5, -2.0)
    assert atom_holds(parse_atom("robot within 1 of 1.5,-2"), s, ZONES, "robot_0") is None
    assert atom_holds(parse_atom("0,0 within 5 of 3,4"), s, ZONES, "robot_0") is True


def test_until_monitor_meets_on_a_point_literal_dwell():
    monitor = UntilMonitor(parse_until("robot within 1 of 10,0 for 2"), "robot_0")
    assert monitor.step(sample(t=0.0, robot=(0.0, 0.0, 0.0)), ZONES) is False
    assert monitor.step(sample(t=1.0, robot=(9.5, 0.0, 0.0)), ZONES) is False
    assert monitor.step(sample(t=3.0, robot=(10.0, 0.5, 0.0)), ZONES) is True


def test_entity_atom_values():
    s = sample(fields={("door", "open"): "true", ("lift", "eta"): "4.0000001"})
    assert atom_holds(parse_atom("door.open == true"), s, ZONES) is True
    assert atom_holds(parse_atom("door.open == false"), s, ZONES) is False
    assert atom_holds(parse_atom("lift.eta == 4"), s, ZONES) is True
    assert atom_holds(parse_atom("lift.floor == 2"), s, ZONES) is None


def _monitor_verdict(op, p, q):
    cond = EpisodeCondition(op=op, p="a.x == 1", q="b.x == 1" if q is not None else None)
    monitor = ConditionMonitor(cond)
    for i in range(len(p)):
        fields = {("a", "x"): "1" if p[i] else "0", ("b", "x"): "1" if (q is not None and q[i]) else "0"}
        monitor.update(sample(t=float(i), fields=fields), ZONES)
    return monitor.finish()


_bools = st.lists(st.booleans(), min_size=1, max_size=12)


@given(_bools, _bools, st.sampled_from(["always", "never", "eventually", "before", "never_during"]))
@settings(max_examples=300)
def test_monitor_matches_series_verdict(p, q, op):
    n = min(len(p), len(q))
    p, q = p[:n], q[:n]
    binary = op in ("before", "never_during")
    expected = operator_verdict(op, np.array(p), True, np.array(q) if binary else None, binary)
    assert _monitor_verdict(op, p, q if binary else None) is expected


def test_monitor_decides_violations_early():
    monitor = ConditionMonitor(EpisodeCondition(op="never", p="robot in hallway"), "robot_0")
    assert monitor.update(sample(robot=(1.0, 1.0, 0.0)), ZONES) is None
    assert monitor.update(sample(robot=(11.0, 1.0, 0.0)), ZONES) is False
    assert monitor.decided and monitor.verdict is False
    assert monitor.update(sample(robot=(1.0, 1.0, 0.0)), ZONES) is False


def test_monitor_unknown_atom_gives_unknown_verdict():
    monitor = ConditionMonitor(EpisodeCondition(op="always", p="robot in attic"), "robot_0")
    monitor.update(sample(), ZONES)
    assert monitor.finish() is None
    assert ConditionMonitor(EpisodeCondition(op="always", p="robot in kitchen"), "robot_0").finish() is None


_floats = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
_pos_floats = st.floats(min_value=1e-6, max_value=10.0, allow_nan=False, allow_infinity=False)
_angles = st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False)


def _goto(x, y, yaw, radius, angle):
    return GoToPhase(pose=Pose(Position(x, y), Orientation.from_yaw(yaw)), tolerance_radius=radius, tolerance_angle=angle)


@given(_floats, _floats, _angles, _pos_floats, _pos_floats)
@settings(max_examples=100)
def test_arrived_at_exact_goal(gx, gy, gyaw, radius, angle):
    phase = _goto(gx, gy, gyaw, radius, angle)
    assert arrived(phase, sample(robot=(gx, gy, gyaw)), ZONES, "robot_0") is True


@given(_floats, _floats, _angles, _pos_floats)
@settings(max_examples=100)
def test_arrived_far_away_is_false(gx, gy, gyaw, radius):
    assume(radius < 100.0)
    phase = _goto(gx, gy, gyaw, radius, 0.0)
    assert arrived(phase, sample(robot=(gx + 1000.0, gy + 1000.0, gyaw)), ZONES, "robot_0") is False


def test_arrived_without_pose_is_unknown():
    assert arrived(_goto(1.0, 2.0, 0.0, 0.5, 0.0), sample(robot=None), ZONES, "robot_0") is None


def test_arrived_angle_tolerance_and_wrap():
    phase = _goto(0.0, 0.0, math.pi - 0.05, 1.0, 0.2)
    assert arrived(phase, sample(robot=(0.0, 0.0, -math.pi + 0.05)), ZONES, "robot_0") is True
    assert arrived(_goto(1.0, 2.0, 0.0, 1.0, 0.1), sample(robot=(1.0, 2.0, 0.5)), ZONES, "robot_0") is False
    assert arrived(_goto(1.0, 2.0, 0.0, 1.0, 0.0), sample(robot=(1.0, 2.0, math.pi - 0.01)), ZONES, "robot_0") is True


def test_arrived_distance_tolerance():
    assert arrived(_goto(1.0, 2.0, 0.0, 0.5, 0.0), sample(robot=(1.3, 2.0, 0.0)), ZONES, "robot_0") is True
    assert arrived(_goto(1.0, 2.0, 0.0, 0.1, 0.0), sample(robot=(2.0, 2.0, 0.0)), ZONES, "robot_0") is False


def test_arrived_zone_target_uses_membership_not_dispatch_pose():
    phase = GoToPhase(target="kitchen", pose=Pose(Position(1.0, 1.0), Orientation.identity()), tolerance_radius=0.2)
    assert arrived(phase, sample(robot=(3.5, 3.5, 0.0)), ZONES, "robot_0") is True
    assert arrived(phase, sample(robot=(5.0, 1.0, 0.0)), ZONES, "robot_0") is False


def test_arrived_ped_target_uses_tolerance_radius():
    phase = GoToPhase(target="alice", tolerance_radius=1.0)
    assert arrived(phase, sample(robot=(0.0, 0.0, 0.0), peds={"alice": (0.5, 0.5)}), ZONES, "robot_0") is True
    assert arrived(phase, sample(robot=(0.0, 0.0, 0.0), peds={"alice": (2.0, 0.0)}), ZONES, "robot_0") is False
    assert arrived(phase, sample(robot=(0.0, 0.0, 0.0)), ZONES, "robot_0") is None


def test_park_timer_counts_only_while_still():
    timer = ParkTimer()
    assert timer.held_for(0.0, 0.0, 0.0, 10.0) == 0.0
    assert timer.held_for(0.01, 0.0, 0.0, 12.0) == 2.0
    assert timer.held_for(PARK_DRIFT_M * 3, 0.0, 0.0, 13.0) == 0.0
    assert timer.held_for(PARK_DRIFT_M * 3, 0.0, 0.0, 15.0) == 2.0
    assert timer.held_for(PARK_DRIFT_M * 3, 0.0, math.radians(20), 16.0) == 0.0


def test_goto_phase_meets_on_arrival_without_hold():
    monitor = PhaseMonitor(_goto(1.0, 1.0, 0.0, 0.5, 0.0), "robot_0")
    assert monitor.step(sample(t=0.0, robot=(5.0, 5.0, 0.0)), ZONES) is False
    assert monitor.waiting is False
    assert monitor.step(sample(t=1.0, robot=(1.1, 1.0, 0.0)), ZONES) is True
    assert monitor.met_at == 1.0 and monitor.arrived_at == 1.0


def test_goto_phase_park_resets_when_leaving():
    phase = GoToPhase(pose=Pose(Position(1.0, 1.0), Orientation.identity()), tolerance_radius=0.5, tolerance_angle=0.0, hold_time=2.0)
    monitor = PhaseMonitor(phase, "robot_0")
    assert monitor.step(sample(t=0.0, robot=(1.0, 1.0, 0.0)), ZONES) is False
    assert monitor.waiting is True
    assert monitor.step(sample(t=1.0, robot=(1.0, 1.0, 0.0)), ZONES) is False
    assert monitor.step(sample(t=1.5, robot=(3.0, 1.0, 0.0)), ZONES) is False
    assert monitor.waiting is False
    assert monitor.step(sample(t=2.0, robot=(1.0, 1.0, 0.0)), ZONES) is False
    assert monitor.step(sample(t=3.9, robot=(1.0, 1.0, 0.0)), ZONES) is False
    assert monitor.step(sample(t=4.0, robot=(1.0, 1.0, 0.0)), ZONES) is True
    assert monitor.met_at == 4.0


def test_goto_phase_until_latches_arrival():
    phase = GoToPhase(pose=Pose(Position(1.0, 1.0), Orientation.identity()), tolerance_radius=0.5, tolerance_angle=0.0, until="not alice in kitchen")
    monitor = PhaseMonitor(phase, "robot_0")
    assert monitor.step(sample(t=0.0, robot=(1.0, 1.0, 0.0), peds={"alice": (2.0, 2.0)}), ZONES) is False
    assert monitor.waiting is True
    assert monitor.step(sample(t=1.0, robot=(20.0, 20.0, 0.0), peds={"alice": (2.0, 2.0)}), ZONES) is False
    assert monitor.waiting is True
    assert monitor.step(sample(t=2.0, robot=(20.0, 20.0, 0.0), peds={"alice": (12.0, 2.0)}), ZONES) is True


def test_until_unknown_never_completes():
    phase = GoToPhase(pose=Pose(Position(1.0, 1.0), Orientation.identity()), tolerance_radius=0.5, tolerance_angle=0.0, until="bob in kitchen")
    monitor = PhaseMonitor(phase, "robot_0")
    assert monitor.step(sample(t=0.0, robot=(1.0, 1.0, 0.0)), ZONES) is False
    assert monitor.step(sample(t=9.0, robot=(1.0, 1.0, 0.0)), ZONES) is False


def test_gesture_phase_completes_on_action_result_then_until():
    monitor = PhaseMonitor(PlayGesturePhase(gesture="wave", until="door.open == true"), "robot_0")
    assert monitor.step(sample(t=0.0), ZONES, action_done=False) is False
    assert monitor.step(sample(t=1.0), ZONES, action_done=True) is False
    assert monitor.waiting is True
    assert monitor.step(sample(t=2.0, fields={("door", "open"): "true"}), ZONES, action_done=True) is True
    assert PhaseMonitor(PlayGesturePhase(gesture="wave"), "robot_0").step(sample(), ZONES, action_done=True) is True


def test_scoped_conditions_judged_over_the_phase():
    phase = TaskPhase.parse({"goto": [1.0, 1.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "conditions": [{"op": "never", "p": "robot in hallway"}, {"op": "eventually", "p": "robot in kitchen"}]})
    monitor = PhaseMonitor(phase, "robot_0")
    monitor.step(sample(t=0.0, robot=(11.0, 1.0, 0.0)), ZONES)
    monitor.step(sample(t=1.0, robot=(1.0, 1.0, 0.0)), ZONES)
    assert monitor.finish() == [False, True]


def test_zone_polygons_from_level():
    level = Level(
        zones=[
            Zone(name="kitchen", polygon=[Position(0, 0), Position(0, 4), Position(4, 4), Position(4, 0)]),
            Zone(name="line", polygon=[Position(0, 0), Position(1, 1)]),
        ],
        doors=[Door(name="kitchen_door", start=Position(4, 1), end=Position(4, 2))],
    )
    polygons = zone_polygons(level)
    assert set(polygons) == {"kitchen", "kitchen_door"}
    assert polygons["kitchen"].covers(shapely.Point(2, 2))
    assert polygons["kitchen_door"].covers(shapely.Point(4.2, 1.5))


def test_park_timer_jitter_inside_bounds_keeps_the_anchor():
    timer = ParkTimer()
    timer.held_for(1.0, 2.0, 0.0, 10.0)
    timer.held_for(1.0 + 0.8 * PARK_DRIFT_M, 2.0, 0.8 * PARK_TURN_RAD, 11.0)
    assert timer.held_for(1.0, 2.0, -0.8 * PARK_TURN_RAD, 13.0) == 3.0


def test_park_timer_turn_wraps_across_pi():
    timer = ParkTimer()
    timer.held_for(0.0, 0.0, math.pi - 0.01, 0.0)
    assert timer.held_for(0.0, 0.0, -math.pi + 0.01, 4.0) == 4.0


def test_park_timer_reset_restarts_the_hold():
    timer = ParkTimer()
    timer.held_for(0.0, 0.0, 0.0, 0.0)
    timer.reset()
    assert timer.held_for(0.0, 0.0, 0.0, 9.0) == 0.0


def test_signal_phase_is_never_met_by_position():
    monitor = PhaseMonitor(GoToPhase(pose=Pose(Position(1.0, 1.0), Orientation.identity()), tolerance_radius=0.5, tolerance_angle=0.0, hold_time=0.0, signal="arrived"), "robot_0")
    assert monitor.step(sample(t=0.0, robot=(1.0, 1.0, 0.0)), ZONES) is False
    assert monitor.step(sample(t=1.0, robot=(1.0, 1.0, 0.0)), ZONES, signal="stop") is False
    assert monitor.waiting is False and monitor.misfire is None


def test_signal_within_tolerance_meets_the_phase():
    monitor = PhaseMonitor(GoToPhase(pose=Pose(Position(1.0, 1.0), Orientation.identity()), tolerance_radius=0.5, tolerance_angle=0.0, hold_time=5.0, signal="arrived"), "robot_0")
    assert monitor.step(sample(t=0.0, robot=(1.2, 1.0, 0.0)), ZONES) is False
    assert monitor.step(sample(t=1.0, robot=(1.2, 1.0, 0.0)), ZONES, signal="arrived") is True
    assert monitor.met_at == 1.0


def test_signal_far_from_goal_misfires():
    monitor = PhaseMonitor(GoToPhase(pose=Pose(Position(1.0, 1.0), Orientation.identity()), tolerance_radius=0.5, tolerance_angle=0.0, hold_time=0.0, signal="arrived"), "robot_0")
    assert monitor.step(sample(t=1.0, robot=(4.0, 5.0, 0.0)), ZONES, signal="arrived") is False
    assert monitor.misfire == "signaled arrived 5.00 m from goal"
    assert monitor.step(sample(t=2.0, robot=(1.0, 1.0, 0.0)), ZONES, signal="arrived") is False


def test_signal_on_named_target_measures_distance_to_the_target():
    monitor = PhaseMonitor(GoToPhase(target="alice", tolerance_radius=1.0, tolerance_angle=0.0, hold_time=0.0, signal="arrived"), "robot_0")
    assert monitor.step(sample(t=0.0, robot=(0.0, 0.0, 0.0), peds={"alice": (3.0, 4.0)}), ZONES, signal="arrived") is False
    assert monitor.misfire == "signaled arrived 5.00 m from goal"


@pytest.mark.parametrize(
    ("op", "p_ok", "q_ok"),
    [("always", False, True), ("never", False, True), ("eventually", False, True), ("before", True, False), ("before", False, True), ("never_during", True, False)],
)
def test_operator_verdict_with_unresolvable_atom_is_unknown(op, p_ok, q_ok):
    series = np.array([True, False])
    assert operator_verdict(op, series, p_ok, series, q_ok) is None


def test_arrived_on_hold_phase_before_pose_fill_is_unknown():
    phase = GoToPhase(hold_time=2.0)
    assert arrived(phase, sample(robot=(1.0, 1.0, 0.0)), ZONES, "robot_0") is None
    monitor = PhaseMonitor(phase, "robot_0")
    assert monitor.step(sample(t=0.0), ZONES) is False
    assert monitor.step(sample(t=5.0), ZONES) is False
    assert monitor.met is False and monitor.arrived_at is None


def test_met_phase_stays_met_while_conditions_keep_judging():
    phase = TaskPhase.parse({"goto": [1.0, 1.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "conditions": [{"op": "never", "p": "robot in hallway"}]})
    monitor = PhaseMonitor(phase, "robot_0")
    assert monitor.met is False
    assert monitor.step(sample(t=0.0, robot=(1.0, 1.0, 0.0)), ZONES) is True
    assert monitor.met is True
    assert monitor.step(sample(t=1.0, robot=(11.0, 1.0, 0.0)), ZONES) is True
    assert monitor.met_at == 0.0
    assert monitor.finish() == [False]


def test_goto_phase_with_until_keeps_arrival_when_leaving_before_the_park():
    phase = GoToPhase(pose=Pose(Position(1.0, 1.0), Orientation.identity()), tolerance_radius=0.5, tolerance_angle=0.0, hold_time=2.0, until="door.open == true")
    monitor = PhaseMonitor(phase, "robot_0")
    assert monitor.step(sample(t=0.0, robot=(5.0, 5.0, 0.0)), ZONES) is False
    assert monitor.arrived_at is None and monitor.waiting is False
    assert monitor.step(sample(t=1.0, robot=(1.0, 1.0, 0.0)), ZONES) is False
    assert monitor.step(sample(t=2.0, robot=(5.0, 5.0, 0.0)), ZONES) is False
    assert monitor.arrived_at == 1.0 and monitor.waiting is True and monitor.held is False
    assert monitor.step(sample(t=3.0, robot=(1.0, 1.0, 0.0)), ZONES) is False
    assert monitor.step(sample(t=5.0, robot=(1.0, 1.0, 0.0)), ZONES) is False
    assert monitor.held is True
    assert monitor.step(sample(t=6.0, robot=(5.0, 5.0, 0.0), fields={("door", "open"): "true"}), ZONES) is True
    assert monitor.met_at == 6.0 and monitor.arrived_at == 1.0


def test_signal_without_robot_pose_misfires_at_unknown_distance():
    monitor = PhaseMonitor(GoToPhase(pose=Pose(Position(1.0, 1.0), Orientation.identity()), tolerance_radius=0.5, tolerance_angle=0.0, signal="arrived"), "robot_0")
    assert monitor.step(sample(t=0.0, robot=None), ZONES, signal="arrived") is False
    assert monitor.misfire == "signaled arrived inf m from goal"


def test_signal_on_unresolvable_target_without_dispatch_pose_misfires_at_unknown_distance():
    monitor = PhaseMonitor(GoToPhase(target="alice", tolerance_radius=1.0, signal="arrived"), "robot_0")
    assert monitor.step(sample(t=0.0, robot=(0.0, 0.0, 0.0)), ZONES, signal="arrived") is False
    assert monitor.misfire == "signaled arrived inf m from goal"


def test_signal_on_absent_target_measures_distance_to_the_dispatch_pose():
    monitor = PhaseMonitor(GoToPhase(target="alice", pose=Pose(Position(3.0, 4.0), Orientation.identity()), tolerance_radius=1.0, signal="arrived"), "robot_0")
    absent = Sample(t=0.0, robots={"robot_0": (0.0, 0.0, 0.0)}, known=frozenset({"alice"}))
    assert monitor.step(absent, ZONES, signal="arrived") is False
    assert monitor.misfire == "signaled arrived 5.00 m from goal"


RESOLUTION = 0.05
SHAPE = (200, 200)


def wall_grid(x: float | None = None) -> WallGrid:
    """A 10 m x 10 m open grid, with a full-height wall three cells wide at world x when given."""
    cells = np.zeros(SHAPE, dtype=bool)
    if x is not None:
        col = int(x / RESOLUTION)
        cells[:, col - 1 : col + 2] = True
    return WallGrid(cells=cells, origin=(0.0, 0.0), resolution=RESOLUTION)


def test_wall_grid_clear_across_open_space_and_blocked_by_a_wall():
    assert wall_grid().clear((1.0, 5.0), (9.0, 5.0))
    assert not wall_grid(5.0).clear((1.0, 5.0), (9.0, 5.0))
    assert wall_grid(5.0).clear((1.0, 5.0), (4.0, 5.0))
    assert not wall_grid(5.0).clear((1.0, 8.0), (9.0, 8.0))


def test_wall_grid_of_level_leaves_doors_open():
    level = Level(
        zones=[Zone(name="room", polygon=[Position(0, 0), Position(0, 4), Position(4, 4), Position(4, 0)])],
        walls=[WallPath(path=[Position(2, 0), Position(2, 4)])],
        doors=[Door(name="room_door", start=Position(2, 1), end=Position(2, 2))],
    )
    grid = WallGrid.of_level(level, resolution=0.05)
    assert not grid.clear((1.0, 3.0), (3.0, 3.0))
    assert grid.clear((1.0, 1.5), (3.0, 1.5))


def test_wall_grid_of_level_blocks_diagonal_walls_at_every_crossing():
    level = Level(
        zones=[Zone(name="room", polygon=[Position(0, 0), Position(0, 4), Position(4, 4), Position(4, 0)])],
        walls=[WallPath(path=[Position(0, 0), Position(4, 4)])],
    )
    grid = WallGrid.of_level(level, resolution=0.05)
    for k in range(40):
        y = 0.5 + k * 0.07
        assert not grid.clear((0.2, y + 1.0), (3.5, y - 1.0))
        assert not grid.clear((0.3, y + 0.013), (3.7, y + 0.013))
    assert grid.clear((0.5, 3.0), (1.0, 3.5))


def test_wall_grid_of_level_defaults_to_the_shared_resolution():
    level = Level(
        zones=[Zone(name="room", polygon=[Position(0, 0), Position(0, 4), Position(4, 4), Position(4, 0)])],
        walls=[WallPath(path=[Position(2, 0), Position(2, 4)])],
    )
    grid = WallGrid.of_level(level)
    explicit = WallGrid.of_level(level, resolution=WALL_GRID_RESOLUTION_M)
    assert grid.resolution == WALL_GRID_RESOLUTION_M
    assert grid.origin == explicit.origin
    assert np.array_equal(grid.cells, explicit.cells)
    assert grid.cells.shape != WallGrid.of_level(level, resolution=2 * WALL_GRID_RESOLUTION_M).cells.shape


def test_sees_needs_range_and_sight():
    near = Sample(t=0.0, robots={"robot_0": (1.0, 5.0, 0.0)}, peds={"caller": (3.0, 5.0)}, walls=wall_grid(5.0))
    assert atom_holds(parse_atom("robot sees caller within 4"), near, ZONES, "robot_0") is True
    behind = Sample(t=0.0, robots={"robot_0": (1.0, 5.0, 0.0)}, peds={"caller": (8.0, 5.0)}, walls=wall_grid(5.0))
    assert atom_holds(parse_atom("robot sees caller within 4"), behind, ZONES, "robot_0") is False
    assert atom_holds(parse_atom("robot sees caller within 12"), behind, ZONES, "robot_0") is False
    assert atom_holds(parse_atom("caller within 12 of robot"), behind, ZONES, "robot_0") is True
    open_space = Sample(t=0.0, robots={"robot_0": (1.0, 5.0, 0.0)}, peds={"caller": (8.0, 5.0)}, walls=wall_grid())
    assert atom_holds(parse_atom("robot sees caller"), open_space, ZONES, "robot_0") is True


def test_sees_unknown_without_walls_or_subject():
    no_walls = Sample(t=0.0, robots={"robot_0": (1.0, 5.0, 0.0)}, peds={"caller": (3.0, 5.0)})
    assert atom_holds(parse_atom("robot sees caller"), no_walls, ZONES, "robot_0") is None
    assert atom_holds(parse_atom("robot sees caller within 1"), no_walls, ZONES, "robot_0") is False
    assert atom_holds(parse_atom("robot sees bob"), Sample(t=0.0, robots={"robot_0": (1.0, 5.0, 0.0)}, walls=wall_grid()), ZONES, "robot_0") is None
    absent = Sample(t=0.0, robots={"robot_0": (1.0, 5.0, 0.0)}, known=frozenset({"caller"}), walls=wall_grid())
    assert atom_holds(parse_atom("robot sees caller"), absent, ZONES, "robot_0") is False


def test_plays_reads_the_pedestrian_clips():
    s = Sample(t=0.0, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (1.0, 1.0), "bystander": (2.0, 2.0)}, clips={"caller": frozenset({"beckon", "gaze"}), "bystander": frozenset()})
    assert atom_holds(parse_atom("caller plays beckon"), s, ZONES, "robot_0") is True
    assert atom_holds(parse_atom("bystander plays beckon"), s, ZONES, "robot_0") is False
    assert atom_holds(parse_atom("not caller plays wave"), s, ZONES, "robot_0") is True
    assert atom_holds(parse_atom("bob plays beckon"), s, ZONES, "robot_0") is None
    absent = Sample(t=0.0, robots={"robot_0": (0.0, 0.0, 0.0)}, known=frozenset({"caller"}))
    assert atom_holds(parse_atom("caller plays beckon"), absent, ZONES, "robot_0") is False


def test_node_holds_on_a_text_conjunction():
    until = parse_until("robot within 2 of caller and caller plays beckon")
    near = Sample(t=0.0, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (1.0, 0.0)}, clips={"caller": frozenset({"beckon"})})
    assert node_holds(until, near, ZONES, "robot_0") is True
    silent = Sample(t=0.0, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (1.0, 0.0)}, clips={"caller": frozenset()})
    assert node_holds(until, silent, ZONES, "robot_0") is False
    far_and_unknown = Sample(t=0.0, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (5.0, 0.0)})
    assert node_holds(until, far_and_unknown, ZONES, "robot_0") is False
    unknown = Sample(t=0.0, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (1.0, 0.0)})
    assert node_holds(until, unknown, ZONES, "robot_0") is None


def test_until_monitor_requires_the_dwell_without_a_gap():
    monitor = UntilMonitor(parse_until("robot sees caller within 12 and caller plays beckon for 0.5"), "robot_0")
    seen = {"caller": frozenset({"beckon"})}
    assert monitor.step(Sample(t=0.0, robots={"robot_0": (1.0, 5.0, 0.0)}, peds={"caller": (3.0, 5.0)}, clips=seen, walls=wall_grid()), ZONES) is False
    assert monitor.step(Sample(t=0.4, robots={"robot_0": (1.0, 5.0, 0.0)}, peds={"caller": (3.0, 5.0)}, clips=seen, walls=wall_grid()), ZONES) is False
    assert monitor.step(Sample(t=0.6, robots={"robot_0": (1.0, 5.0, 0.0)}, peds={"caller": (3.0, 5.0)}, clips={"caller": frozenset()}, walls=wall_grid()), ZONES) is False
    assert monitor.step(Sample(t=0.8, robots={"robot_0": (1.0, 5.0, 0.0)}, peds={"caller": (3.0, 5.0)}, clips=seen, walls=wall_grid()), ZONES) is False
    assert monitor.step(Sample(t=1.2, robots={"robot_0": (1.0, 5.0, 0.0)}, peds={"caller": (3.0, 5.0)}, clips=seen, walls=wall_grid()), ZONES) is False
    assert monitor.step(Sample(t=1.3, robots={"robot_0": (1.0, 5.0, 0.0)}, peds={"caller": (3.0, 5.0)}, clips=seen, walls=wall_grid()), ZONES) is True


def test_goto_phase_until_dwell_counts_after_arrival():
    phase = GoToPhase(pose=Pose(Position(1.0, 1.0), Orientation.identity()), tolerance_radius=0.5, tolerance_angle=0.0, until="alice in kitchen for 2")
    monitor = PhaseMonitor(phase, "robot_0")
    assert monitor.step(sample(t=0.0, robot=(1.0, 1.0, 0.0), peds={"alice": (2.0, 2.0)}), ZONES) is False
    assert monitor.waiting is True
    assert monitor.step(sample(t=1.0, robot=(1.0, 1.0, 0.0), peds={"alice": (2.0, 2.0)}), ZONES) is False
    assert monitor.step(sample(t=1.5, robot=(1.0, 1.0, 0.0), peds={"alice": (12.0, 2.0)}), ZONES) is False
    assert monitor.step(sample(t=2.0, robot=(1.0, 1.0, 0.0), peds={"alice": (2.0, 2.0)}), ZONES) is False
    assert monitor.step(sample(t=4.0, robot=(1.0, 1.0, 0.0), peds={"alice": (2.0, 2.0)}), ZONES) is True
    assert monitor.met_at == 4.0


def test_node_holds_on_a_text_disjunction_of_clauses():
    until = parse_until("robot sees caller within 12 and caller plays beckon or robot within 2.5 of caller")
    near_and_silent = Sample(t=0.0, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (1.0, 0.0)}, clips={"caller": frozenset()}, walls=wall_grid())
    assert node_holds(until, near_and_silent, ZONES, "robot_0") is True
    far_and_seen = Sample(t=0.0, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (5.0, 0.0)}, clips={"caller": frozenset({"beckon"})}, walls=wall_grid())
    assert node_holds(until, far_and_seen, ZONES, "robot_0") is True
    far_and_silent = Sample(t=0.0, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (5.0, 0.0)}, clips={"caller": frozenset()}, walls=wall_grid())
    assert node_holds(until, far_and_silent, ZONES, "robot_0") is False
    far_without_clips = Sample(t=0.0, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (5.0, 0.0)}, walls=wall_grid())
    assert node_holds(until, far_without_clips, ZONES, "robot_0") is None
    near_without_clips = Sample(t=0.0, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (1.0, 0.0)}, walls=wall_grid())
    assert node_holds(until, near_without_clips, ZONES, "robot_0") is True


def test_until_monitor_times_each_clause_on_its_own():
    monitor = UntilMonitor(parse_until("caller plays beckon for 1 or robot within 2.5 of caller for 0.5"), "robot_0")
    calling = {"caller": frozenset({"beckon"})}
    silent = {"caller": frozenset()}
    assert monitor.step(Sample(t=0.0, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (5.0, 0.0)}, clips=calling), ZONES) is False
    assert monitor.step(Sample(t=0.6, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (1.0, 0.0)}, clips=calling), ZONES) is False
    assert monitor.step(Sample(t=0.9, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (5.0, 0.0)}, clips=calling), ZONES) is False
    assert monitor.step(Sample(t=1.0, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (5.0, 0.0)}, clips=calling), ZONES) is True
    restarted = UntilMonitor(parse_until("caller plays beckon for 1 or robot within 2.5 of caller for 0.5"), "robot_0")
    assert restarted.step(Sample(t=0.0, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (1.0, 0.0)}, clips=silent), ZONES) is False
    assert restarted.step(Sample(t=0.3, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (1.0, 0.0)}, clips=calling), ZONES) is False
    assert restarted.step(Sample(t=0.5, robots={"robot_0": (0.0, 0.0, 0.0)}, peds={"caller": (1.0, 0.0)}, clips=silent), ZONES) is True


def field_sample(t=0.0, fields=None, families=None, robot=(1.0, 1.0, 0.0), peds=None):
    values = dict(fields or {})
    members = dict(families or {})
    return Sample(t=t, robots={"robot_0": robot}, peds=dict(peds or {}), field=lambda entity, name: values.get((entity, name)), family=lambda entity, name: members.get(entity, {}).get(name, {}) if entity in members else None)


def holds(value, s):
    return node_holds(parse_node(value), s, ZONES, "robot_0")


@pytest.mark.parametrize(
    ("op", "bound", "below", "at_edge_below", "at", "at_edge_above", "above"),
    [
        ("eq", 0.5, False, True, True, True, False),
        ("ne", 0.5, True, False, False, False, True),
        ("lt", 0.5, True, False, False, False, False),
        ("le", 0.5, True, True, True, True, False),
        ("gt", 0.5, False, False, False, False, True),
        ("ge", 0.5, False, True, True, True, True),
    ],
)
def test_compare_operators_at_the_tolerance_edges(op, bound, below, at_edge_below, at, at_edge_above, above):
    half = FLOAT_TOLERANCE / 2
    twice = FLOAT_TOLERANCE * 2
    assert compare(str(bound - twice), op, bound) is below
    assert compare(str(bound - half), op, bound) is at_edge_below
    assert compare(str(bound), op, bound) is at
    assert compare(str(bound + half), op, bound) is at_edge_above
    assert compare(str(bound + twice), op, bound) is above


def test_compare_within_is_inclusive_with_tolerance():
    assert compare("0.2", "within", (0.2, 0.4)) is True
    assert compare(str(0.4 + FLOAT_TOLERANCE / 2), "within", (0.2, 0.4)) is True
    assert compare(str(0.2 - FLOAT_TOLERANCE / 2), "within", (0.2, 0.4)) is True
    assert compare(str(0.4 + FLOAT_TOLERANCE * 2), "within", (0.2, 0.4)) is False
    assert compare("0.1", "within", (0.2, 0.4)) is False


def test_compare_strings_and_non_numeric_values():
    assert compare("open", "eq", "open") is True
    assert compare("open", "ne", "open") is False
    assert compare("open", "eq", 1.0) is False
    assert compare("1", "eq", 1.0) is True
    assert compare("open", "ge", 0.5) is None
    assert compare("open", "within", (0.0, 1.0)) is None


def test_field_test_reads_the_sample_field():
    s = field_sample(fields={("wardrobe", "openness.door_left"): "0.9", ("wardrobe", "state"): "open"})
    assert holds({"field": "wardrobe.openness.door_left", "ge": 0.8}, s) is True
    assert holds({"field": "wardrobe.openness.door_left", "lt": 0.8}, s) is False
    assert holds({"field": "wardrobe.state", "eq": "open"}, s) is True
    assert holds({"field": "wardrobe.state", "ge": 0.5}, s) is None
    assert holds({"field": "wardrobe.openness.door_right", "ge": 0.8}, s) is None
    assert holds("wardrobe.openness.door_left == 0.9", s) is True


def test_family_test_quantifies_over_the_members():
    s = field_sample(families={"wardrobe": {"openness": {"door_left": "0.9", "door_right": "0.1"}, "setting": {"knob": "off"}}, "box": {}})
    assert holds({"some": "wardrobe.openness", "ge": 0.5}, s) is True
    assert holds({"each": "wardrobe.openness", "ge": 0.5}, s) is False
    assert holds({"each": "wardrobe.openness", "within": [0.0, 1.0]}, s) is True
    assert holds({"some": "wardrobe.openness", "gt": 0.95}, s) is False
    assert holds({"each": "wardrobe.setting", "eq": "off"}, s) is True
    assert holds({"some": "wardrobe.setting", "ge": 0.5}, s) is None
    assert holds({"each": "wardrobe.setting", "ge": 0.5}, s) is None


def test_family_test_on_an_empty_family_is_false_and_on_an_unknown_entity_is_unknown():
    s = field_sample(families={"wardrobe": {"openness": {"door_left": "0.9"}}, "box": {}})
    assert holds({"some": "wardrobe.pos", "ge": 0.5}, s) is False
    assert holds({"each": "wardrobe.pos", "ge": 0.5}, s) is False
    assert holds({"some": "box.openness", "ge": 0.5}, s) is False
    assert holds({"some": "ghost.openness", "ge": 0.5}, s) is None
    assert holds({"each": "ghost.openness", "ge": 0.5}, s) is None


def test_family_test_mixes_known_and_unknown_members():
    s = field_sample(families={"stove": {"angle": {"a": "0.9", "b": "hot"}}})
    assert holds({"some": "stove.angle", "ge": 0.5}, s) is True
    assert holds({"some": "stove.angle", "ge": 0.95}, s) is None
    assert holds({"each": "stove.angle", "ge": 0.95}, s) is False
    assert holds({"each": "stove.angle", "ge": 0.5}, s) is None


def test_three_valued_not_all_and_any():
    s = field_sample(fields={("a", "x"): "1", ("b", "x"): "0"})
    true, false, unknown = "a.x == 1", "b.x == 1", "c.x == 1"
    assert holds({"not": true}, s) is False
    assert holds({"not": false}, s) is True
    assert holds({"not": unknown}, s) is None
    assert holds({"all": [true, true]}, s) is True
    assert holds({"all": [true, false, unknown]}, s) is False
    assert holds({"all": [true, unknown]}, s) is None
    assert holds({"any": [false, true, unknown]}, s) is True
    assert holds({"any": [false, unknown]}, s) is None
    assert holds({"any": [false, false]}, s) is False


def test_node_holds_refuses_a_node_with_a_hold():
    with pytest.raises(ValueError):
        holds({"all": ["robot in kitchen", {"field": "a.x", "eq": 1, "for": 1}]}, field_sample())


def _door(t, value):
    return field_sample(t=t, fields={("door", "open"): value})


def test_hold_starts_on_the_first_true_sample_and_resets_on_false():
    monitor = NodeMonitor(parse_node({"field": "door.open", "eq": True, "for": 2}), "robot_0")
    assert monitor.step(_door(0.0, "false"), ZONES) is False
    assert monitor.step(_door(1.0, "true"), ZONES) is False
    assert monitor.step(_door(2.5, "true"), ZONES) is False
    assert monitor.step(_door(3.0, "true"), ZONES) is True
    assert monitor.step(_door(3.5, "false"), ZONES) is False
    assert monitor.step(_door(4.0, "true"), ZONES) is False
    assert monitor.step(_door(6.0, "true"), ZONES) is True


def test_hold_returns_unknown_and_resets_on_an_unknown_child():
    monitor = NodeMonitor(parse_node({"field": "door.open", "eq": True, "for": 2}), "robot_0")
    assert monitor.step(_door(0.0, "true"), ZONES) is False
    assert monitor.step(field_sample(t=1.0), ZONES) is None
    assert monitor.step(_door(2.0, "true"), ZONES) is False
    assert monitor.step(_door(3.9, "true"), ZONES) is False
    assert monitor.step(_door(4.0, "true"), ZONES) is True


def test_hold_reset_clears_the_timers():
    monitor = NodeMonitor(parse_node({"field": "door.open", "eq": True, "for": 2}), "robot_0")
    monitor.step(_door(0.0, "true"), ZONES)
    monitor.reset()
    assert monitor.step(_door(2.0, "true"), ZONES) is False
    assert monitor.step(_door(4.0, "true"), ZONES) is True


def test_holds_time_independently_per_node_path():
    node = parse_node({"any": [{"field": "a.x", "eq": 1, "for": 1}, {"field": "a.x", "eq": 1, "for": 3}]})
    monitor = NodeMonitor(node, "robot_0")
    first = field_sample(t=0.0, fields={("a", "x"): "1"})
    assert monitor.step(first, ZONES) is False
    assert monitor.step(field_sample(t=1.0, fields={("a", "x"): "1"}), ZONES) is True
    both = NodeMonitor(parse_node({"all": [{"field": "a.x", "eq": 1, "for": 1}, {"field": "b.x", "eq": 1, "for": 2}]}), "robot_0")
    assert both.step(field_sample(t=0.0, fields={("a", "x"): "1", ("b", "x"): "0"}), ZONES) is False
    assert both.step(field_sample(t=1.0, fields={("a", "x"): "1", ("b", "x"): "1"}), ZONES) is False
    assert both.step(field_sample(t=2.0, fields={("a", "x"): "1", ("b", "x"): "1"}), ZONES) is False
    assert both.step(field_sample(t=3.0, fields={("a", "x"): "1", ("b", "x"): "1"}), ZONES) is True


def test_nested_holds_keep_timing_while_a_sibling_is_false():
    node = parse_node({"all": [{"field": "a.x", "eq": 1}, {"field": "b.x", "eq": 1, "for": 2}], "for": 1})
    monitor = NodeMonitor(node, "robot_0")
    assert monitor.step(field_sample(t=0.0, fields={("a", "x"): "0", ("b", "x"): "1"}), ZONES) is False
    assert monitor.step(field_sample(t=2.0, fields={("a", "x"): "1", ("b", "x"): "1"}), ZONES) is False
    assert monitor.step(field_sample(t=2.5, fields={("a", "x"): "1", ("b", "x"): "1"}), ZONES) is False
    assert monitor.step(field_sample(t=3.0, fields={("a", "x"): "1", ("b", "x"): "1"}), ZONES) is True


def test_hold_inside_not_negates_the_timed_child():
    monitor = NodeMonitor(parse_node({"not": {"field": "door.open", "eq": True, "for": 1}}), "robot_0")
    assert monitor.step(_door(0.0, "true"), ZONES) is True
    assert monitor.step(_door(1.0, "true"), ZONES) is False


def test_condition_monitor_judges_structured_p_and_q():
    condition = EpisodeCondition.parse({"op": "never_during", "p": {"subject": "robot", "in": "hallway"}, "q": {"some": "door.openness", "lt": 0.5}})
    monitor = ConditionMonitor(condition, "robot_0")
    shut = {"door": {"openness": {"left": "0.1", "right": "1.0"}}}
    opened = {"door": {"openness": {"left": "1.0", "right": "1.0"}}}
    assert monitor.update(field_sample(t=0.0, robot=(11.0, 1.0, 0.0), families=opened), ZONES) is None
    assert monitor.update(field_sample(t=1.0, robot=(1.0, 1.0, 0.0), families=shut), ZONES) is None
    assert monitor.update(field_sample(t=2.0, robot=(11.0, 1.0, 0.0), families=shut), ZONES) is False


def test_condition_monitor_times_a_held_p_within_its_scope():
    condition = EpisodeCondition.parse({"op": "eventually", "p": {"field": "door.open", "eq": True, "for": 2}})
    monitor = ConditionMonitor(condition, "robot_0")
    assert monitor.update(_door(0.0, "true"), ZONES) is None
    assert monitor.update(_door(1.0, "false"), ZONES) is None
    assert monitor.update(_door(2.0, "true"), ZONES) is None
    assert monitor.update(_door(4.0, "true"), ZONES) is True


def test_until_monitor_is_met_only_after_the_hold():
    monitor = UntilMonitor(parse_node({"all": [{"subject": "robot", "in": "kitchen"}, {"field": "door.open", "eq": True}], "for": 1.5}), "robot_0")
    assert monitor.step(_door(0.0, "true"), ZONES) is False
    assert monitor.step(_door(1.0, "true"), ZONES) is False
    assert monitor.step(_door(1.5, "true"), ZONES) is True


def test_phase_until_takes_a_structured_condition():
    phase = TaskPhase.parse({"gesture": "wave", "until": {"field": "door.open", "eq": True, "for": 1}})
    monitor = PhaseMonitor(phase, "robot_0")
    assert monitor.step(_door(0.0, "true"), ZONES, action_done=True) is False
    assert monitor.step(_door(0.5, "true"), ZONES, action_done=True) is False
    assert monitor.step(_door(1.0, "true"), ZONES, action_done=True) is True


WORLDS_DIR = Path(__file__).resolve().parents[2] / "worlds"


def _authored_conditions() -> list[tuple[str, str, str]]:
    """(scenario path, kind, text) of every text condition p/q and phase until in the shipped scenarios."""
    found: list[tuple[str, str, str]] = []

    def walk(value: object, path: str) -> None:
        if isinstance(value, dict):
            if "op" in value and "p" in value:
                found.extend((path, "condition", value[key]) for key in ("p", "q") if isinstance(value.get(key), str))
            if isinstance(value.get("until"), str):
                found.append((path, "until", value["until"]))
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    for scenario in sorted(WORLDS_DIR.rglob("scenario.yaml")):
        walk(yaml.safe_load(scenario.read_text(encoding="utf-8")), str(scenario.relative_to(WORLDS_DIR)))
    return found


AUTHORED = _authored_conditions()


def _clause_reference(text: str, s: Sample, zones: dict, robot: str) -> bool | None:
    """The pre-node until truth on one sample: any ` or ` clause whose ` and ` atoms all hold, dwell ignored."""
    verdicts = []
    for clause in text.strip().split(" or "):
        body = clause.strip().rsplit(" for ", 1)[0] if " for " in clause else clause
        atoms = [atom_holds(parse_atom(part), s, zones, robot) for part in body.split(" and ")]
        verdicts.append(False if any(v is False for v in atoms) else None if any(v is None for v in atoms) else True)
    return True if any(v is True for v in verdicts) else None if any(v is None for v in verdicts) else False


def _without_holds(node):
    if isinstance(node, HoldNode):
        return _without_holds(node.node)
    if isinstance(node, AllNode):
        return AllNode(tuple(_without_holds(child) for child in node.nodes))
    if isinstance(node, AnyNode):
        return AnyNode(tuple(_without_holds(child) for child in node.nodes))
    assert not node_children(node)
    return node


def _leaves(node):
    children = node_children(node)
    return [leaf for child in children for leaf in _leaves(child)] if children else [node]


def _synthetic_samples(node, seed: int) -> tuple[list[Sample], dict]:
    """Samples that resolve every subject, zone, field and clip of a node, with drawn positions, values and clips."""
    rng = np.random.default_rng(seed)
    leaves = [leaf.atom if isinstance(leaf, NotAtom) else leaf for leaf in _leaves(node)]
    zones = {leaf.zone: SQUARE for leaf in leaves if isinstance(leaf, MembershipAtom)}
    subjects = sorted({subject for leaf in leaves for subject in atom_subjects(leaf)} - {"robot"})
    expected = {(leaf.entity, leaf.field): leaf.value for leaf in leaves if isinstance(leaf, EntityAtom)}
    clips = {leaf.subject: leaf.clip for leaf in leaves if isinstance(leaf, PlaysAtom)}
    samples = []
    for i in range(6):
        values = {key: value if rng.random() < 0.5 else "other" for key, value in expected.items()}
        playing = {name: frozenset({clip}) if rng.random() < 0.5 else frozenset() for name, clip in clips.items()}
        peds = {name: (float(rng.uniform(-2, 8)), float(rng.uniform(-2, 8))) for name in subjects}
        samples.append(
            Sample(
                t=float(i),
                robots={"robot_0": (float(rng.uniform(-2, 8)), float(rng.uniform(-2, 8)), 0.0)},
                peds=peds,
                field=lambda entity, name, values=values: values.get((entity, name)),
                clips=playing,
                walls=wall_grid(),
            )
        )
    return samples, zones


def test_shipped_scenarios_carry_text_conditions():
    assert {kind for _, kind, _ in AUTHORED} == {"condition", "until"}


@pytest.mark.parametrize(("path", "kind", "text"), AUTHORED, ids=[f"{path}:{text}" for path, _, text in AUTHORED])
def test_shipped_text_conditions_parse_into_nodes_and_judge_like_the_atom_path(path, kind, text):
    node = parse_node(text)
    samples, zones = _synthetic_samples(node, seed=len(text))
    if kind == "condition":
        atom = parse_atom(text)
        assert node == atom, path
        for s in samples:
            assert NodeMonitor(node, "robot_0").step(s, zones) == atom_holds(atom, s, zones, "robot_0"), path
        return
    plain = _without_holds(node)
    for s in samples:
        assert node_holds(plain, s, zones, "robot_0") == _clause_reference(text, s, zones, "robot_0"), path
