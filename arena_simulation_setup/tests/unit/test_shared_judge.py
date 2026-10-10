from __future__ import annotations

import math

import numpy as np
import pytest
import shapely
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from arena_simulation_setup.shared.conditions import (
    EntityAtom,
    EpisodeCondition,
    MembershipAtom,
    NotAtom,
    WithinAtom,
    atom_subjects,
    parse_atom,
    render_atom,
)
from arena_simulation_setup.shared.judge import (
    PARK_DRIFT_M,
    PARK_TURN_RAD,
    ConditionMonitor,
    ParkTimer,
    PhaseMonitor,
    Sample,
    arrived,
    atom_holds,
    operator_verdict,
    zone_polygons,
)
from arena_simulation_setup.shared.task import GoToPhase, PlayGesturePhase, TaskPhase
from arena_simulation_setup.tree.World.World import LevelDescription
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

SQUARE = shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)])
ZONES = {"kitchen": SQUARE, "hallway": shapely.Polygon([(10, 0), (10, 4), (14, 4), (14, 0)])}


def sample(t=0.0, robot=(1.0, 1.0, 0.0), peds=None, fields=None, robots=None):
    robots = dict(robots or {})
    if robot is not None:
        robots["robot_0"] = robot
    values = dict(fields or {})
    return Sample(t=t, robots=robots, peds=dict(peds or {}), field=lambda entity, name: values.get((entity, name)))


def test_parse_not_and_within_atoms():
    assert parse_atom("not robot in kitchen") == NotAtom(MembershipAtom("robot", "kitchen"))
    assert parse_atom("robot within 2.5 of alice") == WithinAtom("robot", 2.5, "alice")
    assert parse_atom("not alice within 1 of robot") == NotAtom(WithinAtom("alice", 1.0, "robot"))


@pytest.mark.parametrize("value", ["not not robot in kitchen", "robot within x of alice", "robot within -1 of alice", "robot within 2 of env_0/alice"])
def test_parse_rejects_malformed_new_forms(value):
    with pytest.raises(ValueError):
        parse_atom(value)


@pytest.mark.parametrize("text", ["door.open == true", "robot in kitchen", "robot within 2 of alice", "not alice in kitchen"])
def test_render_atom_round_trips(text):
    assert render_atom(parse_atom(text)) == text


def test_atom_subjects():
    assert atom_subjects(parse_atom("door.open == true")) == frozenset()
    assert atom_subjects(parse_atom("not robot within 2 of alice")) == {"robot", "alice"}


def test_condition_on_failure_round_trip():
    cond = EpisodeCondition.parse({"op": "never", "p": "robot in hallway", "on_failure": "abort_episode"})
    assert cond.on_failure == "abort_episode"
    assert EpisodeCondition.parse(cond.serialize()) == cond
    assert "on_failure" not in EpisodeCondition(op="never", p="robot in hallway").serialize()
    with pytest.raises(ValueError):
        EpisodeCondition.parse({"op": "never", "p": "robot in hallway", "on_failure": "explode"})


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
    level = LevelDescription(
        zones=[
            LevelDescription.Zone(name="kitchen", corners=[Position(0, 0), Position(0, 4), Position(4, 4), Position(4, 0)]),
            LevelDescription.Zone(name="line", corners=[Position(0, 0), Position(1, 1)]),
        ]
    )
    polygons = zone_polygons(level)
    assert set(polygons) == {"kitchen"}
    assert polygons["kitchen"].covers(shapely.Point(2, 2))


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
