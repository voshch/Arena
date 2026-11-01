from __future__ import annotations

import math

import numpy as np
import pytest
import shapely
from arena_simulation_setup.shared.conditions import EpisodeCondition
from arena_simulation_setup.shared.judge import Sample, WallGrid
from arena_simulation_setup.shared.task import GoToPhase, LoopPhase, PlayGesturePhase
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

from arena_runtime.env.robot_manager.task_runner import TaskRunner, standoff_pose

ZONES = {"kitchen": shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)]), "hallway": shapely.Polygon([(10, 0), (10, 4), (14, 4), (14, 0)])}
OK = (0, None)
OPEN_FLOOR = WallGrid(cells=np.zeros((400, 400), dtype=bool), origin=(0.0, 0.0), resolution=0.05)
BECKON = "robot sees caller within 12 and caller plays beckon for 0.5"


def goto(x, y, **kwargs):
    kwargs.setdefault("tolerance_radius", 0.5)
    kwargs.setdefault("tolerance_angle", 0.0)
    kwargs.setdefault("hold_time", 0.0)
    return GoToPhase(pose=Pose(Position(x, y), Orientation.identity()), **kwargs)


def at(t, x, y, peds=None, clips=None):
    return Sample(t=t, robots={"r": (x, y, 0.0)}, peds=dict(peds or {}), field=lambda e, f: None, clips=dict(clips or {}), walls=OPEN_FLOOR)


def beckoning(t, x, y):
    return at(t, x, y, peds={"caller": (x + 3.0, y)}, clips={"caller": frozenset({"beckon"})})


def patrol(until=BECKON):
    return LoopPhase(phases=[goto(1.0, 1.0), goto(3.0, 3.0)], until=until)


def test_sequence_advances_and_finishes():
    runner = TaskRunner("r")
    runner.submit([goto(1.0, 1.0), goto(3.0, 3.0)])
    assert runner.active == 0 and not runner.done
    assert runner.step(at(0.0, 5.0, 5.0), ZONES, None, OK).ended is None
    ended = runner.step(at(1.0, 1.0, 1.0), ZONES, None, OK).ended
    assert ended is not None and ended.index == 0 and ended.outcome == "met"
    assert runner.active == 1
    runner.step(at(2.0, 3.0, 3.0), ZONES, None, OK)
    assert runner.done and runner.outcomes("met") == [0, 1]


def test_failed_phase_policies():
    runner = TaskRunner("r")
    runner.submit([goto(1.0, 1.0, on_failure="continue"), goto(2.0, 2.0, on_failure="stop_task"), goto(3.0, 3.0)])
    t1 = runner.step(at(0.0, 1.0, 1.0), ZONES, None, (2, "aborted"))
    assert t1.ended.outcome == "failed" and t1.ended.reason == "aborted" and not t1.stop
    assert runner.active == 1
    t2 = runner.step(at(1.0, 2.0, 2.0), ZONES, None, (2, "aborted"))
    assert t2.stop and t2.abort is None
    assert runner.done and runner.outcomes("dropped") == [2]


def test_abort_episode_policy_reports_reason():
    runner = TaskRunner("r")
    runner.submit([goto(1.0, 1.0, on_failure="abort_episode")])
    transition = runner.step(at(0.0, 1.0, 1.0), ZONES, None, (3, "timeout"))
    assert transition.abort == "timeout"
    assert runner.outcomes("failed") == [0]


def test_resubmission_keeps_episode_wide_indices_and_drops_pending():
    runner = TaskRunner("r")
    runner.submit([goto(1.0, 1.0), goto(2.0, 2.0)])
    runner.step(at(0.0, 1.0, 1.0), ZONES, None, OK)
    states = runner.submit([goto(3.0, 3.0)])
    assert [s.index for s in states] == [2]
    assert runner.active == 2
    assert runner.outcomes("met") == [0] and runner.outcomes("dropped") == [1]
    assert len(runner.serialize()["phases"]) == 3


def test_stop_drops_the_unfinished_phases_and_judges_nothing_after():
    runner = TaskRunner("r")
    runner.submit([goto(1.0, 1.0), goto(3.0, 3.0), goto(5.0, 5.0)])
    runner.step(at(0.0, 1.0, 1.0), ZONES, None, OK)
    runner.stop()
    assert runner.done and runner.active_state is None
    assert runner.outcomes("met") == [0] and runner.outcomes("dropped") == [1, 2]
    assert runner.step(at(1.0, 3.0, 3.0), ZONES, None, OK).ended is None
    assert runner.outcomes("met") == [0]


def test_begin_episode_clears_everything():
    runner = TaskRunner("r")
    runner.submit([goto(1.0, 1.0)])
    runner.begin_episode()
    assert runner.phases == [] and runner.active is None and runner.done
    assert runner.submit([goto(1.0, 1.0)])[0].index == 0


def test_gesture_phase_uses_action_result():
    runner = TaskRunner("r")
    runner.submit([PlayGesturePhase(gesture="wave")])
    assert runner.step(at(0.0, 0.0, 0.0), ZONES, False, OK).ended is None
    assert runner.step(at(1.0, 0.0, 0.0), ZONES, True, OK).ended.outcome == "met"


def test_waiting_during_hold_and_until():
    runner = TaskRunner("r")
    runner.submit([goto(1.0, 1.0, hold_time=2.0), goto(1.0, 1.0, until="not alice in kitchen")])
    runner.step(at(0.0, 1.0, 1.0), ZONES, None, OK)
    assert runner.waiting
    runner.step(at(2.0, 1.0, 1.0), ZONES, None, OK)
    assert runner.active == 1
    runner.step(at(3.0, 1.0, 1.0, peds={"alice": (2.0, 2.0)}), ZONES, None, OK)
    assert runner.waiting
    runner.step(at(4.0, 1.0, 1.0, peds={"alice": (12.0, 2.0)}), ZONES, None, OK)
    assert runner.done and not runner.waiting


def test_phase_condition_violation_applies_policy_immediately():
    runner = TaskRunner("r")
    runner.submit([goto(1.0, 1.0, conditions=[EpisodeCondition(op="never", p="robot in hallway", on_failure="abort_episode")])])
    transition = runner.step(at(0.0, 11.0, 1.0), ZONES, None, OK)
    assert transition.abort is not None and "0:0" in runner.violated
    assert runner.done and runner.outcomes("dropped") == [0]


def test_request_condition_scoped_to_its_request():
    runner = TaskRunner("r")
    runner.submit([goto(1.0, 1.0)], conditions=[EpisodeCondition(op="never", p="robot in hallway")])
    runner.step(at(0.0, 1.0, 1.0), ZONES, None, OK)
    assert runner.done and runner.violated == []
    runner.submit([goto(12.0, 1.0)])
    runner.step(at(1.0, 12.0, 1.0), ZONES, None, OK)
    assert runner.violated == []


def test_request_condition_violation_recorded_without_policy():
    runner = TaskRunner("r")
    runner.submit([goto(12.0, 1.0)], conditions=[EpisodeCondition(op="never", p="robot in hallway")])
    transition = runner.step(at(0.0, 12.0, 1.0), ZONES, None, OK)
    assert not transition.stop and runner.violated == ["r0:0"]
    assert runner.outcomes("met") == [0]


def test_episode_conditions_finish_at_episode_end():
    runner = TaskRunner("r")
    runner.set_episode_conditions([EpisodeCondition(op="eventually", p="robot in kitchen")])
    runner.submit([goto(12.0, 1.0)])
    runner.step(at(0.0, 12.0, 1.0), ZONES, None, OK)
    assert runner.violated == []
    runner.finish_episode()
    assert runner.violated == ["e0"]


def test_serialize_lists_phases_and_condition_spans():
    runner = TaskRunner("r")
    runner.set_episode_conditions([EpisodeCondition(op="never", p="robot in hallway")])
    runner.submit([goto(1.0, 1.0), goto(2.0, 2.0)], conditions=[EpisodeCondition(op="eventually", p="robot in kitchen")])
    runner.submit([goto(3.0, 3.0)])
    data = runner.serialize()
    assert [p["goto"][0] for p in data["phases"]] == [1.0, 2.0, 3.0]
    spans = {c["id"]: (c["from"], c["to"]) for c in data["conditions"]}
    assert spans == {"e0": (0, None), "r0:0": (0, 1)}


def test_serialize_keeps_goal_inputs_and_the_instruction_per_phase():
    runner = TaskRunner("r")
    states = runner.submit([goto(1.0, 1.0), goto(2.0, 2.0)])
    runner.goal_inputs = ("instruction",)
    states[0].instruction = {"source": "route", "text": "Walk about 1 meter and stop."}
    data = runner.serialize()
    assert data["goal_inputs"] == ["instruction"]
    assert data["instructions"] == [{"source": "route", "text": "Walk about 1 meter and stop."}, None]
    runner.begin_episode()
    assert runner.serialize()["goal_inputs"] == []


def test_signal_phase_waits_for_the_signal_then_meets():
    runner = TaskRunner("r")
    runner.submit([goto(1.0, 1.0, signal="arrived"), goto(3.0, 3.0)])
    assert runner.step(at(0.0, 1.0, 1.0), ZONES, None, OK).ended is None
    ended = runner.step(at(1.0, 1.0, 1.0), ZONES, None, OK, signal="arrived").ended
    assert ended is not None and ended.outcome == "met"
    assert runner.active == 1


def test_signal_far_from_goal_fails_the_phase_and_aborts_the_episode():
    runner = TaskRunner("r")
    runner.submit([goto(1.0, 1.0, signal="arrived"), goto(3.0, 3.0)])
    transition = runner.step(at(0.0, 4.0, 5.0), ZONES, None, OK, signal="arrived")
    assert transition.ended is not None and transition.ended.outcome == "failed"
    assert transition.abort == "signaled arrived 5.00 m from goal"
    assert runner.done and runner.outcomes("dropped") == [1]


def test_loop_body_expands_once_and_repeats_at_the_end_of_the_list():
    runner = TaskRunner("r")
    states = runner.submit([patrol(), goto(12.0, 1.0)])
    assert [s.index for s in states] == [0, 1, 2]
    assert [p["goto"][0] for p in runner.serialize()["phases"]] == [1.0, 3.0, 12.0]
    runner.step(at(0.0, 1.0, 1.0), ZONES, None, OK)
    assert runner.active == 1
    runner.step(at(1.0, 3.0, 3.0), ZONES, None, OK)
    assert runner.active == 3 and runner.outcomes("met") == [0, 1]
    assert [p["goto"][0] for p in runner.serialize()["phases"]] == [1.0, 3.0, 12.0, 1.0, 3.0]
    runner.step(at(2.0, 1.0, 1.0), ZONES, None, OK)
    runner.step(at(3.0, 3.0, 3.0), ZONES, None, OK)
    assert runner.active == 5 and runner.outcomes("met") == [0, 1, 3, 4]


def test_loop_exit_drops_the_iteration_and_resumes_after_the_loop():
    runner = TaskRunner("r")
    runner.submit([patrol(), goto(12.0, 1.0)])
    runner.step(at(0.0, 1.0, 1.0), ZONES, None, OK)
    runner.step(at(1.0, 3.0, 3.0), ZONES, None, OK)
    assert runner.active == 3
    assert runner.step(beckoning(2.0, 2.0, 2.0), ZONES, None, OK).ended is None
    exit_ = runner.step(beckoning(2.5, 2.0, 2.0), ZONES, None, OK)
    assert exit_.ended is not None and exit_.ended.index == 3 and exit_.ended.outcome == "dropped"
    assert runner.outcomes("dropped") == [3, 4] and runner.active == 2
    runner.step(at(3.0, 12.0, 1.0), ZONES, None, OK)
    assert runner.done and runner.outcomes("met") == [0, 1, 2]


def test_loop_exit_in_the_first_iteration_skips_its_remaining_phases():
    runner = TaskRunner("r")
    runner.submit([patrol(), goto(12.0, 1.0)])
    runner.step(beckoning(0.0, 5.0, 5.0), ZONES, None, OK)
    runner.step(beckoning(0.5, 5.0, 5.0), ZONES, None, OK)
    assert runner.outcomes("dropped") == [0, 1] and runner.active == 2


def test_loop_exit_dwell_carries_across_iterations():
    runner = TaskRunner("r")
    runner.submit([patrol(), goto(12.0, 1.0)])
    runner.step(at(0.0, 1.0, 1.0), ZONES, None, OK)
    runner.step(beckoning(1.0, 3.0, 3.0), ZONES, None, OK)
    assert runner.active == 3 and runner.outcomes("met") == [0, 1]
    runner.step(beckoning(1.4, 3.0, 3.0), ZONES, None, OK)
    assert runner.active == 3
    runner.step(beckoning(1.6, 3.0, 3.0), ZONES, None, OK)
    assert runner.outcomes("dropped") == [3, 4] and runner.active == 2


def test_loop_exit_breaks_when_the_caller_stops_beckoning():
    runner = TaskRunner("r")
    runner.submit([patrol()])
    runner.step(beckoning(0.0, 5.0, 5.0), ZONES, None, OK)
    runner.step(at(0.3, 5.0, 5.0, peds={"caller": (8.0, 5.0)}, clips={"caller": frozenset()}), ZONES, None, OK)
    runner.step(beckoning(0.6, 5.0, 5.0), ZONES, None, OK)
    assert runner.active == 0
    runner.step(beckoning(1.1, 5.0, 5.0), ZONES, None, OK)
    assert runner.done and runner.outcomes("dropped") == [0, 1]


def test_loop_behind_a_wall_does_not_exit():
    cells = np.zeros((400, 400), dtype=bool)
    cells[:, 140:143] = True
    walls = WallGrid(cells=cells, origin=(0.0, 0.0), resolution=0.05)
    runner = TaskRunner("r")
    runner.submit([patrol()])
    for t in (0.0, 0.5, 1.0):
        runner.step(Sample(t=t, robots={"r": (5.0, 5.0, 0.0)}, peds={"caller": (9.0, 5.0)}, clips={"caller": frozenset({"beckon"})}, walls=walls), ZONES, None, OK)
    assert runner.active == 0 and runner.outcomes("dropped") == []


def test_loop_without_until_repeats_until_stopped():
    runner = TaskRunner("r")
    runner.submit([LoopPhase(phases=[goto(1.0, 1.0)])])
    for t in range(4):
        runner.step(at(float(t), 1.0, 1.0), ZONES, None, OK)
    assert not runner.done and runner.outcomes("met") == [0, 1, 2, 3] and runner.active == 4
    runner.stop()
    assert runner.done and runner.outcomes("dropped") == [4]


def test_loop_as_last_phase_exits_to_done():
    runner = TaskRunner("r")
    runner.submit([goto(12.0, 1.0), patrol()], conditions=[EpisodeCondition(op="eventually", p="robot in kitchen")])
    runner.step(at(0.0, 12.0, 1.0), ZONES, None, OK)
    runner.step(beckoning(1.0, 12.0, 1.0), ZONES, None, OK)
    runner.step(beckoning(1.5, 12.0, 1.0), ZONES, None, OK)
    assert runner.done and runner.outcomes("met") == [0] and runner.outcomes("dropped") == [1, 2]
    assert runner.violated == ["r0:0"]


def test_loop_request_condition_covers_appended_iterations():
    runner = TaskRunner("r")
    runner.submit([patrol()], conditions=[EpisodeCondition(op="never", p="robot in hallway")])
    runner.step(at(0.0, 1.0, 1.0), ZONES, None, OK)
    runner.step(at(1.0, 3.0, 3.0), ZONES, None, OK)
    assert runner.active == 2 and runner.violated == []
    runner.step(at(2.0, 12.0, 1.0), ZONES, None, OK)
    assert runner.violated == ["r0:0"]


def test_resubmission_drops_a_running_loop():
    runner = TaskRunner("r")
    runner.submit([patrol()])
    runner.step(at(0.0, 1.0, 1.0), ZONES, None, OK)
    states = runner.submit([goto(12.0, 1.0)])
    assert [s.index for s in states] == [2]
    assert runner.outcomes("met") == [0] and runner.outcomes("dropped") == [1]


def test_standoff_pose_stops_short_of_the_pedestrian_and_faces_them():
    goal = standoff_pose((0.0, 0.0), (5.0, 0.0), 1.0)
    assert goal.position.x == pytest.approx(4.0) and goal.position.y == pytest.approx(0.0)
    assert goal.orientation.to_yaw() == pytest.approx(0.0)

    diagonal = standoff_pose((0.0, 0.0), (3.0, 4.0), 1.0)
    assert math.dist((diagonal.position.x, diagonal.position.y), (3.0, 4.0)) == pytest.approx(1.0)
    assert math.dist((diagonal.position.x, diagonal.position.y), (0.0, 0.0)) == pytest.approx(4.0)
    assert diagonal.orientation.to_yaw() == pytest.approx(math.atan2(4.0, 3.0))


def test_standoff_pose_defaults_and_stays_put_when_already_close():
    default = standoff_pose((0.0, 0.0), (5.0, 0.0))
    assert default.position.x == pytest.approx(4.2)

    close = standoff_pose((0.0, 0.0), (0.5, 0.0))
    assert (close.position.x, close.position.y) == pytest.approx((0.0, 0.0))
    assert close.orientation.to_yaw() == pytest.approx(0.0)


def drawer_at(t, x, y, openness):
    return Sample(t=t, robots={"r": (x, y, 0.0)}, field=lambda e, f: None, family=lambda e, f: {"left": openness, "right": "0.0"} if (e, f) == ("drawer", "openness") else None, walls=OPEN_FLOOR)


def test_loop_exits_on_a_structured_family_until_after_its_hold():
    runner = TaskRunner("r")
    runner.submit([patrol(until={"some": "drawer.openness", "ge": 0.5, "for": 1}), goto(12.0, 1.0)])
    runner.step(drawer_at(0.0, 5.0, 5.0, "0.9"), ZONES, None, OK)
    runner.step(drawer_at(0.5, 5.0, 5.0, "0.1"), ZONES, None, OK)
    runner.step(drawer_at(1.0, 5.0, 5.0, "0.9"), ZONES, None, OK)
    runner.step(drawer_at(1.9, 5.0, 5.0, "0.9"), ZONES, None, OK)
    assert runner.active == 0 and runner.outcomes("dropped") == []
    runner.step(drawer_at(2.0, 5.0, 5.0, "0.9"), ZONES, None, OK)
    assert runner.outcomes("dropped") == [0, 1] and runner.active == 2


def test_violation_reason_names_a_structured_condition_in_text():
    runner = TaskRunner("r")
    runner.submit([goto(12.0, 1.0)], conditions=[EpisodeCondition.parse({"op": "never", "p": {"subject": "robot", "in": "hallway"}, "on_failure": "abort_episode"})])
    transition = runner.step(at(0.0, 11.0, 1.0), ZONES, None, OK)
    assert transition.abort == "condition r0:0 (robot in hallway) violated on robot r"
