from __future__ import annotations

import shapely
from arena_simulation_setup.shared.conditions import EpisodeCondition
from arena_simulation_setup.shared.judge import Sample
from arena_simulation_setup.shared.task import GoToPhase, PlayGesturePhase
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

from task_generator.manager.robot_manager.task_runner import TaskRunner

ZONES = {"kitchen": shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)]), "hallway": shapely.Polygon([(10, 0), (10, 4), (14, 4), (14, 0)])}
OK = (0, None)


def goto(x, y, **kwargs):
    kwargs.setdefault("tolerance_radius", 0.5)
    kwargs.setdefault("tolerance_angle", 0.0)
    kwargs.setdefault("hold_time", 0.0)
    return GoToPhase(pose=Pose(Position(x, y), Orientation.identity()), **kwargs)


def at(t, x, y, peds=None):
    return Sample(t=t, robots={"r": (x, y, 0.0)}, peds=dict(peds or {}), field=lambda e, f: None)


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
