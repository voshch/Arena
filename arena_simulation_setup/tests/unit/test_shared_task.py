from __future__ import annotations

import pytest

from arena_simulation_setup.shared.conditions import EpisodeCondition
from arena_simulation_setup.shared.task import GoToPhase, PlayGesturePhase, ReachPhase, TaskPhase, TaskRequest
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position


def test_parse_goto_pose():
    phase = TaskPhase.parse({"goto": [1.0, 2.0, 0.5], "tolerance_radius": 0.3})
    assert isinstance(phase, GoToPhase)
    assert phase.pose.position.x == pytest.approx(1.0)
    assert phase.pose.orientation.to_yaw() == pytest.approx(0.5)
    assert phase.tolerance_radius == 0.3
    assert phase.target is None
    assert not phase.hold


def test_parse_goto_named_target():
    phase = TaskPhase.parse({"goto": "kitchen", "hold_time": 2})
    assert isinstance(phase, GoToPhase)
    assert phase.target == "kitchen"
    assert phase.pose is None
    assert phase.hold_time == 2.0


def test_parse_hold_phase():
    phase = TaskPhase.parse({"until": "not alice in kitchen"})
    assert isinstance(phase, GoToPhase)
    assert phase.hold
    assert phase.until == "not alice in kitchen"


def test_parse_hold_phase_duration_only():
    phase = TaskPhase.parse({"hold_time": 5})
    assert isinstance(phase, GoToPhase)
    assert phase.hold
    assert phase.hold_time == 5.0


def test_parse_rejects_malformed_until():
    with pytest.raises(ValueError):
        TaskPhase.parse({"goto": [0.0, 0.0], "until": "nonsense"})


def test_parse_rejects_unknown_keys():
    with pytest.raises(ValueError):
        TaskPhase.parse({"goto": [0.0, 0.0], "speed": 1.0})


def test_parse_gesture_random_normalizes_to_none():
    assert TaskPhase.parse({"gesture": "random"}).gesture is None
    assert TaskPhase.parse({"gesture": ""}).gesture is None
    assert TaskPhase.parse({"gesture": "wave", "instance": "left"}).instance == "left"


def test_parse_reach_forms():
    named = TaskPhase.parse({"reach": "stow"})
    assert isinstance(named, ReachPhase)
    assert named.named_target == "stow"
    random = TaskPhase.parse({"reach": "random", "planning_time": 2})
    assert random.random and random.planning_time == 2.0
    posed = TaskPhase.parse({"reach": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0], "frame": "base_link"})
    assert posed.target is not None and posed.frame == "base_link"


def test_reach_requires_exactly_one_target():
    with pytest.raises(ValueError):
        ReachPhase()
    with pytest.raises(ValueError):
        ReachPhase(named_target="a", random=True)


def test_scoped_conditions_and_text_parse():
    phase = TaskPhase.parse(
        {
            "goto": "sofa",
            "text": "go to the sofa without entering the hallway",
            "conditions": [{"op": "never", "p": "robot in hallway", "on_failure": "stop_task"}],
        }
    )
    assert phase.text.startswith("go to the sofa")
    assert phase.conditions == [EpisodeCondition(op="never", p="robot in hallway", on_failure="stop_task")]


@pytest.mark.parametrize(
    "value",
    [
        {"goto": [1.0, 2.0, 0.5], "tolerance_radius": 0.3, "tolerance_angle": 0.1, "hold_time": 1.0},
        {"goto": "kitchen", "until": "robot in kitchen", "on_failure": "abort_episode"},
        {"goto": "kitchen", "pose": [3.0, 4.0, 0.0]},
        {"hold_time": 5.0},
        {"until": "not alice in kitchen"},
        {"gesture": "wave", "instance": "left"},
        {"gesture": "random"},
        {"reach": "stow", "planning_time": 2.0, "instance": "left"},
        {"reach": "random"},
        {"reach": [0.4, 0.0, 0.3], "frame": "base_link", "position_tolerance": 0.01},
        {"goto": [0.0, 0.0, 0.0], "text": "stay", "conditions": [{"op": "always", "p": "robot in hallway", "text": "stay in the hallway"}]},
    ],
)
def test_phase_round_trips_through_parse_and_serialize(value):
    phase = TaskPhase.parse(value)
    again = TaskPhase.parse(phase.serialize())
    assert again == phase
    assert again.serialize() == phase.serialize()


def test_phase_serialize_omits_defaults():
    assert TaskPhase.parse({"goto": [1.0, 2.0]}).serialize() == {"goto": [1.0, 2.0, 0.0]}
    assert TaskPhase.parse({"gesture": "random"}).serialize() == {"gesture": "random"}


def test_request_parse_and_serialize():
    request = TaskRequest.parse(
        {
            "phases": [{"goto": "kitchen"}, {"until": "not alice in kitchen"}, {"goto": "sofa"}],
            "conditions": [{"op": "never", "p": "robot in hallway"}],
        }
    )
    assert [type(p) for p in request.phases] == [GoToPhase, GoToPhase, GoToPhase]
    assert request.kind == "goto_pose"
    assert TaskRequest.parse(request.serialize()) == request
    assert TaskRequest.parse([{"gesture": "wave"}]).conditions == []


def test_request_kind_mixed_is_none():
    request = TaskRequest(phases=[GoToPhase(pose=Pose()), PlayGesturePhase(gesture="wave")])
    assert request.kind is None
    assert TaskRequest(phases=[]).kind is None


def test_structure_hook_builds_phases():
    phase = converter.structure({"goto": [1.0, 1.0]}, TaskPhase)
    assert isinstance(phase, GoToPhase)
    assert phase.pose == Pose(Position(1.0, 1.0), Orientation.identity())


def test_parse_goto_signal():
    phase = TaskPhase.parse({"goto": [1.0, 2.0, 0.0], "tolerance_radius": 3, "tolerance_angle": 0.5, "hold_time": 2.0, "signal": "arrived"})
    assert isinstance(phase, GoToPhase)
    assert (phase.tolerance_radius, phase.tolerance_angle, phase.hold_time, phase.signal) == (3.0, 0.5, 2.0, "arrived")
    assert phase.serialize()["signal"] == "arrived"
    assert TaskPhase.parse(phase.serialize()) == phase


def test_parse_goto_leaves_unset_criteria_to_launch_defaults():
    phase = TaskPhase.parse({"goto": [1.0, 2.0, 0.0]})
    assert isinstance(phase, GoToPhase)
    assert (phase.tolerance_radius, phase.tolerance_angle, phase.hold_time, phase.signal) == (None, None, None, None)


def test_parse_instruction_is_text():
    assert TaskPhase.parse({"goto": [1.0, 2.0, 0.0], "instruction": "go to the blue door"}).text == "go to the blue door"
    assert TaskPhase.parse({"goto": [1.0, 2.0, 0.0], "instruction": "ignored", "text": "wins"}).text == "wins"
    assert TaskPhase.parse({"goto": [1.0, 2.0, 0.0]}).text == ""
