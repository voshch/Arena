from __future__ import annotations

import pytest

from arena_simulation_setup.shared.conditions import AllNode, EpisodeCondition, FieldTest, HoldNode, MembershipAtom
from arena_simulation_setup.shared.task import GoToPhase, LoopPhase, PlayGesturePhase, ReachPhase, TaskPhase, TaskRequest
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


@pytest.mark.parametrize("value", [{"goto": [0.0, 0.0], "hold_time": [1.0]}, {"goto": [0.0, 0.0], "conditions": 3}, {"loop": 3}])
def test_parse_rejects_wrong_field_types(value):
    with pytest.raises(TypeError):
        TaskPhase.parse(value)


def test_request_parse_rejects_non_list_phases():
    with pytest.raises(TypeError):
        TaskRequest.parse({"phases": 3})


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


def test_reach_pose_survives_serialization_in_full():
    phase = ReachPhase(target=Pose(Position(0.4, 0.1, 0.9), Orientation.from_euler((0.1, 0.5, -0.3))), frame="base_link")
    again = TaskPhase.parse(phase.serialize())
    assert isinstance(again, ReachPhase) and again.target is not None and again.frame == "base_link"
    assert list(again.target.position) == pytest.approx([0.4, 0.1, 0.9])
    assert again.target.orientation.to_euler() == pytest.approx((0.1, 0.5, -0.3))


@pytest.mark.parametrize(
    "value",
    [
        {"reach": "stow", "motion": "linear"},
        {"reach": "random", "relative": True},
        {"reach": [0.4, 0.0, 0.3], "motion": "curved"},
        {"reach": [0.4, 0.0, 0.3], "speed": 0.0},
        {"reach": [0.4, 0.0, 0.3], "speed": 1.5},
        {"reach": [0.4, 0.0, 0.3], "stop": {"field": "chest.openness.drawer", "between": 0.02}},
    ],
)
def test_reach_rejects_invalid_motion_fields(value):
    with pytest.raises(ValueError):
        TaskPhase.parse(value)


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
        {"goto": "caller", "pose": [3.0, 4.0, 0.0], "dispatch": False},
        {"hold_time": 5.0},
        {"until": "not alice in kitchen"},
        {"gesture": "wave", "instance": "left"},
        {"gesture": "random"},
        {"reach": "stow", "planning_time": 2.0, "instance": "left"},
        {"reach": "random"},
        {"reach": [0.4, 0.0, 0.3], "frame": "base_link", "position_tolerance": 0.01},
        {"reach": [0.3, -0.2, 0.55], "frame": "chest", "motion": "linear", "speed": 0.1, "stop": {"field": "chest.openness.prismatic_joint_left_02", "le": 0.02}},
        {"reach": [-0.12, 0.0, 0.0], "relative": True, "motion": "linear", "stop": "door.open == true for 0.5"},
        {"goto": [0.0, 0.0, 0.0], "text": "stay", "conditions": [{"op": "always", "p": "robot in hallway", "text": "stay in the hallway"}]},
    ],
)
def test_phase_round_trips_through_parse_and_serialize(value):
    phase = TaskPhase.parse(value)
    again = TaskPhase.parse(phase.serialize())
    assert again == phase
    assert again.serialize() == phase.serialize()


def test_a_goto_is_dispatched_unless_marked_judge_only():
    assert TaskPhase.parse({"goto": [1.0, 2.0]}).dispatch
    assert not TaskPhase.parse({"goto": "caller", "dispatch": False}).dispatch


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


def test_parse_until_with_dwell_and_conjunction():
    phase = TaskPhase.parse({"goto": "caller", "until": "robot sees caller within 12 and caller plays beckon for 0.5"})
    assert phase.until == "robot sees caller within 12 and caller plays beckon for 0.5"
    assert TaskPhase.parse(phase.serialize()) == phase
    assert TaskPhase.parse({"goto": "caller", "until": "caller plays beckon for 0.5 or robot within 2.5 of caller"}).until == "caller plays beckon for 0.5 or robot within 2.5 of caller"
    with pytest.raises(ValueError):
        TaskPhase.parse({"goto": "caller", "until": "robot sees caller for x"})
    with pytest.raises(ValueError):
        TaskPhase.parse({"goto": "caller", "until": "robot sees caller or"})


def test_parse_loop_phase():
    phase = TaskPhase.parse({"loop": [{"goto": [10.0, 18.0, 0.0]}, {"goto": [10.0, 2.0, 0.0]}], "until": "robot sees caller within 12 for 0.5", "text": "Patrol the hallway."})
    assert isinstance(phase, LoopPhase)
    assert [type(p) for p in phase.phases] == [GoToPhase, GoToPhase]
    assert phase.until == "robot sees caller within 12 for 0.5"
    assert phase.text == "Patrol the hallway."
    assert TaskPhase.parse(phase.serialize()) == phase
    assert TaskPhase.parse({"loop": [{"goto": "kitchen"}]}).until is None


@pytest.mark.parametrize("value", [{"loop": []}, {"loop": [{"goto": "kitchen"}], "hold_time": 2}, {"loop": [{"goto": "kitchen"}], "conditions": [{"op": "never", "p": "robot in hallway"}]}, {"loop": [{"goto": "kitchen"}], "until": "nonsense"}])
def test_parse_loop_rejects_other_keys_and_empty_loops(value):
    with pytest.raises(ValueError):
        TaskPhase.parse(value)


def test_request_with_loop_round_trips():
    request = TaskRequest.parse({"phases": [{"loop": [{"goto": [10.0, 18.0]}, {"goto": [10.0, 2.0]}], "until": "robot sees caller and caller plays beckon for 0.5"}, {"goto": "caller", "tolerance_radius": 2.5, "hold_time": 0}]})
    assert request.kind is None
    assert TaskRequest.parse(request.serialize()) == request
    assert request.serialize()["phases"][0]["loop"] == [{"goto": [10.0, 18.0, 0.0]}, {"goto": [10.0, 2.0, 0.0]}]


def test_parse_goto_zone_dispatch_pose_remembers_its_zone():
    import numpy as np

    from arena_simulation_setup.utils.geometry import PointResolver
    from arena_simulation_setup.utils.resolution import activate_resolver

    square = [Position(0.0, 0.0), Position(2.0, 0.0), Position(2.0, 2.0), Position(0.0, 2.0)]
    resolver = PointResolver(lookup={"lobby": square}.get, rng=np.random.default_rng(0), candidates=lambda: ["lobby"])
    with activate_resolver(resolver):
        zoned = TaskPhase.parse({"goto": "kitchen", "pose": "lobby"})
    literal = TaskPhase.parse({"goto": "kitchen", "pose": [1.0, 1.0, 0.0]})
    assert isinstance(zoned, GoToPhase) and isinstance(literal, GoToPhase)
    assert zoned.pose_zone == "lobby"
    assert 0.0 <= zoned.pose.position.x <= 2.0 and 0.0 <= zoned.pose.position.y <= 2.0
    assert literal.pose_zone == ""


def test_parse_structured_until_keeps_the_mapping():
    until = {"all": [{"subject": "robot", "in": "kitchen"}, {"field": "wardrobe.openness.door_left", "ge": 0.8}], "for": 2}
    phase = TaskPhase.parse({"goto": "kitchen", "until": until})
    assert phase.until == until
    assert phase.until is not until
    assert phase.until_node == HoldNode(AllNode((MembershipAtom("robot", "kitchen"), FieldTest("wardrobe", "openness.door_left", "ge", 0.8))), 2.0)
    assert phase.serialize()["until"] == until
    assert TaskPhase.parse(phase.serialize()) == phase
    loop = TaskPhase.parse({"loop": [{"goto": "kitchen"}], "until": {"some": "wardrobe.openness", "ge": 0.5}})
    assert isinstance(loop, LoopPhase)
    assert loop.serialize()["until"] == {"some": "wardrobe.openness", "ge": 0.5}
    assert TaskPhase.parse(loop.serialize()) == loop


def test_text_until_stays_text_and_parses_into_a_node():
    phase = TaskPhase.parse({"goto": "kitchen", "until": "robot in kitchen for 2"})
    assert phase.until == "robot in kitchen for 2"
    assert phase.until_node == HoldNode(MembershipAtom("robot", "kitchen"), 2.0)
    assert TaskPhase.parse({"goto": "kitchen"}).until_node is None


@pytest.mark.parametrize("until", [{"field": "wardrobe.state"}, {"some": "wardrobe.openness", "ge": "half"}, {"all": []}, {"subject": "robot", "in": "kitchen", "for": -1}])
def test_parse_rejects_a_malformed_structured_until(until):
    with pytest.raises(ValueError):
        TaskPhase.parse({"goto": "kitchen", "until": until})
    with pytest.raises(ValueError):
        TaskPhase.parse({"loop": [{"goto": "kitchen"}], "until": until})
