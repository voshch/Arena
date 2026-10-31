from __future__ import annotations

import typing

import attrs
import pytest

from arena_simulation_setup.shared.conditions import EpisodeCondition, parse_atom
from arena_simulation_setup.shared.render import render_atom_text, render_condition, render_phase, render_phases
from arena_simulation_setup.shared.task import TaskPhase, TaskRequest

NOUNS = {"kitchen": "the kitchen", "sofa": "the sofa", "hallway": "the hallway", "alice": "the visitor"}


@pytest.mark.parametrize(
    ("atom", "text"),
    [
        ("robot in kitchen", "you are in the kitchen"),
        ("not alice in kitchen", "the visitor is not in the kitchen"),
        ("robot within 2 of alice", "you are within 2 m of the visitor"),
        ("not robot within 1.5 of alice", "you are more than 1.5 m from the visitor"),
        ("door_1.open == true", "door 1 open is true"),
        ("not lift.target == L2", "lift target is not L2"),
        ("robot_1 in hallway", "robot 1 is in the hallway"),
        ("robot sees alice within 12", "you can see the visitor within 12 m"),
        ("not robot sees alice", "you cannot see the visitor within 12 m"),
        ("alice plays beckon", "the visitor is making the beckon gesture"),
        ("not alice plays beckon", "the visitor is not making the beckon gesture"),
        ("robot within 4 of 12.5,3", "you are within 4 m of the point (12.5, 3)"),
    ],
)
def test_render_atom_text(atom, text):
    assert render_atom_text(parse_atom(atom), NOUNS) == text


@pytest.mark.parametrize(
    ("phase", "text"),
    [
        ({"goto": "kitchen"}, "Go to the kitchen."),
        ({"goto": [1.0, 2.0, 0.0]}, "Go to (1.0, 2.0)."),
        ({"goto": "sofa", "hold_time": 2}, "Go to the sofa and stay there for 2 seconds."),
        ({"until": "not alice in kitchen"}, "Stay where you are and wait until the visitor is not in the kitchen."),
        ({"hold_time": 5}, "Stay where you are and stay there for 5 seconds."),
        ({"goto": "elevator_a", "until": "robot in elevator_b"}, "Go to elevator a and wait until you are in elevator b."),
        ({"goto": "sofa", "conditions": [{"op": "never", "p": "robot in hallway"}]}, "Go to the sofa, never letting it happen that you are in the hallway."),
        ({"gesture": "wave"}, "Perform the wave gesture."),
        ({"gesture": "random"}, "Perform a gesture."),
        ({"reach": "stow"}, "Move your arm to the stow position."),
        ({"reach": "random"}, "Reach to a point in your workspace."),
        ({"reach": [0.5, 0.2, 0.0]}, "Reach to the given pose."),
        ({"goto": "sofa", "conditions": [{"op": "always", "p": "robot within 3 of alice"}]}, "Go to the sofa, making sure that you are within 3 m of the visitor the whole time."),
        ({"goto": "sofa", "conditions": [{"op": "eventually", "p": "robot in kitchen"}]}, "Go to the sofa, making sure that at some point you are in the kitchen."),
        ({"goto": "sofa", "conditions": [{"op": "before", "p": "robot in kitchen", "q": "robot in hallway"}]}, "Go to the sofa, making sure that you are in the kitchen before you are in the hallway."),
        ({"goto": "sofa", "conditions": [{"op": "never_during", "p": "robot in hallway", "q": "door_1.open == false"}]}, "Go to the sofa, never letting it happen that you are in the hallway while door 1 open is false."),
        ({"goto": "sofa", "conditions": [{"op": "never", "p": "robot in hallway", "text": "keeping out of the hallway"}]}, "Go to the sofa, keeping out of the hallway."),
        ({"goto": "sofa", "conditions": [{"op": "never", "p": "robot in hallway"}, {"op": "eventually", "p": "alice in kitchen"}]}, "Go to the sofa, never letting it happen that you are in the hallway, making sure that at some point the visitor is in the kitchen."),
        ({"goto": "sofa", "text": "Head over to the couch."}, "Head over to the couch."),
        ({"goto": "sofa", "until": "robot sees alice within 12 and alice plays beckon for 0.5"}, "Go to the sofa and wait until you can see the visitor within 12 m and the visitor is making the beckon gesture for 0.5 seconds."),
        ({"loop": [{"goto": "kitchen"}, {"goto": "sofa"}], "until": "robot sees alice for 0.5"}, "Repeat these steps until you can see the visitor within 12 m for 0.5 seconds: Go to the kitchen. Then go to the sofa."),
        ({"loop": [{"goto": "kitchen"}]}, "Keep repeating these steps: Go to the kitchen."),
        ({"goto": "sofa", "until": "alice plays beckon for 1 or robot within 2.5 of alice"}, "Go to the sofa and wait until the visitor is making the beckon gesture for 1 seconds or you are within 2.5 m of the visitor."),
        ({"loop": [{"goto": "kitchen"}], "text": "Patrol the kitchen."}, "Patrol the kitchen."),
        ({"gesture": "wave", "until": {"field": "wardrobe.openness.door_left", "ge": 0.8}}, "Perform the wave gesture and wait until wardrobe openness door left is at least 0.8."),
        ({"gesture": "wave", "until": {"field": "drawer.pos.slide", "within": [0.2, 0.4], "for": 2}}, "Perform the wave gesture and wait until drawer pos slide is between 0.2 and 0.4 for 2 seconds."),
        ({"gesture": "wave", "until": {"some": "wardrobe.openness", "gt": 0.5}}, "Perform the wave gesture and wait until the openness of some joint of wardrobe is above 0.5."),
        ({"gesture": "wave", "until": {"each": "stove.setting", "eq": "off"}}, "Perform the wave gesture and wait until the setting of every joint of stove is off."),
        ({"gesture": "wave", "until": {"all": [{"subject": "robot", "in": "kitchen"}, {"not": {"subject": "alice", "plays": "beckon"}}]}}, "Perform the wave gesture and wait until you are in the kitchen and the visitor is not making the beckon gesture."),
        ({"gesture": "wave", "until": {"any": [{"field": "door.open", "ne": "true"}, {"field": "lift.eta", "lt": 3, "for": 1}]}}, "Perform the wave gesture and wait until door open is not true or lift eta is below 3 for 1 seconds."),
        ({"gesture": "wave", "until": {"all": [{"any": ["robot in kitchen", "robot in hallway"]}, {"field": "lift.eta", "le": 3}]}}, "Perform the wave gesture and wait until (you are in the kitchen or you are in the hallway) and lift eta is at most 3."),
        ({"gesture": "wave", "until": {"not": {"any": ["robot in kitchen", "robot in hallway"]}}}, "Perform the wave gesture and wait until it is not the case that (you are in the kitchen or you are in the hallway)."),
    ],
)
def test_render_phase(phase, text):
    assert render_phase(TaskPhase.parse(phase), NOUNS) == text


@pytest.mark.parametrize(
    ("condition", "text"),
    [
        ({"op": "never", "p": "robot in hallway"}, "Never let it happen that you are in the hallway."),
        ({"op": "always", "p": "robot within 3 of alice"}, "Make sure that you are within 3 m of the visitor the whole time."),
        ({"op": "eventually", "p": "robot in kitchen"}, "At some point, you are in the kitchen."),
        ({"op": "before", "p": "robot in kitchen", "q": "robot in sofa"}, "Make sure that you are in the kitchen before you are in the sofa."),
        ({"op": "never_during", "p": "robot in hallway", "q": "door_1.open == false"}, "Never let it happen that you are in the hallway while door 1 open is false."),
        ({"op": "never", "p": "robot in hallway", "text": "Avoid the hallway."}, "Avoid the hallway."),
    ],
)
def test_render_condition(condition, text):
    assert render_condition(EpisodeCondition.parse(condition), NOUNS) == text


def test_render_request_joins_phases_and_conditions():
    request = TaskRequest.parse(
        {
            "phases": [{"goto": "kitchen"}, {"until": "not alice in kitchen"}, {"goto": "sofa", "conditions": [{"op": "never", "p": "robot in hallway"}]}],
            "conditions": [{"op": "always", "p": "robot within 5 of alice"}],
        }
    )
    assert render_phases(request.phases, NOUNS, request.conditions) == ("Go to the kitchen. Then stay where you are and wait until the visitor is not in the kitchen. Then go to the sofa, never letting it happen that you are in the hallway. Make sure that you are within 5 m of the visitor the whole time.")


def test_render_without_nouns_uses_names():
    assert render_phase(TaskPhase.parse({"goto": "ward_a"})) == "Go to ward a."


def test_render_request_with_a_loop():
    request = TaskRequest.parse({"phases": [{"loop": [{"goto": "kitchen"}, {"goto": "sofa"}], "until": "alice plays beckon"}, {"goto": "alice", "hold_time": 2}]})
    assert render_phases(request.phases, NOUNS) == "Repeat these steps until the visitor is making the beckon gesture: Go to the kitchen. Then go to the sofa. Then go to the visitor and stay there for 2 seconds."


@attrs.define
class _DockPhase(TaskPhase):
    kind: typing.ClassVar[str] = "dock"


def test_render_phase_of_unknown_kind_uses_the_generic_sentence():
    assert render_phase(_DockPhase(), NOUNS) == "Carry out the step."
    assert render_phase(_DockPhase(until="robot in kitchen"), NOUNS) == "Carry out the step and wait until you are in the kitchen."


def test_render_condition_with_structured_p_and_q():
    condition = EpisodeCondition.parse({"op": "never_during", "p": {"subject": "robot", "in": "hallway"}, "q": {"some": "door.openness", "lt": 0.5}})
    assert render_condition(condition, NOUNS) == "Never let it happen that you are in the hallway while the openness of some joint of door is below 0.5."
