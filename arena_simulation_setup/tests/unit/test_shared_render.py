from __future__ import annotations

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
        ({"goto": "sofa", "text": "Head over to the couch."}, "Head over to the couch."),
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
