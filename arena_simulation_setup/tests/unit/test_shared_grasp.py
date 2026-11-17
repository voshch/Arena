from __future__ import annotations

import pytest

from arena_simulation_setup.shared.conditions import HoldsAtom, NotAtom, atom_subjects, parse_atom, parse_node, render_atom, render_until
from arena_simulation_setup.shared.judge import PhaseMonitor, Sample, atom_holds, holds
from arena_simulation_setup.shared.render import render_atom_text, render_phase, render_phases
from arena_simulation_setup.shared.task import GraspPhase, TaskPhase, TaskRequest

ROBOT = "tiago"


def sample(t=0.0, holding=None):
    return Sample(t=t, robots={ROBOT: (0.0, 0.0, 0.0)}, holding=holding)


@pytest.mark.parametrize(
    ("source", "phase"),
    [
        ({"grasp": "close", "object": "cup_1"}, GraspPhase(action="close", object="cup_1")),
        ({"grasp": "open"}, GraspPhase(action="open")),
        ({"grasp": "close", "instance": "left"}, GraspPhase(action="close", instance="left")),
        ({"grasp": "open", "object": "cup_1", "instance": "left", "until": "robot in kitchen"}, GraspPhase(action="open", object="cup_1", instance="left", until="robot in kitchen")),
    ],
)
def test_grasp_phase_round_trips_through_yaml_form(source, phase):
    parsed = TaskPhase.parse(source)
    assert parsed == phase
    assert parsed.serialize() == source
    assert TaskPhase.parse(parsed.serialize()) == phase


def test_grasp_phase_rejects_unknown_action_and_keys():
    with pytest.raises(ValueError, match="close"):
        TaskPhase.parse({"grasp": "squeeze"})
    with pytest.raises(ValueError, match="unknown phase keys"):
        TaskPhase.parse({"grasp": "close", "force": 3})


def test_grasp_phase_travels_in_a_request():
    request = TaskRequest.parse({"phases": [{"grasp": "close", "object": "cup_1"}, {"grasp": "open"}], "task": "t1"})
    assert [p.kind for p in request.phases] == ["grasp", "grasp"]
    assert TaskRequest.parse(request.serialize()) == request


def test_holds_atom_forms():
    assert parse_atom("robot holds cup_1") == HoldsAtom("robot", "cup_1")
    assert parse_atom("not tiago holds cup_1") == NotAtom(HoldsAtom("tiago", "cup_1"))
    assert parse_node({"subject": "robot", "holds": "cup_1"}) == HoldsAtom("robot", "cup_1")
    assert render_atom(HoldsAtom("robot", "cup_1")) == "robot holds cup_1"
    assert render_until(parse_node("robot holds cup_1 and robot in kitchen")) == "robot holds cup_1 and robot in kitchen"
    assert atom_subjects(HoldsAtom("robot", "cup_1")) == {"robot"}


@pytest.mark.parametrize("text", ["robot holds", "env_0/cup holds cup", "robot holds env_0/cup", "1,2 holds cup"])
def test_holds_atom_rejects_malformed_forms(text):
    with pytest.raises(ValueError):
        parse_atom(text)


def test_holds_atom_truth_on_samples():
    held = sample(holding={ROBOT: frozenset({"cup_1"})})
    empty = sample(holding={ROBOT: frozenset()})
    assert atom_holds(parse_atom("robot holds cup_1"), held, {}, ROBOT) is True
    assert atom_holds(parse_atom("robot holds cup_2"), held, {}, ROBOT) is False
    assert atom_holds(parse_atom("tiago holds cup_1"), held, {}, "other") is True
    assert atom_holds(parse_atom("not robot holds cup_1"), held, {}, ROBOT) is False
    assert atom_holds(parse_atom("robot holds cup_1"), empty, {}, ROBOT) is False
    assert atom_holds(parse_atom("robot holds cup_1"), sample(holding={}), {}, ROBOT) is False
    assert atom_holds(parse_atom("robot holds cup_1"), sample(), {}, ROBOT) is None
    assert holds(held, ROBOT, "cup_1") is True


def test_open_is_met_when_the_action_returns():
    monitor = PhaseMonitor(GraspPhase(action="open", object="cup_1"), ROBOT)
    assert monitor.step(sample(0.0), {}, action_done=False) is False
    assert monitor.step(sample(1.0, holding={ROBOT: frozenset({"cup_1"})}), {}, action_done=True) is True
    assert monitor.met_at == 1.0


def test_close_without_object_is_met_when_the_action_returns():
    monitor = PhaseMonitor(GraspPhase(action="close"), ROBOT)
    assert monitor.step(sample(0.0, holding={ROBOT: frozenset()}), {}, action_done=False) is False
    assert monitor.step(sample(1.0, holding={ROBOT: frozenset()}), {}, action_done=True) is True


def test_close_on_an_object_waits_for_the_robot_to_hold_it():
    monitor = PhaseMonitor(GraspPhase(action="close", object="cup_1"), ROBOT)
    empty = frozenset()
    assert monitor.step(sample(0.0, holding={ROBOT: empty}), {}, action_done=False) is False
    assert monitor.step(sample(1.0, holding={ROBOT: empty}), {}, action_done=True) is False
    assert monitor.arrived_at is None
    assert monitor.step(sample(2.0, holding={ROBOT: frozenset({"cup_2"})}), {}, action_done=True) is False
    assert monitor.step(sample(3.0, holding={ROBOT: frozenset({"cup_1"})}), {}, action_done=True) is True
    assert monitor.met_at == 3.0


def test_close_on_an_object_needs_the_action_to_return_even_when_held():
    monitor = PhaseMonitor(GraspPhase(action="close", object="cup_1"), ROBOT)
    assert monitor.step(sample(0.0, holding={ROBOT: frozenset({"cup_1"})}), {}, action_done=False) is False


def test_close_on_an_object_is_judged_on_the_action_alone_when_holding_is_unmeasured():
    monitor = PhaseMonitor(GraspPhase(action="close", object="cup_1"), ROBOT)
    assert monitor.step(sample(0.0), {}, action_done=True) is True


def test_grasp_until_clause_can_require_holding():
    monitor = PhaseMonitor(GraspPhase(action="close", until="robot holds cup_1"), ROBOT)
    assert monitor.step(sample(0.0, holding={ROBOT: frozenset()}), {}, action_done=True) is False
    assert monitor.waiting
    assert monitor.step(sample(1.0, holding={ROBOT: frozenset({"cup_1"})}), {}, action_done=True) is True


@pytest.mark.parametrize(
    ("source", "text"),
    [
        ({"grasp": "close", "object": "cup_1"}, "Grasp cup 1."),
        ({"grasp": "close", "object": "cup_1", "until": "robot in kitchen"}, "Grasp cup 1 and wait until you are in the kitchen."),
        ({"grasp": "close"}, "Close your gripper."),
        ({"grasp": "open", "object": "cup_1"}, "Release cup 1."),
        ({"grasp": "open"}, "Open your gripper."),
        ({"grasp": "open", "text": "let go"}, "let go"),
    ],
)
def test_grasp_phase_renders_to_instruction_text(source, text):
    assert render_phase(TaskPhase.parse(source), {"kitchen": "the kitchen"}) == text


def test_grasp_phase_renders_authored_nouns():
    phases = [TaskPhase.parse({"grasp": "close", "object": "cup_1"}), TaskPhase.parse({"grasp": "open", "object": "cup_1"})]
    assert render_phases(phases, {"cup_1": "the red cup"}) == "Grasp the red cup. Then release the red cup."


@pytest.mark.parametrize(
    ("atom", "text"),
    [
        ("robot holds cup_1", "you have cup 1 in the gripper"),
        ("tiago holds cup_1", "tiago has cup 1 in the gripper"),
        ("not robot holds cup_1", "you do not hold cup 1"),
        ("not tiago holds cup_1", "tiago does not hold cup 1"),
    ],
)
def test_holds_atom_renders_to_text(atom, text):
    assert render_atom_text(parse_atom(atom)) == text
