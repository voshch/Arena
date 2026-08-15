from __future__ import annotations

import pytest

from arena_simulation_setup.shared.conditions import (
    EntityAtom,
    EpisodeCondition,
    MembershipAtom,
    parse_atom,
)
from arena_simulation_setup.utils.cattrs import converter

# ---------------------------------------------------------------------------
# parse_atom: acceptance
# ---------------------------------------------------------------------------


def test_parse_atom_entity_form():
    atom = parse_atom("ward_a_door.open == false")
    assert atom == EntityAtom(entity="ward_a_door", field="open", value="false")


def test_parse_atom_entity_form_float_value():
    atom = parse_atom("main_elevator.arriving_eta == 4.0")
    assert atom == EntityAtom(entity="main_elevator", field="arriving_eta", value="4.0")


def test_parse_atom_membership_form_robot():
    atom = parse_atom("robot in ward_a")
    assert atom == MembershipAtom(subject="robot", zone="ward_a")


def test_parse_atom_membership_form_ped():
    atom = parse_atom("blocker_1 in ward_a_doorway")
    assert atom == MembershipAtom(subject="blocker_1", zone="ward_a_doorway")


def test_parse_atom_strips_outer_whitespace():
    atom = parse_atom("  robot in pharmacy  ")
    assert atom == MembershipAtom(subject="robot", zone="pharmacy")


# ---------------------------------------------------------------------------
# parse_atom: rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "",
        "robot",
        "robot ward_a",
        "robot.state",
        "robot.state = true",
        ".field == value",
        "entity. == value",
        "entity.field == ",
        " in zone",
        "subject in ",
        "robot in ward_a in pharmacy",
    ],
)
def test_parse_atom_rejects_malformed_shapes(value):
    with pytest.raises(ValueError):
        parse_atom(value)


def test_parse_atom_rejects_env_prefixed_entity():
    with pytest.raises(ValueError):
        parse_atom("env_0/ward_a_door.open == false")


def test_parse_atom_rejects_env_prefixed_subject():
    with pytest.raises(ValueError):
        parse_atom("env_0/blocker_1 in ward_a")


def test_parse_atom_rejects_env_prefixed_zone():
    with pytest.raises(ValueError):
        parse_atom("robot in env_0/ward_a")


# ---------------------------------------------------------------------------
# EpisodeCondition.parse: op arity enforcement
# ---------------------------------------------------------------------------


def test_parse_unary_op_eventually():
    cond = EpisodeCondition.parse({"op": "eventually", "p": "robot in ward_a"})
    assert cond.op == "eventually"
    assert cond.q is None


def test_parse_unary_op_never():
    cond = EpisodeCondition.parse({"op": "never", "p": "robot in pharmacy"})
    assert cond.q is None


def test_parse_unary_op_always():
    cond = EpisodeCondition.parse({"op": "always", "p": "robot in ward_a"})
    assert cond.q is None


def test_parse_binary_op_before():
    cond = EpisodeCondition.parse({"op": "before", "p": "robot in ward_a", "q": "robot in pharmacy"})
    assert cond.q == "robot in pharmacy"


def test_parse_binary_op_never_during():
    cond = EpisodeCondition.parse(
        {"op": "never_during", "p": "robot in ward_a_doorway", "q": "ward_a_door.open == false"},
    )
    assert cond.p == "robot in ward_a_doorway"
    assert cond.q == "ward_a_door.open == false"


def test_parse_unary_op_rejects_q():
    with pytest.raises(ValueError):
        EpisodeCondition.parse({"op": "eventually", "p": "robot in ward_a", "q": "robot in pharmacy"})


def test_parse_binary_op_requires_q():
    with pytest.raises(ValueError):
        EpisodeCondition.parse({"op": "before", "p": "robot in ward_a"})


def test_parse_unknown_op_rejected():
    with pytest.raises(ValueError):
        EpisodeCondition.parse({"op": "until", "p": "robot in ward_a"})


# ---------------------------------------------------------------------------
# EpisodeCondition.parse: malformed dict rejection
# ---------------------------------------------------------------------------


def test_parse_rejects_missing_op():
    with pytest.raises(ValueError):
        EpisodeCondition.parse({"p": "robot in ward_a"})


def test_parse_rejects_non_string_p():
    with pytest.raises(ValueError):
        EpisodeCondition.parse({"op": "eventually", "p": 1.0})


def test_parse_rejects_malformed_p_atom():
    with pytest.raises(ValueError):
        EpisodeCondition.parse({"op": "eventually", "p": "not an atom"})


def test_parse_rejects_malformed_q_atom():
    with pytest.raises(ValueError):
        EpisodeCondition.parse({"op": "before", "p": "robot in ward_a", "q": "not an atom"})


def test_parse_rejects_unknown_keys():
    with pytest.raises(ValueError):
        EpisodeCondition.parse({"op": "eventually", "p": "robot in ward_a", "bogus": 1})


# ---------------------------------------------------------------------------
# Parseable fallback path: converter.structure must enforce validation too
# ---------------------------------------------------------------------------


def test_structure_rejects_binary_op_missing_q():
    with pytest.raises(ValueError):
        converter.structure({"op": "before", "p": "robot in ward_a"}, EpisodeCondition)


def test_structure_rejects_unary_op_with_q():
    with pytest.raises(ValueError):
        converter.structure(
            {"op": "always", "p": "robot in ward_a", "q": "robot in pharmacy"},
            EpisodeCondition,
        )


def test_structure_rejects_malformed_p_atom():
    with pytest.raises(ValueError):
        converter.structure({"op": "eventually", "p": "not an atom"}, EpisodeCondition)


def test_structure_valid_clause_round_trips():
    cond = converter.structure(
        {"op": "before", "p": "robot in ward_a", "q": "robot in pharmacy"},
        EpisodeCondition,
    )
    assert cond.op == "before"
    assert cond.p == "robot in ward_a"
    assert cond.q == "robot in pharmacy"


# ---------------------------------------------------------------------------
# serialize / round-trip
# ---------------------------------------------------------------------------


def test_serialize_omits_absent_q_and_empty_text():
    cond = EpisodeCondition.parse({"op": "eventually", "p": "robot in ward_a"})
    assert cond.serialize() == {"op": "eventually", "p": "robot in ward_a"}


def test_serialize_includes_q_and_text():
    cond = EpisodeCondition.parse(
        {"op": "before", "p": "robot in ward_a", "q": "robot in pharmacy", "text": "Deliver before visiting the pharmacy."},
    )
    assert cond.serialize() == {
        "op": "before",
        "p": "robot in ward_a",
        "q": "robot in pharmacy",
        "text": "Deliver before visiting the pharmacy.",
    }


def test_round_trip_through_parse_serialize():
    cond = EpisodeCondition.parse(
        {"op": "never_during", "p": "robot in ward_a_doorway", "q": "ward_a_door.open == false", "text": "Wait for the door."},
    )
    reparsed = EpisodeCondition.parse(cond.serialize())
    assert reparsed == cond


def test_round_trip_through_cattrs():
    cond = EpisodeCondition.parse({"op": "eventually", "p": "robot in ward_a", "text": "Deliver the package to ward A."})
    raw = converter.unstructure(cond)
    restructured = converter.structure(raw, EpisodeCondition)
    assert restructured == cond


# ---------------------------------------------------------------------------
# M3.2 worked example: "deliver to the ward, never enter the pharmacy, wait for doors"
# ---------------------------------------------------------------------------

_WORKED_EXAMPLE = [
    {"op": "eventually", "p": "robot in ward_a", "text": "Deliver the package to ward A."},
    {"op": "never", "p": "robot in pharmacy", "text": "Never enter the pharmacy."},
    {
        "op": "never_during",
        "p": "robot in ward_a_doorway",
        "q": "ward_a_door.open == false",
        "text": "Wait for the door before passing through.",
    },
]


def test_worked_example_parses_and_round_trips():
    conditions = [EpisodeCondition.parse(item) for item in _WORKED_EXAMPLE]
    assert [c.op for c in conditions] == ["eventually", "never", "never_during"]
    assert [c.serialize() for c in conditions] == _WORKED_EXAMPLE
