from __future__ import annotations

import re

import pytest

from arena_simulation_setup.shared.conditions import (
    SEES_RANGE_M,
    AllNode,
    AnyNode,
    EntityAtom,
    EpisodeCondition,
    FamilyTest,
    FieldTest,
    HoldNode,
    MembershipAtom,
    NotAtom,
    NotNode,
    PlaysAtom,
    SeesAtom,
    WithinAtom,
    atom_subjects,
    check_fields,
    node_families,
    node_fields,
    node_subjects,
    parse_atom,
    parse_node,
    parse_point,
    parse_until,
    render_atom,
    render_node,
    render_until,
)
from arena_simulation_setup.utils.cattrs import converter

# ---------------------------------------------------------------------------
# parse_atom: acceptance
# ---------------------------------------------------------------------------


def test_parse_atom_entity_form():
    atom = parse_atom("ward_a_door.open == false")
    assert atom == EntityAtom(entity="ward_a_door", field="open", value="false")


def test_parse_atom_entity_form_splits_at_the_first_dot():
    assert parse_atom("wardrobe.openness.door_left == 1") == EntityAtom(entity="wardrobe", field="openness.door_left", value="1")
    assert parse_atom("timeline.stimulus.alice.rest == 2.5") == EntityAtom(entity="timeline", field="stimulus.alice.rest", value="2.5")
    assert render_atom(parse_atom("wardrobe.openness.door_left == 1")) == "wardrobe.openness.door_left == 1"


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


def test_atom_subjects_of_membership_is_its_subject():
    assert atom_subjects(parse_atom("blocker_1 in ward_a_doorway")) == {"blocker_1"}
    assert atom_subjects(parse_atom("not robot in pharmacy")) == {"robot"}


def test_parse_atom_sees_form_takes_an_optional_range():
    assert parse_atom("robot sees caller") == SeesAtom("robot", "caller", SEES_RANGE_M)
    assert parse_atom("robot sees caller within 8.5") == SeesAtom("robot", "caller", 8.5)
    assert parse_atom("not robot_1 sees caller within 3") == NotAtom(SeesAtom("robot_1", "caller", 3.0))
    assert atom_subjects(parse_atom("robot sees caller")) == {"robot", "caller"}


def test_parse_atom_plays_form():
    assert parse_atom("caller plays beckon") == PlaysAtom("caller", "beckon")
    assert parse_atom("not caller plays beckon") == NotAtom(PlaysAtom("caller", "beckon"))
    assert atom_subjects(parse_atom("caller plays beckon")) == {"caller"}


def test_parse_not_and_within_atoms():
    assert parse_atom("not robot in kitchen") == NotAtom(MembershipAtom("robot", "kitchen"))
    assert parse_atom("robot within 2.5 of alice") == WithinAtom("robot", 2.5, "alice")
    assert parse_atom("not alice within 1 of robot") == NotAtom(WithinAtom("alice", 1.0, "robot"))


def test_atom_subjects():
    assert atom_subjects(parse_atom("door.open == true")) == frozenset()
    assert atom_subjects(parse_atom("not robot within 2 of alice")) == {"robot", "alice"}


@pytest.mark.parametrize(
    "text",
    [
        "door.open == true",
        "robot in kitchen",
        "robot within 2 of alice",
        "not alice in kitchen",
        "robot sees caller within 12",
        "robot sees caller within 2.5",
        "caller plays beckon",
        "not caller plays beckon",
    ],
)
def test_render_atom_round_trips(text):
    assert render_atom(parse_atom(text)) == text


def test_render_atom_writes_the_default_range():
    assert render_atom(parse_atom("robot sees caller")) == "robot sees caller within 12"


# ---------------------------------------------------------------------------
# parse_until
# ---------------------------------------------------------------------------


def test_parse_until_single_atom_is_the_atom():
    assert parse_until("robot in kitchen") == MembershipAtom("robot", "kitchen")
    assert parse_until("robot in kitchen for 0") == MembershipAtom("robot", "kitchen")


def test_parse_until_joins_atoms_and_takes_a_dwell():
    until = parse_until("robot sees caller within 12 and caller plays beckon for 0.5")
    assert until == HoldNode(AllNode((SeesAtom("robot", "caller", 12.0), PlaysAtom("caller", "beckon"))), 0.5)
    assert parse_until("robot within 2 of alice for 3") == HoldNode(WithinAtom("robot", 2.0, "alice"), 3.0)
    assert parse_until("robot in kitchen and alice in kitchen") == AllNode((MembershipAtom("robot", "kitchen"), MembershipAtom("alice", "kitchen")))


def test_parse_until_or_binds_loosest_and_for_belongs_to_its_clause():
    until = parse_until("robot sees caller within 12 and caller plays beckon for 0.5 or robot within 2.5 of caller")
    assert until == AnyNode(
        (
            HoldNode(AllNode((SeesAtom("robot", "caller", 12.0), PlaysAtom("caller", "beckon"))), 0.5),
            WithinAtom("robot", 2.5, "caller"),
        )
    )
    assert parse_until("robot in kitchen or alice in kitchen for 2") == AnyNode((MembershipAtom("robot", "kitchen"), HoldNode(MembershipAtom("alice", "kitchen"), 2.0)))


@pytest.mark.parametrize(
    "text",
    [
        "robot in kitchen",
        "robot in kitchen for 2",
        "robot sees caller within 12 and caller plays beckon for 0.5",
        "not alice in kitchen and door.open == true",
        "robot sees caller within 12 and caller plays beckon for 0.5 or robot within 2.5 of caller",
        "robot in kitchen for 1 or alice in kitchen for 2 or door.open == true",
    ],
)
def test_render_until_round_trips(text):
    assert render_until(parse_until(text)) == text


@pytest.mark.parametrize(
    "value",
    [
        "robot in kitchen for",
        "robot in kitchen for x",
        "robot in kitchen for -1",
        "robot in kitchen and",
        "and robot in kitchen",
        "for 2",
        "robot in kitchen or",
        "or robot in kitchen",
        "robot in kitchen or or alice in kitchen",
        "robot in kitchen for 2 and alice in kitchen",
        "robot in kitchen for 2 for 3",
    ],
)
def test_parse_until_rejects_malformed_forms(value):
    with pytest.raises(ValueError):
        parse_until(value)


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
        "not not robot in kitchen",
        "robot within x of alice",
        "robot within -1 of alice",
        "robot within 2 of env_0/alice",
        "robot sees",
        "robot sees caller within",
        "robot sees caller within x",
        "robot sees caller within -1",
        "robot sees env_0/caller",
        "caller plays",
        "env_0/caller plays beckon",
        "env_0/ward_a_door.open == false",
        "env_0/blocker_1 in ward_a",
        "robot in env_0/ward_a",
    ],
)
def test_parse_atom_rejects_malformed_shapes(value):
    with pytest.raises(ValueError):
        parse_atom(value)


def test_parse_atom_within_takes_point_literals():
    assert parse_atom("robot within 4.2 of 12.1,4.0") == WithinAtom(subject="robot", radius=4.2, other="12.1,4.0")
    assert parse_atom("-3,.5 within 1 of alice") == WithinAtom(subject="-3,.5", radius=1.0, other="alice")
    assert render_atom(parse_atom("robot within 4.2 of 12.1,4.0")) == "robot within 4.2 of 12.1,4.0"


@pytest.mark.parametrize(
    ("subject", "point"),
    [("12.1,4.0", (12.1, 4.0)), ("-3,.5", (-3.0, 0.5)), ("+1e1,2.", (10.0, 2.0)), ("alice", None), ("a,b", None), ("1,2,3", None), ("1,", None)],
)
def test_parse_point(subject, point):
    assert parse_point(subject) == point


@pytest.mark.parametrize("value", ["1,2 in ward_a", "robot sees 1,2", "1,2 sees alice", "1,2 plays beckon"])
def test_parse_atom_takes_points_only_as_within_subjects(value):
    with pytest.raises(ValueError, match="only as a within subject"):
        parse_atom(value)


def test_atom_subjects_leave_points_out():
    assert atom_subjects(parse_atom("robot within 4 of 1,2")) == {"robot"}
    assert atom_subjects(parse_atom("not 1,2 within 4 of 3,4")) == frozenset()


# ---------------------------------------------------------------------------
# EpisodeCondition.parse: op arity enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "p"),
    [("eventually", "robot in ward_a"), ("never", "robot in pharmacy"), ("always", "robot in ward_a")],
)
def test_parse_unary_op_has_no_q(op, p):
    cond = EpisodeCondition.parse({"op": op, "p": p})
    assert (cond.op, cond.p, cond.q) == (op, p, None)


def test_parse_binary_op_before():
    cond = EpisodeCondition.parse({"op": "before", "p": "robot in ward_a", "q": "robot in pharmacy"})
    assert cond.q == "robot in pharmacy"


def test_parse_binary_op_never_during():
    cond = EpisodeCondition.parse(
        {"op": "never_during", "p": "robot in ward_a_doorway", "q": "ward_a_door.open == false"},
    )
    assert cond.p == "robot in ward_a_doorway"
    assert cond.q == "ward_a_door.open == false"


def test_condition_on_failure_round_trip():
    cond = EpisodeCondition.parse({"op": "never", "p": "robot in hallway", "on_failure": "abort_episode"})
    assert cond.on_failure == "abort_episode"
    assert EpisodeCondition.parse(cond.serialize()) == cond
    assert "on_failure" not in EpisodeCondition(op="never", p="robot in hallway").serialize()
    with pytest.raises(ValueError):
        EpisodeCondition.parse({"op": "never", "p": "robot in hallway", "on_failure": "explode"})


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


# ---------------------------------------------------------------------------
# parse_node: structured conditions
# ---------------------------------------------------------------------------


def test_parse_node_reads_text_with_the_until_grammar():
    assert parse_node("robot in kitchen") == MembershipAtom("robot", "kitchen")
    assert parse_node("a.x == 1 and b.y == 2 for 3 or robot in kitchen") == AnyNode((HoldNode(AllNode((EntityAtom("a", "x", "1"), EntityAtom("b", "y", "2"))), 3.0), MembershipAtom("robot", "kitchen")))


@pytest.mark.parametrize(
    ("value", "node"),
    [
        ({"field": "wardrobe.openness.door_left", "ge": 0.8}, FieldTest("wardrobe", "openness.door_left", "ge", 0.8)),
        ({"field": "wardrobe.state", "eq": "open"}, FieldTest("wardrobe", "state", "eq", "open")),
        ({"field": "wardrobe.moving", "eq": True}, FieldTest("wardrobe", "moving", "eq", "true")),
        ({"field": "wardrobe.q.hinge", "ne": 1}, FieldTest("wardrobe", "q.hinge", "ne", 1.0)),
        ({"field": "stove.angle.knob", "lt": -1}, FieldTest("stove", "angle.knob", "lt", -1.0)),
        ({"field": "stove.angle.knob", "le": 0}, FieldTest("stove", "angle.knob", "le", 0.0)),
        ({"field": "stove.angle.knob", "gt": 0.5}, FieldTest("stove", "angle.knob", "gt", 0.5)),
        ({"field": "drawer.pos.slide", "within": [0.2, 0.4]}, FieldTest("drawer", "pos.slide", "within", (0.2, 0.4))),
        ({"some": "wardrobe.openness", "ge": 0.5}, FamilyTest("wardrobe", "openness", "some", "ge", 0.5)),
        ({"each": "stove.setting", "eq": "off"}, FamilyTest("stove", "setting", "each", "eq", "off")),
        ({"all": ["robot in kitchen", {"field": "door.open", "eq": "true"}]}, AllNode((MembershipAtom("robot", "kitchen"), FieldTest("door", "open", "eq", "true")))),
        ({"any": [{"subject": "robot", "in": "kitchen"}, "alice plays wave"]}, AnyNode((MembershipAtom("robot", "kitchen"), PlaysAtom("alice", "wave")))),
        ({"not": {"subject": "alice", "in": "kitchen"}}, NotNode(MembershipAtom("alice", "kitchen"))),
        ({"subject": "robot", "near": "alice", "radius": 2}, WithinAtom("robot", 2.0, "alice")),
        ({"subject": "robot", "near": "1,2", "radius": 0.5}, WithinAtom("robot", 0.5, "1,2")),
        ({"subject": "robot", "sees": "alice"}, SeesAtom("robot", "alice", SEES_RANGE_M)),
        ({"subject": "robot", "sees": "alice", "range": 4}, SeesAtom("robot", "alice", 4.0)),
        ({"subject": "alice", "plays": "beckon"}, PlaysAtom("alice", "beckon")),
        ({"field": "wardrobe.openness.door_left", "ge": 0.8, "for": 2}, HoldNode(FieldTest("wardrobe", "openness.door_left", "ge", 0.8), 2.0)),
        ({"all": [{"subject": "robot", "in": "kitchen", "for": 1}, "door.open == true"], "for": 3}, HoldNode(AllNode((HoldNode(MembershipAtom("robot", "kitchen"), 1.0), EntityAtom("door", "open", "true"))), 3.0)),
        ({"any": ["robot in kitchen for 2", {"not": "alice in kitchen"}]}, AnyNode((HoldNode(MembershipAtom("robot", "kitchen"), 2.0), NotNode(MembershipAtom("alice", "kitchen"))))),
    ],
)
def test_parse_node_reads_every_structured_kind(value, node):
    assert parse_node(value) == node


@pytest.mark.parametrize(
    ("value", "key"),
    [
        ({"field": "wardrobe.openness.door_left", "ge": 0.8, "bogus": 1}, "bogus"),
        ({"field": "wardrobe.state"}, "field"),
        ({"field": "wardrobe.q.hinge", "ge": 0.8, "le": 1.0}, "field"),
        ({"some": "wardrobe.openness"}, "some"),
        ({"field": "drawer.pos.slide", "within": [0.4, 0.2]}, "within"),
        ({"field": "drawer.pos.slide", "within": [0.4]}, "within"),
        ({"field": "drawer.pos.slide", "within": "0.2..0.4"}, "within"),
        ({"field": "drawer.pos.slide", "within": [0.2, "x"]}, "within"),
        ({"field": "wardrobe.state", "lt": "open"}, "lt"),
        ({"field": "wardrobe.state", "le": "open"}, "le"),
        ({"field": "wardrobe.state", "gt": "open"}, "gt"),
        ({"field": "wardrobe.state", "ge": "open"}, "ge"),
        ({"field": "wardrobe.state", "eq": ["open"]}, "eq"),
        ({"field": "wardrobe.q.hinge", "ge": 0.5, "for": -1}, "for"),
        ({"field": "wardrobe.q.hinge", "ge": 0.5, "for": "long"}, "for"),
        ({"all": []}, "all"),
        ({"any": []}, "any"),
        ({"any": "robot in kitchen"}, "any"),
        ({"field": "wardrobe", "eq": "open"}, "field"),
        ({"field": ".state", "eq": "open"}, "field"),
        ({"some": "wardrobe", "ge": 0.5}, "some"),
        ({"each": "wardrobe.openness.door", "ge": 0.5}, "each"),
        ({"field": "env_0/wardrobe.state", "eq": "open"}, "env_0/wardrobe"),
        ({"some": "env_1/wardrobe.openness", "ge": 0.5}, "env_1/wardrobe"),
        ({"field": "wardrobe.state", "all": ["robot in kitchen"], "eq": "open"}, "all"),
        ({"all": ["robot in kitchen"], "eq": "open"}, "eq"),
        ({"subject": "robot", "in": "kitchen", "plays": "wave"}, "plays"),
        ({"subject": "robot", "near": "alice"}, "radius"),
        ({"subject": "robot", "near": "alice", "radius": -1}, "radius"),
        ({"subject": "robot", "in": "kitchen", "radius": 1}, "radius"),
        ({"subject": "robot", "sees": "alice", "range": "far"}, "range"),
        ({"subject": "robot in kitchen", "in": "kitchen"}, "subject"),
        ({"subject": "env_0/robot", "in": "kitchen"}, "env_0/robot"),
        ({"subject": "1,2", "in": "kitchen"}, "within subject"),
        ({"for": 2}, "field"),
        (3, "3"),
    ],
)
def test_parse_node_errors_name_the_offending_key(value, key):
    with pytest.raises(ValueError, match=re.escape(key)):
        parse_node(value)


def test_render_node_covers_every_kind():
    assert render_node(parse_node({"field": "wardrobe.openness.door_left", "ge": 0.8})) == "wardrobe openness.door_left >= 0.8"
    assert render_node(parse_node({"field": "wardrobe.state", "ne": "open"})) == "wardrobe state != open"
    assert render_node(parse_node({"field": "drawer.pos.slide", "within": [0.2, 0.4]})) == "drawer pos.slide within [0.2, 0.4]"
    assert render_node(parse_node({"some": "wardrobe.openness", "ge": 0.5})) == "some wardrobe openness >= 0.5"
    assert render_node(parse_node({"each": "stove.setting", "eq": "off"})) == "each stove setting == off"
    assert render_node(parse_node({"field": "wardrobe.q.hinge", "lt": 1, "for": 2})) == "wardrobe q.hinge < 1 for 2 s"
    assert render_node(parse_node({"not": {"any": ["robot in kitchen", "alice in kitchen"]}})) == "not (robot in kitchen or alice in kitchen)"
    assert render_node(parse_node({"all": [{"any": ["robot in kitchen", "alice in kitchen"]}, "door.open == true"]})) == "(robot in kitchen or alice in kitchen) and door.open == true"
    assert render_node(parse_node({"all": ["robot in kitchen for 1", "door.open == true"]})) == "(robot in kitchen for 1 s) and door.open == true"
    assert render_node(parse_node({"any": ["robot in kitchen", "alice in kitchen"], "for": 2})) == "(robot in kitchen or alice in kitchen) for 2 s"
    assert render_node(parse_node("robot in kitchen and alice in kitchen for 2 or door.open == true")) == "robot in kitchen and alice in kitchen for 2 s or door.open == true"
    assert render_node(parse_node({"field": "stove.angle.knob", "gt": 0.5})) == "stove angle.knob > 0.5"


def test_render_until_writes_structured_nodes_in_the_text_grammar_when_it_can():
    assert render_until(parse_node({"all": [{"subject": "robot", "in": "kitchen"}, {"subject": "alice", "plays": "wave"}], "for": 2})) == "robot in kitchen and alice plays wave for 2"
    assert render_until(parse_node({"field": "wardrobe.q.hinge", "ge": 0.5})) == "wardrobe q.hinge >= 0.5"


def test_node_subjects_fields_and_families():
    node = parse_node({"all": ["robot within 2 of alice", "not door.open == true", {"field": "wardrobe.openness.door_left", "ge": 0.8}, {"each": "stove.setting", "eq": "off"}, {"not": "caller plays beckon"}]})
    assert node_subjects(node) == {"robot", "alice", "caller"}
    assert node_fields(node) == {("door", "open"), ("wardrobe", "openness.door_left")}
    assert node_families(node) == {("stove", "setting")}
    assert node_subjects(parse_node({"field": "wardrobe.state", "eq": "open"})) == frozenset()


def test_check_fields_reports_missing_fields_and_families_and_skips_other_entities():
    fields = {"wardrobe": frozenset({"state", "target", "moving", "blocked", "open", "closed", "openness.door_left", "q.door_left"})}
    node = parse_node(
        {
            "all": [
                {"field": "wardrobe.openness.door_left", "ge": 0.8},
                {"field": "wardrobe.openness.door_right", "ge": 0.8},
                {"some": "wardrobe.openness", "ge": 0.5},
                {"some": "wardrobe.setting", "eq": "off"},
                "wardrobe.state == open",
                "door.open == true",
                {"each": "door.openness", "ge": 0.5},
            ]
        }
    )
    problems = check_fields(node, fields.get)
    assert len(problems) == 2
    assert "wardrobe has no field 'openness.door_right'" in problems[0]
    assert "wardrobe has no 'setting' fields" in problems[1]
    assert check_fields(parse_node("door.open == true"), fields.get) == []


# ---------------------------------------------------------------------------
# EpisodeCondition: structured p and q
# ---------------------------------------------------------------------------


def test_condition_keeps_the_form_it_was_read_in():
    raw = {"op": "never_during", "p": {"field": "wardrobe.openness.door_left", "ge": 0.5, "for": 1}, "q": "robot in hallway", "text": "Close it."}
    condition = EpisodeCondition.parse(raw)
    assert condition.p == {"field": "wardrobe.openness.door_left", "ge": 0.5, "for": 1}
    assert condition.p_node == HoldNode(FieldTest("wardrobe", "openness.door_left", "ge", 0.5), 1.0)
    assert condition.q_node == MembershipAtom("robot", "hallway")
    assert condition.serialize() == raw
    assert EpisodeCondition.parse(condition.serialize()) == condition
    assert converter.structure(converter.unstructure(condition), EpisodeCondition) == condition


def test_condition_copies_a_mapping_it_reads_and_serializes():
    raw = {"op": "eventually", "p": {"all": ["robot in kitchen", {"field": "door.open", "eq": True}]}}
    condition = EpisodeCondition.parse(raw)
    raw["p"]["all"].append("alice in kitchen")
    assert condition.p == {"all": ["robot in kitchen", {"field": "door.open", "eq": True}]}
    serialized = condition.serialize()
    serialized["p"]["all"].clear()
    assert condition.p["all"] == ["robot in kitchen", {"field": "door.open", "eq": True}]


def test_condition_text_p_takes_the_until_grammar():
    condition = EpisodeCondition.parse({"op": "eventually", "p": "robot in kitchen and alice in kitchen for 2"})
    assert condition.p_node == HoldNode(AllNode((MembershipAtom("robot", "kitchen"), MembershipAtom("alice", "kitchen"))), 2.0)
    assert condition.serialize() == {"op": "eventually", "p": "robot in kitchen and alice in kitchen for 2"}


@pytest.mark.parametrize(
    "value",
    [
        {"op": "eventually", "p": {"field": "wardrobe.state"}},
        {"op": "before", "p": "robot in kitchen", "q": {"all": []}},
        {"op": "eventually", "p": ["robot in kitchen"]},
    ],
)
def test_condition_rejects_malformed_structured_conditions(value):
    with pytest.raises(ValueError):
        EpisodeCondition.parse(value)
