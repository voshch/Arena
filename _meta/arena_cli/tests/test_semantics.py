"""The semantics subcommand of arena env: argument parsing, name matching, value checks and rendering."""

import random

import pytest

from arena_cli.common import CLIError
from arena_cli.semantics import Kind, changes, describe, draw, entity, fitting, index, labels, parse, plan, render, select, split_env

LIGHT = Kind(frozenset({"lit", "level", "dead_fraction"}), {"level": (0.0, 1.0), "dead_fraction": (0.0, 1.0)})
DOOR = Kind(frozenset(), {})
KINDS = {"light": LIGHT, "door": DOOR}


def _light(name: str, level: float = 1.0, lit: bool = True, env: str = "env_0") -> object:
    return entity(env, f"{env}/{name}", "light", [], [("level", level)], [("lit", lit)])


def _world() -> list:
    return [
        _light("hall/0"),
        _light("desk_lamp_light/0", level=0.4),
        _light("exit_strip/0", lit=False),
        entity("env_0", "env_0/front_door/0", "door", [("state", "closed")], [], [("open", False)]),
        entity("env_0", "env_0/office/0", "zone", [("regime", "day")], [("noise_db", 40.0)], []),
    ]


def test_split_env_takes_the_selector_before_semantics() -> None:
    assert split_env(["0", "semantics", "hall", "level=0.3"]) == (["0"], ["hall", "level=0.3"])
    assert split_env(["semantics"]) == ([], [])
    assert split_env(["--all", "semantics", "--kind", "light"]) == (["--all"], ["--kind", "light"])


def test_split_env_leaves_launch_arguments_to_the_launch() -> None:
    assert split_env(["world:=office_1", "robot:=jackal"]) is None
    assert split_env(["world:=semantics_lab", "semantics"]) is None
    assert split_env([]) is None


@pytest.mark.parametrize(
    ("selector", "env", "ns", "all_envs"),
    [
        ([], None, None, False),
        (["0"], "env_0", None, False),
        (["env_3"], "env_3", None, False),
        (["--ns", "/arena/env_1"], None, "/arena/env_1", False),
        (["--all"], None, None, True),
    ],
)
def test_parse_reads_the_env_selector(selector: list[str], env: str | None, ns: str | None, all_envs: bool) -> None:
    query = parse(selector, [])
    assert (query.env, query.ns, query.all_envs) == (env, ns, all_envs)


def test_parse_reads_entity_assignments_and_options() -> None:
    query = parse(["0"], ["hall", "level=0.3", "lit=true", "--seed", "7"])
    assert (query.entity, query.assignments, query.seed) == ("hall", [("level", "0.3"), ("lit", "true")], 7)
    assert parse([], ["--kind", "light", "lit=false"]).kind == "light"
    assert parse([], ["--watch", "--kind", "light"]).watch


@pytest.mark.parametrize(
    ("selector", "argv", "message"),
    [
        (["0", "1"], [], "one ENV"),
        (["0", "--all"], [], "one of ENV"),
        (["--ns"], [], "takes a namespace"),
        (["--bogus"], [], "unknown option '--bogus' before"),
        ([], ["hall", "level:=0.3"], "single ="),
        ([], ["level=0.3"], "name an ENTITY or pass --kind"),
        ([], ["hall", "level=0.3", "lamp"], "unexpected 'lamp'"),
        ([], ["--fast"], "options are --kind KIND, --watch, --seed N"),
        ([], ["--seed", "x"], "integer"),
        ([], ["--watch", "hall"], "--watch takes only --kind"),
        ([], ["hall", "level="], "needs FIELD=VALUE"),
    ],
)
def test_parse_rejects_with_the_accepted_forms(selector: list[str], argv: list[str], message: str) -> None:
    with pytest.raises(CLIError, match=message):
        parse(selector, argv)


def test_select_matches_bare_names_level_names_and_globs() -> None:
    world = _world()
    assert [e.name for e in select(world, "hall", None)] == ["hall/0"]
    assert [e.name for e in select(world, "hall/0", None)] == ["hall/0"]
    assert [e.name for e in select(world, "*_light", None)] == ["desk_lamp_light/0"]
    assert [e.name for e in select(world, None, "light")] == ["hall/0", "desk_lamp_light/0", "exit_strip/0"]


def test_select_lists_what_exists_when_nothing_matches() -> None:
    with pytest.raises(CLIError, match="entities: desk_lamp_light, exit_strip, front_door, hall, office"):
        select(_world(), "lobby", None)
    with pytest.raises(CLIError, match="kinds present: door, light, zone"):
        select(_world(), None, "signal")


def test_draw_keeps_literals_and_draws_ranges_reproducibly() -> None:
    assert draw("0.3", random.Random(1)) == "0.3"
    assert draw("false", random.Random(1)) == "false"
    first, again = draw("0.3..1.0", random.Random(7)), draw("0.3..1.0", random.Random(7))
    assert first == again
    assert 0.3 <= float(first) <= 1.0
    assert 0.0 <= float(draw("random", random.Random(7))) <= 1.0
    with pytest.raises(CLIError, match="two numbers"):
        draw("low..high", random.Random(1))


def test_plan_sets_every_matched_entity_with_its_own_draw() -> None:
    writes = plan(select(_world(), None, "light"), [("level", "0.2..0.6"), ("lit", "TRUE")], KINDS, random.Random(3))
    levels = [value for _, field, value in writes if field == "level"]
    assert len(levels) == 3
    assert len(set(levels)) == 3
    assert all(0.2 <= float(level) <= 0.6 for level in levels)
    assert {value for _, field, value in writes if field == "lit"} == {"true"}


@pytest.mark.parametrize(
    ("pattern", "assignment", "message"),
    [
        ("hall", ("level", "1.5"), r"hall.level takes a number in 0..1, got 1.5"),
        ("hall", ("level", "bright"), "takes a number, got 'bright'"),
        ("hall", ("lit", "maybe"), "takes true or false"),
        ("hall", ("colour", "red"), r"hall.colour does not exist on this light, writable fields: level \(number\), lit \(true\|false\)"),
        ("front_door", ("open", "true"), r"front_door.open is read-only on this door, writable fields: none"),
    ],
)
def test_plan_rejects_with_the_writable_fields_and_ranges(pattern: str, assignment: tuple[str, str], message: str) -> None:
    with pytest.raises(CLIError, match=message):
        plan(select(_world(), pattern, None), [assignment], KINDS, random.Random(0))


def test_plan_treats_every_zone_field_as_writable() -> None:
    ((target, field, value),) = plan(select(_world(), "office", None), [("regime", "night")], KINDS, random.Random(0))
    assert (target.name, field, value) == ("office/0", "regime", "night")


def test_render_lists_each_entity_with_writable_fields_marked() -> None:
    text = render(_world(), KINDS)
    assert "light  hall  " in text
    assert "level=1* lit=true*" in text
    assert "state=closed open=false" in text
    assert "regime=day* noise_db=40*" in text


def test_describe_states_type_range_and_writability() -> None:
    text = describe(_light("hall/0", level=0.25), KINDS, "hall")
    assert text.splitlines()[0] == "hall  (light, env_0/hall/0)"
    assert "  level = 0.25    number in 0..1, writable" in text
    assert "  lit = true    true|false, writable" in text


def test_changes_report_each_changed_field() -> None:
    before = index(_world())
    after = before | index([_light("hall/0", level=0.3)])
    assert changes(before, after) == ["env_0 light hall level: 1 -> 0.3"]
    assert changes({}, index([_light("hall/0")])) == ["env_0 light hall level: - -> 1", "env_0 light hall lit: - -> true"]


def test_labels_keep_the_level_only_where_several_levels_share_a_name() -> None:
    world = [_light("hall/0"), _light("hall/1"), _light("lobby/0"), _light("1_elevator")]
    assert labels(world) == {("light", "env_0/hall/0"): "hall/0", ("light", "env_0/hall/1"): "hall/1", ("light", "env_0/lobby/0"): "lobby", ("light", "env_0/1_elevator"): "1_elevator"}


def _ward() -> list:
    return [_light("ward/0"), entity("env_0", "env_0/ward/0", "zone", [("regime", "day")], [("noise_db", 40.0)], [])]


def test_a_zone_and_its_ceiling_light_of_the_same_name_both_stay_listed() -> None:
    ward = _ward()
    assert sorted(index(ward)) == [("light", "env_0/ward/0"), ("zone", "env_0/ward/0")]
    assert set(labels(ward).values()) == {"ward"}
    text = render(ward, KINDS)
    assert "light  ward  level=1* lit=true*" in text
    assert "zone   ward  regime=day* noise_db=40*" in text
    assert [e.kind for e in select(ward, None, "light")] == ["light"]


def test_a_field_goes_to_the_same_named_entity_that_has_it() -> None:
    ward = select(_ward(), "ward", None)
    assert [e.kind for e in ward] == ["light", "zone"]
    ((target, field, value),) = plan(fitting(ward, [("level", "0.3")]), [("level", "0.3")], KINDS, random.Random(0))
    assert (target.kind, field, value) == ("light", "level", "0.3")
    ((target, field, value),) = plan(fitting(ward, [("regime", "night")]), [("regime", "night")], KINDS, random.Random(0))
    assert (target.kind, field, value) == ("zone", "regime", "night")


def test_a_field_no_matched_entity_has_is_rejected() -> None:
    ward = select(_ward(), "ward", None)
    assert fitting(ward, [("colour", "red")]) == ward
    with pytest.raises(CLIError, match="ward.colour does not exist on this light"):
        plan(fitting(ward, [("colour", "red")]), [("colour", "red")], KINDS, random.Random(0))


def test_a_glob_sets_a_field_on_the_entities_that_have_it() -> None:
    writes = plan(fitting(select(_world(), "*", None), [("level", "0.5")]), [("level", "0.5")], KINDS, random.Random(0))
    assert sorted(target.name for target, _, _ in writes) == ["desk_lamp_light/0", "exit_strip/0", "hall/0"]
