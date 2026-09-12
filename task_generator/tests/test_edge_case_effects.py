from __future__ import annotations

import pytest

from task_generator.tasks.obstacles.edge_case.effects import (
    After, EffectError, Hold, Intercept, Kind, Rally, Shuffle, TrackRobot,
    injected_count, parse_effect, parse_effects, parse_scope, parse_when,
)


def test_unknown_type_names_the_alternatives() -> None:
    with pytest.raises(EffectError, match="is not one of"):
        parse_effect({"type": "sprint"})


def test_typo_in_a_key_is_refused() -> None:
    with pytest.raises(EffectError, match="unknown key"):
        parse_effect({"type": "rally", "target": "lobby", "scop": "all"})


def test_intercept_defaults() -> None:
    e = parse_effect({"type": "intercept"})
    assert isinstance(e, Intercept)
    assert (e.angle, e.route_fraction, e.after, e.waypoint_mode) == (180.0, 0.5, After.CONTINUE, "reverse")


def test_intercept_fraction_must_be_inside_the_route() -> None:
    with pytest.raises(EffectError, match="route_fraction"):
        parse_effect({"type": "intercept", "route_fraction": 1.0})


def test_veer_needs_a_destination() -> None:
    with pytest.raises(EffectError, match="veer_to"):
        parse_effect({"type": "intercept", "after": "veer"})
    e = parse_effect({"type": "intercept", "after": "veer", "veer_to": "reception"})
    assert e.veer_to == "reception"


def test_veer_to_only_with_veer() -> None:
    with pytest.raises(EffectError, match="only applies"):
        parse_effect({"type": "intercept", "veer_to": [1, 2]})


def test_rally_needs_a_target() -> None:
    with pytest.raises(EffectError, match="needs a 'target'"):
        parse_effect({"type": "rally"})


def test_rally_targets_may_be_zones_or_points() -> None:
    e = parse_effect({"type": "rally", "targets": ["exit", [1.0, 2.0]], "scope": ["a", "b"]})
    assert isinstance(e, Rally)
    assert e.targets == ("exit", (1.0, 2.0))
    assert e.scope == ("a", "b")


def test_scatter_is_a_one_shot_shuffle() -> None:
    e = parse_effect({"type": "scatter", "radius": 2.0})
    assert isinstance(e, Shuffle)
    assert e.once and e.radius == 2.0 and e.kind is Kind.SHUFFLE


def test_shuffle_period_is_floored() -> None:
    assert parse_effect({"type": "shuffle", "period": 0.01}).period == 1.0


def test_track_robot_and_hold() -> None:
    t = parse_effect({"type": "track_robot", "duration": 5, "scope": "injected"})
    assert isinstance(t, TrackRobot) and t.duration == 5.0 and t.scope == "injected"
    h = parse_effect({"type": "hold", "duration": 4, "scope": ["x"]})
    assert isinstance(h, Hold) and h.duration == 4.0


def test_hold_duration_must_be_positive() -> None:
    with pytest.raises(EffectError, match="positive"):
        parse_effect({"type": "hold", "duration": 0})


def test_when_gates_are_exclusive() -> None:
    with pytest.raises(EffectError, match="one of"):
        parse_when({"robot_within": 3.0, "route_fraction": 0.5})


def test_when_of_needs_a_proximity_gate() -> None:
    with pytest.raises(EffectError, match="only applies"):
        parse_when({"at": 3.0, "of": "lobby"})


def test_when_round_trips() -> None:
    w = parse_when({"robot_within": 3.0, "of": [1, 2], "at": 2})
    assert w.gated and w.as_dict() == {"at": 2.0, "robot_within": 3.0, "of": [1.0, 2.0]}
    assert parse_when(None).as_dict() == {"at": 0.0}


def test_scope_forms() -> None:
    assert parse_scope(None, where="x") == "all"
    assert parse_scope("Injected", where="x") == "injected"
    assert parse_scope(["a", "b"], where="x") == ("a", "b")
    with pytest.raises(EffectError, match="scope"):
        parse_scope("some", where="x")
    with pytest.raises(EffectError, match="non-empty"):
        parse_scope([], where="x")


def test_effects_list_and_count() -> None:
    effects = parse_effects([{"type": "intercept"}, {"type": "rally", "target": "exit"}, {"type": "intercept", "angle": 90}])
    assert injected_count(effects) == 2
    assert parse_effects(None) == ()
    with pytest.raises(EffectError, match="expected a list"):
        parse_effects({"type": "rally"})


def test_as_dict_is_json_shaped() -> None:
    d = parse_effect({"type": "rally", "targets": [[1, 2]], "scope": ["a"], "when": {"at": 1}}).as_dict()
    assert d["type"] == "rally" and d["targets"] == [[1.0, 2.0]] and d["scope"] == ("a",) or d["scope"] == ["a"]
    assert d["when"] == {"at": 1.0}


def test_retune_needs_a_change() -> None:
    from task_generator.tasks.obstacles.edge_case.effects import Retune

    with pytest.raises(EffectError, match="would change nothing"):
        parse_effect({"type": "retune", "scope": ["a"]})
    e = parse_effect({"type": "retune", "scope": ["a"], "profile": "./types/adult__vision_range_low.yaml", "speed_scale": 0.55, "when": {"at": 10}})
    assert isinstance(e, Retune) and e.speed_scale == 0.55 and e.when.at == 10.0
    assert parse_effect({"type": "retune", "speed": 0.6}).speed == 0.6
