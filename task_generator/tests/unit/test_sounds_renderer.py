from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")

from arena_simulation_setup.shared import Position, Sound
from arena_simulation_setup.shared.semantics import parse_semantics
from task_generator.tasks.modules.sounds.impl import (
    _CatalogEntry,
    _catalog_default,
    _catalog_lookup,
    _index_static_entities,
    _merge_params,
    _parse_catalog,
    _realize_frame,
    _resolve_sound_placement,
    _sound_group_id,
)


def _entity(name: str, x: float, y: float, z: float, yaw: float) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=z),
            orientation=SimpleNamespace(to_yaw=lambda: yaw),
        ),
    )


def _level(static: list) -> SimpleNamespace:
    return SimpleNamespace(all_static_entities=static)


def _world(**levels: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(levels=levels)


# ---------------------------------------------------------------------------
# static entity indexing
# ---------------------------------------------------------------------------


def test_index_static_entities_indexes_by_level() -> None:
    desk = _entity("desk", 1.0, 2.0, 0.0, 0.0)
    world = _world(a=_level([desk]), b=_level([]))
    indexed = _index_static_entities(world)
    assert indexed == {"desk": [("a", desk)]}


def test_index_static_entities_includes_scenario_statics() -> None:
    desk = _entity("desk", 1.0, 2.0, 0.0, 0.0)
    crate = _entity("crate", 3.0, 3.0, 0.0, 0.0)
    crate.level_id = "a"
    loose = _entity("loose", 0.0, 0.0, 0.0, 0.0)
    loose.level_id = None
    indexed = _index_static_entities(_world(a=_level([desk])), [crate, loose])
    assert indexed == {"desk": [("a", desk)], "crate": [("a", crate)], "loose": [("", loose)]}


def test_index_static_entities_tracks_duplicate_names_across_levels() -> None:
    desk_a = _entity("desk", 0.0, 0.0, 0.0, 0.0)
    desk_b = _entity("desk", 5.0, 5.0, 0.0, 0.0)
    world = _world(a=_level([desk_a]), b=_level([desk_b]))
    indexed = _index_static_entities(world)
    assert indexed["desk"] == [("a", desk_a), ("b", desk_b)]


# ---------------------------------------------------------------------------
# sound placement resolution
# ---------------------------------------------------------------------------


def test_resolve_sound_placement_entity_ref_rotates_offset_by_entity_yaw() -> None:
    desk = _entity("desk", 1.0, 2.0, 0.5, math.pi / 2)
    indexed = {"desk": [("a", desk)]}
    sound = Sound(name="chime", asset_id="radio_loop", entity_ref="desk", offset=Position(1.0, 0.0, 0.25))
    world = _world(a=_level([desk]))

    position, yaw, level_id = _resolve_sound_placement(sound, world, indexed, "a")

    assert level_id == "a"
    assert yaw == pytest.approx(math.pi / 2)
    assert position.x == pytest.approx(1.0)
    assert position.y == pytest.approx(3.0)
    assert position.z == pytest.approx(0.75)


def test_resolve_sound_placement_entity_ref_unknown_raises() -> None:
    sound = Sound(name="chime", asset_id="radio_loop", entity_ref="missing")
    world = _world(a=_level([]))
    with pytest.raises(ValueError, match="unknown static entity"):
        _resolve_sound_placement(sound, world, {}, "a")


def test_resolve_sound_placement_entity_ref_ambiguous_raises() -> None:
    desk_a = _entity("desk", 0.0, 0.0, 0.0, 0.0)
    desk_b = _entity("desk", 5.0, 5.0, 0.0, 0.0)
    indexed = {"desk": [("a", desk_a), ("b", desk_b)]}
    sound = Sound(name="chime", asset_id="radio_loop", entity_ref="desk")
    world = _world(a=_level([desk_a]), b=_level([desk_b]))
    with pytest.raises(ValueError, match="ambiguous static entity"):
        _resolve_sound_placement(sound, world, indexed, "a")


def test_resolve_sound_placement_position_uses_context_level() -> None:
    sound = Sound(name="alarm", asset_id="alarm_loop", position=Position(2.0, 3.0, 0.0), offset=Position(0.0, 0.0, 1.0))
    world = _world(a=_level([]), b=_level([]))

    position, yaw, level_id = _resolve_sound_placement(sound, world, {}, "b")

    assert level_id == "b"
    assert yaw == 0.0
    assert position.x == pytest.approx(2.0)
    assert position.y == pytest.approx(3.0)
    assert position.z == pytest.approx(1.0)


def test_resolve_sound_placement_position_infers_sole_level() -> None:
    sound = Sound(name="alarm", asset_id="alarm_loop", position=Position(0.0, 0.0, 0.0))
    world = _world(only=_level([]))

    _, _, level_id = _resolve_sound_placement(sound, world, {}, None)

    assert level_id == "only"


def test_resolve_sound_placement_position_requires_level_in_multi_level_world() -> None:
    sound = Sound(name="alarm", asset_id="alarm_loop", position=Position(0.0, 0.0, 0.0))
    world = _world(a=_level([]), b=_level([]))
    with pytest.raises(ValueError, match="requires level in a multi-level world"):
        _resolve_sound_placement(sound, world, {}, None)


def test_resolve_sound_placement_position_honors_sound_level_without_context() -> None:
    sound = Sound(name="alarm", asset_id="alarm_loop", position=Position(0.0, 0.0, 0.0), level="b")
    world = _world(a=_level([]), b=_level([]))

    _, _, level_id = _resolve_sound_placement(sound, world, {}, None)

    assert level_id == "b"


# ---------------------------------------------------------------------------
# frame realization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "frame", "expected"),
    [
        ("env_0", "jackal/base_link", "env_0/jackal/base_link"),
        ("env_0", "env_0/jackal/base_link", "env_0/jackal/base_link"),
        ("env_0", "env_0", "env_0"),
        ("/env_0/", "jackal/base_link", "env_0/jackal/base_link"),
        ("", "jackal/base_link", "jackal/base_link"),
        ("env_0", "env_01/jackal/base_link", "env_0/env_01/jackal/base_link"),
    ],
)
def test_realize_frame_prefixes_once(env: str, frame: str, expected: str) -> None:
    assert _realize_frame(env, frame) == expected


# ---------------------------------------------------------------------------
# group_id derivation
# ---------------------------------------------------------------------------


def test_merge_params_shallow_merges_across_cfgs() -> None:
    cfgs = parse_semantics(
        [
            {"predicate": "sounding", "params": {"sound_on": "fire_alarm"}},
            {"state": "volume_db", "params": {"regime": "fire_alarm"}},
        ],
    )
    assert _merge_params(cfgs) == {"sound_on": "fire_alarm", "regime": "fire_alarm"}


def test_sound_group_id_prefers_sound_on_regime() -> None:
    cfgs = parse_semantics([{"predicate": "sounding", "params": {"sound_on": "fire_alarm"}}])
    assert _sound_group_id(cfgs, "env_0/1_alarm") == "fire_alarm"


def test_sound_group_id_falls_back_to_realized_name() -> None:
    cfgs = parse_semantics([{"preset": "sound"}])
    assert _sound_group_id(cfgs, "env_0/1_alarm") == "env_0/1_alarm"


# ---------------------------------------------------------------------------
# catalog lookup
# ---------------------------------------------------------------------------


def test_parse_catalog_extracts_category_tags_and_level() -> None:
    raw = {
        "assets": {
            "radio_loop": {"category": "music", "semantic_tags": ["radio", "music"], "reference_level_db": 62.0},
            "no_tags": {"category": "misc", "reference_level_db": 50},
        },
    }
    catalog = _parse_catalog(raw)
    assert catalog == {
        "radio_loop": _CatalogEntry("music", ("radio", "music"), 62.0),
        "no_tags": _CatalogEntry("misc", (), 50.0),
    }


def test_parse_catalog_requires_reference_level() -> None:
    with pytest.raises(KeyError):
        _parse_catalog({"assets": {"x": {"category": "music"}}})


def test_catalog_lookup_known_asset() -> None:
    catalog = {"radio_loop": _CatalogEntry("music", ("radio", "music"), 62.0)}
    sound_type, tags, found = _catalog_lookup(catalog, "radio_loop")
    assert found is True
    assert sound_type == "music"
    assert tags == ("radio", "music")


def test_catalog_default_picks_first_asset_of_type() -> None:
    catalog = {
        "a": _CatalogEntry("alarm", (), 88.0),
        "b": _CatalogEntry("music", (), 62.0),
        "c": _CatalogEntry("music", (), 70.0),
    }
    assert _catalog_default(catalog, "music") == ("b", 62.0)
    assert _catalog_default(catalog, "alarm") == ("a", 88.0)
    assert _catalog_default(catalog, "speech") is None


def test_catalog_lookup_unknown_asset_falls_back_to_asset_id() -> None:
    sound_type, tags, found = _catalog_lookup({}, "mystery_sound")
    assert found is False
    assert sound_type == "mystery_sound"
    assert tags == ()
