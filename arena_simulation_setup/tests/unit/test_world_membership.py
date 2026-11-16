from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.kg import DERIVED, GENERATED, Scene
from arena_simulation_setup.params import DERIVE_MODES
from arena_simulation_setup.tree.World import WorldView, connections, contents, derive, describe, parse_world, zones_of
from arena_simulation_setup.tree.World.model import World

WORLD = """\
format: arena-world/2
levels:
- id: '0'
  zones:
  - {name: a, polygon: [[0, 0], [4, 0], [4, 4], [0, 4]]}
  - {name: b, polygon: [[4, 0], [8, 0], [8, 4], [4, 4]]}
  doors:
  - {name: d, span: [[4, 1], [4, 2]]}
  elevators:
  - {name: e0, position: [2, 2], destinations: [e1]}
  objects:
  - {name: o, model: Desk, pose: [1, 1]}
  lights:
  - {name: l, fixture: bulb, lumens: 10, pose: {x: 5, y: 1, z: 2}}
  - {name: lamp, fixture: bulb, lumens: 10, attach: o, offset: {x: 0, y: 0, z: 1}}
  - {name: sun, fixture: sun, lux: 1000}
  sounds:
  - {name: s, asset_id: alarm, position: [6, 3]}
  - {name: radio, asset_id: radio, attach: o}
- id: '1'
  zones:
  - {name: top, polygon: [[0, 0], [4, 0], [4, 4], [0, 4]]}
  elevators:
  - {name: e1, position: [2, 2], destinations: [e0]}
"""

KG_YAML = """\
format: arena-kg/1
uses: [core@2]
entities:
- {id: room:a}
- {id: room:b, props: {description: the far room}}
- {id: obj:o}
- {id: door:d}
facts:
- {subject: room:b, verb: contains, object: obj:o}
- {subject: room:b, verb: connects, object: room:a, params: {door: door:d}}
"""


def _world() -> World:
    return parse_world(yaml.safe_load(WORLD))


def _derived(scene: Scene) -> set[tuple]:
    return {(fact.subject, fact.verb, fact.object) for fact in scene.facts if fact.source == DERIVED}


def _world_dir(root: Path, kg: str | None = KG_YAML) -> Path:
    (root / 'world.yaml').write_text(WORLD, encoding='utf-8')
    if kg is not None:
        (root / 'kg.yaml').write_text(kg, encoding='utf-8')
    return root


def test_auto_puts_every_zone_on_its_floor() -> None:
    derived = _derived(derive(_world(), 'auto'))
    assert {('floor:0', 'contains', 'room:a'), ('floor:0', 'contains', 'room:b'), ('floor:1', 'contains', 'room:top')} <= derived


def test_auto_puts_placed_things_in_the_zone_that_holds_them() -> None:
    scene = derive(_world(), 'auto')
    assert sorted(contents(scene, 'a')) == ['door:d', 'elevator:e0', 'light:lamp', 'obj:o', 'sound:radio']
    assert sorted(contents(scene, 'b')) == ['door:d', 'light:l', 'sound:s']
    assert contents(scene, 'top') == ['elevator:e1']
    assert zones_of(scene, 'light:sun') == []


def test_auto_connects_the_zones_beside_a_door() -> None:
    scene = derive(_world(), 'auto')
    assert connections(scene) == [('a', 'b', 'd')]
    fact = scene.get_fact('room:a', 'connects', 'room:b', {'door': 'door:d'})
    assert fact is not None and fact.params == {'door': 'door:d'} and fact.source == DERIVED


def test_auto_links_elevator_cabins_across_floors() -> None:
    derived = _derived(derive(_world(), 'auto'))
    assert {
        ('elevator:e0', 'connects_floor', 'elevator:e1'),
        ('elevator:e1', 'connects_floor', 'elevator:e0'),
        ('floor:0', 'contains', 'elevator:e0'),
        ('floor:1', 'contains', 'elevator:e1'),
    } <= derived


def test_off_derives_nothing() -> None:
    scene = derive(_world(), 'off')
    assert scene.facts == []
    assert scene.entities == {}


def test_off_keeps_stated_facts_and_drops_derived_ones() -> None:
    world = _world()
    scene = Scene()
    scene.add_entity('room:b')
    scene.add_entity('obj:o')
    scene.add_entity('room:a')
    scene.add_fact('room:b', 'contains', 'obj:o')
    scene.add_fact('room:a', 'contains', 'obj:o', source=DERIVED)
    world.kg = scene
    result = derive(world, 'off')
    assert [(fact.subject, fact.object, fact.source) for fact in result.facts] == [('room:b', 'obj:o', GENERATED)]


def test_stale_derived_facts_are_recomputed() -> None:
    world = _world()
    scene = Scene()
    scene.add_entity('room:b')
    scene.add_entity('obj:o')
    scene.add_fact('room:b', 'contains', 'obj:o', source=DERIVED)
    world.kg = scene
    assert zones_of(derive(world, 'auto'), 'obj:o') == ['a']


def test_a_stated_contains_is_never_rederived_into_another_zone() -> None:
    world = _world()
    scene = Scene()
    scene.add_entity('room:b')
    scene.add_entity('obj:o')
    scene.add_fact('room:b', 'contains', 'obj:o')
    world.kg = scene
    assert zones_of(derive(world, 'auto'), 'obj:o') == ['b']


def test_a_stated_door_contains_suppresses_the_derived_one_but_not_its_connection() -> None:
    world = _world()
    scene = Scene()
    scene.add_entity('room:a')
    scene.add_entity('door:d')
    scene.add_fact('room:a', 'contains', 'door:d')
    world.kg = scene
    result = derive(world, 'auto')
    assert zones_of(result, 'door:d') == ['a']
    assert connections(result) == [('a', 'b', 'd')]


def test_a_stated_floor_membership_is_kept() -> None:
    world = _world()
    scene = Scene()
    scene.add_entity('floor:1')
    scene.add_entity('room:a')
    scene.add_fact('floor:1', 'contains', 'room:a')
    world.kg = scene
    result = derive(world, 'auto')
    assert [fact.subject for fact in result.facts if fact.verb == 'contains' and fact.object == 'room:a'] == ['floor:1']


def test_a_stated_connects_for_a_door_suppresses_the_derived_one() -> None:
    world = _world()
    scene = Scene()
    scene.add_entity('room:a')
    scene.add_entity('room:b')
    scene.add_entity('door:d')
    scene.add_fact('room:b', 'connects', 'room:a', {'door': 'door:d'})
    world.kg = scene
    result = derive(world, 'auto')
    assert connections(result) == [('b', 'a', 'd')]
    assert result.get_fact('room:a', 'connects', 'room:b', {'door': 'door:d'}) is None


def test_kg_beside_world_yaml_wins_over_geometry(tmp_path: Path) -> None:
    world = WorldView(_world_dir(tmp_path)).load()
    assert world.kg is not None
    assert zones_of(world.kg, 'obj:o') == ['b']
    assert connections(world.kg) == [('b', 'a', 'd')]
    assert describe(world.kg, 'b') == 'the far room'
    assert zones_of(world.kg, 'sound:s') == ['b']


def test_load_with_derive_off_keeps_only_the_kg(tmp_path: Path) -> None:
    world = WorldView(_world_dir(tmp_path)).load(derive='off')
    assert world.kg is not None
    assert {(fact.subject, fact.verb, fact.object) for fact in world.kg.facts} == {('room:b', 'contains', 'obj:o'), ('room:b', 'connects', 'room:a')}


def test_load_with_derive_all_replaces_stated_geometry_and_keeps_the_rest(tmp_path: Path) -> None:
    world = WorldView(_world_dir(tmp_path)).load(derive='all')
    assert world.kg is not None
    assert zones_of(world.kg, 'obj:o') == ['a']
    assert connections(world.kg) == [('a', 'b', 'd')]
    assert describe(world.kg, 'b') == 'the far room'
    assert all(fact.source == DERIVED for fact in world.kg.facts)


def test_derive_all_keeps_stated_facts_geometry_cannot_derive() -> None:
    world = _world()
    world.kg = Scene()
    world.kg.add_entity('room:a')
    world.kg.add_entity('region:queue')
    world.kg.add_fact('room:a', 'contains', 'region:queue')
    world.kg.add_fact('room:b', 'contains', 'obj:o')
    scene = derive(world, 'all')
    assert scene.get_fact('room:a', 'contains', 'region:queue') is not None
    assert scene.get_fact('room:b', 'contains', 'obj:o') is None
    assert zones_of(scene, 'obj:o') == ['a']


def test_load_without_kg_derives_everything(tmp_path: Path) -> None:
    world = WorldView(_world_dir(tmp_path, kg=None)).load()
    assert world.kg is not None
    assert zones_of(world.kg, 'obj:o') == ['a']
    assert describe(world.kg, 'a') == ''


def test_unknown_mode_raises() -> None:
    assert DERIVE_MODES == ('auto', 'all', 'off')
    with pytest.raises(ValueError, match=r"kg.derive 'on' is not one of auto, all, off"):
        derive(_world(), 'on')


def test_queries_read_a_scene_built_in_code() -> None:
    scene = Scene()
    for ref in ('room:kitchen', 'room:hall', 'obj:stove', 'door:arch', 'floor:0'):
        scene.add_entity(ref)
    scene.add_entity('room:kitchen', {'description': 'where food is made'})
    scene.add_fact('room:kitchen', 'contains', 'obj:stove')
    scene.add_fact('room:kitchen', 'contains', 'door:arch')
    scene.add_fact('room:hall', 'contains', 'door:arch')
    scene.add_fact('floor:0', 'contains', 'room:kitchen')
    scene.add_fact('room:kitchen', 'connects', 'room:hall', {'door': 'door:arch'})
    assert zones_of(scene, 'door:arch') == ['kitchen', 'hall']
    assert zones_of(scene, 'room:kitchen') == []
    assert contents(scene, 'kitchen') == ['obj:stove', 'door:arch']
    assert connections(scene) == [('kitchen', 'hall', 'arch')]
    assert describe(scene, 'kitchen') == 'where food is made'
    assert describe(scene, 'nowhere') == ''
