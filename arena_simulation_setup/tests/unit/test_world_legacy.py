from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.shared.task import GoToPhase
from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
from arena_simulation_setup.tree.World import FORMAT, LEGACY_FORMAT, WorldView, connections, describe, parse_world, zones_of
from arena_simulation_setup.tree.World.legacy import is_legacy, upgrade
from arena_simulation_setup.tree.World.model import World
from arena_simulation_setup.tree.World.Scenario import SCENARIO_FORMAT, ScenarioView, upgrade_scenario
from arena_simulation_setup.tree.World.view import world_dir

GENERATOR = """\
zones:
- name: room_0
  corners:
  - {x: 0.0, y: 0.0, z: 0.0}
  - {x: 4.0, y: 0.0, z: 0.0}
  - {x: 4.0, y: 4.0, z: 0.0}
  - {x: 0.0, y: 4.0, z: 0.0}
  material: [Porcelain_Tile_4, {}]
  description: a small office
  walls:
  - start: {x: 0.0, y: 0.0, z: 0.0}
    end: {x: 4.0, y: 0.0, z: 0.0}
    material: [Plaster_Wall, {}]
  - start: {x: 4.0, y: 0.0, z: 0.0}
    end: {x: 4.0, y: 4.0, z: 0.0}
    material: [Plaster_Wall, {}]
  - start: {x: 4.0, y: 4.0, z: 0.0}
    end: {x: 0.0, y: 4.0, z: 0.0}
    material: [Plaster_Wall, {}]
  - start: {x: 0.0, y: 4.0, z: 0.0}
    end: {x: 0.0, y: 0.0, z: 0.0}
    material: [Plaster_Wall, {}]
  doors:
  - name: door_0
    start: {x: 4.0, y: 1.0, z: 0.0}
    end: {x: 4.0, y: 2.0, z: 0.0}
    swing: left
- name: room_1
  corners:
  - {x: 4.0, y: 0.0, z: 0.0}
  - {x: 8.0, y: 0.0, z: 0.0}
  - {x: 8.0, y: 4.0, z: 0.0}
  - {x: 4.0, y: 4.0, z: 0.0}
  material: [Porcelain_Tile_4, {}]
  walls:
  - start: {x: 4.0, y: 0.0, z: 0.0}
    end: {x: 8.0, y: 0.0, z: 0.0}
    material: [Plaster_Wall, {}]
  entities:
    static:
    - {name: desk_1, model: Desk, pose: [6.0, 2.0, 0.0], type: desk, capacity: 2}
"""

EXPORTED = """\
zones:
- name: hall
  corners: [[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]]
  material: {name: Marble, domain: Common, _modifiers: {}}
  ceiling_material: {name: Concrete_Smooth, domain: Common}
  ceiling_height: 3.0
  walls:
  - start: [0, 0]
    end: [5, 0]
    material: {name: Plaster_Wall, domain: Common, _modifiers: {tint: '#ff0000'}}
  - start: [5, 0]
    end: [5, 5]
    material: {name: Plaster_Wall, domain: Common, _modifiers: {tint: '#ff0000'}}
"""

HAND_WRITTEN = """\
zones:
- {name: lobby, corners: [[0, 0], [6, 0], [6, 4], [0, 4]], mat: '', ceiling_mat: Marble, wall_material: ''}
- {name: annex, corners: [[6, 0], [9, 0], [9, 4], [6, 4]], mat: Marble, wall_mat: ''}
"""

LEVEL_0 = """\
zones:
- name: ground
  corners: [[0, 0], [6, 0], [6, 6], [0, 6]]
  description: entrance hall
  elevators:
  - {name: lift_0, position: [2, 2], destination: "1.lift_1a,1.lift_1b", hall_call: true}
"""

LEVEL_1 = """\
zones:
- name: upstairs
  corners: [[0, 0], [6, 0], [6, 6], [0, 6]]
  elevators:
  - {name: lift_1a, position: [2, 2], destination: "0.lift_0"}
  - {name: lift_1b, position: [2, 4], destination: "0.lift_0"}
"""

ROOT_LEVELS = """\
levels:
  '0':
    zones:
    - {name: ground, corners: [[0, 0], [6, 0], [6, 6], [0, 6]]}
    entities:
      static:
      - {name: bench, model: Bench, pose: [1, 1]}
      dynamic:
      - {name: walker, model: adult, pose: [2, 2], waypoints: [[3, 3]], behavior: wander}
  '1':
    zones:
    - {name: roof, corners: [[0, 0], [6, 0], [6, 6], [0, 6]]}
"""


def _world_dir(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    return root


def _load(path: Path) -> World:
    with pytest.warns(DeprecationWarning, match='format 1'):
        return WorldView(path).load()


def _read(path: Path) -> tuple[dict, dict | None, list[str]]:
    with pytest.warns(DeprecationWarning, match='format 1'):
        return WorldView(path).read()


def test_is_legacy_tells_the_formats_apart() -> None:
    assert is_legacy({'zones': []})
    assert is_legacy(None)
    assert not is_legacy({'format': FORMAT, 'levels': []})


def test_generator_dialect_reads_point_blocks_and_pair_materials(tmp_path: Path) -> None:
    data, _, _ = _read(_world_dir(tmp_path, {'world.yaml': GENERATOR}))
    (level,) = data['levels']
    assert level['id'] == '0'
    assert level['zones'][0]['polygon'] == [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
    assert level['zones'][0]['floor'] == 'Porcelain_Tile_4'
    assert level['walls'][0] == {'path': [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]], 'material': 'Plaster_Wall', 'closed': True}
    assert level['doors'] == [{'name': 'door_0', 'span': [[4.0, 1.0], [4.0, 2.0]]}]
    assert level['objects'] == [{'name': 'desk_1', 'model': 'Desk', 'pose': [6.0, 2.0, 0.0], 'humansim': {'type': 'desk', 'capacity': 2}}]


def test_generator_dialect_loads(tmp_path: Path) -> None:
    level = _load(_world_dir(tmp_path, {'world.yaml': GENERATOR})).levels['0']
    assert [zone.name for zone in level.zones] == ['room_0', 'room_1']
    assert level.zones[0].floor == MaterialIdentifier('Porcelain_Tile_4')
    assert level.walls[0].closed is True
    assert {wall.material for wall in level.all_walls} == {MaterialIdentifier('Plaster_Wall')}
    assert level.objects[0].humansim == {'type': 'desk', 'capacity': 2}


def test_exported_dialect_reads_dict_materials_and_opens_corner_rings(tmp_path: Path) -> None:
    level = _load(_world_dir(tmp_path, {'world.yaml': EXPORTED})).levels['0']
    (zone,) = level.zones
    assert len(zone.polygon) == 4
    assert zone.floor == MaterialIdentifier('Marble')
    assert zone.ceiling is not None
    assert (zone.ceiling.height, zone.ceiling.material) == (3.0, MaterialIdentifier('Concrete_Smooth'))
    assert level.walls[0].material == MaterialIdentifier('Plaster_Wall', modifiers={'tint': '#ff0000'})
    assert [(point.x, point.y) for point in level.walls[0].path] == [(0, 0), (5, 0), (5, 5)]


def test_hand_written_dialect_reads_flow_corners_and_aliases(tmp_path: Path) -> None:
    data, _, _ = _read(_world_dir(tmp_path, {'world.yaml': HAND_WRITTEN}))
    lobby, annex = data['levels'][0]['zones']
    assert lobby['polygon'] == [[0, 0], [6, 0], [6, 4], [0, 4]]
    assert lobby['floor'] is False
    assert lobby['ceiling'] == {'material': 'Marble'}
    assert annex['floor'] == 'Marble'
    level = _load(tmp_path).levels['0']
    assert level.zones[0].floor is None
    assert level.zones[1].floor == MaterialIdentifier('Marble')
    assert level.pedestrian_walls is False


def test_empty_wall_material_on_every_zone_turns_pedestrian_walls_off() -> None:
    zones = [{'name': 'a', 'corners': [[0, 0], [1, 0], [1, 1]], 'wall_material': ''}, {'name': 'b', 'corners': [[1, 0], [2, 0], [2, 1]], 'wall_material': ['', {}]}]
    data, _, _ = upgrade({'0': {'zones': zones}})
    assert data['levels'][0]['pedestrian_walls'] is False


def test_one_zone_with_wall_material_keeps_pedestrian_walls() -> None:
    zones = [{'name': 'a', 'corners': [[0, 0], [1, 0], [1, 1]], 'wall_material': ''}, {'name': 'b', 'corners': [[1, 0], [2, 0], [2, 1]], 'wall_material': 'Brick'}]
    data, _, _ = upgrade({'0': {'zones': zones}})
    assert 'pedestrian_walls' not in data['levels'][0]


def test_per_level_directories_become_levels_with_elevator_destinations(tmp_path: Path) -> None:
    world = _load(_world_dir(tmp_path, {'0/world.yaml': LEVEL_0, '1/world.yaml': LEVEL_1}))
    assert list(world.levels) == ['0', '1']
    (lift_0,) = world.levels['0'].elevators
    assert lift_0.destinations == ['lift_1a', 'lift_1b']
    assert lift_0.destination == 'lift_1a'
    assert [elevator.destinations for elevator in world.levels['1'].elevators] == [['lift_0'], ['lift_0']]
    assert world.elevator_levels() == {'lift_0': '0', 'lift_1a': '1', 'lift_1b': '1'}


def test_per_level_directories_keep_their_zones(tmp_path: Path) -> None:
    files = {f'{level}/world.yaml': f"zones:\n- {{name: room_{level}, corners: [[0, 0], [5, 0], [5, 5], [0, 5]]}}\n" for level in ('0', '1')}
    with pytest.warns(DeprecationWarning, match='format 1'):
        world = WorldView(_world_dir(tmp_path, files)).load(validate=False)
    assert {level: [zone.name for zone in world.levels[level].zones] for level in world.levels} == {'0': ['room_0'], '1': ['room_1']}


def test_root_level_file_wraps_as_level_0_and_keeps_its_zone(tmp_path: Path) -> None:
    files = {'world.yaml': "zones:\n- {name: my_hall, corners: [[0, 0], [5, 0], [5, 5], [0, 5]]}\n"}
    with pytest.warns(DeprecationWarning, match='format 1'):
        world = WorldView(_world_dir(tmp_path, files)).load(validate=False)
    assert list(world.levels) == ['0']
    assert [zone.name for zone in world.levels['0'].zones] == ['my_hall']


def test_root_levels_file_and_level_entities(tmp_path: Path) -> None:
    world = _load(_world_dir(tmp_path, {'world.yaml': ROOT_LEVELS}))
    assert list(world.levels) == ['0', '1']
    ground = world.levels['0']
    assert [entity.name for entity in ground.objects] == ['bench']
    (walker,) = ground.humans
    assert walker.name == 'walker'
    assert walker.hunav == {'behavior': 'wander'}
    assert [(point.x, point.y) for point in walker.waypoints] == [(3.0, 3.0)]


def test_zone_description_moves_to_the_kg(tmp_path: Path) -> None:
    world = _load(_world_dir(tmp_path, {'world.yaml': GENERATOR}))
    assert world.kg is not None
    assert describe(world.kg, 'room_0') == 'a small office'
    assert describe(world.kg, 'room_1') == ''


def test_listing_zone_becomes_a_stated_contains_and_doors_are_derived(tmp_path: Path) -> None:
    _, kg, _ = _read(_world_dir(tmp_path, {'world.yaml': GENERATOR}))
    assert kg is not None
    assert kg['facts'] == [{'subject': 'room:room_1', 'verb': 'contains', 'object': 'obj:desk_1'}]
    world = _load(tmp_path)
    assert world.kg is not None
    assert zones_of(world.kg, 'obj:desk_1') == ['room_1']
    assert zones_of(world.kg, 'door:door_0') == ['room_0', 'room_1']
    assert connections(world.kg) == [('room_0', 'room_1', 'door_0')]


def test_identical_repeats_are_dropped() -> None:
    door = {'name': 'shared', 'start': [4, 1], 'end': [4, 2]}
    zones = [{'name': 'a', 'corners': [[0, 0], [4, 0], [4, 4]], 'doors': [door]}, {'name': 'b', 'corners': [[4, 0], [8, 0], [8, 4]], 'doors': [dict(door)]}]
    data, _, warnings = upgrade({'0': {'zones': zones}})
    assert [door['name'] for door in data['levels'][0]['doors']] == ['shared']
    assert warnings == []


def test_different_items_sharing_a_name_in_a_level_are_renamed_with_a_warning() -> None:
    zones = [
        {'name': 'a', 'corners': [[0, 0], [4, 0], [4, 4]], 'doors': [{'name': 'd', 'start': [1, 0], 'end': [2, 0]}]},
        {'name': 'b', 'corners': [[10, 0], [14, 0], [14, 4]], 'doors': [{'name': 'd', 'start': [11, 0], 'end': [12, 0]}]},
    ]
    data, _, warnings = upgrade({'0': {'zones': zones}})
    assert [door['name'] for door in data['levels'][0]['doors']] == ['d', 'd_0']
    assert len(warnings) == 1 and "'d'" in warnings[0] and "'d_0'" in warnings[0]


def test_renamed_object_keeps_the_zone_it_was_listed_under(tmp_path: Path) -> None:
    level = {
        'zones': [
            {'name': 'a', 'corners': [[0, 0], [4, 0], [4, 4], [0, 4]], 'entities': {'static': [{'name': 'chair', 'model': 'Chair', 'pose': [1, 1]}]}},
            {'name': 'b', 'corners': [[10, 0], [14, 0], [14, 4], [10, 4]], 'entities': {'static': [{'name': 'chair', 'model': 'Chair', 'pose': [11, 1]}]}},
        ]
    }
    world = _load(_world_dir(tmp_path, {'world.yaml': yaml.safe_dump(level)}))
    assert [entity.name for entity in world.levels['0'].objects] == ['chair', 'chair_0']
    assert world.kg is not None
    assert zones_of(world.kg, 'obj:chair') == ['a']
    assert zones_of(world.kg, 'obj:chair_0') == ['b']


def test_names_repeated_across_levels_are_renamed_with_a_warning(tmp_path: Path) -> None:
    levels = {
        '0': {'zones': [{'name': 'ground', 'corners': [[0, 0], [4, 0], [4, 4], [0, 4]]}], 'entities': {'static': [{'name': 'chair', 'model': 'Chair', 'pose': [1, 1]}]}},
        '1': {'zones': [{'name': 'roof', 'corners': [[0, 0], [4, 0], [4, 4], [0, 4]]}], 'entities': {'static': [{'name': 'chair', 'model': 'Chair', 'pose': [2, 2]}]}},
    }
    data, _, warnings = upgrade(levels)
    assert [entity['name'] for level in data['levels'] for entity in level['objects']] == ['chair', 'chair_1']
    assert any("'chair'" in warning for warning in warnings)
    _load(_world_dir(tmp_path, {'world.yaml': yaml.safe_dump({'levels': levels})}))


def test_dropped_keys_produce_warnings(tmp_path: Path) -> None:
    _, _, generator_notes = _read(_world_dir(tmp_path / 'generator', {'world.yaml': GENERATOR}))
    assert generator_notes == ["0/world.yaml zones[0].doors[0]: dropped keys ['swing'] that format 2 does not know"]
    _, _, notes = _read(_world_dir(tmp_path / 'levels', {'0/world.yaml': LEVEL_0 + 'bogus_key: 1\n', '1/world.yaml': LEVEL_1}))
    assert "0/world.yaml zones[0].elevators[0]: dropped keys ['hall_call'] that format 2 does not know" in notes
    assert "0/world.yaml: dropped top-level keys ['bogus_key'] that format 2 does not know" in notes


def test_unknown_zone_keys_produce_warnings() -> None:
    _, _, warnings = upgrade({'0': {'zones': [{'name': 'a', 'corners': [[0, 0], [1, 0], [1, 1]], 'color': 'red'}]}})
    assert any('color' in warning for warning in warnings)


def test_legacy_load_warns_deprecation(tmp_path: Path) -> None:
    with pytest.warns(DeprecationWarning, match=r'migrate it with touch_world --migrate'):
        WorldView(_world_dir(tmp_path, {'world.yaml': HAND_WRITTEN})).load()


def test_migrate_writes_format_2_and_removes_the_level_files(tmp_path: Path) -> None:
    root = _world_dir(tmp_path, {'0/world.yaml': LEVEL_0, '1/world.yaml': LEVEL_1})
    legacy = _load(root)
    with pytest.warns(DeprecationWarning):
        notes = WorldView(root).migrate()
    assert notes == ["0/world.yaml zones[0].elevators[0]: dropped keys ['hall_call'] that format 2 does not know"]
    assert not (root / '0').exists()
    assert not (root / '1').exists()
    assert yaml.safe_load((root / 'world.yaml').read_text(encoding='utf-8'))['format'] == FORMAT
    assert (root / 'kg.yaml').is_file()
    assert (root / 'map' / '0.png').is_file() and (root / 'map' / '1.png').is_file()
    data, _, warnings = WorldView(root).read()
    assert warnings == [] and not is_legacy(data)
    migrated = WorldView(root).load()
    assert migrated == legacy
    assert migrated.kg is not None
    assert describe(migrated.kg, 'ground') == 'entrance hall'


def test_migrate_prunes_stated_memberships_that_geometry_derives(tmp_path: Path) -> None:
    root = _world_dir(tmp_path, {'world.yaml': GENERATOR})
    with pytest.warns(DeprecationWarning):
        WorldView(root).migrate()
    kg = yaml.safe_load((root / 'kg.yaml').read_text(encoding='utf-8'))
    assert kg['facts'] == []
    assert kg['entities'] == [{'id': 'room:room_0', 'props': {'description': 'a small office'}}]
    world = WorldView(root).load()
    assert world.kg is not None
    assert zones_of(world.kg, 'obj:desk_1') == ['room_1']


def test_migrate_rewrites_format_1_scenarios(tmp_path: Path) -> None:
    root = _world_dir(tmp_path, {'world.yaml': HAND_WRITTEN, 'scenarios/rush/scenario.yaml': yaml.safe_dump(SCENARIO_WRAPPED)})
    with pytest.warns(DeprecationWarning):
        notes = WorldView(root).migrate()
    assert notes == ["scenario rush: dropped scenario keys ['bogus'] that format 2 does not know"]
    migrated = yaml.safe_load((root / 'scenarios' / 'rush' / 'scenario.yaml').read_text(encoding='utf-8'))
    assert migrated['format'] == SCENARIO_FORMAT
    assert [entity.name for entity in ScenarioView(root / 'scenarios' / 'rush').load().objects] == ['crate', 'cart']


def test_migrate_keeps_other_files_in_a_level_directory(tmp_path: Path) -> None:
    root = _world_dir(tmp_path, {'0/world.yaml': LEVEL_0, '0/notes.txt': 'keep', '1/world.yaml': LEVEL_1})
    with pytest.warns(DeprecationWarning):
        WorldView(root).migrate()
    assert (root / '0' / 'notes.txt').read_text(encoding='utf-8') == 'keep'
    assert not (root / '0' / 'world.yaml').exists()


@pytest.mark.parametrize('dialect', [GENERATOR, EXPORTED, HAND_WRITTEN, ROOT_LEVELS])
def test_migrated_world_loads_equal_to_the_legacy_load(tmp_path: Path, dialect: str) -> None:
    root = _world_dir(tmp_path, {'world.yaml': dialect})
    legacy = _load(root)
    with pytest.warns(DeprecationWarning):
        WorldView(root).migrate()
    assert WorldView(root).load() == legacy


def test_upgrade_splits_elevator_destinations_and_drops_the_default_door_side() -> None:
    data, _, _ = upgrade({'0': {'zones': [{'name': 'z', 'corners': [[0, 0], [1, 0], [1, 1]], 'elevators': [{'name': 'e', 'position': [0, 0], 'door_side': '+x', 'destination': ' 1.a , 1.b '}]}]}})
    assert data['levels'][0]['elevators'] == [{'name': 'e', 'position': [0, 0], 'destinations': ['a', 'b']}]
    assert parse_world(data).levels['0'].elevators[0].door_side is None


SCENARIO_WRAPPED = {
    'obstacles': {
        'static': [{'name': 'crate', 'model': 'Crate', 'pose': [1, 1], 'level_id': 1}],
        'interactive': [{'name': 'cart', 'model': 'Cart', 'pose': [2, 2]}],
        'dynamic': [{'name': 'walker', 'model': 'adult', 'pose': [0, 0], 'level_id': 1, 'waypoints': [{'position': [1, 1], 'level_id': 1}], 'behavior': 'wander'}],
    },
    'robots': [{'start': [0, 0, 0], 'goal': [1, 1, 0], 'start_floor': '0', 'goal_floor': '1'}],
    'bogus': 3,
}


def test_upgrade_scenario_unwraps_obstacles_and_moves_levels() -> None:
    upgraded, warnings = upgrade_scenario(SCENARIO_WRAPPED)
    assert upgraded['format'] == SCENARIO_FORMAT
    assert upgraded['objects'] == [{'name': 'crate', 'model': 'Crate', 'pose': [1, 1], 'level': '1'}, {'name': 'cart', 'model': 'Cart', 'pose': [2, 2]}]
    assert upgraded['humans'] == [{'name': 'walker', 'model': 'adult', 'pose': [0, 0], 'level': '1', 'waypoints': [{'x': 1, 'y': 1, 'level': '1'}], 'hunav': {'behavior': 'wander'}}]
    assert upgraded['robots'] == [{'start': [0, 0, 0], 'level': '0', 'goal_level': '1', 'phases': [{'goto': [1, 1, 0]}]}]
    assert warnings == ["dropped scenario keys ['bogus'] that format 2 does not know"]


def test_upgrade_scenario_reads_top_level_static_and_dynamic() -> None:
    upgraded, warnings = upgrade_scenario({'static': [{'name': 'crate', 'model': 'Crate', 'pose': [1, 1]}], 'dynamic': [{'name': 'walker', 'model': 'adult', 'pose': [0, 0]}]})
    assert upgraded == {'format': SCENARIO_FORMAT, 'objects': [{'name': 'crate', 'model': 'Crate', 'pose': [1, 1]}], 'humans': [{'name': 'walker', 'model': 'adult', 'pose': [0, 0]}]}
    assert warnings == []


def test_upgrade_scenario_keeps_explicit_phases_over_a_goal() -> None:
    upgraded, _ = upgrade_scenario({'robots': [{'start': [0, 0, 0], 'goal': [1, 1, 0], 'phases': [{'goto': [2, 2, 0]}], 'start_floor': ''}]})
    assert upgraded['robots'] == [{'start': [0, 0, 0], 'phases': [{'goto': [2, 2, 0]}]}]


def test_scenario_view_loads_a_format_1_scenario(tmp_path: Path) -> None:
    (tmp_path / 'scenario.yaml').write_text(yaml.safe_dump(SCENARIO_WRAPPED), encoding='utf-8')
    with pytest.warns(DeprecationWarning, match='format 1'):
        scenario = ScenarioView(tmp_path).load()
    assert [(entity.name, entity.level) for entity in scenario.objects] == [('crate', '1'), ('cart', '')]
    (walker,) = scenario.humans
    assert (walker.level, walker.hunav) == ('1', {'behavior': 'wander'})
    assert [(point.x, point.y, point.level) for point in walker.waypoints] == [(1.0, 1.0, '1')]
    (robot,) = scenario.robots
    assert (robot.level, robot.goal_level) == ('0', '1')
    (phase,) = robot.phases
    assert isinstance(phase, GoToPhase)
    assert phase.pose is not None and (phase.pose.position.x, phase.pose.position.y) == (1.0, 1.0)


def test_graph_edges_connect_through_the_door_the_world_names_by_room_pair(tmp_path: Path) -> None:
    graph = {'nodes': [{'id': 'room_0', 'room_type': 'office'}, {'id': 'room_1', 'room_type': 'lab'}], 'edges': [{'id': 'e0', 'source': 'room_1', 'target': 'room_0'}]}
    world = _load(_world_dir(tmp_path, {'0/world.yaml': GENERATOR.replace('door_0', 'door_room_0__room_1'), 'graph.yaml': yaml.safe_dump(graph)}))
    assert world.kg is not None
    fact = world.kg.get_fact('room:room_1', 'connects', 'room:room_0', {'door': 'door:door_room_0__room_1'})
    assert fact is not None and fact.source == 'generated'
    assert world.kg.get_entity('door:door_e0') is None


def test_world_format_tells_format_1_directories_from_format_2(tmp_path: Path) -> None:
    legacy = _world_dir(tmp_path / 'legacy', {'0/world.yaml': LEVEL_0})
    assert WorldView(legacy).world_format == LEGACY_FORMAT
    assert WorldView(_world_dir(tmp_path / 'flat', {'world.yaml': HAND_WRITTEN})).world_format == LEGACY_FORMAT
    current = tmp_path / 'current'
    current.mkdir()
    (current / 'world.yaml').write_text(yaml.safe_dump({'format': FORMAT, 'levels': []}), encoding='utf-8')
    assert WorldView(current).world_format == FORMAT


def test_world_dir_takes_a_directory_or_a_name_under_the_repo_worlds(tmp_path: Path) -> None:
    previous = os.environ.get('ARENA_DIR')
    os.environ['ARENA_DIR'] = str(tmp_path / 'repo')
    try:
        assert world_dir(str(tmp_path)) == tmp_path.resolve()
        assert world_dir('mcp_named') == tmp_path / 'repo' / 'arena_simulation_setup' / 'worlds' / 'mcp_named'
    finally:
        if previous is None:
            del os.environ['ARENA_DIR']
        else:
            os.environ['ARENA_DIR'] = previous
