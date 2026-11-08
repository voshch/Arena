from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.tree.World import FORMAT, WorldView, parse_world, write_world
from arena_simulation_setup.tree.World.model import World

WORLD = """\
format: arena-world/2
defaults:
  floor: Marble
  ceiling: {height: 3.0, material: Concrete_Smooth}
  wall: {material: Brick}
  door: {kind: hinged}
schedules:
- {name: fire_alarm, semantics: [{preset: schedule, params: {regime: alarm}}]}
signals:
- {name: crossing, semantics: [{preset: signal}]}
levels:
- id: '0'
  pedestrian_walls: false
  zones:
  - name: hall
    polygon: [[0.0, 0.0], [8.0, 0.0], [8.0, 4.0], [0.0, 4.0]]
    walls: outline
    semantics: [{preset: gate}, {state: max_speed, value: 1.5}]
  - name: office
    polygon: [[0.0, 4.0], [8.0, 4.0], [8.0, 8.0], [0.0, 8.0]]
    floor: Porcelain_Tile_4
    ceiling: {height: 2.5, lights: {fixture: panel, spacing: 2.4, lumens: 7200.0}}
    walls: {outline: true, open_to: [hall], kind: glass_walls}
  - {name: yard, polygon: [[8.0, 0.0], [12.0, 0.0], [12.0, 4.0], [8.0, 4.0]], floor: false, ceiling: false}
  walls:
  - {path: [[0.0, 8.0], [0.0, 12.0], [8.0, 12.0]], material: Plaster_Wall}
  - {path: [[10.0, 0.0], [12.0, 0.0], [12.0, 2.0]], closed: true}
  doors:
  - {name: main_entrance, span: [[3.0, 0.0], [4.0, 0.0]], semantics: [{preset: gate}]}
  - {name: back_door, span: [[3.0, 8.0], [4.0, 8.0]], kind: sliding, height: 2.2}
  elevators:
  - {name: lift_0, position: [6.0, 6.0], door_side: '-y', destinations: [lift_1]}
  objects:
  - {name: desk_1, model: Desk, pose: [2.0, 1.0, 1.5708], humansim: {type: desk}}
  - {name: shelf, model: Shelf, pose: {x: 6.0, y: 1.0, z: 0.5, yaw: 0.25}, scale: [1.0, 2.0, 1.0]}
  humans:
  - {name: walker, model: adult, pose: [1.0, 1.0], waypoints: [[2.0, 2.0], [3.0, 3.0]], velocity: 0.8}
  lights:
  - {name: exit_strip, fixture: tube, lumens: 300.0, pose: {x: 4.0, y: 3.9, z: 2.2}}
  - {name: desk_lamp, fixture: bulb, lumens: 400.0, attach: desk_1, offset: {x: 0.0, y: 0.0, z: 0.8}}
  sounds:
  - {name: hall_siren, asset_id: alarm_loop, position: [4.0, 2.0]}
  - {name: desk_radio, asset_id: radio, attach: desk_1, loop: false}
- id: '1'
  zones:
  - {name: upstairs, polygon: [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]], walls: outline}
  elevators:
  - {name: lift_1, position: [2.0, 2.0], door_side: '+x', door_kind: sliding, destinations: [lift_0]}
"""


def _world() -> World:
    return parse_world(yaml.safe_load(WORLD))


def _written(world: World) -> dict:
    return yaml.safe_load(write_world(world))


def _one_level(**level: object) -> World:
    return parse_world({'format': FORMAT, 'levels': [{'id': '0', **level}]})


def test_round_trip_gives_an_equal_world() -> None:
    world = _world()
    again = parse_world(_written(world))
    assert again == world
    assert again.levels['0'].zones[0].semantics == world.levels['0'].zones[0].semantics


def test_elevator_door_kind_is_written_only_when_not_split() -> None:
    lifts = {lift['name']: lift for level in _written(_world())['levels'] for lift in level.get('elevators', [])}
    assert 'door_kind' not in lifts['lift_0']
    assert lifts['lift_1']['door_kind'] == 'sliding'
    assert _world().levels['0'].elevators[0].door_kind == 'sliding_split'


def test_writing_is_idempotent() -> None:
    text = write_world(_world())
    assert write_world(parse_world(yaml.safe_load(text))) == text


def test_round_trip_keeps_defaults_block_verbatim() -> None:
    assert _written(_world())['defaults'] == yaml.safe_load(WORLD)['defaults']


def test_poses_take_the_shortest_form() -> None:
    objects = [
        {'name': 'flat', 'model': 'Desk', 'pose': {'x': 1, 'y': 2}},
        {'name': 'turned', 'model': 'Desk', 'pose': [1, 2, 0.5]},
        {'name': 'raised', 'model': 'Desk', 'pose': [1, 2, 0.3, 0, 0, 0]},
        {'name': 'tilted', 'model': 'Desk', 'pose': {'x': 1, 'y': 2, 'roll': 0.1, 'yaw': 0.5}},
        {'name': 'pitched', 'model': 'Desk', 'pose': {'x': 1, 'y': 2, 'pitch': 0.2}},
    ]
    written = _written(_one_level(objects=objects))['levels'][0]['objects']
    assert [entity['pose'] for entity in written] == [
        [1.0, 2.0],
        [1.0, 2.0, 0.5],
        {'x': 1.0, 'y': 2.0, 'z': 0.3},
        {'x': 1.0, 'y': 2.0, 'roll': 0.1, 'yaw': 0.5},
        {'x': 1.0, 'y': 2.0, 'pitch': 0.2},
    ]


def test_positions_round_to_a_tenth_of_a_millimeter_and_angles_to_a_microradian() -> None:
    level = _written(
        _one_level(
            zones=[{'name': 'z', 'polygon': [[0.123456789, 0], [4.00004999, 0], [4, 4], [0, 4]]}],
            objects=[{'name': 'o', 'model': 'Desk', 'pose': [1.000049, -2.00005001, 0.1234567891]}],
        )
    )['levels'][0]
    assert level['zones'][0]['polygon'] == [[0.1235, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
    assert level['objects'][0]['pose'] == [1.0, -2.0001, 0.123457]


def test_positions_keep_z_only_when_nonzero() -> None:
    sounds = [{'name': 'low', 'asset_id': 'alarm', 'position': [1, 2, 0]}, {'name': 'high', 'asset_id': 'alarm', 'position': [1, 2, 1.5]}]
    written = _written(_one_level(sounds=sounds))['levels'][0]['sounds']
    assert [sound['position'] for sound in written] == [[1.0, 2.0], [1.0, 2.0, 1.5]]


def test_field_defaults_are_omitted() -> None:
    level = _written(
        _one_level(
            zones=[{'name': 'z', 'polygon': [[0, 0], [4, 0], [4, 4], [0, 4]]}],
            doors=[{'name': 'd', 'span': [[1, 0], [2, 0]], 'kind': 'sliding', 'width': 0.1, 'height': 2.0, 'activation_distance': 3.0}],
            objects=[{'name': 'o', 'model': 'Desk', 'pose': [1, 1]}],
            sounds=[{'name': 's', 'asset_id': 'alarm', 'position': [1, 1], 'loop': True, 'reference_distance_m': 1.0}],
            lights=[{'name': 'l', 'fixture': 'bulb', 'lumens': 10, 'pose': {'x': 1, 'y': 1, 'z': 2}, 'cct_K': 6500, 'dim': 1.0}],
        )
    )['levels'][0]
    assert level['zones'] == [{'name': 'z', 'polygon': [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]}]
    assert level['doors'] == [{'name': 'd', 'span': [[1.0, 0.0], [2.0, 0.0]]}]
    assert level['objects'] == [{'name': 'o', 'pose': [1.0, 1.0], 'model': 'Desk'}]
    assert level['sounds'] == [{'name': 's', 'asset_id': 'alarm', 'position': [1.0, 1.0]}]
    assert level['lights'] == [{'name': 'l', 'fixture': 'bulb', 'pose': {'x': 1.0, 'y': 1.0, 'z': 2.0}, 'lumens': 10.0}]
    assert 'pedestrian_walls' not in level


def test_values_inherited_from_defaults_are_omitted() -> None:
    level = _written(_world())['levels'][0]
    hall, office, _yard = level['zones']
    assert set(hall) == {'name', 'polygon', 'walls', 'semantics'}
    assert office['floor'] == 'Porcelain_Tile_4'
    assert office['ceiling'] == {'height': 2.5, 'lights': {'lumens': 7200.0}}
    main_entrance, back_door = level['doors']
    assert 'kind' not in main_entrance
    assert (back_door['kind'], back_door['height']) == ('sliding', 2.2)
    assert level['walls'][1] == {'path': [[10.0, 0.0], [12.0, 0.0], [12.0, 2.0]], 'closed': True}
    assert level['walls'][0]['material'] == 'Plaster_Wall'


def test_semantics_presets_stay_unexpanded() -> None:
    written = _written(_world())
    assert written['schedules'] == [{'name': 'fire_alarm', 'semantics': [{'preset': 'schedule', 'params': {'regime': 'alarm'}}]}]
    assert written['signals'] == [{'name': 'crossing', 'semantics': [{'preset': 'signal'}]}]
    level = written['levels'][0]
    assert level['zones'][0]['semantics'] == [{'preset': 'gate'}, {'state': 'max_speed', 'value': 1.5}]
    assert level['doors'][0]['semantics'] == [{'preset': 'gate'}]


def test_shorthands_for_outline_walls_and_absent_floor_ceiling_and_pedestrian_walls() -> None:
    written = _written(_world())
    level = written['levels'][0]
    hall, office, yard = level['zones']
    assert hall['walls'] == 'outline'
    assert office['walls'] == {'outline': True, 'open_to': ['hall'], 'kind': 'glass_walls'}
    assert (yard['floor'], yard['ceiling']) == (False, False)
    assert level['pedestrian_walls'] is False
    assert 'pedestrian_walls' not in written['levels'][1]
    assert written['levels'][1]['zones'][0]['walls'] == 'outline'


def test_light_dim_survives_the_round_trip() -> None:
    world = _one_level(lights=[{'name': 'dim', 'fixture': 'bulb', 'lumens': 100, 'pose': {'x': 1, 'y': 1, 'z': 2}, 'dim': 0.5}])
    assert _written(world)['levels'][0]['lights'][0]['dim'] == 0.5
    assert parse_world(_written(world)) == world


def test_written_world_starts_with_the_format() -> None:
    assert write_world(_world()).startswith(f'format: {FORMAT}\n')


def test_world_view_save_then_load_gives_an_equal_world(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'world.yaml').write_text(WORLD, encoding='utf-8')
    world = WorldView(source).load()
    target = tmp_path / 'target'
    WorldView(target).save(world)
    assert (target / 'world.yaml').read_text(encoding='utf-8').startswith(f'format: {FORMAT}\n')
    assert (target / 'kg.yaml').is_file()
    for level_id in ('0', '1'):
        assert (target / 'map' / f'{level_id}.png').read_bytes().startswith(b'\x89PNG')
        assert yaml.safe_load(WorldView(target).map_yaml(level_id).read_text(encoding='utf-8'))['image'] == f'{level_id}.png'
    assert WorldView(target).load() == world


def test_world_view_save_keeps_the_kg_facts(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'world.yaml').write_text(WORLD, encoding='utf-8')
    world = WorldView(source).load()
    target = tmp_path / 'target'
    WorldView(target).save(world)
    reloaded = WorldView(target).load()
    assert reloaded.kg is not None and world.kg is not None
    assert sorted(fact.key for fact in reloaded.kg.facts) == sorted(fact.key for fact in world.kg.facts)
