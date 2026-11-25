from __future__ import annotations

import json

import pytest
import yaml
from PIL import Image

from arena_simulation_setup.metrics.nav_at_r import DEFAULT_RADIUS, SPAWN_CLEARANCE, main, nav_at_r, resolve_world
from arena_simulation_setup.tree.World import FORMAT, WorldView

R = DEFAULT_RADIUS


def _box_walls(x0, y0, x1, y1):
    return [{'path': [[x0, y1], [x0, y0], [x1, y0]]}, {'path': [[x1, y1], [x0, y1]]}]


def _zone(name, x0, y0, x1, y1):
    return {'name': name, 'polygon': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def _two_rooms(gap):
    """Room a is 4 x 4 m, room b 2 x 4 m, joined through a `gap` wide opening centered at y = 2 in the shared wall x = 4."""
    low, high = 2.0 - gap / 2, 2.0 + gap / 2
    return {
        'zones': [_zone('a', 0.0, 0.0, 4.0, 4.0), _zone('b', 4.0, 0.0, 6.0, 4.0)],
        'walls': [
            *_box_walls(0.0, 0.0, 4.0, 4.0),
            {'path': [[4.0, 0.0], [4.0, low]]},
            {'path': [[4.0, high], [4.0, 4.0]]},
            {'path': [[4.0, 0.0], [6.0, 0.0], [6.0, 4.0], [4.0, 4.0]]},
        ],
        'doors': [{'name': 'a_b', 'span': [[4.0, low], [4.0, high]]}],
    }


def _write_world(root, levels):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'world.yaml').write_text(yaml.safe_dump({'format': FORMAT, 'levels': [{'id': level_id, **level} for level_id, level in levels.items()]}))


def _world(tmp_path, level, name='w'):
    root = tmp_path / name
    _write_world(root, {'0': level})
    return WorldView(root)


def _asset(world, model, annotation):
    directory = world.path / 'assets' / 'Common' / 'Object' / model
    directory.mkdir(parents=True)
    (directory / 'annotation.yaml').write_text(yaml.safe_dump(annotation))


def _entity(name, model, x, y, z=0.0):
    return {'name': name, 'model': model, 'pose': {'x': x, 'y': y, 'z': z}}


def _expectation(room_widths, height, clearance):
    """sum_i w_i m_i with m_i, w_i the shares of each room's area shrunk by the clearance in either direction."""
    traversable = [(w - 2 * R) * (height - 2 * R) for w in room_widths]
    spawn = [(w - 2 * clearance) * (height - 2 * clearance) for w in room_widths]
    return sum(t / sum(traversable) * s / sum(spawn) for t, s in zip(traversable, spawn, strict=True))


def test_open_room_is_fully_reachable(tmp_path):
    view = _world(tmp_path, {'zones': [_zone('room', 0.0, 0.0, 4.0, 4.0)], 'walls': [{'path': [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]], 'closed': True}]})
    result = nav_at_r(view)['0']
    assert result.value == pytest.approx(1.0)
    assert result.components == 1
    assert result.largest_share == pytest.approx(1.0)
    assert result.traversable_area == pytest.approx((4 - 2 * R) ** 2, rel=0.05)
    assert result.cut_off_zones == ()


def test_door_wider_than_footprint_joins_rooms(tmp_path):
    result = nav_at_r(_world(tmp_path, _two_rooms(1.0)))['0']
    assert result.value == pytest.approx(1.0)
    assert result.components == 1
    assert {zone.name: zone.reachable_area > 0 for zone in result.zones} == {'a': True, 'b': True}


def test_door_narrower_than_footprint_splits_rooms(tmp_path):
    result = nav_at_r(_world(tmp_path, _two_rooms(2 * R - 0.1)))['0']
    assert result.components == 2
    assert sum(result.masses) == pytest.approx(1.0)
    assert sum(result.spawn_shares) == pytest.approx(1.0)
    assert result.value == pytest.approx(sum(m * w for m, w in zip(result.masses, result.spawn_shares, strict=True)))
    assert result.value == pytest.approx(_expectation((4.0, 2.0), 4.0, SPAWN_CLEARANCE), abs=0.01)
    assert result.largest_share == pytest.approx(3.466 / (3.466 + 1.466), abs=0.01)
    assert result.cut_off_zones == ('b',)


def test_image_paints_dominant_and_stranded_components(tmp_path):
    image = tmp_path / 'nav.png'
    nav_at_r(_world(tmp_path, _two_rooms(2 * R - 0.1)), image=image)
    colors = {color for _, color in Image.open(image).convert('RGB').getcolors()}
    assert {(44, 162, 95), (230, 85, 13), (215, 215, 215), (20, 20, 20)} <= colors
    joined = tmp_path / 'joined.png'
    nav_at_r(_world(tmp_path, _two_rooms(1.0), name='joined'), image=joined)
    assert (230, 85, 13) not in {color for _, color in Image.open(joined).convert('RGB').getcolors()}


def test_walls_only_reports_narrow_door_as_layout_fault(tmp_path):
    result = nav_at_r(_world(tmp_path, _two_rooms(2 * R - 0.1)), walls_only=True)['0']
    assert result.components == 2
    assert result.cut_off_zones == ('b',)


def test_annotated_entity_in_doorway_cuts_room_off(tmp_path):
    level = _two_rooms(1.0)
    level['objects'] = [_entity('crate_1', 'crate', 4.0, 2.0), _entity('plant_1', 'plant', 1.0, 1.0)]
    view = _world(tmp_path, level)
    _asset(view, 'crate', {'bounding_box': [[-0.3, 0.3], [-0.6, 0.6], [0.0, 0.8]]})
    _asset(view, 'plant', {'name': 'plant'})

    furnished = nav_at_r(view)['0']
    walls_only = nav_at_r(view, walls_only=True)['0']

    assert walls_only.components == 1
    assert walls_only.value == pytest.approx(1.0)
    assert furnished.components == 2
    assert furnished.value < 1.0
    assert furnished.entities == 2
    assert furnished.unresolved == 1
    assert furnished.cut_off_zones == ('b',)
    assert {zone.name: zone.reachable_area for zone in furnished.zones}['b'] == 0.0


def test_overhead_entity_does_not_block_doorway(tmp_path):
    level = _two_rooms(1.0)
    level['objects'] = [_entity('sign_1', 'sign', 4.0, 2.0, z=2.2)]
    view = _world(tmp_path, level)
    _asset(view, 'sign', {'bounding_box': [[-0.3, 0.3], [-0.6, 0.6], [0.0, 0.3]]})
    result = nav_at_r(view)['0']
    assert result.components == 1
    assert result.unresolved == 0


def test_start_share_is_the_start_component_mass(tmp_path):
    view = _world(tmp_path, _two_rooms(2 * R - 0.1))
    big = nav_at_r(view, start=(2.0, 2.0))['0']
    small = nav_at_r(view, start=(5.0, 2.0))['0']
    assert big.start_share == pytest.approx(big.masses[0])
    assert big.start_snap == 0.0
    assert small.start_share == pytest.approx(big.masses[1])


def test_start_off_traversable_space_snaps_to_nearest_cell(tmp_path):
    result = nav_at_r(_world(tmp_path, _two_rooms(2 * R - 0.1)), start=(0.05, 2.0))['0']
    assert result.start_share == pytest.approx(result.masses[0])
    assert result.start_snap == pytest.approx(R, abs=0.1)


def test_levels_are_scored_separately(tmp_path):
    root = tmp_path / 'stack'
    narrow = _two_rooms(2 * R - 0.1)
    narrow['zones'] = [{**zone, 'name': f"{zone['name']}_1"} for zone in narrow['zones']]
    narrow['doors'] = [{**door, 'name': 'a_b_1'} for door in narrow['doors']]
    _write_world(root, {'0': _two_rooms(1.0), '1': narrow})
    view = WorldView(root)
    results = nav_at_r(view)
    assert sorted(results) == ['0', '1']
    assert results['0'].components == 1
    assert results['1'].components == 2
    assert sorted(nav_at_r(view, levels={'1'})) == ['1']


def test_resolve_world_directory_with_level_filter(tmp_path):
    view = _world(tmp_path, _two_rooms(1.0))
    resolved, levels = resolve_world(f'{view.path}[0]')
    assert resolved.path == view.path
    assert levels == {'0'}


def test_cli_table_lists_cut_off_zone(tmp_path, capsys):
    view = _world(tmp_path, _two_rooms(2 * R - 0.1))
    assert main([str(view.path), '--radius', str(R)]) == 0
    out = capsys.readouterr().out
    assert 'NAV@r' in out
    assert 'cut-off zones' in out
    assert out.strip().splitlines()[-1].split()[-1] == 'b'


def test_cli_json_carries_components_and_zones(tmp_path, capsys):
    view = _world(tmp_path, _two_rooms(1.0))
    assert main([str(view.path), '--start', '1,2', '--walls-only', '--json']) == 0
    data = json.loads(capsys.readouterr().out)
    level = data['levels'][0]
    assert data['walls_only'] is True
    assert level['components'] == 1
    assert level['start_share'] == pytest.approx(1.0)
    assert [zone['name'] for zone in level['zones']] == ['a', 'b']
    assert level['value'] == pytest.approx(1.0)


def test_cli_bare_call_prints_help(capsys):
    assert main([]) == 2
    err = capsys.readouterr().err
    assert '--radius' in err
    assert '--walls-only' in err


@pytest.mark.parametrize(
    ('argv', 'message'),
    [
        (['w', '--start', 'north'], 'expected X,Y'),
        (['w', '--radius', '-1'], 'positive length'),
        (['w', '--robot', 'jackal'], 'unrecognized arguments'),
    ],
)
def test_cli_bad_argument_explains_values(argv, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_unknown_world_directory_explains_lookup(tmp_path, capsys):
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(SystemExit) as exit_info:
        main([str(empty)])
    assert exit_info.value.code == 2
    assert 'pass a world directory' in capsys.readouterr().err
