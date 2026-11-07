from __future__ import annotations

import copy
import math
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.shared import Stairs, stair_height
from arena_simulation_setup.tree.World import FORMAT, WorldFormatError, WorldView, parse_world, write_world
from arena_simulation_setup.tree.World.model import World
from arena_simulation_setup.utils.geometry import Position

DEMO = Path(__file__).resolve().parents[2] / 'worlds' / 'ramp_demo'

WORLD: dict = {
    'format': FORMAT,
    'levels': [
        {
            'id': '0',
            'zones': [{'name': 'hall', 'polygon': [[0, 0], [60, 0], [60, 12], [0, 12]]}],
            'ramps': [
                {'name': 'up', 'mouth': [10.0, 6.0], 'size': [24.0, 12.0], 'rise': 1.0},
                {'name': 'deck', 'mouth': [34.0, 6.0], 'size': [26.0, 12.0], 'base': 1.0},
            ],
        },
    ],
}


def _world(change: Callable[[dict], object] | None = None) -> dict:
    data = copy.deepcopy(WORLD)
    if change is not None:
        change(data)
    return data


def _ramps(world: World) -> dict[str, Stairs]:
    return {stair.name: stair for level in world.all_levels for stair in level.stairs}


def _set(index: int, **fields: object) -> Callable[[dict], None]:
    return lambda data: data['levels'][0]['ramps'][index].update(fields)


def test_a_ramp_rises_its_whole_rise_from_its_base_and_a_deck_stays_level() -> None:
    ramps = list(_ramps(parse_world(_world())).values())
    assert stair_height(ramps, 5.0, 6.0) == 0.0
    assert stair_height(ramps, 10.0, 6.0) == 0.0
    assert stair_height(ramps, 16.0, 6.0) == pytest.approx(0.25)
    assert stair_height(ramps, 33.999, 6.0) == pytest.approx(1.0, abs=1e-3)
    assert stair_height(ramps, 50.0, 2.0) == pytest.approx(1.0)


def test_a_ramp_can_run_along_any_axis_from_a_base_of_its_own() -> None:
    ramp = _ramps(parse_world(_world(_set(0, mouth=[6.0, 88.0], mouth_side='+y', size=[50.0, 8.0], base=-2.35, rise=2.15))))['up']
    assert ramp.position.z == -2.35
    assert stair_height([ramp], 6.0, 88.0) == pytest.approx(-2.35)
    assert stair_height([ramp], 6.0, 63.0) == pytest.approx(-2.35 + 2.15 / 2.0)
    assert stair_height([ramp], 6.0, 38.0) == pytest.approx(-0.2)
    assert stair_height([ramp], 6.0, 89.0) == 0.0


def test_a_ramp_stays_on_its_level_without_a_seam_partner_or_walls() -> None:
    world = parse_world(_world())
    ramp = _ramps(world)['up']
    assert ramp.kind == 'ramp' and not ramp.boards and not ramp.top and ramp.belt == 0.0
    assert ramp.walls() == [] and ramp.open_walls() == [] and ramp.floor_openings() == [] and ramp.ceiling_openings() == []
    assert world.levels['0'].all_walls == []
    assert world.stair_levels() == {'up': '0', 'deck': '0'}


def test_asked_for_side_walls_follow_the_surface() -> None:
    walls = _ramps(parse_world(_world(_set(0, sides={'left': {'height': 1.0}}))))['up'].walls()
    assert [(wall.start.y, wall.start.z, wall.end.z, wall.height) for wall in walls] == [(12.0, 0.0, 1.0, 1.0)]


def test_a_ramp_spawns_one_slab_with_its_top_on_the_surface() -> None:
    boxes = _ramps(parse_world(_world()))['up'].static_boxes()
    assert [name for name, _material, _box in boxes] == ['base_0']
    (length, width, height), pose = boxes[0][2]
    slope = math.atan2(1.0, 24.0)
    assert length == pytest.approx(24.0 / math.cos(slope)) and width == 12.0
    top = pose.position + pose.orientation * Position(0.0, 0.0, height / 2.0)
    assert (top.x, top.y, top.z) == pytest.approx((22.0, 6.0, 0.5))


def test_the_floor_key_names_the_slab_material_or_leaves_the_surface_to_the_world() -> None:
    assert _ramps(parse_world(_world(_set(0, floor='Oak_Planks'))))['up'].static_boxes()[0][1].name == 'Oak_Planks'
    assert _ramps(parse_world(_world(_set(0, floor=False))))['up'].static_boxes() == []


def test_a_world_without_floors_has_ramps_without_slabs() -> None:
    def no_floors(data: dict) -> None:
        data['defaults'] = {'floor': False}

    ramps = _ramps(parse_world(_world(no_floors)))
    assert ramps['up'].static_boxes() == [] and ramps['deck'].static_boxes() == []
    assert _ramps(parse_world(_world(lambda data: (no_floors(data), _set(0, floor='Marble')(data)))))['up'].static_boxes() != []


@pytest.mark.parametrize(
    ('change', 'message'),
    [
        (_set(0, rise=-1.0), r'levels\[0\]\.ramps\[0\]\.rise'),
        (_set(1, size=[1.0]), r'levels\[0\]\.ramps\[1\]\.size'),
        (_set(0, mouth_side='up'), r'levels\[0\]\.ramps\[0\]\.mouth_side'),
        (_set(0, to='1'), r'levels\[0\]\.ramps\[0\]'),
        (_set(0, sides={'top': {}}), r'levels\[0\]\.ramps\[0\]\.sides'),
    ],
)
def test_a_broken_ramp_is_named_by_its_yaml_path(change: Callable[[dict], None], message: str) -> None:
    with pytest.raises(WorldFormatError, match=message):
        parse_world(_world(change))


@pytest.mark.parametrize('defaults', [{}, {'floor': False}])
def test_written_ramps_parse_back_equal(defaults: dict) -> None:
    def change(data: dict) -> None:
        data.update({'defaults': defaults} if defaults else {})
        data['levels'][0]['ramps'][0].update(mouth_side='+y', base=-2.35, floor='Oak_Planks', sides={'right': {'height': 0.8}})

    world = parse_world(_world(change))
    text = write_world(world)
    written = yaml.safe_load(text)
    assert written['levels'][0]['ramps'][0] == {'name': 'up', 'mouth': [10.0, 6.0], 'mouth_side': '+y', 'size': [24.0, 12.0], 'base': -2.35, 'rise': 1.0, 'floor': 'Oak_Planks', 'sides': {'right': {'height': 0.8}}}
    again = parse_world(written)
    assert [level.stairs for level in again.all_levels] == [level.stairs for level in world.all_levels]
    assert write_world(again) == text


def test_prefixed_levels_prefix_ramp_names() -> None:
    assert _ramps(parse_world(_world())) and [stair.name for stair in parse_world(_world()).levels['0'].prefixed('a_').stairs] == ['a_up', 'a_deck']


def test_a_ramp_resolves_as_a_named_place() -> None:
    corners = parse_world(_world()).levels['0'].lookup_zone_polygon('up')
    assert corners is not None and {(corner.x, corner.y) for corner in corners} == {(10.0, 0.0), (34.0, 0.0), (34.0, 12.0), (10.0, 12.0)}


def test_the_demo_world_loads_with_its_ramp_and_deck() -> None:
    world = WorldView(DEMO).load()
    ramps = _ramps(world)
    assert {stair.kind for stair in ramps.values()} == {'ramp'}
    assert stair_height(list(ramps.values()), 22.0, 6.0) == pytest.approx(0.5)
    assert stair_height(list(ramps.values()), 50.0, 6.0) == pytest.approx(1.0)
