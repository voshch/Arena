from __future__ import annotations

import asyncio
import zlib
from pathlib import Path

import pytest
import shapely
import yaml

from arena_simulation_setup.shared import CeilingLights, Light
from arena_simulation_setup.shared.walls import Wall
from arena_simulation_setup.tree.World.World import AUTO_CEILING_LIGHTS, Level, LevelDescription, MultiLevelWorldView, WorldDescription
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import Position

_L_SHAPE = [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (2.0, 2.0), (2.0, 6.0), (0.0, 6.0)]

_LEVEL_YAML = """
lights:
- {name: ambient, fixture: dome, lux: 50}
zones:
- name: central_hallway
  corners: [[0, 30], [20, 30], [20, 38], [0, 38]]
  ceiling_height: 2.5
  ceiling_lights: {fixture: panel, spacing: 2.4, lumens: 3600, cct_K: 4000, light_on: "!blackout"}
  lights:
  - {name: exit_strip, position: {x: 10, y: 34, z: 2.2}, fixture: tube, lumens: 300, light_on: blackout}
"""


def _rect(x_length: float, y_length: float, x0: float = 0.0, y0: float = 0.0) -> list[Position]:
    return [
        Position(x0, y0),
        Position(x0 + x_length, y0),
        Position(x0 + x_length, y0 + y_length),
        Position(x0, y0 + y_length),
    ]


def _lit_zone(name: str = 'office', corners: list[Position] | None = None, ceiling_lights: CeilingLights | None = None, **kwargs) -> LevelDescription.Zone:
    return LevelDescription.Zone(
        name=name,
        corners=corners if corners is not None else _rect(10.0, 6.0),
        ceiling_lights=ceiling_lights if ceiling_lights is not None else CeilingLights(),
        **kwargs,
    )


def _strip(name: str = 'exit_strip', **kwargs) -> Light:
    fields = {'name': name, 'fixture': 'tube', 'position': Position(1.0, 1.0, 2.2), 'lumens': 300.0}
    fields.update(kwargs)
    return Light(**fields)


def _only_light(zone: LevelDescription.Zone) -> Light:
    (light,) = asyncio.run(LevelDescription(zones=[zone]).all_lights())
    return light


def _ceiling_fields(level: LevelDescription) -> list[tuple]:
    return [(c.name, c.pos.x, c.pos.y, c.x_length, c.y_length, c.z, c.cast_shadows, c.material.name) for c in asyncio.run(level.all_ceilings())]


def test_zone_without_light_keys_yields_no_lights():
    level = converter.structure({'zones': [{'name': 'room', 'corners': [[0, 0], [4, 0], [4, 3], [0, 3]], 'ceiling_height': 2.5}]}, LevelDescription)
    (zone,) = level.zones
    assert zone.lights == []
    assert zone.ceiling_lights is None
    assert level.lights == []
    assert asyncio.run(level.all_lights()) == []
    level.validate_lights()


def test_zone_without_light_keys_keeps_all_ceilings_result():
    level = LevelDescription(zones=[LevelDescription.Zone(name='room', corners=_rect(4.0, 3.0), ceiling_height=2.5, ceiling_cast_shadows=True)])
    (ceiling,) = asyncio.run(level.all_ceilings())
    assert ceiling.name == 'room'
    assert ceiling.pos.x == pytest.approx(2.0)
    assert ceiling.pos.y == pytest.approx(1.5)
    assert ceiling.x_length == pytest.approx(4.0)
    assert ceiling.y_length == pytest.approx(3.0)
    assert ceiling.z == pytest.approx(2.5)
    assert ceiling.cast_shadows is True


def test_all_ceilings_wall_top_fallback_is_unchanged():
    raised = Wall(start=Position(0.0, 0.0, 1.0), end=Position(4.0, 0.0, 1.0))
    with_wall = LevelDescription(zones=[LevelDescription.Zone(name='tall', corners=_rect(4.0, 3.0), walls=[raised])])
    without_wall = LevelDescription(zones=[LevelDescription.Zone(name='bare', corners=_rect(4.0, 3.0))])
    assert asyncio.run(with_wall.all_ceilings())[0].z == pytest.approx(3.0)
    assert asyncio.run(without_wall.all_ceilings())[0].z == pytest.approx(2.0)


def test_ceiling_lights_do_not_change_all_ceilings():
    plain = LevelDescription(zones=[LevelDescription.Zone(name='office', corners=_rect(10.0, 6.0), ceiling_height=2.5)])
    lit = LevelDescription(zones=[_lit_zone(ceiling_height=2.5, lights=[_strip()])])
    assert _ceiling_fields(lit) == _ceiling_fields(plain)


def test_zone_without_lights_unstructures_like_sounds():
    unstructured = converter.unstructure(LevelDescription.Zone(name='room', corners=_rect(4.0, 3.0)))
    assert unstructured['sounds'] == []
    assert unstructured['lights'] == []
    assert unstructured['ceiling_lights'] is None


def test_ceiling_rig_grid_on_rectangular_zone():
    zone = _lit_zone(corners=_rect(10.0, 6.0, x0=10.0, y0=30.0), ceiling_height=2.5, ceiling_lights=CeilingLights(spacing=2.4))
    level = LevelDescription(zones=[zone])
    (rig,) = asyncio.run(level.all_lights())
    (ceiling,) = asyncio.run(level.all_ceilings())

    assert len(rig.fixtures) == 8
    assert [fixture.x for fixture in rig.fixtures] == pytest.approx([11.4, 13.8, 16.2, 18.6] * 2)
    assert [fixture.y for fixture in rig.fixtures] == pytest.approx([31.8] * 4 + [34.2] * 4)
    assert rig.fixtures[1].x - rig.fixtures[0].x == pytest.approx(2.4)
    assert rig.fixtures[4].y - rig.fixtures[0].y == pytest.approx(2.4)
    assert sum(fixture.x for fixture in rig.fixtures) / 8 == pytest.approx(ceiling.pos.x)
    assert sum(fixture.y for fixture in rig.fixtures) / 8 == pytest.approx(ceiling.pos.y)
    assert ceiling.z == pytest.approx(2.5)
    assert [fixture.z for fixture in rig.fixtures] == pytest.approx([ceiling.z - 0.02] * 8)


def test_ceiling_rig_is_one_rig_light_named_after_the_zone():
    cfg = CeilingLights(fixture='tube', spacing=3.0, lumens=2000.0, cct_K=3000.0, cast_shadows=True, light_on='!blackout', level=0.8, dead_fraction=0.25)
    rig = _only_light(_lit_zone(name='central_hallway', ceiling_height=2.5, ceiling_lights=cfg))
    assert rig.name == 'central_hallway'
    assert rig.rig is True
    assert rig.fixture == 'tube'
    assert rig.lumens == 2000.0
    assert rig.lux is None
    assert rig.cct_K == 3000.0
    assert rig.cast_shadows is True
    assert rig.light_on == '!blackout'
    assert rig.lit is None
    assert rig.level == pytest.approx(0.8)
    assert rig.dead_fraction == pytest.approx(0.25)
    assert rig.position is None
    assert rig.entity_ref == ''
    assert rig.frame == ''
    assert len(rig.fixture_ranks) == len(rig.fixtures)
    assert [(entry.role, entry.name) for entry in rig.semantics] == [('predicate', 'lit'), ('state', 'level'), ('state', 'dead_fraction')]


def test_ceiling_rig_carries_lit_false():
    rig = _only_light(_lit_zone(ceiling_height=2.5, ceiling_lights=CeilingLights(lit=False)))
    assert rig.lit is False
    assert rig.semantics[0].value is False


def test_ceiling_rig_keeps_rect_fixtures_inside_l_shaped_zone():
    zone = _lit_zone(name='ell', corners=[Position(x, y) for x, y in _L_SHAPE], ceiling_height=2.5, ceiling_lights=CeilingLights(spacing=1.5))
    rig = _only_light(zone)
    polygon = shapely.Polygon(_L_SHAPE)

    assert len(rig.fixtures) == 7
    assert [fixture.x for fixture in rig.fixtures] == pytest.approx([0.75, 2.25, 3.75, 5.25, 0.75, 0.75, 0.75])
    assert [fixture.y for fixture in rig.fixtures] == pytest.approx([0.75, 0.75, 0.75, 0.75, 2.25, 3.75, 5.25])
    for fixture in rig.fixtures:
        assert polygon.covers(shapely.box(fixture.x - 0.3, fixture.y - 0.3, fixture.x + 0.3, fixture.y + 0.3))


def test_ceiling_rig_keeps_disk_fixtures_inside_l_shaped_zone():
    zone = _lit_zone(name='ell', corners=[Position(x, y) for x, y in _L_SHAPE], ceiling_height=2.5, ceiling_lights=CeilingLights(fixture='downlight', spacing=1.0))
    rig = _only_light(zone)
    polygon = shapely.Polygon(_L_SHAPE)

    assert len(rig.fixtures) == 20
    for fixture in rig.fixtures:
        assert polygon.covers(shapely.Point(fixture.x, fixture.y).buffer(0.1))


def test_ceiling_rig_falls_back_to_one_fixture_in_a_small_zone():
    zone = _lit_zone(name='closet', corners=_rect(0.4, 0.4, x0=3.0, y0=3.0), ceiling_height=2.5)
    rig = _only_light(zone)
    (fixture,) = rig.fixtures
    assert shapely.Polygon([(3.0, 3.0), (3.4, 3.0), (3.4, 3.4), (3.0, 3.4)]).covers(shapely.Point(fixture.x, fixture.y))
    assert fixture.z == pytest.approx(2.48)
    assert len(rig.fixture_ranks) == 1


def test_ceiling_rig_grid_has_at_least_one_row_and_column():
    zone = _lit_zone(name='corridor', corners=_rect(10.0, 1.5), ceiling_height=2.5, ceiling_lights=CeilingLights(spacing=2.4))
    rig = _only_light(zone)
    assert len(rig.fixtures) == 4
    assert [fixture.y for fixture in rig.fixtures] == pytest.approx([0.75] * 4)


def test_ceiling_rig_z_follows_ceiling_height():
    level = LevelDescription(zones=[_lit_zone(ceiling_height=3.1)])
    (rig,) = asyncio.run(level.all_lights())
    (ceiling,) = asyncio.run(level.all_ceilings())
    assert ceiling.z == pytest.approx(3.1)
    assert [fixture.z for fixture in rig.fixtures] == pytest.approx([3.08] * len(rig.fixtures))


def test_ceiling_rig_z_falls_back_to_wall_top():
    raised = Wall(start=Position(0.0, 0.0, 1.0), end=Position(10.0, 0.0, 1.0))
    level = LevelDescription(zones=[_lit_zone(walls=[raised])])
    (rig,) = asyncio.run(level.all_lights())
    (ceiling,) = asyncio.run(level.all_ceilings())
    assert ceiling.z == pytest.approx(3.0)
    assert [fixture.z for fixture in rig.fixtures] == pytest.approx([ceiling.z - 0.02] * len(rig.fixtures))


def test_ceiling_rig_z_defaults_without_walls_or_height():
    level = LevelDescription(zones=[_lit_zone()])
    (rig,) = asyncio.run(level.all_lights())
    (ceiling,) = asyncio.run(level.all_ceilings())
    assert ceiling.z == pytest.approx(2.0)
    assert [fixture.z for fixture in rig.fixtures] == pytest.approx([1.98] * len(rig.fixtures))


def test_fixture_ranks_identical_across_derivations():
    first = _only_light(_lit_zone(corners=_rect(20.0, 13.0), ceiling_height=2.5))
    second = _only_light(_lit_zone(corners=_rect(20.0, 13.0), ceiling_height=2.5))
    assert len(first.fixtures) == 40
    assert first.fixture_ranks == second.fixture_ranks
    assert first.fixture_ranks == [zlib.crc32(f'office:{i}'.encode()) / 2**32 for i in range(40)]
    assert all(0.0 <= rank < 1.0 for rank in first.fixture_ranks)


def test_fixture_ranks_depend_on_zone_name():
    office = _only_light(_lit_zone(name='office', ceiling_height=2.5))
    lobby = _only_light(_lit_zone(name='lobby', ceiling_height=2.5))
    assert office.fixture_ranks != lobby.fixture_ranks


def test_alive_is_monotone_in_dead_fraction():
    rig = _only_light(_lit_zone(corners=_rect(20.0, 13.0), ceiling_height=2.5))
    alive_at_02 = {i for i, alive in enumerate(rig.alive(0.2)) if alive}
    alive_at_05 = {i for i, alive in enumerate(rig.alive(0.5)) if alive}
    assert alive_at_05 <= alive_at_02
    assert rig.alive(0.0) == [True] * 40
    assert rig.alive(1.0) == [False] * 40

    previous = set(range(40))
    for step in range(1, 11):
        current = {i for i, alive in enumerate(rig.alive(step / 10)) if alive}
        assert current <= previous
        previous = current


def test_all_lights_orders_level_then_explicit_then_rigs():
    ambient = Light(name='ambient', fixture='dome', lux=50.0)
    first = _lit_zone(name='first', ceiling_height=2.5, lights=[_strip('first_strip')])
    second = LevelDescription.Zone(name='second', corners=_rect(4.0, 4.0, x0=20.0), lights=[_strip('second_strip')])
    third = _lit_zone(name='third', corners=_rect(6.0, 6.0, x0=40.0), ceiling_height=2.5)
    level = LevelDescription(zones=[first, second, third], lights=[ambient])
    lights = asyncio.run(level.all_lights())
    assert [light.name for light in lights] == ['ambient', 'first_strip', 'second_strip', 'first', 'third']
    assert [light.rig for light in lights] == [False, False, False, True, True]
    assert lights[0] is ambient


def test_validate_lights_accepts_ambient_explicit_and_rig():
    level = LevelDescription(
        zones=[_lit_zone(ceiling_height=2.5, lights=[_strip()])],
        lights=[Light(name='ambient', fixture='dome', lux=50.0), Light(name='sun', fixture='sun', lux=10000.0)],
    )
    level.validate_lights()


def test_validate_lights_rejects_local_fixture_at_level():
    level = LevelDescription(zones=[LevelDescription.Zone(name='room', corners=_rect(4.0, 4.0))], lights=[_strip()])
    with pytest.raises(ValueError, match='level lights take only dome or sun'):
        level.validate_lights()


@pytest.mark.parametrize('fixture', ['dome', 'sun'])
def test_validate_lights_rejects_ambient_fixture_in_zone(fixture):
    zone = LevelDescription.Zone(name='room', corners=_rect(4.0, 4.0), lights=[Light(name='sky', fixture=fixture, lux=50.0)])
    with pytest.raises(ValueError, match='belongs in the level lights'):
        LevelDescription(zones=[zone]).validate_lights()


def test_validate_lights_rejects_duplicate_names_across_zones():
    first = LevelDescription.Zone(name='first', corners=_rect(4.0, 4.0), lights=[_strip('strip')])
    second = LevelDescription.Zone(name='second', corners=_rect(4.0, 4.0, x0=10.0), lights=[_strip('strip')])
    with pytest.raises(ValueError, match="duplicate light 'strip'"):
        LevelDescription(zones=[first, second]).validate_lights()


def test_validate_lights_rejects_duplicate_ambient_names():
    level = LevelDescription(lights=[Light(name='ambient', fixture='dome', lux=50.0), Light(name='ambient', fixture='sun', lux=100.0)])
    with pytest.raises(ValueError, match="duplicate light 'ambient'"):
        level.validate_lights()


def test_validate_lights_rejects_explicit_light_named_like_a_rig_zone():
    first = LevelDescription.Zone(name='first', corners=_rect(4.0, 4.0), lights=[_strip('office')])
    with pytest.raises(ValueError, match="duplicate light 'office'"):
        LevelDescription(zones=[first, _lit_zone(name='office')]).validate_lights()
    with pytest.raises(ValueError, match="duplicate light 'office'"):
        LevelDescription(zones=[_lit_zone(name='office'), first]).validate_lights()


def test_validate_lights_rejects_ambient_light_named_like_a_rig_zone():
    level = LevelDescription(zones=[_lit_zone(name='office')], lights=[Light(name='office', fixture='dome', lux=50.0)])
    with pytest.raises(ValueError, match="duplicate light 'office'"):
        level.validate_lights()


def test_validate_lights_rejects_two_rig_zones_with_one_name():
    level = LevelDescription(zones=[_lit_zone(name='office'), _lit_zone(name='office', corners=_rect(4.0, 4.0, x0=20.0))])
    with pytest.raises(ValueError, match="duplicate light 'office'"):
        level.validate_lights()


def test_validate_lights_allows_explicit_light_named_like_an_unlit_zone():
    zone = LevelDescription.Zone(name='room', corners=_rect(4.0, 4.0), lights=[_strip('room')])
    LevelDescription(zones=[zone]).validate_lights()


@pytest.mark.parametrize('kwargs', [{'ceiling': False}, {'ceiling_material': ''}])
def test_validate_lights_rejects_ceiling_lights_without_ceiling(kwargs):
    with pytest.raises(ValueError, match='has ceiling_lights but no ceiling'):
        LevelDescription(zones=[_lit_zone(**kwargs)]).validate_lights()


def test_validate_lights_rejects_ceiling_lights_without_polygon():
    with pytest.raises(ValueError, match='has ceiling_lights but no polygon'):
        LevelDescription(zones=[_lit_zone(corners=[])]).validate_lights()


def test_world_validate_runs_validate_lights():
    world = WorldDescription.from_levels(LevelDescription(zones=[_lit_zone(ceiling=False)]))
    with pytest.raises(ValueError, match='has ceiling_lights but no ceiling'):
        world.validate()


def test_shift_all_positions_moves_positioned_lights_only():
    positioned = _strip('positioned', position=Position(1.0, 1.0, 2.2))
    attached = _strip('attached', position=None, entity_ref='desk')
    ambient = Light(name='ambient', fixture='dome', lux=50.0)
    level = LevelDescription(zones=[LevelDescription.Zone(name='hall', corners=_rect(4.0, 4.0), lights=[positioned, attached])], lights=[ambient])

    level.shift_all_positions(3.0, -2.0)

    assert positioned.position.x == pytest.approx(4.0)
    assert positioned.position.y == pytest.approx(-1.0)
    assert positioned.position.z == pytest.approx(2.2)
    assert attached.position is None
    assert attached.offset == Position(0.0, 0.0, 0.0)
    assert ambient.position is None


def test_shift_all_positions_moves_rig_fixtures_with_the_zone():
    level = LevelDescription(zones=[_lit_zone(ceiling_height=2.5)])
    (before,) = asyncio.run(level.all_lights())
    level.shift_all_positions(100.0, 50.0)
    (after,) = asyncio.run(level.all_lights())
    assert [fixture.x for fixture in after.fixtures] == pytest.approx([fixture.x + 100.0 for fixture in before.fixtures])
    assert [fixture.y for fixture in after.fixtures] == pytest.approx([fixture.y + 50.0 for fixture in before.fixtures])
    assert after.fixture_ranks == before.fixture_ranks


def test_level_from_level_description_carries_lights():
    ambient = Light(name='ambient', fixture='dome', lux=50.0)
    level = Level.from_level_description(LevelDescription(zones=[_lit_zone()], lights=[ambient]))
    assert level.lights == [ambient]
    assert level.zones[0].ceiling_lights == CeilingLights()


def test_compact_world_carries_and_shifts_lights():
    ambient = Light(name='ambient', fixture='dome', lux=50.0)
    zone = LevelDescription.Zone(name='hall', corners=_rect(4.0, 4.0), lights=[_strip(position=Position(1.0, 1.0, 2.2))])
    world = WorldDescription.from_levels(LevelDescription(zones=[zone], lights=[ambient]))

    compacted = world.compact_world({'0': (5.0, 7.0)})

    assert [light.name for light in compacted.lights] == ['ambient']
    (strip,) = compacted.zones[0].lights
    assert strip.position.x == pytest.approx(6.0)
    assert strip.position.y == pytest.approx(8.0)
    assert zone.lights[0].position.x == pytest.approx(1.0)


def test_level_yaml_structures_lights_and_ceiling_lights():
    level = converter.structure(yaml.safe_load(_LEVEL_YAML), LevelDescription)
    (zone,) = level.zones

    (ambient,) = level.lights
    assert ambient.name == 'ambient'
    assert ambient.fixture == 'dome'
    assert ambient.lux == 50.0

    assert zone.ceiling_lights == CeilingLights(fixture='panel', spacing=2.4, lumens=3600.0, cct_K=4000.0, light_on='!blackout')

    (strip,) = zone.lights
    assert strip.name == 'exit_strip'
    assert strip.fixture == 'tube'
    assert strip.position == Position(10.0, 34.0, 2.2)
    assert strip.lumens == 300.0
    assert strip.light_on == 'blackout'
    assert strip.rig is False

    level.validate_lights()


def test_level_yaml_derives_rig_and_semantics():
    level = converter.structure(yaml.safe_load(_LEVEL_YAML), LevelDescription)
    ambient, strip, rig = asyncio.run(level.all_lights())

    assert [ambient.name, strip.name, rig.name] == ['ambient', 'exit_strip', 'central_hallway']
    assert rig.rig is True
    assert len(rig.fixtures) == 24
    assert [fixture.z for fixture in rig.fixtures] == pytest.approx([2.48] * 24)
    assert rig.semantics[0].value is None
    assert rig.semantics[0].params == {'light_on': '!blackout'}
    assert strip.semantics[0].params == {'light_on': 'blackout'}
    assert ambient.semantics[0].value is True


def test_zone_yaml_rejects_unknown_ceiling_lights_key():
    raw = {'name': 'room', 'corners': [[0, 0], [4, 0], [4, 3], [0, 3]], 'ceiling_lights': {'fixture': 'panel', 'pitch': 2.0}}
    with pytest.raises(Exception):
        converter.structure(raw, LevelDescription.Zone)


def test_zone_yaml_rejects_invalid_light():
    raw = {'name': 'room', 'corners': [[0, 0], [4, 0], [4, 3], [0, 3]], 'lights': [{'name': 'lamp', 'fixture': 'panel', 'lumens': 3600}]}
    with pytest.raises(Exception):
        converter.structure(raw, LevelDescription.Zone)


def test_level_unstructure_omits_derived_light_fields():
    level = converter.structure(yaml.safe_load(_LEVEL_YAML), LevelDescription)
    unstructured = converter.unstructure(level)

    assert unstructured['lights'] == [{'name': 'ambient', 'fixture': 'dome', 'lux': 50.0}]
    (zone,) = unstructured['zones']
    assert zone['ceiling_lights'] == {'cct_K': 4000.0, 'light_on': '!blackout'}
    (strip,) = zone['lights']
    assert set(strip) == {'name', 'fixture', 'position', 'lumens', 'light_on'}
    assert 'rig' not in strip
    assert 'fixtures' not in strip
    assert 'fixture_ranks' not in strip


def test_level_lights_round_trip_through_yaml():
    level = converter.structure(yaml.safe_load(_LEVEL_YAML), LevelDescription)
    reloaded = converter.structure(yaml.safe_load(yaml.safe_dump(converter.unstructure(level), sort_keys=False)), LevelDescription)

    assert [(light.name, light.fixture, light.lux) for light in reloaded.lights] == [('ambient', 'dome', 50.0)]
    assert reloaded.zones[0].ceiling_lights == level.zones[0].ceiling_lights
    (strip,) = reloaded.zones[0].lights
    assert (strip.name, strip.fixture, strip.lumens, strip.light_on) == ('exit_strip', 'tube', 300.0, 'blackout')
    assert strip.position == Position(10.0, 34.0, 2.2)

    original = asyncio.run(level.all_lights())
    derived = asyncio.run(reloaded.all_lights())
    assert [light.name for light in derived] == [light.name for light in original]
    assert derived[-1].fixtures == original[-1].fixtures
    assert derived[-1].fixture_ranks == original[-1].fixture_ranks


def test_world_export_writes_lights_that_load_from_disk(tmp_path: Path):
    world = WorldDescription.from_levels(converter.structure(yaml.safe_load(_LEVEL_YAML), LevelDescription))
    tarball = world.export()
    level_dir = tmp_path / 'lit_world' / '0'
    level_dir.mkdir(parents=True)
    (level_dir / 'world.yaml').write_bytes(tarball.extractfile('0/world.yaml').read())

    loaded = MultiLevelWorldView(tmp_path / 'lit_world').load()

    level = loaded.levels['0']
    assert [(light.name, light.fixture, light.lux) for light in level.lights] == [('ambient', 'dome', 50.0)]
    assert level.zones[0].ceiling_lights == CeilingLights(cct_K=4000.0, light_on='!blackout')
    assert [light.name for light in level.zones[0].lights] == ['exit_strip']
    assert [light.name for light in asyncio.run(level.all_lights())] == ['ambient', 'exit_strip', 'central_hallway']


def _desk_zone(lights: list[Light], yaw: float = 0.0, desks: int = 1) -> LevelDescription.Zone:
    from arena_simulation_setup.shared import Obstacle
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
    from arena_simulation_setup.utils.geometry import Orientation, Pose

    statics = [Obstacle(name='desk', pose=Pose(Position(4.0, 2.0, 0.0), Orientation.from_yaw(yaw)), model=ObjectIdentifier('test_model'), scale=None) for _ in range(desks)]
    return LevelDescription.Zone(name='office', corners=_rect(10.0, 6.0), lights=lights, entities=LevelDescription.Zone.WorldEntities(static=statics))


def test_entity_ref_light_sits_on_its_static_entity_with_turned_offset() -> None:
    import math

    lamp = Light(name='desk_lamp', fixture='spot', entity_ref='desk', offset=Position(1.0, 0.0, 0.8), direction=(1.0, 0.0, -1.0), lumens=400.0)
    (light,) = asyncio.run(LevelDescription(zones=[_desk_zone([lamp], yaw=math.pi / 2)]).all_lights())
    assert light.entity_ref == ''
    assert (light.position.x, light.position.y, light.position.z) == pytest.approx((4.0, 3.0, 0.8))
    assert light.direction == pytest.approx((0.0, 2**-0.5, -(2**-0.5)))


@pytest.mark.parametrize(('desks', 'word'), [(0, 'unknown'), (2, 'ambiguous')])
def test_entity_ref_light_rejects_unresolvable_entities(desks: int, word: str) -> None:
    lamp = Light(name='desk_lamp', fixture='downlight', entity_ref='desk', lumens=400.0)
    level = LevelDescription(zones=[_desk_zone([lamp], desks=desks)])
    with pytest.raises(ValueError, match=word):
        level.validate_lights()
    with pytest.raises(ValueError, match=word):
        asyncio.run(level.all_lights())


def test_frame_light_keeps_its_frame_and_offset() -> None:
    headlight = Light(name='headlight', fixture='spot', frame='jackal/base_link', offset=Position(0.2, 0.0, 0.3), lumens=800.0)
    (light,) = asyncio.run(LevelDescription(zones=[_desk_zone([headlight])]).all_lights())
    assert light.frame == 'jackal/base_link'
    assert light.position is None
    assert (light.offset.x, light.offset.z) == (0.2, 0.3)


_LAMP_ANNOTATION = {
    'name': 'Desk_Lamp',
    'path': 'Object/Desk_Lamp',
    'bounding_box': [[-0.1, 0.1], [-0.1, 0.1], [0.0, 0.5]],
    'lights': [
        {'name': 'bulb', 'fixture': 'bulb', 'offset': [0.1, 0.0, 0.45], 'lumens': 800.0, 'cct_K': 2700.0, 'glow': 'Shade'},
        {'name': 'reading', 'fixture': 'spot', 'offset': [0.0, 0.0, 0.4], 'direction': [1.0, 0.0, -1.0], 'lumens': 300.0, 'cone_deg': 60.0},
    ],
}


@pytest.fixture
def world_assets(tmp_path: Path):
    from arena_simulation_setup.tree import DynamicPaths

    lamp = tmp_path / 'assets' / 'Common' / 'Object' / 'Desk_Lamp'
    lamp.mkdir(parents=True)
    (lamp / 'annotation.yaml').write_text(yaml.safe_dump(_LAMP_ANNOTATION))
    plain = tmp_path / 'assets' / 'Common' / 'Object' / 'Plain_Desk'
    plain.mkdir(parents=True)
    (plain / 'annotation.yaml').write_text(yaml.safe_dump({'name': 'Plain_Desk', 'path': 'Object/Plain_Desk'}))
    previous = DynamicPaths.WORLD.path
    DynamicPaths.WORLD.path = tmp_path
    yield tmp_path
    DynamicPaths.WORLD.path = previous


def _lamp_zone(*statics: dict) -> LevelDescription.Zone:
    from arena_simulation_setup.shared import Obstacle

    return LevelDescription.Zone(
        name='study',
        corners=_rect(6.0, 6.0),
        entities=LevelDescription.Zone.WorldEntities(static=[converter.structure(raw, Obstacle) for raw in statics]),
    )


def test_object_lights_follow_their_entity_pose_and_scale(world_assets: Path) -> None:
    import math

    lamp = {'name': 'desk_lamp', 'model': 'Common/Desk_Lamp', 'pose': {'position': [2.0, 1.0, 0.7], 'orientation': {'x': 0.0, 'y': 0.0, 'z': math.sin(math.pi / 4), 'w': math.cos(math.pi / 4)}}, 'scale': [2.0, 2.0, 2.0]}
    bulb, reading = asyncio.run(LevelDescription(zones=[_lamp_zone(lamp)]).all_lights())
    assert (bulb.name, reading.name) == ('desk_lamp_bulb', 'desk_lamp_reading')
    assert bulb.intrinsic and reading.intrinsic
    assert bulb.entity_ref == ''
    assert (bulb.owner, bulb.glow) == ('desk_lamp', 'Shade')
    assert (reading.owner, reading.glow) == ('desk_lamp', '')
    assert (bulb.position.x, bulb.position.y, bulb.position.z) == pytest.approx((2.0, 1.2, 1.6))
    assert reading.direction == pytest.approx((0.0, 2**-0.5, -(2**-0.5)))
    assert reading.spec.cone_deg == 60.0
    assert bulb.cct_K == 2700.0


def test_object_lights_take_the_placement_settings(world_assets: Path) -> None:
    lamp = {'name': 'desk_lamp', 'model': 'Common/Desk_Lamp', 'pose': {'position': [2.0, 1.0, 0.7]}, 'light': {'light_on': '!daylight', 'level': 0.5}}
    lights = asyncio.run(LevelDescription(zones=[_lamp_zone(lamp)]).all_lights())
    assert [(light.light_on, light.level) for light in lights] == [('!daylight', 0.5), ('!daylight', 0.5)]


def test_objects_without_annotated_lights_add_none(world_assets: Path) -> None:
    desk = {'name': 'desk', 'model': 'Common/Plain_Desk', 'pose': {'position': [1.0, 1.0, 0.0]}}
    assert asyncio.run(LevelDescription(zones=[_lamp_zone(desk)]).all_lights()) == []


def test_object_lights_follow_declared_lights_and_count_once_per_placement(world_assets: Path) -> None:
    lamps = [{'name': f'lamp_{index}', 'model': 'Common/Desk_Lamp', 'pose': {'position': [float(index), 1.0, 0.0]}} for index in (1, 2)]
    zone = _lamp_zone(*lamps)
    zone.lights = [_strip()]
    names = [light.name for light in asyncio.run(LevelDescription(zones=[zone]).all_lights())]
    assert names == ['exit_strip', 'lamp_1_bulb', 'lamp_1_reading', 'lamp_2_bulb', 'lamp_2_reading']


def test_object_light_named_like_a_declared_light_is_rejected(world_assets: Path) -> None:
    zone = _lamp_zone({'name': 'desk_lamp', 'model': 'Common/Desk_Lamp', 'pose': {'position': [2.0, 1.0, 0.0]}})
    zone.lights = [_strip(name='desk_lamp_bulb')]
    with pytest.raises(ValueError, match='collides'):
        asyncio.run(LevelDescription(zones=[zone]).all_lights())


def test_validate_lights_leaves_object_lights_to_spawn_time(world_assets: Path) -> None:
    zone = _lamp_zone({'name': 'desk_lamp', 'model': 'Common/Desk_Lamp', 'pose': {'position': [2.0, 1.0, 0.0]}})
    zone.lights = [_strip(name='desk_lamp_bulb')]
    LevelDescription(zones=[zone]).validate_lights()


def _auto_level() -> LevelDescription:
    own = CeilingLights(lumens=3000.0)
    lamp = Light(name='lamp', fixture='bulb', position=Position(1.0, 1.0, 1.0), lumens=800.0)
    return LevelDescription(
        zones=[
            LevelDescription.Zone(name='bare', corners=_rect(4.0, 4.0)),
            LevelDescription.Zone(name='rigged', corners=_rect(4.0, 4.0, 5.0), ceiling_lights=own),
            LevelDescription.Zone(name='lamped', corners=_rect(4.0, 4.0, 10.0), lights=[lamp]),
            LevelDescription.Zone(name='open', corners=_rect(4.0, 4.0, 15.0), ceiling=False),
        ]
    )


def test_auto_lighting_rigs_only_ceilinged_zones_without_lights() -> None:
    zones = {zone.name: zone for zone in _auto_level().with_lighting('auto').zones}
    assert zones['bare'].ceiling_lights == AUTO_CEILING_LIGHTS
    assert zones['rigged'].ceiling_lights.lumens == 3000.0
    assert zones['lamped'].ceiling_lights is None
    assert zones['open'].ceiling_lights is None
    rig = next(light for light in asyncio.run(_auto_level().with_lighting('auto').all_lights()) if light.name == 'bare')
    assert (rig.fixture, rig.lumens, rig.cct_K, rig.rig) == ('panel', 7200.0, 6500.0, True)


def test_authored_lighting_leaves_the_level_alone() -> None:
    level = _auto_level()
    assert level.with_lighting('authored') is level


def test_unknown_lighting_mode_names_the_modes() -> None:
    with pytest.raises(ValueError, match="world.lighting 'bright' is not one of authored, auto"):
        _auto_level().with_lighting('bright')
