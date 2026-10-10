from __future__ import annotations

import math

import pytest
import yaml

from arena_simulation_setup.shared import LIGHT_FIXTURES, CeilingLights, Light, LightFixture, ObjectLightSettings, Obstacle, cct_to_rgb, object_light
from arena_simulation_setup.shared.semantics import SemanticCfg, parse_semantics
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import Position


def _panel(**kwargs) -> Light:
    fields = {'name': 'lamp', 'fixture': 'panel', 'position': Position(1.0, 2.0, 2.4), 'lumens': 3600.0}
    fields.update(kwargs)
    return Light(**fields)


def _rig(**kwargs) -> Light:
    fields = {
        'name': 'office',
        'fixture': 'panel',
        'lumens': 3600.0,
        'rig': True,
        'fixtures': [Position(1.0, 1.0, 2.48), Position(3.4, 1.0, 2.48)],
        'fixture_ranks': [0.25, 0.75],
    }
    fields.update(kwargs)
    return Light(**fields)


def test_light_fixture_table_shapes():
    assert set(LIGHT_FIXTURES) == {'panel', 'tube', 'downlight', 'spot', 'bulb', 'dome', 'sun'}
    assert LIGHT_FIXTURES['bulb'] == LightFixture(shape='sphere', radius=0.03)
    assert LIGHT_FIXTURES['panel'] == LightFixture(shape='rect', x_length=0.6, y_length=0.6)
    assert LIGHT_FIXTURES['tube'] == LightFixture(shape='rect', x_length=1.2, y_length=0.1)
    assert LIGHT_FIXTURES['downlight'] == LightFixture(shape='disk', radius=0.1)
    assert LIGHT_FIXTURES['spot'] == LightFixture(shape='disk', radius=0.05, cone_deg=40.0)
    assert LIGHT_FIXTURES['dome'] == LightFixture(shape='dome')
    assert LIGHT_FIXTURES['sun'] == LightFixture(shape='distant')


def test_light_fixture_ambient_only_for_dome_and_sun():
    assert {name for name, spec in LIGHT_FIXTURES.items() if spec.ambient} == {'dome', 'sun'}


def test_light_fixture_area():
    assert LIGHT_FIXTURES['panel'].area == pytest.approx(0.36)
    assert LIGHT_FIXTURES['tube'].area == pytest.approx(0.12)
    assert LIGHT_FIXTURES['downlight'].area == pytest.approx(math.pi * 0.01)
    assert LIGHT_FIXTURES['spot'].area == pytest.approx(math.pi * 0.0025)
    assert LIGHT_FIXTURES['bulb'].area == pytest.approx(4.0 * math.pi * 0.0009)
    assert LIGHT_FIXTURES['dome'].area == 0.0
    assert LIGHT_FIXTURES['sun'].area == 0.0


def test_light_fixture_is_frozen():
    with pytest.raises(AttributeError):
        LIGHT_FIXTURES['panel'].radius = 1.0


@pytest.mark.parametrize('fixture', ['panel', 'tube', 'downlight', 'spot'])
def test_light_parses_local_fixture(fixture):
    raw = {'name': 'lamp', 'fixture': fixture, 'position': {'x': 1, 'y': 2, 'z': 2.4}, 'lumens': 800}
    light = converter.structure(raw, Light)
    assert light.name == 'lamp'
    assert light.fixture == fixture
    assert light.spec is LIGHT_FIXTURES[fixture]
    assert light.position == Position(1.0, 2.0, 2.4)
    assert light.lumens == 800.0
    assert isinstance(light.lumens, float)
    assert light.lux is None
    assert light.entity_ref == ''
    assert light.frame == ''
    assert light.offset == Position(0.0, 0.0, 0.0)
    assert light.direction == (0.0, 0.0, -1.0)
    assert light.cct_K == 6500.0
    assert light.cast_shadows is False
    assert light.light_on == ''
    assert light.lit is None
    assert light.level == 1.0
    assert light.dead_fraction == 0.0
    assert light.rig is False
    assert light.fixtures == []
    assert light.fixture_ranks == []


def test_light_parses_position_list():
    light = converter.structure({'name': 'lamp', 'fixture': 'tube', 'position': [4.0, 2.5, 2.2], 'lumens': 300}, Light)
    assert light.position == Position(4.0, 2.5, 2.2)


@pytest.mark.parametrize('fixture', ['dome', 'sun'])
def test_light_parses_ambient_fixture(fixture):
    light = converter.structure({'name': 'ambient', 'fixture': fixture, 'lux': 50}, Light)
    assert light.fixture == fixture
    assert light.spec.ambient
    assert light.lux == 50.0
    assert isinstance(light.lux, float)
    assert light.lumens is None
    assert light.position is None
    assert light.direction == (0.0, 0.0, -1.0)


def test_light_parses_spot_direction_normalized():
    raw = {'name': 'beam', 'fixture': 'spot', 'position': [0.0, 0.0, 2.0], 'lumens': 500, 'direction': [3, 0, -4]}
    light = converter.structure(raw, Light)
    assert light.direction == pytest.approx((0.6, 0.0, -0.8))


def test_light_parses_sun_direction_normalized():
    light = converter.structure({'name': 'sun', 'fixture': 'sun', 'lux': 10000, 'direction': [1, 1, -1]}, Light)
    assert light.direction == pytest.approx((1 / math.sqrt(3), 1 / math.sqrt(3), -1 / math.sqrt(3)))
    assert math.sqrt(sum(c * c for c in light.direction)) == pytest.approx(1.0)


def test_light_accepts_unnormalized_downward_direction_on_panel():
    assert _panel(direction=(0.0, 0.0, -2.0)).direction == (0.0, 0.0, -1.0)


def test_light_parses_entity_ref():
    light = converter.structure({'name': 'desk_lamp', 'fixture': 'downlight', 'entity_ref': ' desk ', 'lumens': 400}, Light)
    assert light.entity_ref == 'desk'
    assert light.position is None


def test_light_parses_frame_with_offset():
    raw = {'name': 'headlight', 'fixture': 'spot', 'frame': '/jackal/base_link', 'offset': [0.3, 0.0, 0.2], 'direction': [1, 0, 0], 'lumens': 900}
    light = converter.structure(raw, Light)
    assert light.frame == 'jackal/base_link'
    assert light.position is None
    assert light.offset == Position(0.3, 0.0, 0.2)
    assert light.direction == (1.0, 0.0, 0.0)


def test_light_parses_state_fields():
    raw = {'name': 'lamp', 'fixture': 'panel', 'position': [0, 0, 2.4], 'lumens': 3600, 'cct_K': 2700, 'cast_shadows': True, 'lit': False, 'level': 0.4}
    light = converter.structure(raw, Light)
    assert light.cct_K == 2700.0
    assert light.cast_shadows is True
    assert light.lit is False
    assert light.level == pytest.approx(0.4)


def test_light_on_is_stripped():
    assert _panel(light_on='  !blackout ').light_on == '!blackout'


def test_light_evolve_keeps_validated_state():
    import attrs

    light = _panel(fixture='spot', direction=(3.0, 0.0, -4.0), light_on='blackout')
    moved = attrs.evolve(light, name='env/lamp', position=Position(5.0, 5.0, 2.4))
    assert moved.direction == pytest.approx(light.direction)
    assert moved.light_on == 'blackout'
    assert moved.position == Position(5.0, 5.0, 2.4)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'name': ''}, 'must be non-empty'),
        ({'name': 'a:b'}, "contain no ':'"),
        ({'fixture': 'lantern'}, 'unknown fixture'),
        ({'position': None}, 'requires exactly one of position, entity_ref or frame'),
        ({'entity_ref': 'desk'}, 'requires exactly one of position, entity_ref or frame'),
        ({'frame': 'jackal/base_link'}, 'requires exactly one of position, entity_ref or frame'),
        ({'position': None, 'entity_ref': 'desk', 'frame': 'jackal/base_link'}, 'requires exactly one of position, entity_ref or frame'),
        ({'lumens': None}, 'requires lumens >= 0'),
        ({'lumens': -1.0}, 'requires lumens >= 0'),
        ({'lux': 50.0}, 'takes lumens, not lux'),
        ({'fixtures': [Position(0.0, 0.0, 2.0)]}, 'fixture positions only as a rig'),
        ({'fixture_ranks': [0.5]}, 'fixture positions only as a rig'),
        ({'fixture': 'bulb', 'direction': (1.0, 0.0, 0.0)}, 'takes no direction'),
        ({'cone_deg': 60.0}, 'cone_deg only as a spot'),
        ({'fixture': 'spot', 'cone_deg': 0.0}, 'cone_deg only as a spot'),
        ({'fixture': 'spot', 'cone_deg': 180.0}, 'cone_deg only as a spot'),
        ({'fixture': 'spot', 'direction': (0.0, 0.0, 0.0)}, 'direction must be non-zero'),
        ({'cct_K': 0.0}, 'cct_K must be positive'),
        ({'cct_K': -2700.0}, 'cct_K must be positive'),
        ({'level': 1.5}, 'level must be within 0..1'),
        ({'level': -0.1}, 'level must be within 0..1'),
        ({'dead_fraction': 1.5}, 'dead_fraction must be within 0..1'),
        ({'dead_fraction': -0.1}, 'dead_fraction must be within 0..1'),
        ({'dead_fraction': 0.2}, 'dead_fraction only as a rig'),
        ({'lit': True, 'light_on': 'blackout'}, 'either lit or light_on'),
        ({'lit': False, 'light_on': '!blackout'}, 'either lit or light_on'),
        ({'light_on': '!'}, "regime name after '!'"),
        ({'light_on': ' ! '}, "regime name after '!'"),
    ],
)
def test_local_light_rejects_invalid(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _panel(**kwargs)


@pytest.mark.parametrize('fixture', ['dome', 'sun'])
@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'position': Position(0.0, 0.0, 3.0)}, 'takes no position, entity_ref or frame'),
        ({'entity_ref': 'desk'}, 'takes no position, entity_ref or frame'),
        ({'frame': 'jackal/base_link'}, 'takes no position, entity_ref or frame'),
        ({'lux': None}, 'requires lux >= 0'),
        ({'lux': -1.0}, 'requires lux >= 0'),
        ({'lumens': 3600.0}, 'takes lux, not lumens'),
        ({'rig': True}, 'cannot be a rig'),
        ({'dead_fraction': 0.5}, 'dead_fraction only as a rig'),
        ({'fixtures': [Position(0.0, 0.0, 2.0)]}, 'fixture positions only as a rig'),
    ],
)
def test_ambient_light_rejects_invalid(fixture, kwargs, message):
    fields = {'name': 'ambient', 'fixture': fixture, 'lux': 50.0}
    fields.update(kwargs)
    with pytest.raises(ValueError, match=message):
        Light(**fields)


def test_dome_rejects_direction():
    with pytest.raises(ValueError, match='takes no direction'):
        Light(name='ambient', fixture='dome', lux=50.0, direction=(1.0, 0.0, -1.0))


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'position': Position(0.0, 0.0, 2.0)}, 'takes no position, entity_ref or frame'),
        ({'entity_ref': 'desk'}, 'takes no position, entity_ref or frame'),
        ({'frame': 'jackal/base_link'}, 'takes no position, entity_ref or frame'),
        ({'fixtures': [], 'fixture_ranks': []}, 'at least one fixture position'),
        ({'fixture_ranks': [0.25]}, 'one fixture rank per fixture position'),
        ({'fixture_ranks': [0.25, 0.5, 0.75]}, 'one fixture rank per fixture position'),
        ({'lumens': None}, 'requires lumens >= 0'),
        ({'lumens': -5.0}, 'requires lumens >= 0'),
        ({'lux': 50.0}, 'takes lumens, not lux'),
        ({'dead_fraction': 1.01}, 'dead_fraction must be within 0..1'),
    ],
)
def test_rig_light_rejects_invalid(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _rig(**kwargs)


def test_rig_light_accepts_dead_fraction():
    assert _rig(dead_fraction=0.3).dead_fraction == pytest.approx(0.3)


def test_light_validation_error_names_the_light():
    with pytest.raises(ValueError, match="'exit_strip'"):
        _panel(name='exit_strip', lumens=None)


@pytest.mark.parametrize(
    'raw',
    [
        {'name': 'lamp', 'fixture': 'panel', 'lumens': 3600},
        {'name': 'lamp', 'fixture': 'panel', 'position': [0, 0, 2.4]},
        {'name': 'lamp', 'fixture': 'lantern', 'position': [0, 0, 2.4], 'lumens': 3600},
        {'name': 'lamp', 'fixture': 'panel', 'position': [0, 0, 2.4], 'lumens': 3600, 'lit': True, 'light_on': 'blackout'},
        {'name': 'ambient', 'fixture': 'dome', 'lumens': 3600},
        {'name': 'ambient', 'fixture': 'dome', 'lux': 50, 'position': [0, 0, 3]},
        {'fixture': 'panel', 'position': [0, 0, 2.4], 'lumens': 3600},
    ],
)
def test_light_structure_rejects_invalid(raw):
    with pytest.raises(Exception):
        converter.structure(raw, Light)


def test_light_semantics_plain_light_is_lit_at_full_level():
    assert _panel().semantics == [
        SemanticCfg(role='predicate', name='lit', value=True),
        SemanticCfg(role='state', name='level', value=1.0),
    ]


def test_light_semantics_light_on_leaves_lit_to_the_regime():
    assert _panel(light_on='!blackout', level=0.5).semantics == [
        SemanticCfg(role='predicate', name='lit', value=None, params={'light_on': '!blackout'}),
        SemanticCfg(role='state', name='level', value=0.5),
    ]


def test_light_semantics_lit_false():
    lit, level = _panel(lit=False).semantics
    assert lit == SemanticCfg(role='predicate', name='lit', value=False)
    assert lit.params == {}
    assert level == SemanticCfg(role='state', name='level', value=1.0)


def test_light_semantics_ambient_has_no_dead_fraction():
    light = Light(name='ambient', fixture='dome', lux=50.0)
    assert [(cfg.role, cfg.name) for cfg in light.semantics] == [('predicate', 'lit'), ('state', 'level')]


def test_light_semantics_rig_adds_dead_fraction():
    assert _rig(dead_fraction=0.25, light_on='!blackout').semantics == [
        SemanticCfg(role='predicate', name='lit', value=None, params={'light_on': '!blackout'}),
        SemanticCfg(role='state', name='level', value=1.0),
        SemanticCfg(role='state', name='dead_fraction', value=0.25),
    ]


def test_light_semantics_rig_matches_light_preset_vocabulary():
    preset = parse_semantics([{'preset': 'light'}])
    assert [(cfg.role, cfg.name) for cfg in _rig().semantics] == [(cfg.role, cfg.name) for cfg in preset]


def test_light_alive_compares_rank_against_dead_fraction():
    rig = _rig(
        fixtures=[Position(float(i), 0.0, 2.48) for i in range(4)],
        fixture_ranks=[0.1, 0.3, 0.6, 0.9],
    )
    assert rig.alive(0.0) == [True, True, True, True]
    assert rig.alive(0.2) == [False, True, True, True]
    assert rig.alive(0.3) == [False, True, True, True]
    assert rig.alive(0.5) == [False, False, True, True]
    assert rig.alive(1.0) == [False, False, False, False]


def test_light_alive_empty_for_explicit_light():
    assert _panel().alive(0.5) == []


def test_cct_to_rgb_2700_is_warm():
    red, green, blue = cct_to_rgb(2700)
    assert red == 1.0
    assert green == pytest.approx(0.654, abs=0.01)
    assert blue == pytest.approx(0.343, abs=0.01)
    assert blue < green - 0.2


def test_cct_to_rgb_4000_is_neutral_warm():
    red, green, blue = cct_to_rgb(4000)
    assert red == 1.0
    assert green == pytest.approx(0.807, abs=0.01)
    assert blue == pytest.approx(0.651, abs=0.01)
    assert red > green > blue


def test_cct_to_rgb_6500_is_near_white():
    red, green, blue = cct_to_rgb(6500)
    assert red > 0.95
    assert green > 0.95
    assert blue > 0.95


def test_cct_to_rgb_cool_side_drops_red():
    red, green, blue = cct_to_rgb(10000)
    assert blue == 1.0
    assert red < green < blue


@pytest.mark.parametrize('cct_K', [500, 1000, 1900, 2700, 4000, 6500, 6600, 6700, 10000, 40000, 100000])
def test_cct_to_rgb_channels_stay_in_unit_range(cct_K):
    assert all(0.0 <= channel <= 1.0 for channel in cct_to_rgb(cct_K))


def test_ceiling_lights_defaults():
    cfg = CeilingLights()
    assert cfg.fixture == 'panel'
    assert cfg.spacing == 2.4
    assert cfg.lumens == 3600.0
    assert cfg.cct_K == 6500.0
    assert cfg.cast_shadows is False
    assert cfg.light_on == ''
    assert cfg.lit is None
    assert cfg.level == 1.0
    assert cfg.dead_fraction == 0.0


def test_ceiling_lights_structure_from_mapping():
    raw = {'fixture': 'tube', 'spacing': 3, 'lumens': 2000, 'cct_K': 3000, 'cast_shadows': True, 'light_on': '!blackout', 'level': 0.8, 'dead_fraction': 0.1}
    cfg = converter.structure(raw, CeilingLights)
    assert cfg == CeilingLights(fixture='tube', spacing=3.0, lumens=2000.0, cct_K=3000.0, cast_shadows=True, light_on='!blackout', level=0.8, dead_fraction=0.1)
    assert isinstance(cfg.spacing, float)
    assert isinstance(cfg.lumens, float)


def test_ceiling_lights_structure_empty_mapping_uses_defaults():
    assert converter.structure({}, CeilingLights) == CeilingLights()


def test_ceiling_lights_structure_rejects_unknown_key():
    with pytest.raises(Exception):
        converter.structure({'fixture': 'panel', 'spaceing': 2.0}, CeilingLights)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'fixture': 'dome'}, 'ceiling_lights fixture'),
        ({'fixture': 'sun'}, 'ceiling_lights fixture'),
        ({'fixture': 'lantern'}, 'ceiling_lights fixture'),
        ({'spacing': 0.0}, 'spacing must be positive'),
        ({'spacing': -1.0}, 'spacing must be positive'),
        ({'lumens': -1.0}, 'requires lumens >= 0'),
        ({'cct_K': 0.0}, 'cct_K must be positive'),
        ({'level': 2.0}, 'level must be within 0..1'),
        ({'level': -0.5}, 'level must be within 0..1'),
        ({'dead_fraction': 1.2}, 'dead_fraction must be within 0..1'),
        ({'dead_fraction': -0.1}, 'dead_fraction must be within 0..1'),
        ({'lit': True, 'light_on': 'blackout'}, 'either lit or light_on'),
        ({'light_on': '!'}, "regime name after '!'"),
    ],
)
def test_ceiling_lights_rejects_invalid(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CeilingLights(**kwargs)


def test_ceiling_lights_serialize_omits_defaults():
    assert CeilingLights().serialize() == {}
    assert CeilingLights(spacing=3.0, light_on='!blackout').serialize() == {'spacing': 3.0, 'light_on': '!blackout'}


def test_ceiling_lights_round_trip():
    cfg = CeilingLights(fixture='downlight', spacing=1.2, lumens=800.0, cct_K=3000.0, lit=False, level=0.5, dead_fraction=0.2)
    assert converter.structure(yaml.safe_load(yaml.safe_dump(converter.unstructure(cfg))), CeilingLights) == cfg


def test_explicit_light_serializes_without_derived_fields():
    raw = {'name': 'exit_strip', 'position': {'x': 10, 'y': 34, 'z': 2.2}, 'fixture': 'tube', 'lumens': 300, 'light_on': 'blackout'}
    unstructured = converter.unstructure(converter.structure(raw, Light))
    assert 'rig' not in unstructured
    assert 'fixtures' not in unstructured
    assert 'fixture_ranks' not in unstructured
    assert set(unstructured) == {'name', 'fixture', 'position', 'lumens', 'light_on'}
    assert unstructured['name'] == 'exit_strip'
    assert unstructured['fixture'] == 'tube'
    assert unstructured['lumens'] == 300.0
    assert unstructured['light_on'] == 'blackout'


def test_explicit_light_round_trips_through_yaml():
    raw = {
        'name': 'beam',
        'fixture': 'spot',
        'position': [1.0, 2.0, 2.4],
        'direction': [3, 0, -4],
        'lumens': 500,
        'cct_K': 3000,
        'cast_shadows': True,
        'lit': False,
        'level': 0.6,
    }
    light = converter.structure(raw, Light)
    reparsed = converter.structure(yaml.safe_load(yaml.safe_dump(converter.unstructure(light))), Light)
    assert reparsed.name == 'beam'
    assert reparsed.fixture == 'spot'
    assert reparsed.position == light.position
    assert reparsed.direction == pytest.approx(light.direction)
    assert reparsed.lumens == 500.0
    assert reparsed.cct_K == 3000.0
    assert reparsed.cast_shadows is True
    assert reparsed.lit is False
    assert reparsed.level == pytest.approx(0.6)
    assert reparsed.rig is False
    assert reparsed.semantics == light.semantics


def test_ambient_light_round_trips_through_yaml():
    light = converter.structure({'name': 'ambient', 'fixture': 'dome', 'lux': 50}, Light)
    unstructured = converter.unstructure(light)
    assert set(unstructured) == {'name', 'fixture', 'lux'}
    reparsed = converter.structure(yaml.safe_load(yaml.safe_dump(unstructured)), Light)
    assert reparsed.fixture == 'dome'
    assert reparsed.lux == 50.0
    assert reparsed.lumens is None


def test_attached_light_round_trips_through_yaml():
    raw = {'name': 'headlight', 'fixture': 'spot', 'frame': 'jackal/base_link', 'offset': [0.3, 0.0, 0.2], 'direction': [1, 0, 0], 'lumens': 900}
    light = converter.structure(raw, Light)
    reparsed = converter.structure(yaml.safe_load(yaml.safe_dump(converter.unstructure(light))), Light)
    assert reparsed.frame == 'jackal/base_link'
    assert reparsed.position is None
    assert reparsed.offset == Position(0.3, 0.0, 0.2)
    assert reparsed.direction == (1.0, 0.0, 0.0)


@pytest.mark.parametrize('fixture', ['panel', 'tube', 'downlight'])
def test_area_fixtures_take_a_direction(fixture):
    assert _panel(fixture=fixture, direction=(0.0, 3.0, -4.0)).direction == pytest.approx((0.0, 0.6, -0.8))


def test_spot_cone_override_reaches_the_spec():
    spot = _panel(fixture='spot', cone_deg=70.0)
    assert spot.spec.cone_deg == 70.0
    assert spot.spec.radius == LIGHT_FIXTURES['spot'].radius
    assert _panel(fixture='spot').spec.cone_deg == 40.0


def test_bulb_sits_at_its_position():
    bulb = _panel(fixture='bulb', lumens=800.0)
    assert bulb.spec.shape == 'sphere'
    assert bulb.direction == (0.0, 0.0, -1.0)


def test_object_light_is_named_after_its_entity_and_anchored_on_it():
    light = object_light('desk_lamp_3', {'name': 'bulb', 'fixture': 'bulb', 'offset': [0.1, 0.0, 0.45], 'lumens': 800.0, 'cct_K': 2700.0}, None)
    assert light.name == 'desk_lamp_3_bulb'
    assert light.entity_ref == 'desk_lamp_3'
    assert (light.offset.x, light.offset.y, light.offset.z) == (0.1, 0.0, 0.45)
    assert light.intrinsic
    assert light.cct_K == 2700.0
    assert light.lit is None
    assert light.light_on == ''
    assert light.level == 1.0


def test_object_light_defaults_to_a_bulb_named_light():
    light = object_light('lamp', {'lumens': 400.0}, None)
    assert (light.name, light.fixture, light.cct_K) == ('lamp_light', 'bulb', 6500.0)
    assert (light.offset.x, light.offset.y, light.offset.z) == (0.0, 0.0, 0.0)


def test_object_light_carries_its_glow_material():
    assert object_light('lamp', {'lumens': 400.0, 'glow': 'Shade'}, None).glow == 'Shade'
    assert object_light('lamp', {'lumens': 400.0, 'glow': None}, None).glow == ''
    assert object_light('lamp', {'lumens': 400.0}, None).glow == ''


def test_object_light_drops_a_direction_on_a_bulb_and_keeps_it_on_a_spot():
    bulb = object_light('lamp', {'fixture': 'bulb', 'lumens': 400.0, 'direction': [1.0, 0.0, 0.0]}, None)
    spot = object_light('lamp', {'fixture': 'spot', 'lumens': 400.0, 'direction': [1.0, 0.0, 0.0], 'cone_deg': 70.0}, None)
    assert bulb.direction == (0.0, 0.0, -1.0)
    assert spot.direction == (1.0, 0.0, 0.0)
    assert spot.spec.cone_deg == 70.0


def test_object_light_takes_per_placement_settings():
    settings = ObjectLightSettings(light_on='!blackout', level=0.4)
    light = object_light('lamp', {'lumens': 400.0}, settings)
    assert (light.light_on, light.level, light.lit) == ('!blackout', 0.4, None)
    assert object_light('lamp', {'lumens': 400.0}, ObjectLightSettings(lit=False)).semantics[0].value is False


@pytest.mark.parametrize(
    ('entry', 'message'),
    [
        ({'lumens': 400.0, 'colour': 'red'}, 'unknown keys'),
        ({'fixture': 'dome', 'lumens': 400.0}, 'has fixture'),
        ({'fixture': 'lantern', 'lumens': 400.0}, 'has fixture'),
        ({}, 'requires lumens'),
    ],
)
def test_object_light_rejects_invalid_entries(entry, message):
    with pytest.raises(ValueError, match=message):
        object_light('lamp', entry, None)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'level': 1.5}, 'level must be within 0..1'),
        ({'lit': True, 'light_on': 'blackout'}, 'either lit or light_on'),
        ({'light_on': '!'}, "regime name after '!'"),
    ],
)
def test_object_light_settings_reject_invalid(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ObjectLightSettings(**kwargs)


def test_obstacle_structures_its_light_settings():
    raw = {'name': 'desk_lamp', 'model': 'Common/Desk_Lamp', 'pose': {'position': [1.0, 2.0, 0.0]}, 'light': {'light_on': 'evening', 'level': 0.5}}
    obstacle = converter.structure(raw, Obstacle)
    assert obstacle.light == ObjectLightSettings(light_on='evening', level=0.5)
    assert converter.structure({key: value for key, value in raw.items() if key != 'light'}, Obstacle).light is None
