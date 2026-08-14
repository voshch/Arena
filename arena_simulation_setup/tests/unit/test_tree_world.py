from __future__ import annotations

import pytest

from arena_simulation_setup.shared.world import Door
from arena_simulation_setup.shared.walls import Wall
from arena_simulation_setup.tree.World.World import LevelDescription as WorldDescription, WorldMicrophone
from arena_simulation_setup.tree.World.World import Schedule, Signal
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import Position


def _make_zone(name: str = "zone", corners=None, walls=None, doors=None, elevators=None) -> WorldDescription.Zone:
    return WorldDescription.Zone(
        name=name,
        corners=corners or [],
        walls=walls or [],
        doors=doors or [],
        elevators=elevators or [],
    )


def test_zone_floor_rectangle():
    zone = _make_zone(
        name="room",
        corners=[
            Position(0.0, 0.0), Position(4.0, 0.0),
            Position(4.0, 3.0), Position(0.0, 3.0),
        ],
    )
    floor = zone.floor
    assert floor.x_length == pytest.approx(4.0)
    assert floor.y_length == pytest.approx(3.0)
    assert floor.pos.x == pytest.approx(2.0)
    assert floor.pos.y == pytest.approx(1.5)


def test_zone_floor_triangular():
    zone = _make_zone(
        name="tri",
        corners=[
            Position(0.0, 0.0), Position(2.0, 0.0), Position(1.0, 2.0),
        ],
    )
    floor = zone.floor
    assert floor.x_length == pytest.approx(2.0)
    assert floor.y_length == pytest.approx(2.0)


def test_zone_floor_empty_corners_raises():
    zone = _make_zone(name="empty")
    with pytest.raises((ValueError, Exception)):
        _ = zone.floor


def test_all_walls_empty():
    wd = WorldDescription(zones=[])
    assert list(wd.all_walls) == []


def test_all_walls_populated():
    w = Wall(start=Position(0, 0), end=Position(1, 0))
    zone = _make_zone(walls=[w])
    wd = WorldDescription(zones=[zone])
    assert list(wd.all_walls) == [w]


def test_all_walls_multiple_zones():
    w1 = Wall(start=Position(0, 0), end=Position(1, 0))
    w2 = Wall(start=Position(2, 0), end=Position(3, 0))
    z1 = _make_zone("z1", walls=[w1])
    z2 = _make_zone("z2", walls=[w2])
    wd = WorldDescription(zones=[z1, z2])
    assert len(list(wd.all_walls)) == 2


# ---------------------------------------------------------------------------
# Zone semantics
# ---------------------------------------------------------------------------


def test_zone_semantics_default_emitted_as_empty_list():
    zone = _make_zone()
    unstructured = converter.unstructure(zone)
    assert unstructured['semantics'] == []


def test_zone_semantics_primitives_round_trip():
    raw = {
        'name': 'lobby',
        'corners': [],
        'semantics': [
            {'state': 'max_speed', 'value': 1.5},
            {'predicate': 'quiet', 'value': True},
            {'predicate': 'restricted', 'value': False},
        ],
    }
    zone = converter.structure(raw, WorldDescription.Zone)
    assert [c.name for c in zone.semantics] == ['max_speed', 'quiet', 'restricted']
    assert [c.value for c in zone.semantics] == [1.5, True, False]
    unstructured = converter.unstructure(zone)
    zone2 = converter.structure(unstructured, WorldDescription.Zone)
    assert zone2.semantics == zone.semantics


def test_zone_semantics_author_defined_name_accepted():
    zone = converter.structure({'name': 'z', 'semantics': [{'predicate': 'custom_flag'}]}, WorldDescription.Zone)
    assert zone.semantics[0].name == 'custom_flag'


def test_all_doors_empty():
    wd = WorldDescription(zones=[])
    assert list(wd.all_doors) == []


def test_all_doors_populated():
    d = Door(name="d", start=Position(0, 0), end=Position(1, 0))
    zone = _make_zone(doors=[d])
    wd = WorldDescription(zones=[zone])
    assert list(wd.all_doors) == [d]


def test_all_elevators_empty():
    wd = WorldDescription(zones=[])
    assert list(wd.all_elevators) == []


# ---------------------------------------------------------------------------
# Schedule / Signal (M2 standalone semantic entities)
# ---------------------------------------------------------------------------


def test_schedule_round_trip():
    raw = {
        'name': 'fire_alarm',
        'semantics': [{'preset': 'schedule', 'params': {'windows': [], 'regime': 'alarm'}}],
    }
    schedule = converter.structure(raw, Schedule)
    assert schedule.name == 'fire_alarm'
    assert [(c.role, c.name) for c in schedule.semantics] == [
        ('state', 'state'),
        ('predicate', 'active'),
        ('state', 'window_remaining'),
    ]
    unstructured = converter.unstructure(schedule)
    reparsed = converter.structure(unstructured, Schedule)
    assert reparsed.name == schedule.name
    assert reparsed.semantics == schedule.semantics


def test_signal_round_trip():
    raw = {
        'name': 'crosswalk_light',
        'semantics': [{'preset': 'signal', 'params': {'phases': [{'name': 'go', 'duration': 5.0}]}}],
    }
    signal = converter.structure(raw, Signal)
    assert signal.name == 'crosswalk_light'
    assert [(c.role, c.name) for c in signal.semantics] == [
        ('state', 'state'),
        ('state', 'phase_remaining'),
        ('predicate', 'stop'),
    ]
    unstructured = converter.unstructure(signal)
    reparsed = converter.structure(unstructured, Signal)
    assert reparsed.name == signal.name
    assert reparsed.semantics == signal.semantics


def test_schedule_default_semantics_empty():
    schedule = converter.structure({'name': 'bare'}, Schedule)
    assert schedule.semantics == []


def test_zone_schedules_and_signals_default_empty():
    zone = _make_zone()
    assert zone.schedules == []
    assert zone.signals == []


def test_all_schedules_populated():
    schedule = Schedule(name='fire_alarm')
    zone = _make_zone()
    zone.schedules = [schedule]
    wd = WorldDescription(zones=[zone])
    assert list(wd.all_schedules) == [schedule]


def test_all_signals_populated():
    signal = Signal(name='crosswalk_light')
    zone = _make_zone()
    zone.signals = [signal]
    wd = WorldDescription(zones=[zone])
    assert list(wd.all_signals) == [signal]


def test_all_floors_count():
    z1 = _make_zone("z1", corners=[Position(0, 0), Position(2, 0), Position(2, 2), Position(0, 2)])
    z2 = _make_zone("z2", corners=[Position(0, 0), Position(4, 0), Position(4, 4), Position(0, 4)])
    wd = WorldDescription(zones=[z1, z2])
    assert len(list(wd.all_floors)) == 2


def test_all_static_entities_empty():
    wd = WorldDescription(zones=[_make_zone()])
    assert list(wd.all_static_entities) == []


def test_all_dynamic_entities_empty():
    wd = WorldDescription(zones=[_make_zone()])
    assert list(wd.all_dynamic_entities) == []


def _make_square_zone(name: str = "room", size: float = 10.0) -> WorldDescription.Zone:
    return _make_zone(
        name=name,
        corners=[
            Position(0.0, 0.0), Position(size, 0.0),
            Position(size, size), Position(0.0, size),
        ],
        walls=[
            Wall(start=Position(0.0, 0.0), end=Position(size, 0.0)),
        ],
    )


def test_render_returns_bytes_and_origin_tuple():
    wd = WorldDescription(zones=[_make_square_zone()])
    result = wd.render()
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_render_bytes_is_valid_png():
    import io
    import PIL.Image
    wd = WorldDescription(zones=[_make_square_zone()])
    png_bytes, _ = wd.render()
    assert isinstance(png_bytes, bytes)
    img = PIL.Image.open(io.BytesIO(png_bytes))
    assert img.format == 'PNG'


def test_render_image_mode_is_rgb():
    import io
    import PIL.Image
    wd = WorldDescription(zones=[_make_square_zone()])
    png_bytes, _ = wd.render()
    img = PIL.Image.open(io.BytesIO(png_bytes))
    assert img.mode == 'RGB'


def test_render_image_size_reflects_zone_aabb():
    import io
    import math
    import PIL.Image
    size = 10.0
    resolution = 0.05
    padding = 5
    wd = WorldDescription(zones=[_make_square_zone(size=size)])
    png_bytes, _ = wd.render(resolution=resolution)
    img = PIL.Image.open(io.BytesIO(png_bytes))
    expected_px = math.ceil(size / resolution) + 2 * padding
    assert img.width == expected_px
    assert img.height == expected_px


def test_render_origin_reflects_padding():
    size = 10.0
    resolution = 0.05
    padding = 5
    wd = WorldDescription(zones=[_make_square_zone(size=size)])
    _, origin = wd.render(resolution=resolution)
    expected_origin_x = 0.0 - padding * resolution
    expected_origin_y = 0.0 - padding * resolution
    assert origin[0] == pytest.approx(expected_origin_x)
    assert origin[1] == pytest.approx(expected_origin_y)


def _square_corners(size: float = 2.0) -> list[Position]:
    return [Position(0.0, 0.0), Position(size, 0.0), Position(size, size), Position(0.0, size)]


def test_all_floors_skips_empty_material():
    kept = WorldDescription.Zone(name="kept", corners=_square_corners())
    dropped = WorldDescription.Zone(name="dropped", corners=_square_corners(), material='')
    wd = WorldDescription(zones=[kept, dropped])
    assert [f.name for f in wd.all_floors] == ["kept"]


def test_all_ceilings_skips_empty_material():
    import asyncio
    kept = WorldDescription.Zone(name="kept", corners=_square_corners(), ceiling_height=2.5)
    dropped = WorldDescription.Zone(name="dropped", corners=_square_corners(), ceiling_height=2.5, ceiling_material='')
    wd = WorldDescription(zones=[kept, dropped])
    ceilings = asyncio.run(wd.all_ceilings())
    assert [c.name for c in ceilings] == ["kept"]


def test_all_walls_skips_empty_material():
    from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
    default_wall = Wall(start=Position(0, 0), end=Position(1, 0))
    named_wall = Wall(start=Position(1, 0), end=Position(2, 0), material=MaterialIdentifier('Marble'))
    dropped_wall = Wall(start=Position(2, 0), end=Position(3, 0), material=MaterialIdentifier(''))
    zone = _make_zone(walls=[default_wall, named_wall, dropped_wall])
    wd = WorldDescription(zones=[zone])
    assert list(wd.all_walls) == [default_wall, named_wall]


def test_zone_mat_key_aliases():
    from arena_simulation_setup.utils.cattrs import converter
    zone = converter.structure({'name': 'z', 'mat': '', 'ceiling_mat': '', 'wall_mat': ''}, WorldDescription.Zone)
    assert zone.material.name == ''
    assert zone.ceiling_material.name == ''
    assert zone.wall_material.name == ''


def test_zone_canonical_key_wins_over_alias():
    from arena_simulation_setup.utils.cattrs import converter
    zone = converter.structure({'name': 'z', 'material': 'Porcelain_Tile_4', 'mat': ''}, WorldDescription.Zone)
    assert zone.material.name == 'Porcelain_Tile_4'


def test_zone_material_defaults_nonempty():
    zone = _make_zone()
    assert zone.material.name
    assert zone.ceiling_material.name
    assert zone.wall_material.name


def test_world_microphones_validate_zone_position_and_indices():
    world = WorldDescription(
        zones=[
            WorldDescription.Zone(
                name="reception",
                corners=_square_corners(10.0),
                ceiling_height=2.9,
            )
        ],
        microphones=[
            WorldMicrophone(
                zone="reception",
                placement="ceiling",
                position=Position(2.0, 2.0, 2.9),
                index=1,
            ),
            WorldMicrophone(
                zone="reception",
                placement="ceiling",
                position=Position(4.0, 2.0, 2.9),
                index=2,
            ),
        ],
    )

    world.validate_microphones()
    assert [
        microphone.listener_id for microphone in world.microphones
    ] == [
        "microphone:zone:reception:ceiling:1",
        "microphone:zone:reception:ceiling:2",
    ]


def test_world_microphones_structure_from_world_yaml_shape():
    from arena_simulation_setup.utils.cattrs import converter

    world = converter.structure(
        {
            "zones": [
                {
                    "name": "reception",
                    "corners": [[0, 0], [5, 0], [5, 5], [0, 5]],
                    "ceiling_height": 2.9,
                }
            ],
            "microphones": [
                {
                    "zone": "reception",
                    "placement": "ceiling",
                    "position": [2, 2, 2.9],
                    "index": 1,
                }
            ],
        },
        WorldDescription,
    )

    world.validate_microphones()
    assert world.microphones[0].position == Position(2, 2, 2.9)


@pytest.mark.parametrize(
    ("microphone", "message"),
    [
        (
            WorldMicrophone(
                zone="missing",
                placement="ceiling",
                position=Position(2.0, 2.0, 2.9),
            ),
            "unknown zone",
        ),
        (
            WorldMicrophone(
                zone="reception",
                placement="ceiling",
                position=Position(20.0, 2.0, 2.9),
            ),
            "outside zone",
        ),
        (
            WorldMicrophone(
                zone="reception",
                placement="ceiling",
                position=Position(2.0, 2.0, 2.0),
            ),
            "does not match ceiling height",
        ),
    ],
)
def test_world_microphones_reject_invalid_construction(
    microphone,
    message,
):
    world = WorldDescription(
        zones=[
            WorldDescription.Zone(
                name="reception",
                corners=_square_corners(10.0),
                ceiling_height=2.9,
            )
        ],
        microphones=[microphone],
    )

    with pytest.raises(ValueError, match=message):
        world.validate_microphones()
