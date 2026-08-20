from __future__ import annotations

import random

import pytest
from arena_simulation_setup.utils.generative import (
    BaseConfiguration,
    WorldGenerator,
    WorldGeneratorType,
)
from arena_simulation_setup.utils.generative.barn import WorldGeneratorBarn
from arena_simulation_setup.utils.generative.empty import WorldGeneratorEmpty
from arena_simulation_setup.utils.generative.hallway import WorldGeneratorHallway
from arena_simulation_setup.utils.generative.utils import line_pairs, to_corners, to_walls
from shapely import LineString, MultiLineString, Polygon

# ---------------------------------------------------------------------------
# WorldGeneratorType enum
# ---------------------------------------------------------------------------


def test_world_generator_type_empty_value():
    assert WorldGeneratorType.EMPTY.value == "empty"


def test_world_generator_type_hallway_value():
    assert WorldGeneratorType.HALLWAY.value == "hallway"


def test_world_generator_type_barn_value():
    assert WorldGeneratorType.BARN.value == "barn"


def test_barn_registered():
    assert WorldGeneratorType.BARN in WorldGenerator.available()
    assert WorldGenerator.config_model(WorldGeneratorType.BARN) is WorldGeneratorBarn.Configuration


# ---------------------------------------------------------------------------
# BaseConfiguration defaults
# ---------------------------------------------------------------------------


def test_extent_belongs_to_the_generators_that_have_one():
    """A sketch or a text sizes itself from what is drawn, so the base carries no width or height."""
    assert 'width' not in BaseConfiguration.model_fields
    assert WorldGeneratorEmpty.Configuration(width=20.0, height=30.0).width == pytest.approx(20.0)


def test_base_configuration_custom_resolution():
    cfg = BaseConfiguration(resolution=0.1)
    assert cfg.resolution == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# WorldGeneratorEmpty.compute
# ---------------------------------------------------------------------------


def test_world_generator_empty_compute_single_zone():
    gen = WorldGeneratorEmpty({"width": 10.0, "height": 8.0}, random.Random(0))
    wd = gen.compute()
    assert len(wd.zones) == 1


def test_world_generator_empty_compute_4_corners():
    gen = WorldGeneratorEmpty({"width": 10.0, "height": 8.0}, random.Random(0))
    wd = gen.compute()
    corners = wd.zones[0].corners
    # shapely Polygon.exterior.coords includes closing point = 5 for rectangle
    assert len(corners) >= 4


def test_world_generator_empty_compute_has_walls():
    gen = WorldGeneratorEmpty({"width": 10.0, "height": 8.0}, random.Random(0))
    wd = gen.compute()
    walls = wd.zones[0].walls
    assert len(walls) > 0


# ---------------------------------------------------------------------------
# WorldGeneratorHallway.Configuration
# ---------------------------------------------------------------------------


def test_hallway_config_hallway_top():
    cfg = WorldGeneratorHallway.Configuration()
    assert cfg.hallway_top == pytest.approx(cfg.height / 2 + cfg.hallway_height / 2)


def test_hallway_config_hallway_bottom():
    cfg = WorldGeneratorHallway.Configuration()
    assert cfg.hallway_bottom == pytest.approx(cfg.height / 2 - cfg.hallway_height / 2)


def test_hallway_config_hallway_top_custom():
    cfg = WorldGeneratorHallway.Configuration(height=100.0, hallway_height=10.0)
    assert cfg.hallway_top == pytest.approx(55.0)
    assert cfg.hallway_bottom == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# WorldGeneratorHallway.compute (seeded random for determinism)
# ---------------------------------------------------------------------------


def test_hallway_compute_room_count_per_side():
    gen = WorldGeneratorHallway(
        {
            "rooms_per_side": 3,
            "width": 40.0,
            "height": 30.0,
        },
        random.Random(42),
    )
    wd = gen.compute()
    # 2 sides, each with (1 hallway + rooms_per_side) zones
    assert len(wd.zones) == 2 * (1 + 3)


def test_hallway_compute_doors_per_room():
    gen = WorldGeneratorHallway(
        {
            "rooms_per_side": 2,
            "width": 40.0,
            "height": 30.0,
        },
        random.Random(0),
    )
    wd = gen.compute()
    door_zones = [z for z in wd.zones if z.doors]
    assert len(door_zones) > 0


# ---------------------------------------------------------------------------
# utils.line_pairs
# ---------------------------------------------------------------------------


def test_line_pairs_polygon_exterior():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    pairs = list(line_pairs(poly))
    assert len(pairs) == 4  # 4 edges in square


def test_line_pairs_polygon_with_interior():
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    inner = [(3, 3), (7, 3), (7, 7), (3, 7)]
    poly = Polygon(outer, [inner])
    pairs = list(line_pairs(poly))
    # exterior: 4 + interior: 4 = 8
    assert len(pairs) == 8


def test_line_pairs_linestring():
    ls = LineString([(0, 0), (1, 0), (2, 0)])
    pairs = list(line_pairs(ls))
    assert len(pairs) == 2


def test_line_pairs_multilinestring():
    mls = MultiLineString([[(0, 0), (1, 0)], [(2, 0), (3, 0)]])
    pairs = list(line_pairs(mls))
    assert len(pairs) == 2


# ---------------------------------------------------------------------------
# utils.to_corners
# ---------------------------------------------------------------------------


def test_to_corners_square():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    corners = to_corners(poly)
    # Polygon closes itself, so exterior.coords has 5 points
    assert len(corners) == 5


def test_to_corners_empty():
    poly = Polygon()
    corners = to_corners(poly)
    assert corners == []


# ---------------------------------------------------------------------------
# utils.to_walls
# ---------------------------------------------------------------------------


def test_to_walls_polygon():
    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    walls = to_walls(poly)
    assert len(walls) == 4


def test_to_walls_linestring():
    ls = LineString([(0, 0), (1, 0), (2, 0)])
    walls = to_walls(ls)
    assert len(walls) == 2


def test_to_walls_multilinestring():
    mls = MultiLineString([[(0, 0), (1, 0)], [(2, 0), (4, 0)]])
    walls = to_walls(mls)
    assert len(walls) == 2


# ---------------------------------------------------------------------------
# WorldGeneratorBarn
# ---------------------------------------------------------------------------


def test_barn_config_pitch_derived():
    cfg = WorldGeneratorBarn.Configuration()
    # one box row separates parallel lane strands
    assert cfg.pitch == pytest.approx(cfg.passage_width + cfg.box_size + cfg.box_gap)


def test_barn_compute_single_zone_with_walls():
    gen = WorldGeneratorBarn({}, random.Random(0))
    wd = gen.compute()
    assert len(wd.zones) == 1
    assert wd.zones[0].name == "barn"
    assert len(wd.zones[0].walls) > 0


def test_barn_too_small_raises():
    with pytest.raises(ValueError):
        WorldGeneratorBarn({"width": 1.0, "height": 1.0}, random.Random(0)).compute()


def test_barn_files_emit_scenario():
    import yaml

    gen = WorldGeneratorBarn({}, random.Random(0))
    files = gen.files()
    key = f"scenarios/{WorldGeneratorBarn.SCENARIO}/scenario.yaml"
    assert key in files

    scenario = yaml.safe_load(files[key])
    robots = scenario["robots"]
    assert len(robots) == 1
    assert len(robots[0]["start"]) == 3  # [x, y, yaw]
    goto = robots[0]["phases"][0]["goto"]
    assert len(goto) == 3


def test_barn_seed_deterministic():
    a = WorldGeneratorBarn({}, random.Random(42))
    b = WorldGeneratorBarn({}, random.Random(42))
    assert len(a.compute().zones[0].walls) == len(b.compute().zones[0].walls)
    assert a.files() == b.files()


def test_barn_episode_binding():
    params = WorldGeneratorBarn({}, random.Random(0)).params()
    assert params["tm_robots"] == "scenario"
    assert params["robots_params"] == {"file": WorldGeneratorBarn.SCENARIO}


def test_default_episode_binding_empty():
    assert WorldGeneratorEmpty({}, random.Random(0)).params() == {}
