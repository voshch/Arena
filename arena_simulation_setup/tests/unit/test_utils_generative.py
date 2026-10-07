from __future__ import annotations

import random

import pytest
from arena_simulation_setup.generative.world import (
    BaseConfiguration,
    WorldGenerator,
    WorldGeneratorType,
)
from arena_simulation_setup.generative.world.barn import WorldGeneratorBarn
from arena_simulation_setup.generative.world.barn_cylinder import WorldGeneratorBarnCylinder
from arena_simulation_setup.generative.world.empty import WorldGeneratorEmpty
from arena_simulation_setup.generative.world.hallway import WorldGeneratorHallway
from arena_simulation_setup.generative.world.utils import line_pairs, to_corners, to_walls
from arena_simulation_setup.generative.world.world_generator import save_generated
from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
from arena_simulation_setup.tree.World import WorldView, describe
from arena_simulation_setup.tree.World.Scenario import ScenarioView
from shapely import GeometryCollection, LineString, MultiLineString, Point, Polygon

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
    world = gen.compute()
    assert len(world.all_levels[0].zones) == 1
    assert describe(world.kg, "empty_zone") == "An empty zone"


def test_world_generator_empty_compute_4_corners():
    gen = WorldGeneratorEmpty({"width": 10.0, "height": 8.0}, random.Random(0))
    polygon = gen.compute().all_levels[0].zones[0].polygon
    assert len(polygon) == 4


def test_world_generator_empty_compute_has_walls():
    gen = WorldGeneratorEmpty({"width": 10.0, "height": 8.0}, random.Random(0))
    level = gen.compute().all_levels[0]
    assert [(len(path.path), path.closed) for path in level.walls] == [(4, True)]
    assert len(level.all_walls) == 4


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
    level = gen.compute().all_levels[0]
    assert len(level.zones) == 2 * (1 + 3)


def test_hallway_compute_doors_per_room():
    gen = WorldGeneratorHallway(
        {
            "rooms_per_side": 2,
            "width": 40.0,
            "height": 30.0,
        },
        random.Random(0),
    )
    world = gen.compute()
    level = world.all_levels[0]
    assert [door.name for door in level.doors] == [f"{side}_room_{index}_door" for side in ("top", "bottom") for index in (1, 2)]
    assert describe(world.kg, "top_room_0") == "hallway"
    assert describe(world.kg, "top_room_1").startswith("top room 1 of size")


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
    assert [(len(path.path), path.closed) for path in walls] == [(4, True)]
    assert len([segment for path in walls for segment in path.segments()]) == 4


def test_to_walls_linestring():
    ls = LineString([(0, 0), (1, 0), (2, 0)])
    walls = to_walls(ls)
    assert [(len(path.path), path.closed) for path in walls] == [(3, False)]
    assert len(walls[0].segments()) == 2


def test_to_walls_multilinestring():
    mls = MultiLineString([[(0, 0), (1, 0)], [(2, 0), (4, 0)]])
    walls = to_walls(mls)
    assert len(walls) == 2


def test_to_walls_skips_points_of_a_collection_and_carries_the_material():
    collection = GeometryCollection([LineString([(0, 0), (1, 0)]), Point(3, 3)])
    walls = to_walls(collection, material=MaterialIdentifier("Cardboard"))
    assert [path.material.name for path in walls] == ["Cardboard"]


def test_generated_world_saves_as_format_2_with_its_scenario(tmp_path):
    gen = WorldGeneratorBarn({}, random.Random(0))
    save_generated(WorldView(tmp_path / "barn"), gen.compute(), gen.files())
    assert (tmp_path / "barn" / "world.yaml").read_text().startswith("format: arena-world/2")
    loaded = WorldView(tmp_path / "barn").load()
    assert [zone.name for zone in loaded.all_levels[0].zones] == ["barn"]
    assert describe(loaded.kg, "barn").startswith("BARN passage")
    scenario = ScenarioView(tmp_path / "barn" / "scenarios" / WorldGeneratorBarn.SCENARIO).load()
    assert len(scenario.robots) == 1


def test_generated_files_stay_inside_the_world(tmp_path):
    gen = WorldGeneratorEmpty({}, random.Random(0))
    with pytest.raises(ValueError, match="outside the world directory"):
        save_generated(WorldView(tmp_path / "empty"), gen.compute(), {"../escape.yaml": b""})


# ---------------------------------------------------------------------------
# WorldGeneratorBarn
# ---------------------------------------------------------------------------


def test_barn_config_pitch_derived():
    cfg = WorldGeneratorBarn.Configuration()
    # one box row separates parallel lane strands
    assert cfg.pitch == pytest.approx(cfg.passage_width + cfg.box_size + cfg.box_gap)


def test_barn_compute_single_zone_with_walls():
    gen = WorldGeneratorBarn({}, random.Random(0))
    level = gen.compute().all_levels[0]
    assert [zone.name for zone in level.zones] == ["barn"]
    assert len(level.walls) > 0
    assert {path.material.name for path in level.walls if path.material is not None} == {"Cardboard"}


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
    assert scenario["format"] == "arena-scenario/2"
    assert len(robots[0]["start"]) == 3
    goto = robots[0]["phases"][0]["goto"]
    assert len(goto) == 3


def test_barn_seed_deterministic():
    a = WorldGeneratorBarn({}, random.Random(42))
    b = WorldGeneratorBarn({}, random.Random(42))
    assert len(a.compute().all_levels[0].walls) == len(b.compute().all_levels[0].walls)
    assert a.files() == b.files()


def test_barn_episode_binding():
    params = WorldGeneratorBarn({}, random.Random(0)).params()
    assert params["tm_robots"] == "scenario"
    assert params["robots_params"] == {"file": WorldGeneratorBarn.SCENARIO}


def test_default_episode_binding_empty():
    assert WorldGeneratorEmpty({}, random.Random(0)).params() == {}


# ---------------------------------------------------------------------------
# WorldGeneratorBarnCylinder
# ---------------------------------------------------------------------------


def test_world_generator_type_barn_cylinder_value():
    assert WorldGeneratorType.BARN_CYLINDER.value == "barn_cylinder"


def test_barn_cylinder_registered():
    assert WorldGeneratorType.BARN_CYLINDER in WorldGenerator.available()
    assert WorldGenerator.config_model(WorldGeneratorType.BARN_CYLINDER) is WorldGeneratorBarnCylinder.Configuration


def test_barn_cylinder_config_defaults():
    cfg = WorldGeneratorBarnCylinder.Configuration()
    assert cfg.cell_size == pytest.approx(0.3)
    assert cfg.cylinder_radius == pytest.approx(0.075)
    assert cfg.robot_clearance == pytest.approx(0.28)
    assert cfg.max_tries == 50


def test_barn_cylinder_compute_single_zone_with_walls():
    gen = WorldGeneratorBarnCylinder({}, random.Random(0))
    level = gen.compute().all_levels[0]
    assert [zone.name for zone in level.zones] == ["barn_cylinder"]
    assert len(level.walls) > 0


def test_barn_cylinder_connectivity_across_seeds():
    # compute() rejects unconnected fields internally, so a clean return proves the
    # reject sampler found a start->goal-connected field within max_tries
    for seed in range(6):
        WorldGeneratorBarnCylinder({}, random.Random(seed)).compute()


def test_barn_cylinder_too_small_raises():
    with pytest.raises(ValueError):
        WorldGeneratorBarnCylinder({"width": 1.0, "height": 1.0}, random.Random(0)).compute()


def test_barn_cylinder_files_emit_scenario():
    import yaml

    gen = WorldGeneratorBarnCylinder({}, random.Random(0))
    files = gen.files()
    key = f"scenarios/{WorldGeneratorBarnCylinder.SCENARIO}/scenario.yaml"
    assert key in files

    scenario = yaml.safe_load(files[key])
    robots = scenario["robots"]
    assert len(robots) == 1
    assert len(robots[0]["start"]) == 3
    assert len(robots[0]["phases"][0]["goto"]) == 3


def test_barn_cylinder_seed_deterministic():
    a = WorldGeneratorBarnCylinder({}, random.Random(7))
    b = WorldGeneratorBarnCylinder({}, random.Random(7))
    assert len(a.compute().all_levels[0].walls) == len(b.compute().all_levels[0].walls)
    assert a.files() == b.files()


def test_barn_cylinder_episode_binding():
    params = WorldGeneratorBarnCylinder({}, random.Random(0)).params()
    assert params["tm_robots"] == "scenario"
    assert params["robots_params"] == {"file": WorldGeneratorBarnCylinder.SCENARIO}
