"""Open fields: square obstacles in a walled box."""

import functools
import random

import pytest
import shapely
from arena_simulation_setup.generative.world.layout import diagnostics_of
from arena_simulation_setup.generative.world.open import WorldGeneratorOpen
from arena_simulation_setup.tree.World import Level


def build(seed: int = 0, **config: float) -> tuple[WorldGeneratorOpen, Level]:
    generator = WorldGeneratorOpen(config, random.Random(seed))
    return generator, generator.compute().all_levels[0]


def fill(level: Level, area: float) -> float:
    """Occupied share of the box. The walls are the rings of the free space, so a point is free inside an odd number of them."""
    free = functools.reduce(shapely.symmetric_difference, (shapely.Polygon([(p.x, p.y) for p in wall.path]) for wall in level.walls))
    return 1 - free.area / area


def test_no_obstacles_is_the_bare_box():
    generator, level = build(width=12.0, height=8.0, obstacles=0)
    assert generator.diagnostics.islands == 0
    assert len(level.zones) == 1 and len(level.walls) == 1
    assert fill(level, 96.0) == pytest.approx(0.0)


def test_an_obstacle_inside_is_walled_and_an_island():
    generator, level = next(built for built in (build(seed, width=20.0, height=20.0, obstacles=1, obstacle_size=4.0) for seed in range(20)) if built[0].diagnostics.islands)
    assert len(level.walls) == 2
    assert fill(level, 400.0) == pytest.approx(16.0 / 400.0)


def test_an_obstacle_is_solid_on_the_map():
    _, level = next(built for built in (build(seed, width=20.0, height=20.0, obstacles=1, obstacle_size=4.0) for seed in range(20)) if built[0].diagnostics.islands)
    center = min((shapely.Polygon([(p.x, p.y) for p in wall.path]) for wall in level.walls), key=lambda ring: ring.area).centroid
    resolution = 0.1
    grid, (origin_x, origin_y) = level.render_grid(resolution)
    assert grid[grid.shape[0] - 1 - round((center.y - origin_y) / resolution), round((center.x - origin_x) / resolution)] == 0


def test_a_cleared_spot_is_free_on_the_map():
    _, level = build(width=12.0, height=8.0, obstacles=0)
    resolution = 0.1
    grid, (origin_x, origin_y) = level.render_grid(resolution)
    assert grid[grid.shape[0] - 1 - round((4.0 - origin_y) / resolution), round((6.0 - origin_x) / resolution)] == 255


def test_no_zone_covers_an_obstacle():
    _, level = build(obstacles=40)
    rings = sorted((shapely.Polygon([(p.x, p.y) for p in wall.path]) for wall in level.walls), key=lambda ring: ring.area)
    free = shapely.union_all([zone.shape() for zone in level.zones])
    assert all(not free.contains(ring.representative_point()) for ring in rings[:-1])


def test_zones_show_the_islands():
    _, level = build(obstacles=300, obstacle_size=0.5)
    diagnostics = diagnostics_of(level)
    assert diagnostics.components == 1 and diagnostics.islands > 100


def test_a_dense_field_renders():
    for seed in range(10):
        _, level = build(seed, obstacles=300, obstacle_size=0.5, size_spread=0.5)
        level.render(0.05)


def test_a_dense_field_stays_in_one_piece():
    for seed in range(5):
        generator, _ = build(seed, obstacles=60, obstacle_size=1.5)
        assert generator.diagnostics.components == 1, seed


def test_a_box_the_obstacles_cover_is_refused():
    with pytest.raises(ValueError):
        build(obstacles=1, obstacle_size=50.0)


def test_fill_grows_with_the_count():
    fills = []
    for n in (5, 20, 60):
        fills.append(fill(build(1, obstacles=n)[1], 400.0))
    assert 0 < fills[0] < fills[1] < fills[2] < 0.5


def test_a_seed_repeats():
    assert build(4, obstacles=25, size_spread=0.5)[1].walls == build(4, obstacles=25, size_spread=0.5)[1].walls
