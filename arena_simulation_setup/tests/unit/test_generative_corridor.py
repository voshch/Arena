"""Corridor trees: branches joined to the nearest node through one elbow, or winding through kinks at 45 and 90 degrees."""

import math
import random

import pytest
import shapely
from arena_simulation_setup.generative.world.corridor import WorldGeneratorCorridor


def build(seed: int = 0, **config: float) -> tuple[WorldGeneratorCorridor, shapely.Polygon | shapely.MultiPolygon]:
    generator = WorldGeneratorCorridor(config, random.Random(seed))
    level = generator.compute().all_levels[0]
    return generator, shapely.union_all([zone.shape() for zone in level.zones])


def test_a_tree_is_one_component():
    for seed in range(5):
        generator, _ = build(seed, branches=20)
        assert generator.diagnostics.components == 1, seed


def test_corridors_stay_inside_the_extent():
    _, free = build(width=40.0, height=25.0, corridor_width=3.0, branches=15)
    minx, miny, maxx, maxy = free.bounds
    assert minx >= -1e-6 and miny >= -1e-6 and maxx <= 40.0 + 1e-6 and maxy <= 25.0 + 1e-6


def test_one_branch_is_an_l_of_the_set_width():
    _, free = build(width=30.0, height=30.0, corridor_width=2.0, branches=1)
    assert free.area == pytest.approx(2.0 * (free.length / 2 - 2.0), rel=0.05)


def test_more_branches_cover_more_ground():
    areas = [build(3, branches=n)[1].area for n in (2, 8, 30)]
    assert areas[0] < areas[1] < areas[2]


def test_a_seed_repeats():
    assert build(7, branches=12)[1].equals(build(7, branches=12)[1])


def test_too_small_an_extent_is_refused():
    with pytest.raises(ValueError):
        build(width=3.0, height=30.0, corridor_width=3.0)


def angled(free: shapely.Polygon | shapely.MultiPolygon) -> float:
    """Share of the free space's outline that runs off the axes."""
    polys = free.geoms if isinstance(free, shapely.MultiPolygon) else [free]
    edges = [(a, b) for poly in polys for ring in (poly.exterior, *poly.interiors) for a, b in zip(ring.coords, ring.coords[1:])]
    off = sum(shapely.LineString((a, b)).length for a, b in edges if abs(a[0] - b[0]) > 1e-6 and abs(a[1] - b[1]) > 1e-6)
    return off / sum(shapely.LineString((a, b)).length for a, b in edges)


def test_without_bends_every_wall_is_axis_aligned():
    _, free = build(4, branches=12, width_spread=0.3)
    assert angled(free) == 0


def octilinear(free: shapely.Polygon | shapely.MultiPolygon) -> float:
    """Share of the outline running at a multiple of 45 degrees."""
    polys = free.geoms if isinstance(free, shapely.MultiPolygon) else [free]
    edges = [(a, b) for poly in polys for ring in (poly.exterior, *poly.interiors) for a, b in zip(ring.coords, ring.coords[1:])]
    on = sum(math.dist(a, b) for a, b in edges if abs((math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) + 22.5) % 45 - 22.5) < 1)
    return on / sum(math.dist(a, b) for a, b in edges)


def test_bends_wind_the_corridors_and_keep_one_tree():
    for seed in range(5):
        generator, free = build(seed, branches=12, bends=3, wiggle=0.2)
        assert generator.diagnostics.components == 1, seed
        assert angled(free) > 0.1, seed


def test_bends_in_an_extent_too_small_for_a_kink_keep_every_branch():
    for seed in range(5):
        generator, free = build(seed, width=7.0, height=7.0, branches=8, bends=2, wiggle=0.2)
        assert generator.diagnostics.components == 1, seed
        assert angled(free) == 0, seed


def test_a_dense_winding_tree_with_a_width_spread_renders():
    for seed in range(20):
        WorldGeneratorCorridor({'branches': 60, 'bends': 2, 'wiggle': 0.2, 'width_spread': 0.3}, random.Random(seed)).compute().all_levels[0].render(0.05)


def test_winding_corridors_turn_at_multiples_of_45_degrees():
    for seed in range(5):
        _, free = build(seed, branches=12, bends=2, wiggle=0.2)
        assert octilinear(free) > 0.98, seed


def test_winding_corridors_stay_inside_the_extent():
    _, free = build(width=40.0, height=25.0, corridor_width=3.0, branches=15, bends=4, wiggle=0.5)
    minx, miny, maxx, maxy = free.bounds
    assert minx >= -1e-6 and miny >= -1e-6 and maxx <= 40.0 + 1e-6 and maxy <= 25.0 + 1e-6


def test_bends_without_diagonals_keep_right_angles():
    for seed in range(5):
        generator, free = build(seed, branches=12, bends=2, wiggle=0.2, diagonal=False)
        assert generator.diagnostics.components == 1, seed
        assert angled(free) == 0, seed
