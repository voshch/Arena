"""Tests for CollisionGrid: labelled occupancy -> footprint hits."""

from __future__ import annotations

import numpy as np
import pytest
import shapely

try:
    from arena_rclpy_mixins.Time import Time

    from task_generator.manager.collision_grid import MAP_ID, WALL_ID, Cell, CollisionGrid
    from task_generator.manager.world_manager.utils import WorldLayers, WorldMap, WorldOccupancy
    from task_generator.shared import Position, Wall
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")


def _world_map(rows: list[str], origin: Position | None = None) -> WorldMap:
    cell = {"#": WorldOccupancy.FULL, ".": WorldOccupancy.EMPTY}
    grid = np.array([[cell[c] for c in row] for row in rows], dtype=np.uint8)
    return WorldMap(occupancy=WorldLayers(walls=WorldOccupancy(grid)), origin=origin or Position(x=0.0, y=0.0), resolution=1.0, time=Time(-1, 0))


def _box(x: float, y: float, half: float = 0.4) -> shapely.Polygon:
    return shapely.box(x - half, y - half, x + half, y + half)


# 4x4 map, one occupied cell at grid (row 1, col 2) = map (2..3, 2..3)
_ROWS = [
    "....",
    "..#.",
    "....",
    "....",
]


def test_map_cell_hits_as_map():
    grid = CollisionGrid.build(_world_map(_ROWS), origin=Position(x=0.0, y=0.0), walls=())
    assert grid.hit(_box(2.5, 2.5)) == (Cell.MAP, MAP_ID)
    assert grid.hit(_box(0.5, 0.5)) is None


def test_off_grid_is_clear():
    grid = CollisionGrid.build(_world_map(_ROWS), origin=Position(x=0.0, y=0.0), walls=())
    assert grid.hit(_box(40.0, 40.0)) is None
    assert grid.hit(_box(-40.0, -40.0)) is None


def test_authored_wall_outranks_map():
    wall = Wall(start=Position(x=0.5, y=2.5), end=Position(x=3.5, y=2.5))
    grid = CollisionGrid.build(_world_map(_ROWS), origin=Position(x=0.0, y=0.0), walls=[wall])
    assert grid.hit(_box(2.5, 2.5)) == (Cell.WALL, WALL_ID)
    assert grid.hit(_box(0.5, 2.5)) == (Cell.WALL, WALL_ID)
    assert grid.hit(_box(0.5, 0.5)) is None


def test_cutout_clears_map_cells():
    grid = CollisionGrid.build(_world_map(_ROWS), origin=Position(x=0.0, y=0.0), walls=(), cutouts=[shapely.box(2.0, 2.0, 3.0, 3.0)])
    assert grid.hit(_box(2.5, 2.5)) is None


def test_static_stamp_and_sync():
    grid = CollisionGrid.build(_world_map(_ROWS), origin=Position(x=0.0, y=0.0), walls=())
    grid.stamp("crate", shapely.box(0.0, 0.0, 1.0, 1.0))
    grid.stamp("bench", shapely.box(2.0, 2.0, 3.0, 3.0))
    assert grid.hit(_box(0.5, 0.5)) == (Cell.STATIC, "crate")
    assert grid.hit(_box(2.5, 2.5)) == (Cell.STATIC, "bench")
    grid.sync(["bench"])
    assert grid.hit(_box(0.5, 0.5)) is None
    assert grid.hit(_box(2.5, 2.5)) == (Cell.STATIC, "bench")
    grid.sync([])
    assert grid.hit(_box(2.5, 2.5)) == (Cell.MAP, MAP_ID)


def test_sub_half_cell_overlap_misses():
    grid = CollisionGrid.build(_world_map(_ROWS), origin=Position(x=0.0, y=0.0), walls=())
    # covers x in [1.7, 2.3]: only 0.3 of the occupied cell, its centre (2.5) is outside
    assert grid.hit(shapely.box(1.7, 2.2, 2.3, 2.8)) is None


def test_realized_origin_shift():
    grid = CollisionGrid.build(_world_map(_ROWS), origin=Position(x=10.0, y=-5.0), walls=())
    assert grid.hit(_box(12.5, -2.5)) == (Cell.MAP, MAP_ID)
    assert grid.hit(_box(2.5, 2.5)) is None


def test_tf_poly2mask_matches_window():
    world_map = _world_map(_ROWS)
    poly = shapely.box(1.0, 1.0, 3.0, 3.0)
    mask = world_map.tf_poly2mask(poly)
    window = world_map.frame.poly_window(poly)
    assert window is not None
    rows, cols, inside = window
    expected = np.zeros(world_map.shape[:2], dtype=bool)
    expected[rows, cols] = inside
    assert np.array_equal(mask, expected)
    assert mask.sum() == 4
