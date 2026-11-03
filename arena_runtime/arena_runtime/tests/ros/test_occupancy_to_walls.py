"""Tests for occupancy_to_walls: occupancy grid -> wall line segments."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

try:
    from arena_runtime.env.world_manager.utils import WorldOccupancy, occupancy_to_walls
    from arena_runtime.shared import Position, Wall
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")


def _grid(rows: list[str]) -> np.ndarray:
    cell = {"#": WorldOccupancy.FULL, ".": WorldOccupancy.EMPTY}
    return np.array([[cell[c] for c in row] for row in rows], dtype=np.uint8)


def _segments(walls: Iterable[Wall]) -> set:
    return {
        frozenset(
            {
                (round(w.start.x, 6), round(w.start.y, 6)),
                (round(w.end.x, 6), round(w.end.y, 6)),
            }
        )
        for w in walls
    }


def test_all_free_grid_yields_outer_perimeter_only():
    grid = _grid(
        [
            "..",
            "..",
        ]
    )
    assert _segments(occupancy_to_walls(grid)) == {
        frozenset({(0, 0), (2, 0)}),
        frozenset({(0, 2), (2, 2)}),
        frozenset({(0, 0), (0, 2)}),
        frozenset({(2, 0), (2, 2)}),
    }


def test_single_occupied_cell_gives_outer_boundary_and_cell_square():
    grid = _grid(
        [
            "...",
            ".#.",
            "...",
        ]
    )
    assert _segments(occupancy_to_walls(grid)) == {
        frozenset({(0, 0), (3, 0)}),
        frozenset({(0, 3), (3, 3)}),
        frozenset({(0, 0), (0, 3)}),
        frozenset({(3, 0), (3, 3)}),
        frozenset({(1, 1), (2, 1)}),
        frozenset({(1, 2), (2, 2)}),
        frozenset({(1, 1), (1, 2)}),
        frozenset({(2, 1), (2, 2)}),
    }


def test_transform_applied_to_every_endpoint():
    grid = _grid(
        [
            "...",
            ".#.",
            "...",
        ]
    )

    def tf(p: tuple[float, float]) -> Position:
        return Position(x=p[0] * 0.5 + 10.0, y=p[1] * 0.5 - 5.0)

    expected = {
        frozenset(
            {
                (round(w.start.x * 0.5 + 10.0, 6), round(w.start.y * 0.5 - 5.0, 6)),
                (round(w.end.x * 0.5 + 10.0, 6), round(w.end.y * 0.5 - 5.0, 6)),
            }
        )
        for w in occupancy_to_walls(grid)
    }
    assert _segments(occupancy_to_walls(grid, transform=tf)) == expected
