"""The global costmap's door mask over elevator and door openings."""

from __future__ import annotations

import shapely

from arena_runtime.env.world_manager.utils import door_mask
from arena_runtime.shared import Position, Wall

RESOLUTION = 0.05
ORIGIN = (0.0, -3.0)
WIDTH, HEIGHT = 200, 200
FACE = [Wall(start=Position(2.0, 0.0), end=Position(4.25, 0.0)), Wall(start=Position(5.75, 0.0), end=Position(8.0, 0.0))]
CABIN = shapely.Polygon([(4.0, -2.0), (6.0, -2.0), (6.0, 0.0), (4.0, 0.0)])
DOORWAY = shapely.Polygon([(4.0, 0.0), (6.0, 0.0), (6.0, 0.3), (4.0, 0.3)])


def _cell(data, x: float, y: float) -> int:
    return int(data[int((y - ORIGIN[1]) / RESOLUTION) * WIDTH + int((x - ORIGIN[0]) / RESOLUTION)])


def test_an_elevator_mask_frees_the_opening_and_cabin_but_never_the_wall_jambs():
    data = door_mask([CABIN, DOORWAY], FACE, WIDTH, HEIGHT, RESOLUTION, ORIGIN, 0.2)

    assert _cell(data, 5.0, 0.0) == 0
    assert _cell(data, 5.0, -1.0) == 0
    assert _cell(data, 5.6, 0.15) == 0
    assert _cell(data, 5.9, 0.0) == -1
    assert _cell(data, 4.1, 0.0) == -1
    assert _cell(data, 7.0, 1.0) == -1


def test_a_mask_without_walls_frees_the_buffered_opening():
    data = door_mask([DOORWAY], [], WIDTH, HEIGHT, RESOLUTION, ORIGIN, 0.2)

    assert _cell(data, 5.9, 0.0) == 0
    assert _cell(data, 5.0, 0.45) == 0
    assert _cell(data, 5.0, 0.6) == -1
