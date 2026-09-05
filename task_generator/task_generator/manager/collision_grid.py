"""Per-world labelled occupancy driving robot static-collision detection."""

from __future__ import annotations

import enum
from collections.abc import Iterable

import numpy as np
import shapely

from task_generator.manager.world_manager.utils import GridFrame, WorldMap, WorldOccupancy
from task_generator.shared import Position, Wall

MAP_ID = '<map>'
WALL_ID = '<wall>'

Window = tuple[slice, slice, np.ndarray]


class Cell(enum.IntEnum):
    """Cell provenance, higher wins when a footprint covers several."""

    FREE = 0
    MAP = 1  # occupied in the world map, nothing authored behind it
    WALL = 2  # authored wall segment
    STATIC = 3  # stamped static-obstacle footprint


def _overlap(a: Window, b: Window) -> bool:
    (ar, ac, am), (br, bc, bm) = a, b
    r0, r1 = max(ar.start, br.start), min(ar.stop, br.stop)
    c0, c1 = max(ac.start, bc.start), min(ac.stop, bc.stop)
    if r0 >= r1 or c0 >= c1:
        return False
    sa = am[r0 - ar.start : r1 - ar.start, c0 - ac.start : c1 - ac.start]
    sb = bm[r0 - br.start : r1 - br.start, c0 - bc.start : c1 - bc.start]
    return bool(np.any(sa & sb))


class CollisionGrid:
    """`base` holds MAP/WALL cells for the loaded world, `live` adds the STATIC stamps of the current episode."""

    def __init__(self, base: np.ndarray, frame: GridFrame):
        self._base = base
        self._live = base.copy()
        self._frame = frame
        self._statics: dict[str, Window] = {}

    @classmethod
    def build(
        cls,
        world_map: WorldMap,
        *,
        origin: Position,
        walls: Iterable[Wall],
        cutouts: Iterable[shapely.Polygon] = (),
    ) -> CollisionGrid:
        """Label the map's walls layer as MAP, clear `cutouts` (doors, elevators), then rasterize authored `walls` on top."""
        occupied = WorldOccupancy.full(world_map.occupancy.walls)
        frame = GridFrame(shape=occupied.shape, origin=origin, resolution=world_map.resolution)
        base = np.where(occupied, np.uint8(Cell.MAP), np.uint8(Cell.FREE))
        for poly in cutouts:
            window = frame.poly_window(poly)
            if window is not None:
                rows, cols, inside = window
                base[rows, cols][inside] = Cell.FREE
        base[frame.lines_mask(((w.start.x, w.start.y), (w.end.x, w.end.y)) for w in walls)] = Cell.WALL
        return cls(base, frame)

    def stamp(self, name: str, poly: shapely.Polygon) -> None:
        window = self._frame.poly_window(poly)
        if window is None:
            return
        self._statics[name] = window
        rows, cols, inside = window
        self._live[rows, cols][inside] = Cell.STATIC

    def sync(self, names: Iterable[str]) -> None:
        """Keep only the stamps in `names`, rebuilding `live` from `base`."""
        keep = set(names)
        self._statics = {n: w for n, w in self._statics.items() if n in keep}
        self._live = self._base.copy()
        for rows, cols, inside in self._statics.values():
            self._live[rows, cols][inside] = Cell.STATIC

    def hit(self, poly: shapely.Polygon) -> tuple[Cell, str] | None:
        """Highest-ranked cell under `poly` and its obstacle id, None when clear or off-grid."""
        window = self._frame.poly_window(poly)
        if window is None:
            return None
        rows, cols, inside = window
        codes = self._live[rows, cols][inside]
        if codes.size == 0:
            return None
        top = Cell(int(codes.max()))
        if top is Cell.FREE:
            return None
        if top is Cell.WALL:
            return top, WALL_ID
        if top is Cell.MAP:
            return top, MAP_ID
        for name, stamped in self._statics.items():
            if _overlap(window, stamped):
                return top, name
        return top, MAP_ID
