from __future__ import annotations

from collections.abc import Callable

from arena_runtime.env.task_api import WorldMap, WorldOccupancy
from arena_runtime.shared import Position, PositionRadius

IsValid = Callable[[Position], bool]


def make_is_valid(world_map: WorldMap, safe_dist: float) -> IsValid | None:
    if safe_dist <= 0:
        return None

    occupancy_grid = world_map.occupancy.grid
    rows, cols = occupancy_grid.shape

    def is_valid(pt: Position) -> bool:
        (lo_r, lo_c), (hi_r, hi_c) = world_map.tf_posr2rect(
            PositionRadius(x=pt.x, y=pt.y, radius=safe_dist),
        )
        r0 = max(0, int(min(lo_r, hi_r)))
        r1 = min(rows, int(max(lo_r, hi_r)) + 1)
        c0 = max(0, int(min(lo_c, hi_c)))
        c1 = min(cols, int(max(lo_c, hi_c)) + 1)
        if r0 >= r1 or c0 >= c1:
            return False
        return bool(WorldOccupancy.empty(occupancy_grid[r0:r1, c0:c1]).all())

    return is_valid
