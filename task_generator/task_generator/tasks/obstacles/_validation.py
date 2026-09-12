"""Occupancy-based pose validation, shared by obstacle task modes.

Extracted from ``TM_Scenario.reset`` so modes that *generate* poses can reject invalid
ones instead of emitting them. Placing an agent inside a wall confounds "the perturbed
parameter caused the failure" with "the agent spawned somewhere impossible" - see
README section 7, where ~35 % of LLM-generated positions fell outside every navigable zone.
"""

from __future__ import annotations

from collections.abc import Callable

from task_generator.manager.world_manager.utils import WorldOccupancy
from task_generator.shared import Position, PositionRadius

IsValid = Callable[[Position], bool]


def in_map_bounds(world_map: object, pt: Position) -> bool:
    """Whether ``pt`` falls inside the occupancy grid's extent at all.

    Distinct from :func:`make_is_valid`, and needed for a different question. ``is_valid``
    asks "can something stand here" - it fails for a point behind a wall, under a desk, or
    simply outside the map, and cannot tell those apart. A point *outside the map* means the
    caller is holding a coordinate in the wrong frame, which is a bug in the caller rather
    than a property of the world, and is worth refusing loudly.
    """
    rows, cols = world_map.occupancy.grid.shape  # type: ignore[attr-defined]
    r, c = world_map.tf_pos2grid(pt)  # type: ignore[attr-defined]
    return 0 <= int(r) < rows and 0 <= int(c) < cols


def make_is_traversable(world_map: object, safe_dist: float) -> IsValid | None:
    """Like :func:`make_is_valid`, but over *physical* geometry only.

    ``make_is_valid`` also honours the forbidden layer - space task modes have reserved so no
    new obstacle spawns there, such as the robot's own start and goal. That is the right
    question for "may something stand here" and the wrong one for "can the robot drive here":
    the robot passes through its own exclusion zone every episode. Asking the wrong one makes
    the robot look sealed into a few hundred cells of free space.
    """
    return make_is_valid(world_map, safe_dist, grid=world_map.occupancy.physical)  # type: ignore[attr-defined]


def make_is_valid(world_map: object, safe_dist: float, *, grid: object = None) -> IsValid | None:
    """Predicate testing whether a disc of ``safe_dist`` around a point is free space.

    Returns ``None`` when ``safe_dist <= 0``, matching the convention the zone converter
    expects (no predicate = no filtering).
    """
    if safe_dist <= 0:
        return None

    occupancy_grid = world_map.occupancy.grid if grid is None else grid  # type: ignore[attr-defined]
    rows, cols = occupancy_grid.shape

    def is_valid(pt: Position) -> bool:
        (lo_r, lo_c), (hi_r, hi_c) = world_map.tf_posr2rect(  # type: ignore[attr-defined]
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
