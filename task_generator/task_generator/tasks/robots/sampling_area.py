"""The area a task mode draws positions from: a named zone or stair group, a disc around a point, or the whole map."""

from __future__ import annotations

import math
import typing

import shapely
from arena_simulation_setup.shared.judge import zone_polygons

if typing.TYPE_CHECKING:
    from arena_simulation_setup.tree.World.model import Level


class ZonedWorld(typing.Protocol):
    def world_compacted(self) -> Level: ...


def sampling_area(*, prefix: str, world: ZonedWorld, zone: str, center_x: float, center_y: float, radius: float) -> shapely.Geometry | None:
    """The polygon sampled positions must lie in, or None for the whole map.

    prefix is the parameter name up to its leaf, for example task.random.goal_, so the errors name the parameters that were set.
    """
    has_center = math.isfinite(center_x) and math.isfinite(center_y)
    if zone and has_center:
        raise ValueError(f"set either {prefix}zone or {prefix}center_x and {prefix}center_y, not both")
    if zone:
        polygon = zone_polygons(world.world_compacted()).get(zone)
        if polygon is None:
            raise ValueError(f"{prefix}zone {zone!r} is not a zone of the loaded world")
        return polygon
    if has_center:
        return shapely.Point(center_x, center_y).buffer(radius)
    return None
