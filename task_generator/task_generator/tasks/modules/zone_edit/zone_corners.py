"""Write edited zone polygons back into a world through the canonical world writer."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from arena_simulation_setup.tree.World import WorldView
from arena_simulation_setup.utils.geometry import Position

_PRECISION = 4


def write_zone_polygons(world_root: Path, polygons: Mapping[str, Sequence[tuple[float, float]]], derive: str = "auto") -> int:
    """Set the x and y of the named zones' polygon points, saving the world only when one moved. Returns the count of coordinates changed."""
    view = WorldView(world_root)
    world = view.load(derive=derive)
    zones = {zone.name: zone for level in world.all_levels for zone in level.zones}
    changed = 0
    for name, points in polygons.items():
        zone = zones.get(name)
        if zone is None:
            raise ValueError(f"{world_root} has no zone {name!r}")
        if len(zone.polygon) != len(points):
            raise ValueError(f"zone {name!r} in {world_root} does not have {len(points)} polygon points")
        polygon: list[Position] = []
        for corner, (x, y) in zip(zone.polygon, points, strict=True):
            moved = [abs(old - new) > 10 ** -(_PRECISION + 1) for old, new in ((corner.x, x), (corner.y, y))]
            changed += sum(moved)
            polygon.append(Position(x if moved[0] else corner.x, y if moved[1] else corner.y, corner.z))
        zone.polygon = polygon
    if changed:
        view.save(world)
    return changed
