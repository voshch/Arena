"""World-geometry MarkerArray: zones, walls, doors, elevators, one marker namespace per layer."""

import colorsys
import typing
from collections.abc import Iterable, Sequence

import arena_simulation_setup.tree.World as World
import attrs
import shapely
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from task_generator.manager.realizer import Realizer

if typing.TYPE_CHECKING:
    from arena_runtime.sim._semantics import SemanticEntitySnapshot

_NAMESPACES = ("zones", "zone_labels", "walls", "doors", "elevators")

_FRAME = "map"
_GOLDEN_RATIO = 0.618033988749895

_Z_ZONES = 0.02
_Z_WALLS = 0.03
_Z_MECHANISMS = 0.04
_Z_LABELS = 0.5

_ZONE_INSET = 0.2
_DOOR_MIN_WIDTH = 0.3

_WALL_COLOR = ColorRGBA(r=0.1, g=0.15, b=0.35, a=1.0)
_UNKNOWN_COLOR = ColorRGBA(r=0.5, g=0.5, b=0.6, a=0.8)
_CLOSED_COLOR = ColorRGBA(r=0.85, g=0.2, b=0.2, a=0.8)
_TRANSIT_COLOR = ColorRGBA(r=0.95, g=0.7, b=0.1, a=0.8)
_OPEN_COLOR = ColorRGBA(r=0.2, g=0.8, b=0.3, a=0.8)

_DOOR_STATE_COLORS = {
    "closed": _CLOSED_COLOR,
    "opening": _TRANSIT_COLOR,
    "closing": _TRANSIT_COLOR,
    "open": _OPEN_COLOR,
}


def _marker(ns: str, marker_id: int, marker_type: int, stamp: Time) -> Marker:
    marker = Marker()
    marker.header.frame_id = _FRAME
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.color.a = 1.0
    return marker


def _zone_color(index: int) -> ColorRGBA:
    r, g, b = colorsys.hsv_to_rgb((index * _GOLDEN_RATIO) % 1.0, 0.75, 0.95)
    return ColorRGBA(r=r, g=g, b=b, a=1.0)


def _quad(marker: Marker, corners: Sequence[tuple[float, float]], z: float, color: ColorRGBA) -> None:
    a, b, c, d = (Point(x=x, y=y, z=z) for x, y in corners)
    marker.points.extend((a, b, c, a, c, d))
    marker.colors.extend([color] * 6)


def deleteall_markers() -> list[Marker]:
    markers = []
    for ns in _NAMESPACES:
        marker = Marker()
        marker.header.frame_id = _FRAME
        marker.ns = ns
        marker.action = Marker.DELETEALL
        markers.append(marker)
    return markers


def static_markers(world: World.WorldDescription, realizer: Realizer, stamp: Time) -> list[Marker]:
    """Zones and walls, fixed for the lifetime of a loaded world."""
    zones = _marker("zones", 0, Marker.LINE_LIST, stamp)
    zones.scale.x = 0.12
    walls = _marker("walls", 0, Marker.LINE_LIST, stamp)
    walls.scale.x = 0.08
    walls.color = _WALL_COLOR
    labels: list[Marker] = []

    rings = sorted((realizer.prefix(zone.name, level_id), zone.name, realizer.realize_polygon(zone.corners, level_id)) for level_id, level in world.levels.items() for zone in level.zones if len(zone.corners) >= 3)
    for index, (_, name, ring) in enumerate(rings):
        color = _zone_color(index)
        polygon = shapely.Polygon(ring)
        inset = polygon.buffer(-_ZONE_INSET, join_style="mitre")
        outlines = [polygon] if inset.is_empty else shapely.get_parts(inset)
        for outline in outlines:
            coords = list(outline.exterior.coords)
            for (x0, y0), (x1, y1) in zip(coords, coords[1:], strict=False):
                zones.points.extend((Point(x=x0, y=y0, z=_Z_ZONES), Point(x=x1, y=y1, z=_Z_ZONES)))
                zones.colors.extend((color, color))
        anchor = polygon.representative_point()
        label = _marker("zone_labels", index, Marker.TEXT_VIEW_FACING, stamp)
        label.text = name
        label.scale.z = 0.8
        label.color = color
        label.pose.position.x = anchor.x
        label.pose.position.y = anchor.y
        label.pose.position.z = _Z_LABELS
        labels.append(label)

    for level_id, level in world.levels.items():
        for wall in level.all_walls:
            realized = realizer.realize(wall, level_id)
            walls.points.extend(
                (
                    Point(x=realized.start.x, y=realized.start.y, z=_Z_WALLS),
                    Point(x=realized.end.x, y=realized.end.y, z=_Z_WALLS),
                )
            )

    return [m for m in (zones, walls) if m.points] + labels


def _door_color(snapshot: "SemanticEntitySnapshot | None") -> ColorRGBA:
    if snapshot is None:
        return _UNKNOWN_COLOR
    return _DOOR_STATE_COLORS.get(snapshot.discrete.get("state", ""), _UNKNOWN_COLOR)


def _elevator_color(snapshot: "SemanticEntitySnapshot | None") -> ColorRGBA:
    if snapshot is None:
        return _UNKNOWN_COLOR
    if snapshot.predicates.get("in_transit", False):
        return _TRANSIT_COLOR
    return _DOOR_STATE_COLORS.get(snapshot.discrete.get("cabin_door", ""), _UNKNOWN_COLOR)


def mechanism_markers(
    world: World.WorldDescription,
    realizer: Realizer,
    snapshots: "Iterable[SemanticEntitySnapshot]",
    stamp: Time,
) -> list[Marker]:
    """Doors and elevators, colored by their current semantic state."""
    states = {(s.kind, s.entity): s for s in snapshots}
    doors = _marker("doors", 0, Marker.TRIANGLE_LIST, stamp)
    elevators = _marker("elevators", 0, Marker.TRIANGLE_LIST, stamp)
    for marker in (doors, elevators):
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0

    for level_id, level in world.levels.items():
        for door in level.all_doors:
            realized = attrs.evolve(realizer.realize(door, level_id), width=max(door.width, _DOOR_MIN_WIDTH))
            _quad(doors, [(c.x, c.y) for c in realized.corners], _Z_MECHANISMS, _door_color(states.get(("door", realized.name))))
        for elevator in level.all_elevators:
            realized_elevator = realizer.realize(elevator, level_id)
            _quad(elevators, [(c.x, c.y) for c in realized_elevator.cabin_corners()], _Z_MECHANISMS, _elevator_color(states.get(("elevator", realized_elevator.name))))

    return [m for m in (doors, elevators) if m.points]
