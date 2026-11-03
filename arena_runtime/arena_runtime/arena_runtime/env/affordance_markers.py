"""Affordance slot MarkerArray: per slot its box with the top diagonals and an arrow from the box edge per allowed facing, colored by occupancy."""

import math
import zlib
from collections.abc import Iterable, Sequence

from arena_simulation_setup.utils.affordance import AffordanceClaims, AffordanceKind, AffordanceSlot
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from arena_runtime.shared import Pose, Position

_FRAME = "map"
_ARROW_LENGTH = 0.25

_KIND_COLORS = {AffordanceKind.SIT: ColorRGBA(r=0.1, g=0.75, b=0.8, a=0.9)}
_OCCUPIED_COLOR = ColorRGBA(r=1.0, g=0.35, b=0.7, a=0.95)
_BLOCKED_COLOR = ColorRGBA(r=0.55, g=0.55, b=0.6, a=0.6)


def _id(*parts: object) -> int:
    return zlib.crc32("/".join(map(str, parts)).encode()) & 0x7FFFFFFF


def _marker(ns: str, marker_id: int, marker_type: int, stamp: Time, color: ColorRGBA) -> Marker:
    marker = Marker()
    marker.header.frame_id = _FRAME
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.color = color
    return marker


def _point(pose: Pose, x: float, y: float, z: float) -> Point:
    placed = (pose * Pose(Position(x, y, z))).position
    return Point(x=placed.x, y=placed.y, z=placed.z)


def _edge(slot: AffordanceSlot, yaw: float) -> tuple[float, float]:
    """Where a ray from the box center along yaw leaves the box footprint."""
    (x_min, x_max), (y_min, y_max), _ = slot.bbox
    cx, cy = slot.center
    dx, dy = math.cos(yaw), math.sin(yaw)
    reach = min((x_max - x_min) / 2 / abs(dx) if abs(dx) > 1e-9 else math.inf, (y_max - y_min) / 2 / abs(dy) if abs(dy) > 1e-9 else math.inf)
    return cx + reach * dx, cy + reach * dy


def deleteall_markers() -> list[Marker]:
    markers = []
    for kind in AffordanceKind:
        marker = Marker()
        marker.header.frame_id = _FRAME
        marker.ns = str(kind)
        marker.action = Marker.DELETEALL
        markers.append(marker)
    return markers


def claims_of(slots: Sequence[AffordanceSlot], held: Iterable[int]) -> AffordanceClaims:
    """Claims on the held indices into the sit slots, a held slot overlapping an earlier claim left out."""
    claims = AffordanceClaims(list(slots))
    sit = [slot for slot in slots if slot.kind is AffordanceKind.SIT]
    for index in held:
        if 0 <= index < len(sit) and claims.blocking(sit[index]) is None:
            claims.claim(sit[index])
    return claims


def slot_markers(name: str, pose: Pose, slot: AffordanceSlot, stamp: Time, color: ColorRGBA | None = None) -> list[Marker]:
    """Map-frame markers of one slot of the object name at pose, ids stable per object, slot and part."""
    color = color or _KIND_COLORS[slot.kind]
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = slot.bbox
    outline = _marker(str(slot.kind), _id(name, slot.kind, slot.index, "box"), Marker.LINE_LIST, stamp, color)
    outline.scale.x = 0.02
    corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    lines: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1], strict=True):
        lines += [((ax, ay, z_min), (bx, by, z_min)), ((ax, ay, z_max), (bx, by, z_max))]
    lines += [((x, y, z_min), (x, y, z_max)) for x, y in corners]
    lines += [((x_min, y_min, z_max), (x_max, y_max, z_max)), ((x_min, y_max, z_max), (x_max, y_min, z_max))]
    outline.points = [_point(pose, *end) for line in lines for end in line]
    markers = [outline]
    for number, yaw in enumerate(slot.facing.yaws):
        x, y = _edge(slot, yaw)
        arrow = _marker(str(slot.kind), _id(name, slot.kind, slot.index, "facing", number), Marker.ARROW, stamp, color)
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.03, 0.07, 0.08
        arrow.points = [_point(pose, x, y, z_max), _point(pose, x + _ARROW_LENGTH * math.cos(yaw), y + _ARROW_LENGTH * math.sin(yaw), z_max)]
        markers.append(arrow)
    return markers


def affordance_markers(objects: Iterable[tuple[str, Pose, Sequence[AffordanceSlot], Sequence[int]]], stamp: Time) -> list[Marker]:
    """The latched overlay: a clear of every kind, then every slot of every object, occupied ones and those overlapping them recolored."""
    markers = deleteall_markers()
    for name, pose, slots, held in objects:
        claims = claims_of(slots, held)
        occupied = claims.occupied()
        for slot in slots:
            color = _OCCUPIED_COLOR if slot in occupied else _BLOCKED_COLOR if claims.blocking(slot) is not None else None
            markers.extend(slot_markers(name, pose, slot, stamp, color))
    return markers
