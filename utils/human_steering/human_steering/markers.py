"""Static MarkerArray geometry for the canvas: a marker cache per topic and each marker's outlines in its header frame."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from visualization_msgs.msg import Marker

CUBE = 1
LINE_STRIP = 4
LINE_LIST = 5
DELETE = 2
DELETEALL = 3

Polyline = list[tuple[float, float]]


def merge(cache: dict[tuple[str, int], Marker], markers: Iterable[Marker]) -> None:
    """Apply one MarkerArray to the cache of its topic: DELETEALL clears it, DELETE drops one marker, anything else adds or replaces it."""
    for marker in markers:
        if marker.action == DELETEALL:
            cache.clear()
        elif marker.action == DELETE:
            cache.pop((marker.ns, marker.id), None)
        else:
            cache[(marker.ns, marker.id)] = marker


def outlines(marker: Marker) -> list[Polyline]:
    """The marker's 2D outlines with its pose applied: one per LINE_LIST pair, one open LINE_STRIP, one closed CUBE footprint, none for other types."""
    q = marker.pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    ox, oy = marker.pose.position.x, marker.pose.position.y

    def place(x: float, y: float) -> tuple[float, float]:
        return ox + cos_yaw * x - sin_yaw * y, oy + sin_yaw * x + cos_yaw * y

    if marker.type == LINE_LIST:
        points = [place(p.x, p.y) for p in marker.points]
        return [[points[i], points[i + 1]] for i in range(0, len(points) - 1, 2)]
    if marker.type == LINE_STRIP:
        points = [place(p.x, p.y) for p in marker.points]
        return [points] if len(points) >= 2 else []
    if marker.type == CUBE:
        hx, hy = marker.scale.x / 2.0, marker.scale.y / 2.0
        corners = [place(-hx, -hy), place(hx, -hy), place(hx, hy), place(-hx, hy)]
        return [[*corners, corners[0]]]
    return []
