"""Task overlay MarkerArray: a state label per agent, and for working agents a link to and the outline of the zones of their task."""

from collections.abc import Callable, Mapping, Sequence

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from arena_runtime.env.realizer import Realizer
from arena_runtime.env.task_board import AgentEntry

_NAMESPACES = ("task_labels", "task_links", "task_zones")

_FRAME = "map"

_Z_LABEL = 2.0
_Z_LINK = 0.3
_Z_ZONE = 0.05

_STATE_COLORS = {
    "working": ColorRGBA(r=0.2, g=0.8, b=0.3, a=1.0),
    "assigned": ColorRGBA(r=0.95, g=0.8, b=0.1, a=1.0),
    "met": ColorRGBA(r=0.6, g=0.6, b=0.6, a=1.0),
    "failed": ColorRGBA(r=0.9, g=0.2, b=0.2, a=1.0),
    "dropped": ColorRGBA(r=0.9, g=0.2, b=0.2, a=1.0),
    "idle": ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
}


def _marker(ns: str, marker_id: int, marker_type: int, stamp: Time, frame: str = _FRAME) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.color.a = 1.0
    return marker


def _label_text(entry: AgentEntry, activity: str) -> str:
    if entry.state == "idle" or not entry.task:
        return f"{entry.name}: idle"
    return f"{entry.name}: {entry.task} ({activity if entry.state == 'working' and activity else entry.state})"


def task_markers(
    agents: Sequence[AgentEntry],
    frames: Mapping[str, str],
    positions: Mapping[str, tuple[float, float]],
    zone_corners: Callable[[str], Sequence | None],
    activity: Mapping[str, str],
    realizer: Realizer,
    stamp: Time,
) -> list[Marker]:
    """Clears the previous overlay, then labels every positioned agent in its own TF frame and draws the zones of the working ones."""
    markers = []
    for ns in _NAMESPACES:
        clear = Marker()
        clear.header.frame_id = _FRAME
        clear.ns = ns
        clear.action = Marker.DELETEALL
        markers.append(clear)

    for index, entry in enumerate(agents):
        position = positions.get(entry.name)
        frame = frames.get(entry.name)
        if position is None or frame is None:
            continue
        color = _STATE_COLORS.get(entry.state, _STATE_COLORS["idle"])
        label = _marker("task_labels", index, Marker.TEXT_VIEW_FACING, Time(), frame)
        label.frame_locked = True
        label.text = _label_text(entry, activity.get(entry.name, ""))
        label.scale.z = 0.4
        label.color = color
        label.pose.position.z = _Z_LABEL
        markers.append(label)

        if entry.state != "working":
            continue
        rings = []
        for zone in entry.zones:
            corners = zone_corners(zone)
            if corners is not None and len(corners) >= 3:
                rings.append([realizer.realize_xy(corner.x, corner.y) for corner in corners])
        if not rings:
            continue

        first = rings[0]
        center = (sum(x for x, _ in first) / len(first), sum(y for _, y in first) / len(first))
        link = _marker("task_links", index, Marker.LINE_LIST, stamp)
        link.scale.x = 0.05
        link.color = color
        link.points.extend((Point(x=position[0], y=position[1], z=_Z_LINK), Point(x=center[0], y=center[1], z=_Z_LINK)))  # pyright: ignore[reportAttributeAccessIssue]
        markers.append(link)

        outline = _marker("task_zones", index, Marker.LINE_LIST, stamp)
        outline.scale.x = 0.1
        outline.color = color
        for ring in rings:
            for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1], strict=True):
                outline.points.extend((Point(x=x0, y=y0, z=_Z_ZONE), Point(x=x1, y=y1, z=_Z_ZONE)))  # pyright: ignore[reportAttributeAccessIssue]
        markers.append(outline)
    return markers
