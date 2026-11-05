"""DisplayKind.MARKER_ARRAY renderer: generic visualization_msgs/MarkerArray → rerun.

Keeps the markers the array has added (by namespace and id) until it deletes them or their `lifetime` passes, so
incremental arrays work, and logs them per namespace and frame in that frame: cubes, spheres, cylinders, arrows, lines, points, cube and sphere
lists, triangle lists and text. Mesh resources are skipped. Makes no pedestrian-namespace assumptions, so it suits
debug overlays and static-geometry layers; `subscribe_marker_array` is the reusable subscriber.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import rerun as rr
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from rerun_utils.entity_paths import display_path, frame_name, in_frame
from rerun_utils.renderers._registry import RendererCtx, register

Color = tuple[int, int, int, int]


def _marker_color(m: Marker) -> Color:
    c = m.color
    return (int(c.r * 255), int(c.g * 255), int(c.b * 255), int(c.a * 255))


def _point_colors(m: Marker, n: int) -> list[Color]:
    if len(m.colors) == n:
        return [(int(c.r * 255), int(c.g * 255), int(c.b * 255), int(c.a * 255)) for c in m.colors]
    return [_marker_color(m)] * n


def _rotation(m: Marker) -> np.ndarray:
    q = m.pose.orientation
    x, y, z, w = q.x, q.y, q.z, q.w
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def _posed(m: Marker, points: Iterable) -> np.ndarray:
    """`points` of a list marker from its pose into its frame."""
    local = np.array([[p.x, p.y, p.z] for p in points], dtype=float).reshape(-1, 3)
    p = m.pose.position
    return local @ _rotation(m).T + [p.x, p.y, p.z]


def _center(m: Marker) -> list[float]:
    p = m.pose.position
    return [p.x, p.y, p.z]


def _quat(m: Marker) -> rr.Quaternion:
    q = m.pose.orientation
    return rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w] if (q.x, q.y, q.z, q.w) != (0.0, 0.0, 0.0, 0.0) else [0.0, 0.0, 0.0, 1.0])


def _archetypes(markers: list[Marker]) -> dict[str, rr.AsComponents]:
    """One archetype per marker kind for markers that share a namespace and frame."""
    boxes, spheres, cylinders, arrows, strips, points, meshes, texts = (defaultdict(list) for _ in range(8))
    for m in markers:
        t = m.type
        if t == Marker.CUBE:
            boxes["centers"].append(_center(m))
            boxes["sizes"].append([m.scale.x, m.scale.y, m.scale.z])
            boxes["quaternions"].append(_quat(m))
            boxes["colors"].append(_marker_color(m))
        elif t == Marker.SPHERE:
            spheres["centers"].append(_center(m))
            spheres["half_sizes"].append([m.scale.x / 2, m.scale.y / 2, m.scale.z / 2])
            spheres["quaternions"].append(_quat(m))
            spheres["colors"].append(_marker_color(m))
        elif t == Marker.CYLINDER:
            cylinders["centers"].append(_center(m))
            cylinders["lengths"].append(m.scale.z)
            cylinders["radii"].append(max(m.scale.x, m.scale.y) / 2)
            cylinders["quaternions"].append(_quat(m))
            cylinders["colors"].append(_marker_color(m))
        elif t == Marker.ARROW:
            if len(m.points) >= 2:
                a, b = _posed(m, m.points[:2])
                arrows["origins"].append(a.tolist())
                arrows["vectors"].append((b - a).tolist())
            else:
                arrows["origins"].append(_center(m))
                arrows["vectors"].append((_rotation(m) @ [m.scale.x, 0.0, 0.0]).tolist())
            arrows["colors"].append(_marker_color(m))
        elif t in (Marker.LINE_STRIP, Marker.LINE_LIST):
            pts = _posed(m, m.points)
            segments = [pts] if t == Marker.LINE_STRIP else [pts[i : i + 2] for i in range(0, len(pts) - 1, 2)]
            for seg in segments:
                if len(seg) >= 2:
                    strips["strips"].append(seg.tolist())
                    strips["colors"].append(_marker_color(m))
                    strips["radii"].append(max(m.scale.x / 2, 0.005))
        elif t in (Marker.POINTS, Marker.SPHERE_LIST):
            pts = _posed(m, m.points)
            points["positions"].extend(pts.tolist())
            points["colors"].extend(_point_colors(m, len(pts)))
            points["radii"].extend([max(m.scale.x / 2, 0.005)] * len(pts))
        elif t == Marker.CUBE_LIST:
            pts = _posed(m, m.points)
            boxes["centers"].extend(pts.tolist())
            boxes["sizes"].extend([[m.scale.x, m.scale.y, m.scale.z]] * len(pts))
            boxes["quaternions"].extend([_quat(m)] * len(pts))
            boxes["colors"].extend(_point_colors(m, len(pts)))
        elif t == Marker.TRIANGLE_LIST and len(m.points) >= 3:
            pts = _posed(m, m.points[: len(m.points) // 3 * 3])
            meshes["vertex_positions"].extend(pts.tolist())
            meshes["vertex_colors"].extend(_point_colors(m, len(m.points))[: len(pts)])
        elif t == Marker.TEXT_VIEW_FACING and m.text:
            texts["positions"].append(_center(m))
            texts["labels"].append(m.text)
            texts["colors"].append(_marker_color(m))
    out: dict[str, rr.AsComponents] = {}
    if boxes:
        out["boxes"] = rr.Boxes3D(fill_mode="solid", centers=boxes["centers"], sizes=boxes["sizes"], quaternions=boxes["quaternions"], colors=boxes["colors"])
    if spheres:
        out["spheres"] = rr.Ellipsoids3D(fill_mode="solid", centers=spheres["centers"], half_sizes=spheres["half_sizes"], quaternions=spheres["quaternions"], colors=spheres["colors"])
    if cylinders:
        out["cylinders"] = rr.Cylinders3D(fill_mode="solid", centers=cylinders["centers"], lengths=cylinders["lengths"], radii=cylinders["radii"], quaternions=cylinders["quaternions"], colors=cylinders["colors"])
    if arrows:
        out["arrows"] = rr.Arrows3D(origins=arrows["origins"], vectors=arrows["vectors"], colors=arrows["colors"])
    if strips:
        out["lines"] = rr.LineStrips3D(strips["strips"], colors=strips["colors"], radii=strips["radii"])
    if points:
        out["points"] = rr.Points3D(points["positions"], colors=points["colors"], radii=points["radii"])
    if meshes:
        n = len(meshes["vertex_positions"])
        out["triangles"] = rr.Mesh3D(vertex_positions=meshes["vertex_positions"], vertex_colors=meshes["vertex_colors"], triangle_indices=np.arange(n).reshape(-1, 3))
    if texts:
        out["text"] = rr.Points3D(texts["positions"], radii=0.01, colors=texts["colors"], labels=texts["labels"], show_labels=True)
    return out


#: Path part of the empty namespace or frame; names starting with it get it once more, so the mapping stays one to one.
EMPTY_PART = "~"
#: How often markers with a lifetime are checked for lapsing between messages, seconds of the node's clock.
EXPIRY_PERIOD_S = 0.25


def _safe(name: str) -> str:
    """`name` as one entity path part, distinct names to distinct parts: escaped as rerun does, `~` for the empty name."""
    if not name or name.startswith(EMPTY_PART):
        name = EMPTY_PART + name
    return rr.escape_entity_path_part(name)


class MarkerStore:
    """The markers an array topic has added and not yet deleted, by namespace and id, each lapsing `lifetime` after it
    arrived (on the node's clock, as rviz2 does), never with a zero lifetime."""

    def __init__(self) -> None:
        self._live: dict[tuple[str, int], tuple[Marker, int | None]] = {}

    def ingest(self, markers: Iterable[Marker], now_ns: int) -> None:
        for m in markers:
            if m.action == Marker.DELETEALL:
                self._live.clear()
            elif m.action == Marker.DELETE:
                self._live.pop((m.ns, m.id), None)
            else:
                lifetime_ns = m.lifetime.sec * 1_000_000_000 + m.lifetime.nanosec
                self._live[(m.ns, m.id)] = (m, now_ns + lifetime_ns if lifetime_ns > 0 else None)

    def expire(self, now_ns: int) -> bool:
        """Drop the markers whose lifetime has passed; whether any went."""
        gone = [key for key, (_, until) in self._live.items() if until is not None and until <= now_ns]
        for key in gone:
            del self._live[key]
        return bool(gone)

    def lapsing(self) -> bool:
        """Whether any live marker has a lifetime."""
        return any(until is not None for _, until in self._live.values())

    def groups(self) -> dict[tuple[str, str], list[Marker]]:
        """The live markers by namespace and frame."""
        out: dict[tuple[str, str], list[Marker]] = defaultdict(list)
        for (ns, _), (m, _) in self._live.items():
            out[(ns, frame_name(m.header.frame_id))].append(m)
        return out


def subscribe_marker_array(ctx: RendererCtx, d: AdapterDisplay, base: str) -> None:
    """Subscribe to a MarkerArray topic and log its markers under `base/<namespace>/<frame>/<kind>`."""
    store = MarkerStore()
    logged: set[str] = set()
    timer = None

    def now_ns() -> int:
        return ctx.node.get_clock().now().nanoseconds

    def publish() -> None:
        now: set[str] = set()
        for (ns, frame), markers in store.groups().items():
            for kind, archetype in _archetypes(markers).items():
                entity = f"{base}/{_safe(ns)}/{_safe(frame)}/{kind}"
                rr.log(entity, archetype, *in_frame(frame))
                now.add(entity)
        for entity in logged - now:
            rr.log(entity, rr.Clear(recursive=False))
        logged.clear()
        logged.update(now)

    def tick() -> None:
        if store.expire(now_ns()):
            publish()

    def cb(msg: MarkerArray) -> None:
        nonlocal timer
        t = now_ns()
        store.ingest(msg.markers, t)
        store.expire(t)
        publish()
        if timer is None and store.lapsing():
            timer = ctx.node.create_timer(EXPIRY_PERIOD_S, tick)

    if StyleSpec.from_json(d.style_json).latched:
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    else:
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
    ctx.node.create_subscription(MarkerArray, d.topic, cb, qos)


@register(DisplayKind.MARKER_ARRAY)
def render_marker_array(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    if not style.enabled:
        return
    subscribe_marker_array(ctx, d, display_path(ctx.env_id, robot, d.name))
