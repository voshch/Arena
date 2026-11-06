"""DisplayKind.MARKER_ARRAY renderer: generic visualization_msgs/MarkerArray → rerun.

Keeps the markers the array has added (by namespace and id) until it deletes them or their `lifetime` passes, so
incremental arrays work, and logs them per namespace and frame in that frame: cubes, spheres, cylinders, arrows, lines, points, cube and sphere
lists, triangle lists and text. Mesh resources are skipped. Makes no pedestrian-namespace assumptions, so it suits
debug overlays and static-geometry layers; `subscribe_marker_array` is the reusable subscriber.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import rclpy.node
import rerun as rr
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from rerun_utils.entity_paths import display_path, frame_name, in_frame
from rerun_utils.marker_geometry import convert
from rerun_utils.renderers._registry import RendererCtx, register

Group = tuple[str, str]

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
    arrived (on the node's clock, as rviz2 does), never with a zero lifetime.

    Remembers which namespace and frame groups changed since `take_dirty`.
    """

    def __init__(self) -> None:
        self._live: dict[tuple[str, int], tuple[Marker, int | None]] = {}
        self._dirty: set[Group] = set()

    @staticmethod
    def _group(m: Marker) -> Group:
        return (m.ns, frame_name(m.header.frame_id))

    def ingest(self, markers: Iterable[Marker], now_ns: int) -> None:
        for m in markers:
            if m.action == Marker.DELETEALL:
                self._dirty.update(self._group(live) for live, _ in self._live.values())
                self._live.clear()
            elif m.action == Marker.DELETE:
                gone = self._live.pop((m.ns, m.id), None)
                if gone is not None:
                    self._dirty.add(self._group(gone[0]))
            else:
                old = self._live.get((m.ns, m.id))
                if old is not None:
                    self._dirty.add(self._group(old[0]))
                lifetime_ns = m.lifetime.sec * 1_000_000_000 + m.lifetime.nanosec
                self._live[(m.ns, m.id)] = (m, now_ns + lifetime_ns if lifetime_ns > 0 else None)
                self._dirty.add(self._group(m))

    def expire(self, now_ns: int) -> bool:
        """Drop the markers whose lifetime has passed; whether any went."""
        gone = [key for key, (_, until) in self._live.items() if until is not None and until <= now_ns]
        for key in gone:
            self._dirty.add(self._group(self._live.pop(key)[0]))
        return bool(gone)

    def lapsing(self) -> bool:
        """Whether any live marker has a lifetime."""
        return any(until is not None for _, until in self._live.values())

    def groups(self) -> dict[Group, list[Marker]]:
        """The live markers by namespace and frame."""
        out: dict[Group, list[Marker]] = defaultdict(list)
        for m, _ in self._live.values():
            out[self._group(m)].append(m)
        return out

    def take_dirty(self) -> dict[Group, list[Marker]]:
        """The groups that changed since the last call with their live markers, empty for a group that lost them all."""
        live = self.groups()
        out = {group: live.get(group, []) for group in self._dirty}
        self._dirty.clear()
        return out


@dataclass(frozen=True)
class Entry:
    """One log call: `archetype` at `entity` in `frame`, or a clear of `entity` when there is none."""

    entity: str
    frame: str
    archetype: rr.AsComponents | None


class MarkerLog:
    """Turns changed groups into log entries under `base/<namespace>/<frame>/<kind>`, clearing the kinds a group lost."""

    def __init__(self, base: str) -> None:
        self._base = base
        self._kinds: dict[Group, set[str]] = {}

    def _entity(self, group: Group, kind: str) -> str:
        return f"{self._base}/{_safe(group[0])}/{_safe(group[1])}/{kind}"

    def entries(self, groups: dict[Group, list[Marker]]) -> list[Entry]:
        out: list[Entry] = []
        for group, markers in groups.items():
            archetypes = convert(markers).archetypes()
            out.extend(Entry(self._entity(group, kind), group[1], archetype) for kind, archetype in archetypes.items())
            out.extend(Entry(self._entity(group, kind), group[1], None) for kind in self._kinds.get(group, set()) - set(archetypes))
            if archetypes:
                self._kinds[group] = set(archetypes)
            else:
                self._kinds.pop(group, None)
        return out


class ExpiryTimer:
    """Checks for lapsed markers every `EXPIRY_PERIOD_S` once a marker with a lifetime has arrived."""

    def __init__(self, node: rclpy.node.Node, store: MarkerStore, on_expired: Callable[[], None]) -> None:
        self._node = node
        self._store = store
        self._on_expired = on_expired
        self._timer = None

    def now_ns(self) -> int:
        return self._node.get_clock().now().nanoseconds

    def arm(self) -> None:
        if self._timer is None and self._store.lapsing():
            self._timer = self._node.create_timer(EXPIRY_PERIOD_S, self._tick)

    def _tick(self) -> None:
        if self._store.expire(self.now_ns()):
            self._on_expired()


def subscribe_marker_array(ctx: RendererCtx, d: AdapterDisplay, base: str) -> None:
    """Subscribe to a MarkerArray topic and log its markers under `base/<namespace>/<frame>/<kind>`."""
    store = MarkerStore()
    log = MarkerLog(base)

    def flush() -> None:
        for entry in log.entries(store.take_dirty()):
            if entry.archetype is None:
                rr.log(entry.entity, rr.Clear(recursive=False))
            else:
                rr.log(entry.entity, entry.archetype, *in_frame(entry.frame))

    timer = ExpiryTimer(ctx.node, store, flush)

    def cb(msg: MarkerArray) -> None:
        now = timer.now_ns()
        store.ingest(msg.markers, now)
        store.expire(now)
        flush()
        timer.arm()

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
