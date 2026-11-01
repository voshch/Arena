"""Per-env interactive marker server: handle registry, drag coalescing, reset gating."""

from __future__ import annotations

import asyncio
import enum
import math
import threading
from collections.abc import Awaitable, Callable, Sequence

import attrs
from arena_rclpy_mixins import ArenaMixinNode
from geometry_msgs.msg import Pose as PoseMsg
from interactive_markers import InteractiveMarkerServer
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker, MenuEntry

PoseFn = Callable[[PoseMsg], Awaitable[PoseMsg | None]]
ActionFn = Callable[[], Awaitable[None]]

FRAME = "map"


class Apply(enum.Enum):
    LIVE = "live"
    RELEASE = "release"


@attrs.define
class Menu:
    title: str
    action: ActionFn | None = None
    children: Sequence[Menu] = ()


@attrs.define
class _Handle:
    marker: InteractiveMarker
    on_pose: PoseFn | None
    apply: Apply
    on_click: ActionFn | None
    actions: dict[int, ActionFn]
    dragging: bool = False
    busy: bool = False
    pending: PoseMsg | None = None


def _flatten(menu: Sequence[Menu], parent_id: int, entries: list[MenuEntry], actions: dict[int, ActionFn]) -> None:
    for item in menu:
        entry = MenuEntry(id=len(entries) + 1, parent_id=parent_id, title=item.title, command_type=MenuEntry.FEEDBACK)
        entries.append(entry)
        if item.action is not None:
            actions[entry.id] = item.action
        _flatten(item.children, entry.id, entries, actions)


def _z_axis_control(name: str, mode: int) -> InteractiveMarkerControl:
    control = InteractiveMarkerControl(name=name, interaction_mode=mode)
    half = math.sqrt(0.5)
    control.orientation.w = half
    control.orientation.y = half
    return control


def planar_marker(
    name: str,
    pose: PoseMsg,
    *,
    description: str = "",
    scale: float = 1.0,
    rotate: bool = True,
    visuals: Sequence[Marker] = (),
    frame: str = FRAME,
) -> InteractiveMarker:
    """Marker dragged in the ground plane, optionally with a yaw ring, drawing `visuals` on the drag control."""
    marker = InteractiveMarker(name=name, description=description, scale=scale, pose=pose)
    marker.header.frame_id = frame
    move = _z_axis_control("move", InteractiveMarkerControl.MOVE_PLANE)
    move.always_visible = True
    move.markers = list(visuals)
    controls = [move]
    if rotate:
        controls.append(_z_axis_control("yaw", InteractiveMarkerControl.ROTATE_AXIS))
    marker.controls = controls
    return marker


def static_marker(name: str, pose: PoseMsg, visuals: Sequence[Marker], *, description: str = "") -> InteractiveMarker:
    """Marker without drag controls, clickable and carrying a menu when the hub registers one."""
    marker = InteractiveMarker(name=name, description=description, pose=pose)
    marker.header.frame_id = FRAME
    control = InteractiveMarkerControl(name="body", interaction_mode=InteractiveMarkerControl.BUTTON, always_visible=True)
    control.markers = list(visuals)
    marker.controls = [control]
    return marker


def visual(kind: int, scale: tuple[float, float, float], color: tuple[float, float, float, float]) -> Marker:
    marker = Marker(type=kind)
    marker.scale.x, marker.scale.y, marker.scale.z = scale
    marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
    marker.pose.orientation.w = 1.0
    return marker


class MarkerHub:
    """Owns one InteractiveMarkerServer. Owners put markers from the event loop, feedback comes back on it."""

    def __init__(self, node: ArenaMixinNode, namespace: str, reset_lock: asyncio.Lock) -> None:
        self._node = node
        self._namespace = namespace
        self._reset_lock = reset_lock
        self._server = InteractiveMarkerServer(node, namespace)
        self._handles: dict[str, _Handle] = {}
        self._lock = threading.Lock()
        self._generation = 0
        self._logger = node.get_logger().get_child("markers")

    @property
    def namespace(self) -> str:
        return self._namespace

    def put(
        self,
        marker: InteractiveMarker,
        *,
        on_pose: PoseFn | None = None,
        apply: Apply = Apply.RELEASE,
        on_click: ActionFn | None = None,
        menu: Sequence[Menu] = (),
    ) -> None:
        entries: list[MenuEntry] = []
        actions: dict[int, ActionFn] = {}
        _flatten(menu, 0, entries, actions)
        marker.menu_entries = entries
        with self._lock:
            self._handles[marker.name] = _Handle(marker=marker, on_pose=on_pose, apply=apply, on_click=on_click, actions=actions)
        self._server.insert(marker, feedback_callback=self._on_feedback)
        self._server.applyChanges()

    def set_pose(self, name: str, pose: PoseMsg) -> None:
        """Move a marker from the owner's side, ignored while the user holds or applies it."""
        with self._lock:
            handle = self._handles.get(name)
            if handle is None or handle.dragging or handle.busy:
                return
            handle.marker.pose = pose
        self._server.setPose(name, pose)
        self._server.applyChanges()

    def erase(self, name: str) -> None:
        with self._lock:
            if self._handles.pop(name, None) is None:
                return
        self._server.erase(name)
        self._server.applyChanges()

    def erase_prefix(self, prefix: str) -> None:
        with self._lock:
            names = [n for n in self._handles if n.startswith(prefix)]
            for n in names:
                del self._handles[n]
        for n in names:
            self._server.erase(n)
        self._server.applyChanges()

    def names(self, prefix: str = "") -> list[str]:
        with self._lock:
            return [n for n in self._handles if n.startswith(prefix)]

    def clear(self) -> None:
        with self._lock:
            self._handles.clear()
            self._generation += 1
        self._server.clear()
        self._server.applyChanges()

    async def move(self, name: str, pose: PoseMsg) -> None:
        """Apply `pose` through the marker's owner, as a release would. Raises KeyError for unknown or fixed markers."""
        with self._lock:
            handle = self._handles.get(name)
            on_pose = handle.on_pose if handle is not None else None
        if on_pose is None:
            raise KeyError(name)
        async with self._reset_lock:
            settled = await on_pose(pose)
        self.set_pose(name, pose if settled is None else settled)

    def _on_feedback(self, feedback: InteractiveMarkerFeedback) -> None:
        event = feedback.event_type
        name = feedback.marker_name
        with self._lock:
            handle = self._handles.get(name)
            if handle is None:
                return
            generation = self._generation
            if event == InteractiveMarkerFeedback.MOUSE_DOWN:
                handle.dragging = True
                return
            if event == InteractiveMarkerFeedback.BUTTON_CLICK:
                action = handle.on_click
            elif event == InteractiveMarkerFeedback.MENU_SELECT:
                action = handle.actions.get(feedback.menu_entry_id)
            else:
                action = None
                if event == InteractiveMarkerFeedback.MOUSE_UP:
                    handle.dragging = False
                live = event == InteractiveMarkerFeedback.POSE_UPDATE and handle.apply is Apply.LIVE
                if handle.on_pose is None or not (live or event == InteractiveMarkerFeedback.MOUSE_UP):
                    return
                handle.pending = feedback.pose
                if handle.busy:
                    return
                handle.busy = True
        loop = self._node.event_loop
        if action is not None:
            asyncio.run_coroutine_threadsafe(self._run_action(action, generation), loop)
        elif event in (InteractiveMarkerFeedback.POSE_UPDATE, InteractiveMarkerFeedback.MOUSE_UP):
            asyncio.run_coroutine_threadsafe(self._drain(name, generation), loop)

    async def _run_action(self, action: ActionFn, generation: int) -> None:
        if self._reset_lock.locked():
            return
        async with self._reset_lock:
            if generation != self._generation:
                return
            try:
                await action()
            except Exception as e:
                self._logger.error(f"marker action failed: {e!r}")

    async def _drain(self, name: str, generation: int) -> None:
        while True:
            with self._lock:
                handle = self._handles.get(name)
                if handle is None:
                    return
                pose, handle.pending = handle.pending, None
                if pose is None:
                    handle.busy = False
                    return
                on_pose = handle.on_pose
                previous = handle.marker.pose
            applied = False
            settled: PoseMsg | None = None
            if on_pose is not None and not self._reset_lock.locked():
                async with self._reset_lock:
                    if generation != self._generation:
                        return
                    try:
                        settled = await on_pose(pose)
                        applied = True
                    except Exception as e:
                        self._logger.error(f"marker {name!r} rejected pose: {e!r}")
            with self._lock:
                current = self._handles.get(name)
                if current is not handle:
                    return
                if applied:
                    handle.marker.pose = pose if settled is None else settled
                    if settled is None:
                        continue
                elif handle.pending is not None:
                    continue
            self._server.setPose(name, previous if settled is None else settled)
            self._server.applyChanges()
