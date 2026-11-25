"""QGraphicsView canvas: map/marker underlay, ped rendering, direct-manipulation tools.

ROS callbacks stash only. The QTimer tick's apply_pending() is the sole Qt-thread mutator.
"""

from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING

from python_qt_binding.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from python_qt_binding.QtGui import QBrush, QColor, QImage, QKeyEvent, QMouseEvent, QPainterPath, QPen, QPixmap, QWheelEvent
from python_qt_binding.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QMenu,
)

from human_steering import markers as static_markers

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    import rclpy.node
    from nav_msgs.msg import OccupancyGrid
    from rclpy.client import Client
    from rclpy.qos import QoSProfile
    from rclpy.subscription import Subscription
    from rclpy.task import Future
    from visualization_msgs.msg import Marker, MarkerArray

    from human_steering.driver import Namespaces
else:
    try:
        from nav_msgs.msg import OccupancyGrid
        from visualization_msgs.msg import MarkerArray
    except ImportError:  # pragma: no cover - exercised only without a sourced ROS install
        OccupancyGrid = MarkerArray = None

PED_RADIUS_M = 0.25
STATIC_MARKERS_PREFIX = "pedestrian_markers/static"

PED_MARKER_PX = 7.0
PED_PEN_WIDTH_PX = 1.4
PED_RING_WIDTH_PX = 3.0
PED_PEN_COLOR_RGB = (25, 25, 25)
PED_SELECTED_RING_RGB = (224, 149, 79)
PED_HELD_RING_RGB = (64, 196, 180)
WAYPOINT_NODE_PX = 5.0
WAYPOINT_COLOR_RGB = (220, 160, 40)
CROSSHAIR_PX = 10.0
CROSSHAIR_COLOR_RGB = (120, 124, 128)
CROSSHAIR_Z = -19.0
FIT_MARGIN_FRAC = 0.1
PAN_MARGIN_MIN_M = 20.0
ZOOM_STEP = 1.15
REQUIRE_MAP_RETRY_TICKS = 60  # ~3 s at the panel's 50 ms tick cadence
HINT_LABEL_STYLE = "background-color: rgba(20, 23, 25, 160); color: #9aa0a6; padding: 2px 6px; font-size: 10px;"
HINT_LABEL_MARGIN_PX = 6


class Tool(enum.Enum):
    SELECT = "select"
    WALK_TO = "walk_to"
    WAYPOINT = "waypoint"
    TELEPORT = "teleport"
    TELEOP = "teleop_tool"
    GAZE = "gaze"


_ARROW_KEYS = frozenset({Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right})


def _latched_qos() -> QoSProfile:
    import rclpy.qos

    return rclpy.qos.QoSProfile(
        reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
        durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
        history=rclpy.qos.HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def _occupancy_to_pixmap(grid: OccupancyGrid) -> QPixmap:
    """Grayscale render: unknown (-1) mid-gray, 0 white, 100 black."""
    width, height = grid.info.width, grid.info.height
    buf = bytearray(width * height)
    for i, cell in enumerate(grid.data):
        buf[i] = 128 if cell < 0 else max(0, min(255, 255 - round(cell * 2.55)))
    return QPixmap.fromImage(QImage(bytes(buf), width, height, width, QImage.Format_Grayscale8))


class Canvas(QGraphicsView):
    """Top-down 2D view: map/marker underlay, pedestrians, and direct-manipulation tools."""

    ped_selected = Signal(object)  # str | None, mouse-driven selection including clearing
    walk_to_requested = Signal(str, float, float)
    waypoint_added = Signal(str, float, float)
    teleport_requested = Signal(str, float, float)
    gaze_requested = Signal(str, float, float)
    stop_requested = Signal(str)
    state_requested = Signal(str, int)
    play_clip_requested = Signal(str)
    clear_joints_requested = Signal(str)
    clear_gaze_requested = Signal(str)
    release_requested = Signal(str)

    def __init__(self, parent: object = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.scale(1.0, -1.0)  # ROS x-right/y-up -> Qt y-down screen space
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self.tool = Tool.SELECT
        self._selected: str | None = None
        self._dragging_teleport = False
        self._auto_fit_done = False

        self._ped_items: dict[str, tuple[QGraphicsEllipseItem, QGraphicsLineItem]] = {}
        self._waypoint_items: dict[str, list[QGraphicsItem]] = {}
        self._gaze_items: dict[str, QGraphicsLineItem] = {}
        self._map_item: QGraphicsPixmapItem | None = None
        self._static_items: list[QGraphicsPathItem] = []
        self._static_markers: dict[str, dict[tuple[str, int], Marker]] = {}

        self._node: rclpy.node.Node | None = None
        self._namespaces: Namespaces | None = None
        self._pending_static: dict[str, MarkerArray] = {}
        self._pending_map: OccupancyGrid | None = None
        self._static_subs: list[Subscription] = []
        self._map_sub: Subscription | None = None
        self._map_client: Client | None = None
        self._ticks_since_map_request = 0

        self._build_background()
        self._hint_label = QLabel(self.viewport())
        self._hint_label.setStyleSheet(HINT_LABEL_STYLE)
        self._update_hint_label()
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def _build_background(self) -> None:
        """Origin crosshair so the canvas is never a blank void."""
        crosshair_pen = QPen(QColor(*CROSSHAIR_COLOR_RGB))
        crosshair_pen.setCosmetic(True)
        crosshair_pen.setWidthF(1.5)
        for line_args in ((-CROSSHAIR_PX, 0.0, CROSSHAIR_PX, 0.0), (0.0, -CROSSHAIR_PX, 0.0, CROSSHAIR_PX)):
            cross = QGraphicsLineItem(*line_args)
            cross.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            cross.setPen(crosshair_pen)
            cross.setZValue(CROSSHAIR_Z)
            self._scene.addItem(cross)

    # -- tools --

    def set_tool(self, tool: Tool) -> None:
        self.tool = tool
        self.setDragMode(QGraphicsView.ScrollHandDrag if tool == Tool.SELECT else QGraphicsView.NoDrag)

    def select(self, name: str | None) -> None:
        self._selected = name

    def has_map(self) -> bool:
        return self._map_item is not None

    # -- view fit / zoom --

    def fit_view(self) -> None:
        """Fit the view to map/static/ped bounds with a margin."""
        rect = self._content_bounds()
        if rect is None or rect.isEmpty():
            return
        mx, my = rect.width() * FIT_MARGIN_FRAC, rect.height() * FIT_MARGIN_FRAC
        self.fitInView(rect.adjusted(-mx, -my, mx, my), Qt.KeepAspectRatio)

    def _content_bounds(self) -> QRectF | None:
        rect: QRectF | None = None
        if self._map_item is not None:
            rect = self._map_item.sceneBoundingRect()
        for item in self._static_items:
            rect = item.sceneBoundingRect() if rect is None else rect.united(item.sceneBoundingRect())
        for body, _heading in self._ped_items.values():
            px, py = body.pos().x(), body.pos().y()
            point_rect = QRectF(px - PED_RADIUS_M, py - PED_RADIUS_M, 2 * PED_RADIUS_M, 2 * PED_RADIUS_M)
            rect = point_rect if rect is None else rect.united(point_rect)
        return rect

    def _update_scene_rect(self) -> None:
        """Inflate the scroll range one content-size past the content so panning is not map-bound."""
        rect = self._content_bounds()
        if rect is None or rect.isEmpty():
            return
        margin = max(rect.width(), rect.height(), PAN_MARGIN_MIN_M)
        self.setSceneRect(rect.adjusted(-margin, -margin, margin, margin))

    def _maybe_auto_fit(self) -> None:
        """One-shot fit once the viewport is sized and first content arrives."""
        if self._auto_fit_done:
            return
        if self.viewport().width() <= 0 or self.viewport().height() <= 0:
            return
        if self._map_item is None and not self._static_items and not self._ped_items:
            return
        self._auto_fit_done = True
        self.fit_view()

    # -- ROS attach: static markers always, map only after runtime/require_map --

    def attach_ros(self, node: rclpy.node.Node, namespaces: Namespaces) -> None:
        self._node = node
        self._namespaces = namespaces
        prefix = f"{namespaces.env_ns.rstrip('/')}/{STATIC_MARKERS_PREFIX}"
        for topic, _types in node.get_topic_names_and_types():
            if topic.startswith(prefix):
                sub = node.create_subscription(
                    MarkerArray,
                    topic,
                    lambda msg, t=topic: self._pending_static.__setitem__(t, msg),  # pyright: ignore[reportArgumentType]
                    _latched_qos(),
                )
                self._static_subs.append(sub)

        from std_srvs.srv import Trigger

        self._map_client = node.create_client(Trigger, f"{namespaces.node_ns}/runtime/require_map")
        self._request_map()

    def _request_map(self) -> None:
        """Fire one require_map Trigger, retried by _maybe_retry_require_map() until subscribed."""
        from std_srvs.srv import Trigger

        assert self._map_client is not None
        self._ticks_since_map_request = 0
        future = self._map_client.call_async(Trigger.Request())
        future.add_done_callback(self._on_map_ready)

    def _on_map_ready(self, future: Future) -> None:
        if self._map_sub is not None:
            return
        assert self._node is not None and self._namespaces is not None
        result = future.result()
        if result is None or not result.success:
            return
        self._map_sub = self._node.create_subscription(
            OccupancyGrid,
            self._namespaces.map_topic,
            lambda msg: setattr(self, "_pending_map", msg),
            _latched_qos(),
        )

    def _maybe_retry_require_map(self) -> None:
        if self._map_sub is not None or self._map_client is None:
            return
        self._ticks_since_map_request += 1
        if self._ticks_since_map_request >= REQUIRE_MAP_RETRY_TICKS:
            self._request_map()

    def detach_ros(self) -> None:
        if self._node is None:
            return
        for sub in self._static_subs:
            self._node.destroy_subscription(sub)
        self._static_subs.clear()
        if self._map_sub is not None:
            self._node.destroy_subscription(self._map_sub)
            self._map_sub = None
        if self._map_client is not None:
            self._node.destroy_client(self._map_client)
            self._map_client = None
        self._ticks_since_map_request = 0

    def apply_pending(self) -> None:
        """Drain stashed ROS messages into scene items. Call from the Qt tick."""
        content_changed = False
        if self._pending_map is not None:
            grid = self._pending_map
            self._pending_map = None
            content_changed = True
            if self._map_item is not None:
                self._scene.removeItem(self._map_item)
            map_item = self._scene.addPixmap(_occupancy_to_pixmap(grid))
            map_item.setZValue(-10.0)
            map_item.setScale(grid.info.resolution)
            map_item.setPos(grid.info.origin.position.x, grid.info.origin.position.y)
            self._map_item = map_item

        if self._pending_static:
            pending, self._pending_static = self._pending_static, {}
            content_changed = True
            for item in self._static_items:
                self._scene.removeItem(item)
            self._static_items.clear()
            for topic, msg in pending.items():
                static_markers.merge(self._static_markers.setdefault(topic, {}), msg.markers)
            pen = QPen(QColor(90, 90, 90))
            pen.setCosmetic(True)
            for cache in self._static_markers.values():
                for marker in cache.values():
                    for outline in static_markers.outlines(marker):
                        path = QPainterPath(QPointF(*outline[0]))
                        for point in outline[1:]:
                            path.lineTo(QPointF(*point))
                        item = QGraphicsPathItem(path)
                        item.setPen(pen)
                        item.setZValue(-5.0)
                        self._scene.addItem(item)
                        self._static_items.append(item)

        if content_changed:
            self._update_scene_rect()
        self._maybe_retry_require_map()
        self._maybe_auto_fit()
        self._update_hint_label()

    # -- ped rendering --

    def update_peds(self, poses: dict[str, tuple[float, float, float]], held: AbstractSet[str] = frozenset()) -> None:
        """poses: name -> (x, y, yaw), world frame. held: driver's claim set, previews and gaze markers are swept for peds outside it."""
        stale = set(self._ped_items) - set(poses)
        for name in stale:
            body, heading = self._ped_items.pop(name)
            self._scene.removeItem(body)
            self._scene.removeItem(heading)
        for name in [n for n in self._waypoint_items if n not in held]:
            self.update_waypoint_preview(name, [])
        for name in [n for n in self._gaze_items if n not in held]:
            self.update_gaze(name, None, None)

        for name, (x, y, yaw) in poses.items():
            if name not in self._ped_items:
                body = QGraphicsEllipseItem(-PED_MARKER_PX, -PED_MARKER_PX, 2 * PED_MARKER_PX, 2 * PED_MARKER_PX)
                body.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                body.setZValue(1.0)
                heading = QGraphicsLineItem(0.0, 0.0, PED_MARKER_PX * 1.8, 0.0)
                heading.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                heading_pen = QPen(QColor(*PED_PEN_COLOR_RGB), 2.0)
                heading_pen.setCosmetic(True)
                heading.setPen(heading_pen)
                heading.setZValue(1.1)
                self._scene.addItem(body)
                self._scene.addItem(heading)
                self._ped_items[name] = (body, heading)
            body, heading = self._ped_items[name]
            is_selected = name == self._selected
            body.setBrush(QBrush(QColor(220, 60, 60) if is_selected else QColor(60, 120, 220)))
            if is_selected:
                ring_width, ring_rgb = PED_RING_WIDTH_PX, PED_SELECTED_RING_RGB
            elif name in held:
                ring_width, ring_rgb = PED_RING_WIDTH_PX, PED_HELD_RING_RGB
            else:
                ring_width, ring_rgb = PED_PEN_WIDTH_PX, PED_PEN_COLOR_RGB
            pen = QPen(QColor(*ring_rgb), ring_width)
            pen.setCosmetic(True)
            body.setPen(pen)
            body.setPos(x, y)
            heading.setPos(x, y)
            heading.setRotation(-math.degrees(yaw))  # ItemIgnoresTransformations: rotation runs in screen (y-down) space

        self._maybe_auto_fit()
        self._update_hint_label()

    def update_waypoint_preview(self, name: str, points: list[tuple[float, float]]) -> None:
        for item in self._waypoint_items.pop(name, []):
            self._scene.removeItem(item)
        if not points:
            return
        items: list[QGraphicsItem] = []
        if len(points) >= 2:
            pen = QPen(QColor(*WAYPOINT_COLOR_RGB))
            pen.setCosmetic(True)
            for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
                line = QGraphicsLineItem(QLineF(x0, y0, x1, y1))
                line.setPen(pen)
                line.setZValue(0.5)
                self._scene.addItem(line)
                items.append(line)
        node_pen = QPen(QColor(*WAYPOINT_COLOR_RGB))
        node_pen.setCosmetic(True)
        for x, y in points:
            node = QGraphicsEllipseItem(-WAYPOINT_NODE_PX, -WAYPOINT_NODE_PX, 2 * WAYPOINT_NODE_PX, 2 * WAYPOINT_NODE_PX)
            node.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            node.setPen(node_pen)
            node.setBrush(QBrush(QColor(40, 42, 45)))
            node.setPos(x, y)
            node.setZValue(0.6)
            self._scene.addItem(node)
            items.append(node)
        self._waypoint_items[name] = items

    def update_gaze(self, name: str, origin: tuple[float, float] | None, target: tuple[float, float] | None) -> None:
        existing = self._gaze_items.pop(name, None)
        if existing is not None:
            self._scene.removeItem(existing)
        if origin is None or target is None:
            return
        line = QGraphicsLineItem(QLineF(origin[0], origin[1], target[0], target[1]))
        pen = QPen(QColor(40, 200, 120), 0)
        pen.setCosmetic(True)
        line.setPen(pen)
        line.setZValue(0.6)
        self._scene.addItem(line)
        self._gaze_items[name] = line

    # -- corner status hint --

    def _update_hint_label(self) -> None:
        map_status = "ready" if self._map_item is not None else "waiting"
        self._hint_label.setText(
            f"map: {map_status} · static: {len(self._static_items)} walls · peds: {len(self._ped_items)}",
        )
        self._hint_label.adjustSize()
        self._reposition_hint_label()

    def _reposition_hint_label(self) -> None:
        self._hint_label.move(
            HINT_LABEL_MARGIN_PX,
            self.viewport().height() - self._hint_label.height() - HINT_LABEL_MARGIN_PX,
        )

    # -- mouse / tools --

    def _ped_at(self, wx: float, wy: float) -> str | None:
        best: str | None = None
        best_dist = PED_RADIUS_M
        for name, (body, _heading) in self._ped_items.items():
            dx, dy = body.pos().x() - wx, body.pos().y() - wy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= best_dist:
                best, best_dist = name, dist
        return best

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = self.mapToScene(event.pos())
        wx, wy = pos.x(), pos.y()
        if event.button() == Qt.RightButton:
            ped = self._ped_at(wx, wy)
            if ped is not None:
                self._open_context_menu(ped, event.globalPos())
            return
        if self.tool == Tool.SELECT:
            ped = self._ped_at(wx, wy)
            self._selected = ped
            self.ped_selected.emit(ped)
        elif self.tool == Tool.WALK_TO and self._selected is not None:
            self.walk_to_requested.emit(self._selected, wx, wy)
        elif self.tool == Tool.WAYPOINT and self._selected is not None:
            self.waypoint_added.emit(self._selected, wx, wy)
        elif self.tool == Tool.TELEPORT and self._selected is not None:
            self._dragging_teleport = True
            self.teleport_requested.emit(self._selected, wx, wy)
        elif self.tool == Tool.GAZE and self._selected is not None:
            self.gaze_requested.emit(self._selected, wx, wy)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.tool == Tool.TELEPORT and self._dragging_teleport and self._selected is not None:
            pos = self.mapToScene(event.pos())
            self.teleport_requested.emit(self._selected, pos.x(), pos.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging_teleport = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        """Zoom anchored under the cursor."""
        factor = ZOOM_STEP if event.angleDelta().y() > 0 else 1.0 / ZOOM_STEP
        self.scale(factor, factor)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """Arrow keys never scroll the view, Teleop owns them via the app-wide filter."""
        if event.key() in _ARROW_KEYS:
            event.ignore()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event: object) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._reposition_hint_label()

    def _open_context_menu(self, name: str, global_pos: object) -> None:
        menu = QMenu(self)
        menu.addAction("Stop", lambda: self.stop_requested.emit(name))
        state_menu = menu.addMenu("State")
        for label, value in (("Idle", 0), ("Walking", 1), ("Running", 2)):
            state_menu.addAction(label, lambda v=value: self.state_requested.emit(name, v))
        menu.addAction("Play clip", lambda: self.play_clip_requested.emit(name))
        menu.addAction("Clear joints", lambda: self.clear_joints_requested.emit(name))
        menu.addAction("Clear gaze", lambda: self.clear_gaze_requested.emit(name))
        menu.addAction("Release", lambda: self.release_requested.emit(name))
        menu.exec_(global_pos)
