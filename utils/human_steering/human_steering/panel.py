"""Main panel widget: toolbar, roster rail, canvas, controls rail, status bar.

Owns the QTimer tick (plugin.py) and wires canvas tool signals to driver calls.
"""

from __future__ import annotations

import math
import os
import subprocess
from typing import TYPE_CHECKING

from python_qt_binding.QtCore import QEvent, Qt
from python_qt_binding.QtGui import QColor, QPainter, QPen
from python_qt_binding.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from human_steering import fk
from human_steering.canvas import PED_HELD_RING_RGB, Canvas, Tool
from human_steering.driver import RUN_THRESHOLD_MPS, Driver

try:
    from task_generator.simulators.human.gait import LIMITS, GaitGenerator
except ImportError:  # pragma: no cover - exercised only without a sourced ROS install
    GaitGenerator = None  # type: ignore[assignment,misc]
    LIMITS = ()  # type: ignore[assignment]

if TYPE_CHECKING:
    import rclpy.node

    from human_steering.driver import Namespaces, RosterStatus

_SLIDER_FACTOR = 1000.0
_SPEED_RANGE_MPS = (0.0, 3.0)
_STATE_LABELS = (("Auto", None), ("Idle", 0), ("Walking", 1), ("Running", 2))
_JOINT_GROUPS = (
    ("Torso / Head", ("r_waist", "y_waist", "waist", "r_spine", "y_spine", "spine", "r_chest", "y_chest", "chest", "r_head", "y_head", "p_head")),
    ("Left arm", ("l_y_collar", "l_p_collar", "l_y_shoulder", "l_p_shoulder", "l_r_shoulder", "l_elbow")),
    ("Right arm", ("r_y_collar", "r_p_collar", "r_y_shoulder", "r_p_shoulder", "r_r_shoulder", "r_elbow")),
    ("Left leg", ("l_y_hip", "l_p_hip", "l_r_hip", "l_knee", "l_y_ankle", "l_ankle")),
    ("Right leg", ("r_y_hip", "r_p_hip", "r_r_hip", "r_knee", "r_y_ankle", "r_ankle")),
)
_COLLAPSED_GROUPS = frozenset({"Left leg", "Right leg"})
_BACKEND_BANNER_TEXT = "this env exposes no human control endpoints"
_BASE_TITLE = "human_steering"
_NO_CLIPS_PLACEHOLDER = "no clips in bundle"

_ROSTER_RAIL_WIDTH = 170
_CONTROLS_RAIL_WIDTH = 460
_CANVAS_DEFAULT_WIDTH = 1210

_VALUE_DECIMALS = 3
_VALUE_STEP = 0.01
_ENGAGED_STYLE = "color: #cf7430; font-weight: bold;"
_DISENGAGED_STYLE = "color: #7a7a7a;"

_PILL_COLORS = {
    "IDLE": "#9aa0a6",
    "WALKING": "#79b87a",
    "RUNNING": "#79b87a",
    "TELEOP": "#e0954f",
}

_HELD_MARKER = "●"  # filled circle, matches the canvas held ring color
_HELD_NAME_STYLE = "color: rgb({}, {}, {});".format(*PED_HELD_RING_RGB)

_FK_PREVIEW_HEIGHT_PX = 140
_FK_STROKE_WIDTH_PX = 2.0
_FK_MARGIN_FRAC = 0.12
_FK_STROKE_RGB = (198, 202, 205)
_FK_BASELINE_RGB = (90, 94, 98)


class _JointRow(QWidget):
    """One pose joint: slider + editable value box, moving either engages it."""

    def __init__(self, joint: str, lo: float, hi: float, on_change: object, parent: object = None) -> None:
        super().__init__(parent)
        self.joint = joint
        self.engaged = False
        self._on_change = on_change

        self.name_label = QLabel(joint)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(lo * _SLIDER_FACTOR), int(hi * _SLIDER_FACTOR))
        self.slider.valueChanged.connect(self._slider_changed)
        self.value_box = QDoubleSpinBox()
        self.value_box.setDecimals(_VALUE_DECIMALS)
        self.value_box.setRange(lo, hi)
        self.value_box.setSingleStep(_VALUE_STEP)
        self.value_box.valueChanged.connect(self._box_changed)
        self.clear_button = QToolButton()
        self.clear_button.setText("x")
        self.clear_button.setToolTip(f"Release {joint} back to bus-tracking")
        self.clear_button.setVisible(False)
        self.clear_button.clicked.connect(self._clear_clicked)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.name_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_box)
        layout.addWidget(self.clear_button)
        self._restyle()

    def _slider_changed(self, raw: int) -> None:
        value = raw / _SLIDER_FACTOR
        self.value_box.blockSignals(True)
        self.value_box.setValue(value)
        self.value_box.blockSignals(False)
        self._engage(value)

    def _box_changed(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(value * _SLIDER_FACTOR))
        self.slider.blockSignals(False)
        self._engage(value)

    def _engage(self, value: float) -> None:
        self.engaged = True
        self.clear_button.setVisible(True)
        self._restyle()
        self._on_change(self.joint, True, value)

    def _clear_clicked(self) -> None:
        self.release()
        self._on_change(self.joint, False, self.slider.value() / _SLIDER_FACTOR)

    def release(self) -> None:
        """Release back to bus-tracking: called by the clear button or Clear pose."""
        self.engaged = False
        self.clear_button.setVisible(False)
        self._restyle()

    def set_live_value(self, value: float) -> None:
        """Read-only display update from the bus, ignored while engaged."""
        if self.engaged:
            return
        self.slider.blockSignals(True)
        self.slider.setValue(int(value * _SLIDER_FACTOR))
        self.slider.blockSignals(False)
        self.value_box.blockSignals(True)
        self.value_box.setValue(value)
        self.value_box.blockSignals(False)

    def _restyle(self) -> None:
        style = _ENGAGED_STYLE if self.engaged else _DISENGAGED_STYLE
        self.name_label.setStyleSheet(style)
        self.value_box.setStyleSheet(style)


class _CollapsibleSection(QWidget):
    """Arrow-toggle group header: show/hide only, no state semantics."""

    def __init__(self, title: str, content: QWidget, collapsed: bool, parent: object = None) -> None:
        super().__init__(parent)
        self._content = content
        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(False)
        self._toggle.setAutoRaise(True)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.clicked.connect(self._on_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toggle)
        layout.addWidget(content)
        self._render(collapsed)

    def _on_clicked(self) -> None:
        self._render(self._content.isVisible())

    def _render(self, collapsed: bool) -> None:
        self._content.setVisible(not collapsed)
        self._toggle.setArrowType(Qt.RightArrow if collapsed else Qt.DownArrow)


class _FKPreviewWidget(QWidget):
    """Fixed-height front/side FK stick-figure preview from fk.preview() segments."""

    def __init__(self, parent: object = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(_FK_PREVIEW_HEIGHT_PX)
        self._front: list[fk.Segment] = []
        self._side: list[fk.Segment] = []

    def set_segments(self, front: list[fk.Segment], side: list[fk.Segment]) -> None:
        self._front = front
        self._side = side
        self.update()

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        painter = QPainter(self)
        try:
            half_w = self.width() / 2.0
            self._draw_baseline(painter, 0.0, half_w)
            self._draw_baseline(painter, half_w, half_w)
            self._draw(painter, self._front, 0.0, half_w)
            self._draw(painter, self._side, half_w, half_w)
        finally:
            painter.end()

    def _draw_baseline(self, painter: QPainter, x_offset: float, panel_w: float) -> None:
        pen = QPen(QColor(*_FK_BASELINE_RGB), 1.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        y = self.height() * 0.88
        painter.drawLine(int(x_offset + panel_w * 0.1), int(y), int(x_offset + panel_w * 0.9), int(y))

    def _draw(self, painter: QPainter, segments: list[fk.Segment], x_offset: float, panel_w: float) -> None:
        if not segments:
            return
        us = [p[0] for seg in segments for p in seg]
        vs = [p[1] for seg in segments for p in seg]
        u_min, u_max = min(us), max(us)
        v_min, v_max = min(vs), max(vs)
        u_span = max(u_max - u_min, 1e-6)
        v_span = max(v_max - v_min, 1e-6)
        avail_w = panel_w * (1.0 - 2.0 * _FK_MARGIN_FRAC)
        avail_h = self.height() * (1.0 - 2.0 * _FK_MARGIN_FRAC)
        scale = min(avail_w / u_span, avail_h / v_span)
        cx = x_offset + panel_w / 2.0 - (u_min + u_max) / 2.0 * scale
        cy = self.height() / 2.0 + (v_min + v_max) / 2.0 * scale
        pen = QPen(QColor(*_FK_STROKE_RGB), _FK_STROKE_WIDTH_PX)
        pen.setCosmetic(True)
        painter.setPen(pen)
        for (u0, v0), (u1, v1) in segments:
            painter.drawLine(
                int(cx + u0 * scale),
                int(cy - v0 * scale),
                int(cx + u1 * scale),
                int(cy - v1 * scale),
            )


class _RosterRow(QWidget):
    """One roster line: name + colored state pill + speed/waypoint-progress detail."""

    def __init__(self, parent: object = None) -> None:
        super().__init__(parent)
        self.name_label = QLabel()
        self.pill_label = QLabel()
        self.pill_label.setAlignment(Qt.AlignCenter)
        self.detail_label = QLabel()
        self.detail_label.setStyleSheet(_DISENGAGED_STYLE)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(self.name_label)
        layout.addWidget(self.pill_label)
        layout.addWidget(self.detail_label, 1)

    def set_status(self, name: str, status: RosterStatus, held: bool) -> None:
        self.name_label.setText(f"{_HELD_MARKER} {name}" if held else name)
        self.name_label.setStyleSheet(_HELD_NAME_STYLE if held else "")
        self.pill_label.setText(f" {status.state_label} ")
        color = _PILL_COLORS.get(status.state_label, "#9aa0a6")
        self.pill_label.setStyleSheet(
            f"background-color: {color}; color: #16181a; border-radius: 6px; font-size: 9px; font-weight: 600;",
        )
        detail = f"{status.speed:.1f} m/s"
        if status.waypoint_progress:
            detail = f"{detail} · {status.waypoint_progress}"
        self.detail_label.setText(detail)


class Panel(QWidget):
    """Toolbar, roster/canvas/controls splitter, status bar, ticks driver + canvas."""

    def __init__(self, parent: object = None, unlimited: bool = False) -> None:
        super().__init__(parent)
        self._unlimited = unlimited
        self._driver: Driver | None = None
        self._namespaces: Namespaces | None = None
        self._selected: str | None = None
        self._teleop_held: list[int] = []
        self._teleop_filter_installed = False
        self._instance_suffix = ""
        self.setWindowTitle(_BASE_TITLE)

        self.canvas = Canvas()
        self.canvas.ped_selected.connect(self._select)
        self.canvas.walk_to_requested.connect(self._on_walk_to)
        self.canvas.waypoint_added.connect(self._on_waypoint_added)
        self.canvas.teleport_requested.connect(self._on_teleport)
        self.canvas.gaze_requested.connect(self._on_gaze)
        self.canvas.stop_requested.connect(self._stop_ped)
        self.canvas.state_requested.connect(self._on_state_requested)
        self.canvas.play_clip_requested.connect(lambda name: self._play_clip(name))
        self.canvas.clear_joints_requested.connect(self._on_clear_joints)
        self.canvas.clear_gaze_requested.connect(self._on_clear_gaze)
        self.canvas.release_requested.connect(self._release_ped)

        self._build_toolbar()
        self._build_roster()
        self._build_drive_group()
        self._build_clips_group()
        self._build_pose_group()
        self.fk_preview = _FKPreviewWidget()
        self._build_status_bar()

        self.roster_list.setMinimumWidth(120)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumWidth(200)

        controls = QVBoxLayout()
        controls.addWidget(self.drive_group)
        controls.addWidget(self.clips_group)
        controls.addWidget(self.pose_group)
        controls.addWidget(QLabel("FK preview"))
        controls.addWidget(self.fk_preview)
        controls.addStretch(1)
        controls_widget = QWidget()
        controls_widget.setLayout(controls)

        controls_scroll = QScrollArea()
        controls_scroll.setWidget(controls_widget)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(380)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.roster_list)
        splitter.addWidget(self.canvas)
        splitter.addWidget(controls_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setCollapsible(1, False)
        splitter.setSizes([_ROSTER_RAIL_WIDTH, _CANVAS_DEFAULT_WIDTH, _CONTROLS_RAIL_WIDTH])

        root = QVBoxLayout(self)
        root.addWidget(self.toolbar)
        root.addWidget(self.backend_banner)
        root.addWidget(splitter, 1)
        root.addWidget(self.status_bar)

        self._set_controls_enabled(False)

    # -- construction helpers --

    def _build_toolbar(self) -> None:
        self.toolbar = QToolBar()
        self._tool_actions: dict[Tool, QAction] = {}
        for tool, label in (
            (Tool.SELECT, "Select"),
            (Tool.WALK_TO, "Walk to"),
            (Tool.WAYPOINT, "+ Waypoint"),
            (Tool.TELEPORT, "Teleport"),
            (Tool.TELEOP, "Teleop"),
            (Tool.GAZE, "Gaze"),
        ):
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(lambda _checked, t=tool: self._set_tool(t))
            self.toolbar.addAction(action)
            self._tool_actions[tool] = action
        self._tool_actions[Tool.SELECT].setChecked(True)

        self.toolbar.addSeparator()
        self.fit_action = QAction("Fit", self)
        self.fit_action.setShortcut(Qt.Key_F)
        self.fit_action.setToolTip("Fit view to content (F)")
        self.fit_action.triggered.connect(lambda: self.canvas.fit_view())
        self.toolbar.addAction(self.fit_action)

        self.state_combo = QComboBox()
        for label, _value in _STATE_LABELS:
            self.state_combo.addItem(label)
        self.state_combo.currentIndexChanged.connect(self._on_state_combo)
        self.toolbar.addWidget(self.state_combo)

        self.stop_button = QPushButton("STOP")
        self.stop_button.setToolTip("Kill motion, keep authorship: posed joints stay engaged")
        self.stop_button.clicked.connect(lambda: self._selected and self._stop_ped(self._selected))
        self.toolbar.addWidget(self.stop_button)

        self.release_button = QPushButton("Release")
        self.release_button.setToolTip("Hand the ped back entirely: clears mode, route, teleop, pose, clip, gaze")
        self.release_button.clicked.connect(lambda: self._selected and self._release_ped(self._selected))
        self.toolbar.addWidget(self.release_button)

        self.backend_banner = QLabel(_BACKEND_BANNER_TEXT)
        self.backend_banner.setStyleSheet("background-color: #a33; color: white; padding: 4px;")
        self.backend_banner.setVisible(False)

    def _build_roster(self) -> None:
        self.roster_list = QListWidget()
        self.roster_list.currentTextChanged.connect(self._select)
        self._roster_rows: dict[str, _RosterRow] = {}

    def _build_drive_group(self) -> None:
        self.drive_group = QGroupBox("Drive")
        lo, hi = _SPEED_RANGE_MPS
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(int(lo * _SLIDER_FACTOR), int(hi * _SLIDER_FACTOR))
        self.speed_slider.setValue(int(1.2 * _SLIDER_FACTOR))
        self.speed_label = QLabel("1.20 m/s")
        self.speed_slider.valueChanged.connect(self._on_speed_changed)

        layout = QVBoxLayout(self.drive_group)
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed"))
        speed_row.addWidget(self.speed_slider)
        speed_row.addWidget(self.speed_label)
        layout.addLayout(speed_row)

    def _build_clips_group(self) -> None:
        self.clips_group = QGroupBox("Clips")
        self.clip_combo = QComboBox()
        self.clip_combo.addItem(_NO_CLIPS_PLACEHOLDER)
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.stop_clip_button = QPushButton("Stop")
        self.stop_clip_button.setEnabled(False)
        self.play_button.clicked.connect(lambda: self._selected and self._play_clip(self._selected))
        self.stop_clip_button.clicked.connect(lambda: self._selected and self._driver and self._driver.stop_clip(self._selected))

        layout = QHBoxLayout(self.clips_group)
        layout.addWidget(self.clip_combo)
        layout.addWidget(self.play_button)
        layout.addWidget(self.stop_clip_button)

    def _build_pose_group(self) -> None:
        self.pose_group = QGroupBox("Pose")
        outer = QVBoxLayout(self.pose_group)

        header_row = QHBoxLayout()
        header_row.addStretch(1)
        self.clear_pose_button = QPushButton("Clear pose")
        self.clear_pose_button.clicked.connect(self._on_clear_pose_clicked)
        header_row.addWidget(self.clear_pose_button)
        outer.addLayout(header_row)

        self._joint_rows: dict[str, _JointRow] = {}
        joint_names = GaitGenerator.JOINT_NAMES if GaitGenerator is not None else ()
        for label, names in _JOINT_GROUPS:
            group_widget = QWidget()
            group_layout = QVBoxLayout(group_widget)
            group_layout.setContentsMargins(0, 0, 0, 0)
            for joint in names:
                if joint not in joint_names:
                    continue
                if self._unlimited:
                    lo, hi = 0.0, 2.0 * math.pi
                else:
                    lo, hi = LIMITS[joint_names.index(joint)] if LIMITS else (-3.14, 3.14)
                row = _JointRow(joint, lo, hi, self._on_joint_changed)
                self._joint_rows[joint] = row
                group_layout.addWidget(row)
            outer.addWidget(_CollapsibleSection(label, group_widget, label in _COLLAPSED_GROUPS))

    def _build_status_bar(self) -> None:
        self.status_bar = QStatusBar()
        self.stream_hz_label = QLabel("stream: -- Hz")
        self.backend_label = QLabel("human/move: unknown")
        self.map_label = QLabel("map: unknown")
        self.status_bar.addWidget(self.stream_hz_label)
        self.status_bar.addWidget(self.backend_label)
        self.status_bar.addWidget(self.map_label)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.drive_group.setEnabled(enabled)
        self.clips_group.setEnabled(enabled)
        self.pose_group.setEnabled(enabled)
        for action in self._tool_actions.values():
            action.setEnabled(enabled)
        self.state_combo.setEnabled(enabled)
        self.stop_button.setEnabled(enabled)
        self.release_button.setEnabled(enabled)

    # -- attach / detach --

    def attach(self, node: rclpy.node.Node, namespaces: Namespaces) -> None:
        self._namespaces = namespaces
        self._driver = Driver(node, namespaces, unlimited=self._unlimited)
        self.canvas.attach_ros(node, namespaces)
        self._refresh_window_title()

    def detach(self) -> None:
        self._uninstall_teleop_filter()
        self._teleop_held.clear()
        if self._driver is not None:
            self._driver.close()
            self._driver = None
        self.canvas.detach_ros()

    # -- window title --

    def set_instance_number(self, serial_number: int) -> None:
        """Fold rqt's multi-instance serial number into the window title."""
        self._instance_suffix = f" ({serial_number})" if serial_number > 1 else ""
        self._refresh_window_title()

    def _refresh_window_title(self) -> None:
        if self._namespaces is not None:
            ns_label = os.path.basename(self._namespaces.env_ns.rstrip("/")) or self._namespaces.env_ns
            title = f"{_BASE_TITLE} - {ns_label}"
        else:
            title = _BASE_TITLE
        self.setWindowTitle(f"{title}{self._instance_suffix}")

    # -- tool / roster wiring --

    def _set_tool(self, tool: Tool) -> None:
        for t, action in self._tool_actions.items():
            action.setChecked(t is tool)
        self.canvas.set_tool(tool)
        if tool == Tool.TELEOP:
            self._install_teleop_filter()
        else:
            self._uninstall_teleop_filter()
            self._teleop_held.clear()

    def _install_teleop_filter(self) -> None:
        if self._teleop_filter_installed:
            return
        app = QApplication.instance()
        if app is None:
            return
        app.installEventFilter(self)
        self._teleop_filter_installed = True

    def _uninstall_teleop_filter(self) -> None:
        if not self._teleop_filter_installed:
            return
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._teleop_filter_installed = False

    def _select(self, name: str | None) -> None:
        """Single source of truth for selection: roster clicks and canvas clicks both land here."""
        if name == "":
            return
        self._selected = name
        self.canvas.select(name)
        if self._driver is None:
            return
        if name is None:
            self.clip_combo.clear()
            self.clip_combo.addItem(_NO_CLIPS_PLACEHOLDER)
            self.play_button.setEnabled(False)
            self.stop_clip_button.setEnabled(False)
            return
        clips = self._driver.clip_inventory(name)
        self.clip_combo.clear()
        self.clip_combo.addItems(clips if clips else [_NO_CLIPS_PLACEHOLDER])
        self.play_button.setEnabled(bool(clips))
        self.stop_clip_button.setEnabled(bool(clips))
        self.canvas.update_waypoint_preview(name, self._driver.waypoints(name))

    def _on_walk_to(self, name: str, x: float, y: float) -> None:
        if self._driver is None:
            return
        speed = self.speed_slider.value() / _SLIDER_FACTOR
        self._driver.set_waypoints(name, [(x, y)], loop=False, speed=speed)

    def _on_waypoint_added(self, name: str, x: float, y: float) -> None:
        if self._driver is None:
            return
        speed = self.speed_slider.value() / _SLIDER_FACTOR
        self._driver.append_waypoint(name, (x, y), speed)
        self.canvas.update_waypoint_preview(name, self._driver.waypoints(name))

    def _on_teleport(self, name: str, x: float, y: float) -> None:
        if self._driver is not None:
            self._driver.teleport(name, x, y)

    def _on_gaze(self, name: str, x: float, y: float) -> None:
        if self._driver is not None:
            self._driver.set_gaze(name, (x, y, 0.0))

    def _stop_ped(self, name: str) -> None:
        if self._driver is not None:
            self._driver.stop(name)

    def _release_ped(self, name: str) -> None:
        if self._driver is not None:
            self._driver.release(name)
        if name == self._selected:
            self.roster_list.setCurrentRow(-1)
            self._select(None)

    def _on_state_requested(self, name: str, state: int) -> None:
        if self._driver is not None:
            self._driver.set_state_override(name, state)

    def _play_clip(self, name: str) -> None:
        if self._driver is not None and self.clip_combo.currentText():
            self._driver.start_clip(name, self.clip_combo.currentText())

    def _on_clear_joints(self, name: str) -> None:
        for row in self._joint_rows.values():
            row.release()
        if self._driver is not None:
            for joint in list(self._joint_rows):
                self._driver.disengage_joint(name, joint)

    def _on_clear_pose_clicked(self) -> None:
        if self._selected is not None:
            self._on_clear_joints(self._selected)

    def _on_clear_gaze(self, name: str) -> None:
        if self._driver is not None:
            self._driver.set_gaze(name, None)

    def _on_state_combo(self, index: int) -> None:
        if self._selected is None or self._driver is None:
            return
        _label, value = _STATE_LABELS[index]
        self._driver.set_state_override(self._selected, value)

    def _on_speed_changed(self, raw: int) -> None:
        self.speed_label.setText(f"{raw / _SLIDER_FACTOR:.2f} m/s")

    def _on_joint_changed(self, joint: str, engaged: bool, value: float) -> None:
        if self._selected is None or self._driver is None:
            return
        if engaged:
            self._driver.engage_joint(self._selected, joint, value)
        else:
            self._driver.disengage_joint(self._selected, joint)

    # -- Teleop tool: app-wide key capture, arrow keys never reach focused widgets --

    def eventFilter(self, obj: object, event: object) -> bool:  # noqa: N802
        """Swallow every arrow KeyPress/KeyRelease, including auto-repeats, so none reach the canvas."""
        if event.type() == QEvent.KeyPress and event.key() in _ARROW_TWIST:
            if not event.isAutoRepeat():
                if event.key() in self._teleop_held:
                    self._teleop_held.remove(event.key())
                self._teleop_held.append(event.key())
            return True
        if event.type() == QEvent.KeyRelease and event.key() in _ARROW_TWIST:
            if not event.isAutoRepeat() and event.key() in self._teleop_held:
                self._teleop_held.remove(event.key())
            return True
        if event.type() == QEvent.WindowDeactivate:
            self._teleop_held.clear()
        return super().eventFilter(obj, event)

    # -- tick: called by plugin.py's QTimer --

    def tick(self) -> None:
        if self._driver is None:
            self._update_status()
            return

        if self._teleop_held and self._selected is not None:
            vx = vy = wz = 0.0
            for key in self._teleop_held:  # press order, later keys win their axis
                kx, ky, kw = _ARROW_TWIST[key]
                vx, vy, wz = (kx or vx), (ky or vy), (kw or wz)
            speed = self.speed_slider.value() / _SLIDER_FACTOR
            if QApplication.queryKeyboardModifiers() & Qt.ShiftModifier:
                speed = max(2.0 * speed, RUN_THRESHOLD_MPS)
            self._driver.teleop_input(self._selected, vx * speed, vy * speed, wz)

        self._driver.tick()
        self.canvas.apply_pending()
        held = self._driver.held_names()
        self._refresh_roster(held)
        self.canvas.update_peds(self._driver.all_poses(), held)
        self._refresh_pose_rows()
        self._refresh_fk_preview()
        self._update_status()

    def _refresh_roster(self, held: set[str]) -> None:
        if self._driver is None:
            return
        names = self._driver.roster()
        if list(self._roster_rows) != names:
            self.roster_list.clear()
            self._roster_rows.clear()
            for name in names:
                row = _RosterRow()
                item = QListWidgetItem()
                item.setText(name)
                item.setSizeHint(row.sizeHint())
                self.roster_list.addItem(item)
                self.roster_list.setItemWidget(item, row)
                self._roster_rows[name] = row
        for name, row in self._roster_rows.items():
            row.set_status(name, self._driver.roster_status(name), name in held)

    def _refresh_pose_rows(self) -> None:
        if self._driver is None or self._selected is None:
            return
        angles = self._driver.current_joint_state(self._selected)
        if angles is None:
            return
        for joint, row in self._joint_rows.items():
            if joint in angles:
                row.set_live_value(angles[joint])

    def _refresh_fk_preview(self) -> None:
        if self._driver is None or self._selected is None:
            return
        angles = self._driver.current_joint_state(self._selected)
        if angles is None:
            return
        try:
            front, side = fk.preview(angles)
        except (RuntimeError, OSError, subprocess.CalledProcessError):
            return
        self.fk_preview.set_segments(front, side)

    def _update_status(self) -> None:
        if self._driver is None:
            self.backend_banner.setVisible(True)
            self._set_controls_enabled(False)
            return
        available = self._driver.human_move_available()
        self.backend_banner.setVisible(not available)
        self._set_controls_enabled(available)
        self.stream_hz_label.setText(f"stream: {self._driver.stream_hz:.1f} Hz")
        self.backend_label.setText(f"human/move: {'up' if available else 'absent'}")
        self.map_label.setText(f"map: {'ready' if self.canvas.has_map() else 'pending'}")


_ARROW_TWIST: dict[int, tuple[float, float, float]] = {
    Qt.Key_Up: (1.0, 0.0, 0.0),
    Qt.Key_Down: (-1.0, 0.0, 0.0),
    Qt.Key_Left: (0.0, 0.0, 1.0),
    Qt.Key_Right: (0.0, 0.0, -1.0),
    # WASD alternates: arrow pairs (e.g. Left+Down) ghost on many keyboard matrices
    Qt.Key_W: (1.0, 0.0, 0.0),
    Qt.Key_S: (-1.0, 0.0, 0.0),
    Qt.Key_A: (0.0, 0.0, 1.0),
    Qt.Key_D: (0.0, 0.0, -1.0),
}
