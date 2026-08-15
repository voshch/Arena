"""Keyboard panel: held keys to intent, QTimer to `Driver.tick`.

The camera is driven while the panel's window is active and released when it is
not, so there is nothing to take or hand back. Keys are read through an
application filter rather than widget focus, and skipped while a text field has
focus so the target box stays typeable. Auto-repeat is filtered, so a held key is
one press rather than a pulse train.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from python_qt_binding.QtCore import QEvent, Qt
from python_qt_binding.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from arena_cam import drive, fly
from arena_cam.drive import Driver, EntityRoster
from arena_cam.fly import Mode

if TYPE_CHECKING:
    import rclpy.node

    from arena_cam.surfaces import TargetSelection

_HELD_KEYS = {
    Qt.Key_W: "w",
    Qt.Key_A: "a",
    Qt.Key_S: "s",
    Qt.Key_D: "d",
    Qt.Key_Q: "q",
    Qt.Key_E: "e",
    Qt.Key_Left: "left",
    Qt.Key_Right: "right",
    Qt.Key_Up: "up",
    Qt.Key_Down: "down",
    Qt.Key_BracketLeft: "[",
    Qt.Key_BracketRight: "]",
}

_TAP_KEYS = {
    Qt.Key_Tab: "tab",
    Qt.Key_F: "f",
    Qt.Key_H: "h",
    Qt.Key_1: "1",
    Qt.Key_3: "3",
    Qt.Key_7: "7",
    Qt.Key_Space: "space",
    Qt.Key_P: "p",
}

# xkb keycodes (evdev + 8): the physical key, identical under any layout. The Qt keys
# above stay as the fallback for platforms that report no scancode.
_HELD_SCANCODES = {
    25: "w",
    38: "a",
    39: "s",
    40: "d",
    24: "q",
    26: "e",
    113: "left",
    114: "right",
    111: "up",
    116: "down",
    34: "[",
    35: "]",
}

_TAP_SCANCODES = {
    23: "tab",
    41: "f",
    43: "h",
    10: "1",
    12: "3",
    16: "7",
    65: "space",
    33: "p",
}

_TAB_ROW = {Mode.FLY: "switch to orbit", Mode.ORBIT: "switch to fly"}

_MODE_ROWS = {
    Mode.FLY: (("W / S", "dolly"), ("A / D", "truck"), ("Q / E", "down / up (world Z)"), ("arrows", "look")),
    Mode.ORBIT: (("W / S", "elevation"), ("A / D", "azimuth"), ("Q / E", "radius"), ("arrows", "pan focus")),
}

_COMMON_ROWS = (
    ("Shift / Ctrl", "boost / crawl"),
    ("[ / ]", "hold to widen / narrow fov"),
    ("F / Shift+F", "frame target / cycle reference mode"),
    ("H", "back to world"),
    ("1 / 3 / 7", "front / right / top"),
    ("Space", "brake, drop the carried momentum"),
    ("P", "still capture"),
)

_LEAD_RANGE_MS = (10, 200)
_HEADER_STYLE = "color: #7a7a7a;"
_ACTIVATE_EVENTS = (QEvent.ApplicationActivate, QEvent.WindowActivate)
_DEACTIVATE_EVENTS = (QEvent.ApplicationDeactivate, QEvent.WindowDeactivate)


class Panel(QWidget):
    """Camera readout, target picker and speed knobs around the keyboard driving."""

    def __init__(self, selection: TargetSelection) -> None:
        super().__init__()
        self._selection = selection
        self._driver: Driver | None = None
        self._roster: EntityRoster | None = None
        self._last = time.monotonic()
        self._since_discovery = 0.0
        self._keys: set[str] = set()
        self._boost = False
        self._crawl = False
        self._active = False
        QApplication.instance().installEventFilter(self)

        self.mode_label = QLabel("fly")
        self._tab_label = QLabel("")
        self._mode_labels = [(QLabel(""), QLabel("")) for _ in _MODE_ROWS[Mode.FLY]]
        self._shown_mode: Mode | None = None
        self.summary_label = QLabel("no viewport cameras found")
        self.status_label = QLabel("")

        self.entity = QComboBox()
        self.entity.setEditable(True)
        self.entity.setMinimumWidth(220)
        self.reference_mode = QComboBox()
        self.reference_mode.addItems(drive.REFERENCE_MODES)
        self.frame_button = QPushButton("Frame (F)")
        self.frame_button.clicked.connect(self._on_frame)
        self.world_button = QPushButton("World (H)")
        self.world_button.clicked.connect(self._on_world)

        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.2, 50.0)
        self.speed.setSingleStep(0.5)
        self.speed.setSuffix(" m/s")
        self.speed.valueChanged.connect(self._on_speed)

        self.lead = QSlider(Qt.Horizontal)
        self.lead.setRange(*_LEAD_RANGE_MS)
        self.lead.setValue(int(drive.DRIVE_LEAD * 1000))
        self.lead.valueChanged.connect(self._on_lead)
        self.lead_label = QLabel(f"{self.lead.value()} ms")

        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)

        head = QHBoxLayout()
        head.addWidget(QLabel("mode:"))
        head.addWidget(self.mode_label)
        head.addStretch(1)
        head.addWidget(self.summary_label)
        layout.addLayout(head)

        target = QHBoxLayout()
        target.addWidget(QLabel("target:"))
        target.addWidget(self.entity)
        target.addWidget(self.reference_mode)
        target.addWidget(self.frame_button)
        target.addWidget(self.world_button)
        target.addStretch(1)
        layout.addLayout(target)

        knobs = QHBoxLayout()
        knobs.addWidget(QLabel("speed:"))
        knobs.addWidget(self.speed)
        knobs.addSpacing(12)
        knobs.addWidget(QLabel("lead:"))
        knobs.addWidget(self.lead)
        knobs.addWidget(self.lead_label)
        knobs.addStretch(1)
        layout.addLayout(knobs)

        keymap = QGridLayout()
        tab_key = QLabel("Tab")
        tab_key.setStyleSheet(_HEADER_STYLE)
        self._tab_label.setStyleSheet(_HEADER_STYLE)
        keymap.addWidget(tab_key, 0, 0)
        keymap.addWidget(self._tab_label, 0, 1)
        for row, (key_label, action_label) in enumerate(self._mode_labels, start=1):
            keymap.addWidget(key_label, row, 0)
            keymap.addWidget(action_label, row, 1)
        for row, (key, action) in enumerate(_COMMON_ROWS, start=1 + len(self._mode_labels)):
            keymap.addWidget(QLabel(key), row, 0)
            keymap.addWidget(QLabel(action), row, 1)
        layout.addLayout(keymap)
        self._sync_keymap(Mode.FLY)

        layout.addWidget(self.status_label)

    # lifecycle ------------------------------------------------------------

    def attach(self, node: rclpy.node.Node) -> None:
        self._driver = Driver(node, self._selection)
        self._roster = EntityRoster(node)
        self.speed.setValue(self._driver.fly.speed)
        self.reference_mode.setCurrentText(self._driver.reference_mode)
        self._set_active(self.isActiveWindow())

    def detach(self) -> None:
        if self._driver is not None:
            self._driver.release()

    def command(self, label: str) -> None:
        if self._driver is None:
            return
        if label in ("f", "shift+f"):
            self._driver.entity = self.entity.currentText().strip()
        self._driver.command(label)
        self.reference_mode.setCurrentText(self._driver.reference_mode)

    # input ----------------------------------------------------------------

    def eventFilter(self, obj: object, event: object) -> bool:  # noqa: N802
        kind = event.type()
        if kind in _ACTIVATE_EVENTS or kind in _DEACTIVATE_EVENTS:
            self._set_active(kind in _ACTIVATE_EVENTS)
            return False
        if kind == QEvent.KeyPress:
            return self._on_key(event, pressed=True)
        if kind == QEvent.KeyRelease:
            return self._on_key(event, pressed=False)
        return False

    def _on_key(self, event: object, pressed: bool) -> bool:
        if self._driver is None or not self._active or self._typing():
            return False
        if event.isAutoRepeat():
            return True
        mods = event.modifiers()
        self._boost = bool(mods & Qt.ShiftModifier)
        self._crawl = bool(mods & Qt.ControlModifier)
        scan = event.nativeScanCode()
        held_map, tap_map = (_HELD_SCANCODES, _TAP_SCANCODES) if scan else (_HELD_KEYS, _TAP_KEYS)
        code = scan or event.key()
        held = held_map.get(code)
        if held is not None:
            if pressed:
                self._keys.add(held)
            else:
                self._keys.discard(held)
            return True
        tap = tap_map.get(code)
        if tap is not None:
            if pressed:
                self.command("shift+f" if tap == "f" and mods & Qt.ShiftModifier else tap)
            return True
        return False

    def _typing(self) -> bool:
        widget = QApplication.focusWidget()
        if isinstance(widget, QComboBox):
            return widget.isEditable()
        return isinstance(widget, (QLineEdit, QAbstractSpinBox))

    def _set_active(self, active: bool) -> None:
        self._active = active
        if self._driver is None:
            return
        if active:
            self._driver.engage()
            return
        self._keys.clear()
        self._boost = self._crawl = False
        self._driver.release()

    # tick -----------------------------------------------------------------

    def tick(self) -> None:
        now = time.monotonic()
        dt = min(0.1, max(1e-3, now - self._last))
        self._last = now
        if self._driver is None:
            return

        self._since_discovery += dt
        if self._since_discovery >= drive.REDISCOVER_S:
            self._since_discovery = 0.0
            self._driver.rediscover()
            if self._roster is not None:
                self._roster.refresh()
                self._sync_entities(self._roster.names())

        self._driver.tick(dt, fly.intent_from_keys(self._keys, self._boost, self._crawl))
        self._sync_keymap(self._driver.fly.mode)
        self.summary_label.setText(self._driver.summary())
        self.status_label.setText(self._driver.status)

    def _sync_keymap(self, mode: Mode) -> None:
        """Show one mode's bindings at a time, so the grid reads as the current control set."""
        if mode is self._shown_mode:
            return
        self._shown_mode = mode
        self.mode_label.setText(mode.value)
        self._tab_label.setText(_TAB_ROW[mode])
        for (key_label, action_label), (key, action) in zip(self._mode_labels, _MODE_ROWS[mode], strict=True):
            key_label.setText(key)
            action_label.setText(action)

    def _sync_entities(self, names: list[str]) -> None:
        current = self.entity.currentText()
        if names == [self.entity.itemText(i) for i in range(self.entity.count())]:
            return
        self.entity.clear()
        self.entity.addItems(names)
        self.entity.setEditText(current)

    # widget callbacks -----------------------------------------------------

    def _on_frame(self) -> None:
        self.command("f")

    def _on_world(self) -> None:
        self.command("h")

    def _on_speed(self, value: float) -> None:
        if self._driver is not None:
            self._driver.fly.speed = float(value)

    def _on_lead(self, value: int) -> None:
        self.lead_label.setText(f"{value} ms")
        if self._driver is not None:
            self._driver.lead = value / 1000.0
