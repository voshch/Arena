"""Key tables for the drive panel: physical position first, Qt key as the fallback."""

from __future__ import annotations

from python_qt_binding.QtCore import Qt

# xkb keycodes (evdev + 8): the physical key, identical under any layout.
_SCANCODES = {
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
    23: "tab",
    41: "f",
    43: "h",
    10: "1",
    12: "3",
    16: "7",
    65: "space",
    33: "p",
}

# Layout-dependent Qt keys, for platforms that report no scancode or one outside the
# table (xorgxrdp numbers the arrows 98/100/102/104 rather than evdev's 111-116).
_KEYS = {
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
    Qt.Key_Tab: "tab",
    Qt.Key_F: "f",
    Qt.Key_H: "h",
    Qt.Key_1: "1",
    Qt.Key_3: "3",
    Qt.Key_7: "7",
    Qt.Key_Space: "space",
    Qt.Key_P: "p",
}


def label(event: object) -> str | None:
    """Logical label of a key event, None when it is not bound."""
    return _SCANCODES.get(event.nativeScanCode()) or _KEYS.get(event.key())
