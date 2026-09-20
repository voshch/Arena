from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("python_qt_binding")

from python_qt_binding.QtCore import Qt

from arena_cam.keys import label


@dataclasses.dataclass(frozen=True)
class _Event:
    scan: int
    code: int

    def nativeScanCode(self) -> int:
        return self.scan

    def key(self) -> int:
        return self.code


_ARROWS = {Qt.Key_Up: "up", Qt.Key_Left: "left", Qt.Key_Right: "right", Qt.Key_Down: "down"}
# X server keycodes for the arrow row: evdev (local X11 / XWayland) vs xfree86 (xorgxrdp).
_EVDEV = {111: Qt.Key_Up, 113: Qt.Key_Left, 114: Qt.Key_Right, 116: Qt.Key_Down}
_XFREE86 = {98: Qt.Key_Up, 100: Qt.Key_Left, 102: Qt.Key_Right, 104: Qt.Key_Down}


@pytest.mark.parametrize("arrows", [_EVDEV, _XFREE86], ids=["evdev", "xfree86"])
def test_arrows_under_both_scancode_numberings(arrows: dict[int, int]) -> None:
    for scan, key in arrows.items():
        assert label(_Event(scan, key)) == _ARROWS[key]


def test_no_scancode_falls_back_to_qt_key() -> None:
    assert label(_Event(0, Qt.Key_W)) == "w"
    assert label(_Event(0, Qt.Key_Space)) == "space"


def test_physical_position_wins_over_layout() -> None:
    # colemak types F on the qwerty E key: the held axis wins over the tap command
    assert label(_Event(26, Qt.Key_F)) == "e"


def test_unbound_is_none() -> None:
    assert label(_Event(200, Qt.Key_Z)) is None
