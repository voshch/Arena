"""One stable color per robot, shared by its handles and its path displays."""

from __future__ import annotations

from collections.abc import Collection

PALETTE: tuple[tuple[int, int, int], ...] = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (227, 119, 194),
    (23, 190, 207),
    (188, 189, 34),
)


class RobotColors:
    """Palette slot per robot name: the first free slot on first sight, kept until retain drops the name."""

    def __init__(self) -> None:
        self._slots: dict[str, int] = {}

    def rgb(self, name: str) -> tuple[int, int, int]:
        slot = self._slots.get(name)
        if slot is None:
            used = set(self._slots.values())
            slot = next((i for i in range(len(PALETTE)) if i not in used), len(self._slots) % len(PALETTE))
            self._slots[name] = slot
        return PALETTE[slot]

    def rgba(self, name: str, alpha: float) -> tuple[float, float, float, float]:
        r, g, b = self.rgb(name)
        return (r / 255, g / 255, b / 255, alpha)

    def retain(self, names: Collection[str]) -> None:
        self._slots = {name: slot for name, slot in self._slots.items() if name in names}
