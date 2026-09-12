"""``halt``: the ``point`` swing held at a fixed elevation on the target's bearing, azimuth-only tracking."""

from __future__ import annotations

import math

import numpy as np

from task_generator.simulators.human.pointing import skeleton as S
from task_generator.simulators.human.pointing.generator import direction_from_angles

from .point import PointGesture

HOLD_ELEVATION_DEG = 15.0


class HaltGesture(PointGesture):
    SLOT_HAND = {"halt_l": "l", "halt_r": "r"}

    def _aim(self, local: np.ndarray) -> tuple[np.ndarray, dict]:
        d = S.unit(np.asarray(local, dtype=float))
        azimuth = math.degrees(math.atan2(d[1], d[0]))
        return direction_from_angles(azimuth, HOLD_ELEVATION_DEG, deg=True), {"azimuth": azimuth, "elevation": HOLD_ELEVATION_DEG}
