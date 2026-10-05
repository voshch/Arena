"""Whether a robot has parked: held one pose within a drift and turn bound, ROS-independent."""

from __future__ import annotations

import math

import attrs

PARK_DRIFT_M = 0.05
PARK_TURN_RAD = math.radians(5.0)


@attrs.define
class ParkTimer:
    """Sim seconds a robot has stayed within PARK_DRIFT_M and PARK_TURN_RAD of one pose."""

    _anchor: tuple[float, float, float] | None = None
    _since: float = 0.0

    def reset(self) -> None:
        self._anchor = None

    def held_for(self, x: float, y: float, yaw: float, t: float) -> float:
        anchor = self._anchor
        if anchor is None or math.hypot(x - anchor[0], y - anchor[1]) > PARK_DRIFT_M or abs((yaw - anchor[2] + math.pi) % (2 * math.pi) - math.pi) > PARK_TURN_RAD:
            self._anchor = (x, y, yaw)
            self._since = t
        return t - self._since
