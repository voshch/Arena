"""Per-robot goal progress tracking across an episode, ROS-independent."""

from __future__ import annotations

import math

import attrs


@attrs.define
class GoalProgress:
    """Start distance, running-minimum distance, and path length for one robot's active goal."""

    start_dist: float = 0.0
    min_dist: float = 0.0
    path_length: float = 0.0
    _last_xy: tuple[float, float] | None = attrs.field(default=None, init=False, repr=False)

    @property
    def closed_fraction(self) -> float:
        """Fraction of start distance closed so far; 1.0 when start distance was zero."""
        if self.start_dist <= 0.0:
            return 1.0
        return (self.start_dist - self.min_dist) / self.start_dist

    def sample(self, xy: tuple[float, float], goal_xy: tuple[float, float]) -> None:
        dist = math.hypot(xy[0] - goal_xy[0], xy[1] - goal_xy[1])
        if self._last_xy is None:
            self.start_dist = dist
            self.min_dist = dist
        else:
            self.min_dist = min(self.min_dist, dist)
            self.path_length += math.hypot(xy[0] - self._last_xy[0], xy[1] - self._last_xy[1])
        self._last_xy = xy


class GoalProgressTracker:
    """Per-robot GoalProgress keyed by robot name, reset once per episode."""

    def __init__(self) -> None:
        self._by_robot: dict[str, GoalProgress] = {}

    def reset(self) -> None:
        self._by_robot.clear()

    def sample(self, name: str, xy: tuple[float, float], goal_xy: tuple[float, float]) -> None:
        self._by_robot.setdefault(name, GoalProgress()).sample(xy, goal_xy)

    def least_progress(self) -> GoalProgress | None:
        """Tracked robot with the lowest closed fraction, or None if nothing was sampled."""
        if not self._by_robot:
            return None
        return min(self._by_robot.values(), key=lambda p: p.closed_fraction)
