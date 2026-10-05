"""Per-robot goal progress tracking across an episode, ROS-independent."""

from __future__ import annotations

import math

import attrs

STALL_RADIUS = 1.0


@attrs.define
class GoalProgress:
    """Start distance, running-minimum distance, and path length for one robot's active goal."""

    start_dist: float = 0.0
    min_dist: float = 0.0
    path_length: float = 0.0
    _last_xy: tuple[float, float] | None = attrs.field(default=None, init=False, repr=False)
    _anchor_xy: tuple[float, float] | None = attrs.field(default=None, init=False, repr=False)
    _anchor_t: float = attrs.field(default=0.0, init=False, repr=False)

    @property
    def closed_fraction(self) -> float:
        """Fraction of start distance closed so far; 1.0 when start distance was zero."""
        if self.start_dist <= 0.0:
            return 1.0
        return (self.start_dist - self.min_dist) / self.start_dist

    def stalled_for(self, t: float) -> float:
        """Sim seconds the robot has stayed within STALL_RADIUS of one spot."""
        return t - self._anchor_t

    def sample(self, xy: tuple[float, float], goal_xy: tuple[float, float], t: float) -> None:
        dist = math.hypot(xy[0] - goal_xy[0], xy[1] - goal_xy[1])
        if self._anchor_xy is None or math.hypot(xy[0] - self._anchor_xy[0], xy[1] - self._anchor_xy[1]) > STALL_RADIUS:
            self._anchor_xy = xy
            self._anchor_t = t
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

    def sample(self, name: str, xy: tuple[float, float], goal_xy: tuple[float, float], t: float) -> None:
        self._by_robot.setdefault(name, GoalProgress()).sample(xy, goal_xy, t)

    def longest_stall(self, t: float, goal_radius: float) -> float:
        """Longest stall among robots not yet within goal_radius, 0.0 if there are none."""
        return max((p.stalled_for(t) for p in self._by_robot.values() if p.min_dist > goal_radius), default=0.0)

    def least_progress(self) -> GoalProgress | None:
        """Tracked robot with the lowest closed fraction, or None if nothing was sampled."""
        if not self._by_robot:
            return None
        return min(self._by_robot.values(), key=lambda p: p.closed_fraction)
