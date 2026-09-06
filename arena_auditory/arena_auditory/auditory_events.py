from __future__ import annotations

import math
from collections.abc import Callable

from arena_people_msgs.msg import Pedestrian, Pedestrians


class AuditoryEventDetector:
    """Derive transient pedestrian sounds from the published pedestrian state."""

    def __init__(
        self,
        emit: Callable[[str, Pedestrian], None],
        *,
        walking_speed_threshold: float = 0.05,
        footstep_interval_sec: float = 0.45,
        greeting_distance_m: float = 1.5,
        greeting_fov_deg: float = 90.0,
        greeting_cooldown_sec: float = 5.0,
    ) -> None:
        self._emit = emit
        self._walking_speed_threshold = walking_speed_threshold
        self._footstep_interval_sec = footstep_interval_sec
        self._greeting_distance_m = greeting_distance_m
        self._greeting_fov_rad = math.radians(greeting_fov_deg)
        self._greeting_cooldown_sec = greeting_cooldown_sec

        self._last_footstep_by_ped: dict[int, float] = {}
        self._last_greeting_by_pair: dict[tuple[int, int], float] = {}
        self._last_pose_by_ped: dict[int, tuple[float, float, float]] = {}

    def update(self, pedestrians: Pedestrians, now_sec: float) -> None:
        peds = list(pedestrians.pedestrians)
        current_ids = {int(ped.id) for ped in peds}

        for ped in peds:
            reported_speed = math.hypot(
                ped.twist.linear.x,
                ped.twist.linear.y,
            )
            derived_speed = self._pose_delta_speed(ped, now_sec)
            if max(reported_speed, derived_speed) > self._walking_speed_threshold:
                self._maybe_emit_footstep(ped, now_sec)

        for stale_id in set(self._last_pose_by_ped) - current_ids:
            self._last_pose_by_ped.pop(stale_id, None)
            self._last_footstep_by_ped.pop(stale_id, None)

        for index, ped_a in enumerate(peds):
            for ped_b in peds[index + 1 :]:
                self._maybe_emit_greeting(ped_a, ped_b, now_sec)

    def _pose_delta_speed(
        self,
        ped: Pedestrian,
        now_sec: float,
    ) -> float:
        """Measure motion when a simulator/model does not populate twist."""
        ped_id = int(ped.id)
        x = float(ped.pose.position.x)
        y = float(ped.pose.position.y)
        previous = self._last_pose_by_ped.get(ped_id)
        self._last_pose_by_ped[ped_id] = (x, y, now_sec)
        if previous is None:
            return 0.0
        previous_x, previous_y, previous_time = previous
        elapsed = now_sec - previous_time
        if elapsed <= 1e-4:
            return 0.0
        return math.hypot(x - previous_x, y - previous_y) / elapsed

    def _maybe_emit_footstep(self, ped: Pedestrian, now_sec: float) -> None:
        last = self._last_footstep_by_ped.get(ped.id, -math.inf)
        if now_sec - last < self._footstep_interval_sec:
            return
        self._last_footstep_by_ped[ped.id] = now_sec
        self._emit("footstep", ped)

    def _maybe_emit_greeting(
        self,
        ped_a: Pedestrian,
        ped_b: Pedestrian,
        now_sec: float,
    ) -> None:
        emitter: Pedestrian | None = None
        if self._sees(ped_a, ped_b):
            emitter = ped_a
        elif self._sees(ped_b, ped_a):
            emitter = ped_b

        if emitter is None:
            return

        pair = tuple(sorted((ped_a.id, ped_b.id)))
        last = self._last_greeting_by_pair.get(pair, -math.inf)
        if now_sec - last < self._greeting_cooldown_sec:
            return
        self._last_greeting_by_pair[pair] = now_sec
        self._emit("greeting", emitter)

    def _sees(self, observer: Pedestrian, target: Pedestrian) -> bool:
        dx = target.pose.position.x - observer.pose.position.x
        dy = target.pose.position.y - observer.pose.position.y
        distance = math.hypot(dx, dy)
        if distance <= 1e-6 or distance > self._greeting_distance_m:
            return False

        target_angle = math.atan2(dy, dx)
        observer_yaw = self._yaw_from_quaternion(observer.pose.orientation)
        angle_error = math.atan2(
            math.sin(target_angle - observer_yaw),
            math.cos(target_angle - observer_yaw),
        )
        return abs(angle_error) <= self._greeting_fov_rad / 2.0

    @staticmethod
    def _yaw_from_quaternion(quaternion: object) -> float:
        siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)
