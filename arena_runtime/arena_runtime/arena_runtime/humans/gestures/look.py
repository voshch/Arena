"""``look``: closed-form head aim (yaw, pitch) at a ped-local target."""

from __future__ import annotations

import math

import attrs
import numpy as np

from arena_runtime.humans.pointing import skeleton as S
from arena_runtime.humans.pointing.contract import LIMITS, ROS_JOINT_ORDER

from . import BODY_HEIGHT, GestureClip

FPS = 20.0
EASE_IN_S = 0.25
EASE_OUT_S = 0.3
HOLD_FRAMES = 20
HEAD = ("r_head", "y_head", "p_head")
FOLLOW_OMEGA_RAD_S = 7.0
PEAK_RAD_S = math.radians(200.0)
GAIT_HEAD_MAX_RAD = 0.3


@attrs.frozen
class LookHold:
    yaw: float
    pitch: float


def _ease(n: int) -> list[float]:
    return [0.5 - 0.5 * math.cos(math.pi * (i + 1) / n) for i in range(n)]


def _critical(x: float, v: float, target: float, dt: float) -> tuple[float, float]:
    """Exact ``dt`` step of a critically damped follower toward a target held over the step."""
    e = x - target
    c = v + FOLLOW_OMEGA_RAD_S * e
    k = math.exp(-FOLLOW_OMEGA_RAD_S * dt)
    return target + (e + c * dt) * k, (v - FOLLOW_OMEGA_RAD_S * c * dt) * k


class LookGesture:
    """Head triple only. The aim anchor is the rest-pose eye point (torso assumed upright, spine pose ignored)."""

    def __init__(self, height: float = BODY_HEIGHT) -> None:
        pos, _ = S.fk(dict.fromkeys(ROS_JOINT_ORDER, 0.0), S.Body(height))
        self.anchor = np.asarray(pos["eye"], dtype=float)

    def joints(self, side: str, moving: bool) -> set[str]:
        return set(HEAD)

    def bind(self, slot: str, local: np.ndarray, opts: dict) -> dict:
        return dict(opts)

    def breathing(self, side: str) -> dict[str, float]:
        return {}

    def aim(self, local: np.ndarray) -> LookHold:
        d = np.asarray(local, dtype=float) - self.anchor
        yaw = math.atan2(d[1], d[0])
        pitch = math.atan2(d[2], math.hypot(d[0], d[1]))
        return LookHold(yaw=self._clamp("y_head", yaw), pitch=self._clamp("p_head", pitch))

    @staticmethod
    def _clamp(name: str, value: float) -> float:
        lo, hi = LIMITS[name]
        return min(hi, max(lo, value))

    @staticmethod
    def _frame(yaw: float, pitch: float) -> dict:
        return {"angles": {"r_head": 0.0, "y_head": yaw, "p_head": pitch}}

    def _clip(self, start: LookHold, end: LookHold) -> GestureClip:
        n_in = max(1, int(round(EASE_IN_S * FPS)))
        frames = [self._frame(start.yaw + w * (end.yaw - start.yaw), start.pitch + w * (end.pitch - start.pitch)) for w in _ease(n_in)]
        frames += [self._frame(end.yaw, end.pitch) for _ in range(HOLD_FRAMES)]
        n_out = max(1, int(round(EASE_OUT_S * FPS)))
        release = [self._frame(end.yaw * (1.0 - w), end.pitch * (1.0 - w)) for w in _ease(n_out)]
        return GestureClip(frames=frames, fps=FPS, hold_start=n_in, hold_end=len(frames) - 1, side="", hold=end, report={"ok": True}, release=release)

    def start(self, local: np.ndarray, opts: dict) -> GestureClip:
        return self._clip(LookHold(0.0, 0.0), self.aim(local))

    def retarget(self, hold: object, local: np.ndarray, opts: dict) -> GestureClip:
        assert isinstance(hold, LookHold)
        return self._clip(hold, self.aim(local))

    def follower(self, local: np.ndarray) -> HeadFollower:
        aim = self.aim(local)
        return HeadFollower(yaw=aim.yaw, pitch=aim.pitch, aim=aim)


@attrs.define
class HeadFollower:
    """Critically damped head aim and overlay weight of a live look."""

    yaw: float
    pitch: float
    aim: LookHold
    engaged: float = 0.0
    weight: float = 0.0
    yaw_rate: float = 0.0
    pitch_rate: float = 0.0
    weight_rate: float = 0.0

    def step(self, dt: float) -> None:
        if dt <= 0.0:
            return
        yaw, yaw_rate = _critical(self.yaw, self.yaw_rate, self.aim.yaw, dt)
        pitch, pitch_rate = _critical(self.pitch, self.pitch_rate, self.aim.pitch, dt)
        swept = math.hypot(yaw - self.yaw, pitch - self.pitch)
        if swept > PEAK_RAD_S * dt:
            s = PEAK_RAD_S * dt / swept
            yaw, pitch, swept = self.yaw + (yaw - self.yaw) * s, self.pitch + (pitch - self.pitch) * s, PEAK_RAD_S * dt
        speed = math.hypot(yaw_rate, pitch_rate)
        if speed > PEAK_RAD_S:
            yaw_rate, pitch_rate = yaw_rate * PEAK_RAD_S / speed, pitch_rate * PEAK_RAD_S / speed
        w, w_rate = _critical(self.weight, self.weight_rate, self.engaged, dt)
        span = max(math.hypot(self.yaw, self.pitch), math.hypot(yaw, pitch)) + GAIT_HEAD_MAX_RAD
        budget = max(0.0, PEAK_RAD_S * dt - max(self.weight, w) * swept) / span
        if abs(w - self.weight) > budget:
            w = self.weight + math.copysign(budget, w - self.weight)
        w_rate = max(-budget / dt, min(budget / dt, w_rate))
        self.yaw, self.pitch, self.yaw_rate, self.pitch_rate = yaw, pitch, yaw_rate, pitch_rate
        self.weight, self.weight_rate = min(1.0, max(0.0, w)), w_rate

    def frame(self) -> dict:
        return LookGesture._frame(LookGesture._clamp("y_head", self.yaw), LookGesture._clamp("p_head", self.pitch))
