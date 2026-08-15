"""Interactive camera motion: keyboard intent in, (position, quat, fov) out.

Keys set a target velocity per axis, never a pose delta, and each axis damps toward
it, so held keys accelerate and released keys coast down. FLY translates in the view
frame with world Z as up, never rolling; ORBIT moves on a sphere about a focus, the
same model the `orbit` verb uses, so a mode switch preserves the pose.

The mode reinterprets the fixed axes, which keeps one keymap for both:

| axis | FLY | ORBIT |
| --- | --- | --- |
| forward | dolly along the view axis | elevation over the focus |
| right | truck right | azimuth about the focus |
| up | rise along world Z | radius (multiplicative) |
| yaw / pitch | look | pan the focus, scaled by radius |
"""

from __future__ import annotations

import dataclasses
import enum
import math

from . import curves
from .curves import Quat, Vec3

# velocity damping time constant, ~63% of the gap to target per tau
TAU = 0.15

SPEED_MPS = 4.0
TURN_RATE = 1.4
ORBIT_RATE = 1.2
RADIUS_RATE = 1.0
# focus pan speed as a fraction of the radius per second, so far out pans faster
PAN_FRACTION = 0.6
FOV_RATE = 0.5

BOOST = 4.0
CRAWL = 0.25

# matches the tilt verb: short of vertical, where yaw goes singular
PITCH_LIMIT = 1.5533
MIN_RADIUS = 0.2
FOV_RANGE = (0.1, 2.6)
DEFAULT_FOV = 1.047

_WORLD_Z: Vec3 = (0.0, 0.0, 1.0)


# Held keys, as logical labels the input layer maps its own key codes onto.
AXES: dict[str, tuple[str, float]] = {
    "w": ("forward", 1.0),
    "s": ("forward", -1.0),
    "d": ("right", 1.0),
    "a": ("right", -1.0),
    "e": ("up", 1.0),
    "q": ("up", -1.0),
    "left": ("yaw", 1.0),
    "right": ("yaw", -1.0),
    "up": ("pitch", 1.0),
    "down": ("pitch", -1.0),
    "[": ("fov", -1.0),
    "]": ("fov", 1.0),
}

# Tapped keys, dispatched once per press by `drive.Driver.command`.
COMMANDS = ("tab", "f", "shift+f", "h", "1", "3", "7", "space", "p")

_INTENT_AXES = ("forward", "right", "up", "yaw", "pitch", "fov")


class Mode(enum.Enum):
    FLY = "fly"
    ORBIT = "orbit"


class View(enum.Enum):
    """Canonical framings, each named for the side the camera sits on."""

    FRONT = "front"
    RIGHT = "right"
    TOP = "top"


@dataclasses.dataclass
class Intent:
    """Held-key state as per-axis targets in [-1, 1] plus the speed modifiers."""

    forward: float = 0.0
    right: float = 0.0
    up: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    fov: float = 0.0
    boost: bool = False
    crawl: bool = False

    def scale(self) -> float:
        if self.boost:
            return BOOST
        if self.crawl:
            return CRAWL
        return 1.0


def intent_from_keys(keys: set[str], boost: bool = False, crawl: bool = False) -> Intent:
    """Sum the held axis keys into one intent, so opposing keys cancel."""
    intent = Intent(boost=boost, crawl=crawl)
    for key in keys:
        binding = AXES.get(key)
        if binding is None:
            continue
        axis, value = binding
        setattr(intent, axis, getattr(intent, axis) + value)
    for axis in _INTENT_AXES:
        setattr(intent, axis, max(-1.0, min(1.0, getattr(intent, axis))))
    return intent


@dataclasses.dataclass
class _Velocity:
    """Damped, unitless axis values, converted to rates by the active mode."""

    forward: float = 0.0
    right: float = 0.0
    up: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0


def _damp(current: float, target: float, dt: float, tau: float = TAU) -> float:
    if tau <= 0.0:
        return target
    return target + (current - target) * math.exp(-dt / tau)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _right_vec(yaw: float) -> Vec3:
    """The camera's right axis for a given yaw (gz body frame: +X forward, +Y left)."""
    return (math.sin(yaw), -math.cos(yaw), 0.0)


class Fly:
    """Interactive camera state: integrate `Intent` over time, read back a pose."""

    def __init__(self, speed: float = SPEED_MPS, default_fov: float = DEFAULT_FOV) -> None:
        self.mode = Mode.FLY
        self.speed = speed
        self.default_fov = default_fov
        self.pos: Vec3 = (0.0, 0.0, 2.0)
        # View elevation, positive looks up (the euler pitch is its negation).
        self.yaw = 0.0
        self.pitch = 0.0
        # 0 leaves the sim's own fov alone
        self.fov = 0.0
        self.focus: Vec3 = (0.0, 0.0, 0.0)
        self.radius = 5.0
        self.azimuth = 0.0
        self.elevation = 0.0
        self._vel = _Velocity()

    # state ----------------------------------------------------------------

    def forward_vec(self) -> Vec3:
        cp = math.cos(self.pitch)
        return (cp * math.cos(self.yaw), cp * math.sin(self.yaw), math.sin(self.pitch))

    def quat(self) -> Quat:
        return curves.quat_from_euler(0.0, -self.pitch, self.yaw)

    def pose(self) -> tuple[Vec3, Quat, float]:
        return self.pos, self.quat(), self.fov

    def seed(self, pos: Vec3, quat: Quat) -> None:
        """Adopt a live camera pose, keeping the focus the same distance ahead."""
        fwd = curves.quat_forward(quat)
        self.pos = pos
        self.yaw = math.atan2(fwd[1], fwd[0])
        self.pitch = _clamp(math.asin(_clamp(fwd[2], -1.0, 1.0)), -PITCH_LIMIT, PITCH_LIMIT)
        self.focus = curves.vadd(pos, curves.vscale(self.forward_vec(), self.radius))
        self._sync_orbit()

    def stop(self) -> None:
        self._vel = _Velocity()

    def set_mode(self, mode: Mode) -> None:
        if mode is self.mode:
            return
        self.mode = mode
        self.stop()
        if mode is Mode.ORBIT:
            self.focus = curves.vadd(self.pos, curves.vscale(self.forward_vec(), self.radius))
            self._sync_orbit()

    def toggle_mode(self) -> None:
        self.set_mode(Mode.FLY if self.mode is Mode.ORBIT else Mode.ORBIT)

    def frame(self, center: Vec3, radius: float | None = None) -> None:
        """Orbit `center`, either at `radius` or from wherever the camera already is."""
        self.focus = center
        distance = curves.vlen(curves.vsub(self.pos, center))
        self.radius = max(MIN_RADIUS, radius if radius is not None else distance)
        self.mode = Mode.ORBIT
        self.stop()
        self._sync_orbit()
        self._apply_orbit()

    def canonical(self, view: View) -> None:
        """Snap to a canonical framing of the current focus at the current radius."""
        self.mode = Mode.ORBIT
        self.stop()
        if view is View.FRONT:
            self.azimuth, self.elevation = 0.0, 0.0
        elif view is View.RIGHT:
            self.azimuth, self.elevation = math.pi / 2.0, 0.0
        else:
            self.azimuth, self.elevation = self.azimuth, PITCH_LIMIT
        self._apply_orbit()

    # integration ----------------------------------------------------------

    def tick(self, dt: float, intent: Intent) -> tuple[Vec3, Quat, float]:
        scale = intent.scale()
        self._vel.forward = _damp(self._vel.forward, intent.forward * scale, dt)
        self._vel.right = _damp(self._vel.right, intent.right * scale, dt)
        self._vel.up = _damp(self._vel.up, intent.up * scale, dt)
        self._vel.yaw = _damp(self._vel.yaw, intent.yaw * scale, dt)
        self._vel.pitch = _damp(self._vel.pitch, intent.pitch * scale, dt)

        if self.mode is Mode.FLY:
            self._tick_fly(dt)
        else:
            self._tick_orbit(dt)
        self._tick_fov(dt, intent, scale)
        return self.pose()

    def _tick_fly(self, dt: float) -> None:
        step = self.speed * dt
        right = _right_vec(self.yaw)
        move = curves.vscale(self.forward_vec(), self._vel.forward * step)
        move = curves.vadd(move, curves.vscale(right, self._vel.right * step))
        move = curves.vadd(move, curves.vscale(_WORLD_Z, self._vel.up * step))
        self.pos = curves.vadd(self.pos, move)
        self.yaw += self._vel.yaw * TURN_RATE * dt
        self.pitch = _clamp(self.pitch + self._vel.pitch * TURN_RATE * dt, -PITCH_LIMIT, PITCH_LIMIT)

    def _tick_orbit(self, dt: float) -> None:
        # turntable convention: D swings the subject to the right, so the camera goes left
        self.azimuth -= self._vel.right * ORBIT_RATE * dt
        self.elevation = _clamp(self.elevation + self._vel.forward * ORBIT_RATE * dt, -PITCH_LIMIT, PITCH_LIMIT)
        self.radius = max(MIN_RADIUS, self.radius * math.exp(self._vel.up * RADIUS_RATE * dt))
        pan = self.radius * PAN_FRACTION * dt
        right = _right_vec(self.azimuth + math.pi)
        self.focus = curves.vadd(self.focus, curves.vscale(right, -self._vel.yaw * pan))
        self.focus = curves.vadd(self.focus, curves.vscale(_WORLD_Z, self._vel.pitch * pan))
        self._apply_orbit()

    def _tick_fov(self, dt: float, intent: Intent, scale: float) -> None:
        if intent.fov == 0.0:
            return
        if self.fov <= 0.0:
            self.fov = self.default_fov
        self.fov = _clamp(self.fov + intent.fov * FOV_RATE * scale * dt, *FOV_RANGE)

    def _sync_orbit(self) -> None:
        """Derive the orbit angles from the current position. The radius is the caller's."""
        offset = curves.vsub(self.pos, self.focus)
        distance = curves.vlen(offset)
        if distance < MIN_RADIUS:
            return
        self.azimuth = math.atan2(offset[1], offset[0])
        self.elevation = math.asin(_clamp(offset[2] / distance, -1.0, 1.0))

    def _apply_orbit(self) -> None:
        """Place the camera on its sphere and aim it at the focus."""
        ce = math.cos(self.elevation)
        offset = (self.radius * ce * math.cos(self.azimuth), self.radius * ce * math.sin(self.azimuth), self.radius * math.sin(self.elevation))
        self.pos = curves.vadd(self.focus, offset)
        self.yaw = self.azimuth + math.pi
        self.pitch = -self.elevation
