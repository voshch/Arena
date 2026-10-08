"""Viewport camera drive model: each frame the camera world pose is reference * local, in body axes (forward +X, up +Z)."""

from __future__ import annotations

import math
from collections import deque

import attrs

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # w, x, y, z

IDENTITY: Quat = (1.0, 0.0, 0.0, 0.0)
ORIGIN: Vec3 = (0.0, 0.0, 0.0)

FULL = 0
YAW_ONLY = 1
POSITION_ONLY = 2

# must exceed the publisher's keyframe lead so the buffer plays out before releasing
STREAM_TIMEOUT = 0.4
MANUAL_POS_EPS = 0.01  # m
MANUAL_ANG_EPS = 0.01  # rad


def q_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def q_conj(q: Quat) -> Quat:
    w, x, y, z = q
    return (w, -x, -y, -z)


def q_rotate(q: Quat, v: Vec3) -> Vec3:
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def q_normalize(q: Quat) -> Quat:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return IDENTITY
    return (w / n, x / n, y / n, z / n)


def q_from_euler(roll: float, pitch: float, yaw: float) -> Quat:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def q_yaw(q: Quat) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def q_slerp(alpha: float, a: Quat, b: Quat) -> Quat:
    """Shortest-path slerp, lerping for nearly parallel inputs."""
    dot = sum(p * q for p, q in zip(a, b, strict=True))
    if dot < 0.0:
        b = (-b[0], -b[1], -b[2], -b[3])
        dot = -dot
    if dot > 0.9995:
        return q_normalize(tuple(p + alpha * (q - p) for p, q in zip(a, b, strict=True)))
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta = math.sin(theta)
    wa = math.sin((1.0 - alpha) * theta) / sin_theta
    wb = math.sin(alpha * theta) / sin_theta
    return q_normalize(tuple(wa * p + wb * q for p, q in zip(a, b, strict=True)))


def q_angle(a: Quat, b: Quat) -> float:
    """Absolute angle between two orientations, in radians."""
    dot = abs(sum(p * q for p, q in zip(a, b, strict=True)))
    return 2.0 * math.acos(max(0.0, min(1.0, dot)))


def look_at(eye: Vec3, target: Vec3) -> Quat:
    """Zero-roll orientation aiming the body +X axis from eye toward target."""
    dx, dy, dz = target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-9:
        return IDENTITY
    dx, dy, dz = dx / length, dy / length, dz / length
    return q_from_euler(0.0, -math.asin(max(-1.0, min(1.0, dz))), math.atan2(dy, dx))


@attrs.define
class Pose:
    position: Vec3 = ORIGIN
    orientation: Quat = IDENTITY


def reduce_reference(pose: Pose, mode: int) -> Pose:
    """Drop the reference rotation channels the caller did not ask to inherit."""
    if mode == YAW_ONLY:
        return Pose(pose.position, q_from_euler(0.0, 0.0, q_yaw(pose.orientation)))
    if mode == POSITION_ONLY:
        return Pose(pose.position, IDENTITY)
    return pose


def resolve_reference(entity: str, ref_pose: Pose, mode: int, target: Pose | None) -> Pose:
    """Reference world pose: a tracked entity once sampled, else ref_pose, else the world origin while the entity is unresolved."""
    if not entity:
        return reduce_reference(ref_pose, mode)
    if target is not None:
        return reduce_reference(target, mode)
    return Pose()


def compose_world(ref: Pose, local: Pose, world_orientation: bool) -> Pose:
    """Camera world pose as reference * local, keeping the aim world-stable when asked."""
    offset = q_rotate(ref.orientation, local.position)
    return Pose(
        (ref.position[0] + offset[0], ref.position[1] + offset[1], ref.position[2] + offset[2]),
        local.orientation if world_orientation else q_mul(ref.orientation, local.orientation),
    )


@attrs.define
class Keyframe:
    """A local-frame pose tagged with the time it is due."""

    time: float
    local: Pose
    world_orientation: bool
    fov: float  # <= 0 leaves the fov unchanged


@attrs.define
class Sample:
    local: Pose
    world_orientation: bool
    fov: float


def sample_buffer(buffer: deque[Keyframe], now: float) -> Sample | None:
    """Sample the keyframes at now, clamping outside the buffered span."""
    if not buffer:
        return None
    if now <= buffer[0].time:
        head = buffer[0]
        return Sample(head.local, head.world_orientation, head.fov)
    if now >= buffer[-1].time:
        tail = buffer[-1]
        return Sample(tail.local, tail.world_orientation, tail.fov)
    i = 1
    while i < len(buffer) and buffer[i].time < now:
        i += 1
    a, b = buffer[i - 1], buffer[i]
    span = b.time - a.time
    alpha = (now - a.time) / span if span > 1e-9 else 1.0
    position = tuple(pa + (pb - pa) * alpha for pa, pb in zip(a.local.position, b.local.position, strict=True))
    return Sample(
        Pose(position, q_slerp(alpha, a.local.orientation, b.local.orientation)),
        b.world_orientation,
        b.fov if b.fov > 0.0 else a.fov,
    )


@attrs.define
class Frame:
    """What the backend applies this frame. A pose of None releases the camera."""

    pose: Pose | None = None
    fov: float | None = None
    projection: str | None = None


class ViewportController:
    """Drive-model state machine: requests in from the ROS surface, a Frame out per tick."""

    def __init__(self) -> None:
        self._one_shot = False
        self._streaming = False
        self._local_set = False
        self._last_view = 0.0
        self._local = Pose()
        self._world_orientation = False
        self._buffer: deque[Keyframe] = deque()
        self._pending_fov: float | None = None
        self._pending_projection: str | None = None

        self._ref_entity = ""
        self._ref_pose = Pose()
        self._ref_mode = FULL
        self._ref_target: Pose | None = None

        self._applied: Pose | None = None
        self.released: tuple[float, float] | None = None

    @property
    def tracked_entity(self) -> str:
        return self._ref_entity

    def set_reference_target(self, pose: Pose | None) -> None:
        """Feed the tracked entity's world pose, sampled once per frame."""
        self._ref_target = pose

    def set_view(self, eye: Vec3, target: Vec3, fov: float) -> None:
        self._local = Pose(eye, look_at(eye, target))
        self._world_orientation = False
        self._one_shot = True
        self._streaming = False
        self._local_set = True
        if fov > 0.0:
            self._pending_fov = fov

    def set_local(self, local: Pose, world_orientation: bool, fov: float) -> None:
        """Snap to an explicit local pose, the capture path's set_view."""
        self._local = local
        self._world_orientation = world_orientation
        self._one_shot = True
        self._streaming = False
        self._local_set = True
        if fov > 0.0:
            self._pending_fov = fov

    def set_reference_frame(self, entity: str, pose: Pose, has_pose: bool, mode: int) -> str:
        if entity:
            self._ref_entity = entity
            self._ref_mode = mode
            self._ref_target = None
            return f"tracking {entity}"
        if has_pose:
            self._ref_pose = reduce_reference(pose, mode)
            self._ref_entity = ""
            self._ref_mode = FULL
            self._ref_target = None
            return "constant reference set"
        current = self._ref_pose if not self._ref_entity else reduce_reference(self._ref_target or self._ref_pose, self._ref_mode)
        self._ref_entity = ""
        self._ref_pose = current
        self._ref_mode = FULL
        self._ref_target = None
        return "latched current pose"

    def set_projection(self, projection: str) -> bool:
        if projection not in ("perspective", "orthographic"):
            return False
        self._pending_projection = projection
        return True

    def push_keyframe(self, keyframe: Keyframe, now: float) -> None:
        self._buffer.append(keyframe)
        self._streaming = True
        self._last_view = now
        self._local_set = True

    def apply(self, now: float, camera_pose: Pose) -> Frame:
        one_shot, self._one_shot = self._one_shot, False

        while len(self._buffer) > 1 and self._buffer[1].time <= now:
            self._buffer.popleft()

        if self._streaming and now - self._last_view > STREAM_TIMEOUT:
            self._streaming = False

        sampled_fov = 0.0
        if self._streaming and not one_shot:
            sample = sample_buffer(self._buffer, now)
            if sample:
                self._local, self._world_orientation, sampled_fov = sample.local, sample.world_orientation, sample.fov
        local, world_orientation = self._local, self._world_orientation

        fov, self._pending_fov = self._pending_fov, None
        if fov is None and sampled_fov > 0.0:
            fov = sampled_fov
        projection, self._pending_projection = self._pending_projection, None

        follow_only = self._ref_target is not None and self._local_set and not self._streaming and not one_shot
        if follow_only and self._applied is not None and self._moved(camera_pose, self._applied):
            self.released = (math.dist(camera_pose.position, self._applied.position), q_angle(camera_pose.orientation, self._applied.orientation))
            self._ref_entity = ""
            self._ref_pose = Pose()
            self._ref_mode = FULL
            self._ref_target = None

        ref = resolve_reference(self._ref_entity, self._ref_pose, self._ref_mode, self._ref_target)

        if one_shot or self._streaming or (self._ref_target is not None and self._local_set):
            self._applied = compose_world(ref, local, world_orientation)
        else:
            self._applied = None
        return Frame(self._applied, fov, projection)

    @staticmethod
    def _moved(a: Pose, b: Pose) -> bool:
        offset = math.dist(a.position, b.position)
        return offset > MANUAL_POS_EPS or q_angle(a.orientation, b.orientation) > MANUAL_ANG_EPS
