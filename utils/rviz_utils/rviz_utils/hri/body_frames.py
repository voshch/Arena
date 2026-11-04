"""ROS4HRI joint frames of the bodies in `hri_producer`, by the rules of robot_state_publisher."""

from __future__ import annotations

import dataclasses
import logging
import math

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from urdf_parser_py import urdf as urdf_parser

from .urdf import body_urdf

_LOG = logging.getLogger(__name__)

PUBLISH_INTERVAL_NS = 50_000_000
_PLACEHOLDER = "__BODY__"


def _origin_matrix(origin: urdf_parser.Pose | None) -> np.ndarray:
    """4x4 homogeneous transform for a URDF joint origin (xyz + rpy)."""
    m = np.eye(4)
    if origin is None:
        return m
    r, p, y = origin.rpy or (0.0, 0.0, 0.0)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    m[:3, :3] = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    m[:3, 3] = origin.xyz or (0.0, 0.0, 0.0)
    return m


def _hip_to_foot_offset(robot: urdf_parser.URDF) -> float:
    """Distance to lift the hip-origin root so the lowest link frame sits at z=0.

    Forward kinematics with every joint at zero (the standing pose), composing
    only joint origin transforms.  The rig root `body` is the REP-155 hip frame,
    so the legs reach below it, this is how far down the lowest link extends.
    """
    root = robot.get_root()
    world: dict[str, np.ndarray] = {root: np.eye(4)}
    min_z = 0.0
    stack = [root]
    while stack:
        parent = stack.pop()
        for joint_name, child in robot.child_map.get(parent, ()):
            world[child] = world[parent] @ _origin_matrix(robot.joint_map[joint_name].origin)
            min_z = min(min_z, world[child][2, 3])
            stack.append(child)
    return -min_z


def _quat_from_rpy(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """x y z w of a URDF origin's fixed-axis roll, pitch, yaw."""
    cr, sr = math.cos(rpy[0] / 2.0), math.sin(rpy[0] / 2.0)
    cp, sp = math.cos(rpy[1] / 2.0), math.sin(rpy[1] / 2.0)
    cy, sy = math.cos(rpy[2] / 2.0), math.sin(rpy[2] / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _transform(parent: str, child: str, xyz: tuple[float, float, float], quat: tuple[float, float, float, float]) -> TransformStamped:
    tf = TransformStamped()
    tf.header.frame_id = parent
    tf.child_frame_id = child
    t, r = tf.transform.translation, tf.transform.rotation
    t.x, t.y, t.z = xyz
    r.x, r.y, r.z, r.w = quat
    return tf


@dataclasses.dataclass(frozen=True)
class _JointSpec:
    name: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    axis: tuple[float, float, float] | None
    prismatic: bool


@dataclasses.dataclass
class _MovingJoint:
    tf: TransformStamped
    xyz: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    axis: tuple[float, float, float]
    prismatic: bool


@dataclasses.dataclass
class _Body:
    urdf: str
    moving: dict[str, _MovingJoint] = dataclasses.field(default_factory=dict)
    fixed: list[TransformStamped] = dataclasses.field(default_factory=list)
    last_stamp_ns: int | None = None


def _joint_specs(robot: urdf_parser.URDF) -> list[_JointSpec]:
    specs = []
    for joint in robot.joints:
        origin = joint.origin
        xyz = tuple(origin.xyz) if origin is not None and origin.xyz else (0.0, 0.0, 0.0)
        quat = _quat_from_rpy(tuple(origin.rpy)) if origin is not None and origin.rpy else (0.0, 0.0, 0.0, 1.0)
        axis = None if joint.type == "fixed" else tuple(joint.axis)
        specs.append(_JointSpec(joint.name, joint.parent, joint.child, xyz, quat, axis, joint.type == "prismatic"))
    return specs


def _animate(body: _Body, specs: list[_JointSpec], body_id: str) -> None:
    """Build the body's transforms from the specs of the placeholder body."""
    for spec in specs:
        tf = _transform(spec.parent.replace(_PLACEHOLDER, body_id), spec.child.replace(_PLACEHOLDER, body_id), spec.xyz, spec.quat)
        if spec.axis is None:
            body.fixed.append(tf)
        else:
            body.moving[spec.name.replace(_PLACEHOLDER, body_id)] = _MovingJoint(tf, spec.xyz, spec.quat, spec.axis, spec.prismatic)


def _pose_moving(joint: _MovingJoint, position: float) -> None:
    """Set the joint's transform to its origin followed by `position` along or about its axis."""
    t, r = joint.tf.transform.translation, joint.tf.transform.rotation
    ax, ay, az = joint.axis
    qx, qy, qz, qw = joint.quat
    if joint.prismatic:
        dx, dy, dz = ax * position, ay * position, az * position
        ux, uy, uz = 2.0 * (qy * dz - qz * dy), 2.0 * (qz * dx - qx * dz), 2.0 * (qx * dy - qy * dx)
        t.x = joint.xyz[0] + dx + qw * ux + qy * uz - qz * uy
        t.y = joint.xyz[1] + dy + qw * uy + qz * ux - qx * uz
        t.z = joint.xyz[2] + dz + qw * uz + qx * uy - qy * ux
        return
    half = position / 2.0
    s, c = math.sin(half), math.cos(half)
    bx, by, bz = ax * s, ay * s, az * s
    r.x = qw * bx + qx * c + qy * bz - qz * by
    r.y = qw * by - qx * bz + qy * c + qz * bx
    r.z = qw * bz + qx * by - qy * bx + qz * c
    r.w = qw * c - qx * bx - qy * by - qz * bz


class BodyFrames:
    """URDF and foot offset of every assigned body, joint transforms of the first `max_bodies`."""

    def __init__(self, max_bodies: int = 32) -> None:
        self._max_bodies = max_bodies
        self._urdf = body_urdf(_PLACEHOLDER)
        robot = urdf_parser.URDF.from_xml_string(self._urdf)
        self._foot_offset = _hip_to_foot_offset(robot)
        self._specs = _joint_specs(robot)
        self._bodies: dict[str, _Body] = {}
        self._animated: set[str] = set()
        self._static_dirty = False
        self._over_cap = False

    def urdf_for(self, body_id: str) -> str | None:
        body = self._bodies.get(body_id)
        return body.urdf if body is not None else None

    def foot_offset(self, body_id: str) -> float:
        """Hip-to-floor lift for body_id, 0.0 if not assigned."""
        return self._foot_offset if body_id in self._bodies else 0.0

    def active_ids(self) -> frozenset[str]:
        return frozenset(self._bodies)

    def sync(self, current_ids: set[str]) -> None:
        """Forget ids no longer present, build new ids and hand free animation slots to waiting bodies."""
        for bid in self.active_ids() - current_ids:
            del self._bodies[bid]
            if bid in self._animated:
                self._animated.remove(bid)
                self._static_dirty = True
        for bid in current_ids - self.active_ids():
            self._bodies[bid] = _Body(self._urdf.replace(_PLACEHOLDER, bid))
        waiting = sorted(current_ids - self._animated)
        free = self._max_bodies - len(self._animated)
        for bid in waiting[:free]:
            _animate(self._bodies[bid], self._specs, bid)
            self._animated.add(bid)
            self._static_dirty = True
        over_cap = len(waiting) > free
        if over_cap and not self._over_cap:
            _LOG.warning("%d bodies over max_bodies=%d have no joint frames", len(waiting) - free, self._max_bodies)
        self._over_cap = over_cap

    def moving_transforms(self, body_id: str, positions: dict[str, float], stamp: Time) -> TFMessage | None:
        """The transforms of the joints in `positions`, None if not animated or within the 20 Hz publish interval."""
        if body_id not in self._animated:
            return None
        body = self._bodies[body_id]
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        last = body.last_stamp_ns
        if last is not None and 0 <= stamp_ns - last < PUBLISH_INTERVAL_NS:
            return None
        body.last_stamp_ns = stamp_ns
        transforms = []
        for name, position in positions.items():
            joint = body.moving.get(name)
            if joint is not None:
                _pose_moving(joint, position)
                joint.tf.header.stamp = stamp
                transforms.append(joint.tf)
        return TFMessage(transforms=transforms)

    def static_transforms(self, stamp: Time) -> TFMessage | None:
        """The fixed-joint transforms of the animated bodies, None unless they changed since the last call."""
        if not self._static_dirty:
            return None
        self._static_dirty = False
        transforms = [tf for bid in self._animated for tf in self._bodies[bid].fixed]
        for tf in transforms:
            tf.header.stamp = stamp
        return TFMessage(transforms=transforms)
