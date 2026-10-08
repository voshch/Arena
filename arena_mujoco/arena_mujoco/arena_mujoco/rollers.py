"""Passive rollers for wheels whose URDF declares grip along one ground direction only."""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

_GZ_EXPRESSED_IN = '{http://gazebosim.org/schema}expressed_in'

_ROLLERS = 12
_SPHERES = 2
_CENTER_RADIUS_SHARE = 0.28
_MASS_SHARE = 0.25

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RollerWheel:
    """A wheel link gripping only along `grip`, both vectors in the link frame at zero joint angles."""

    link: str
    grip: tuple[float, float, float]
    down: tuple[float, float, float]
    friction: float


def _rpy_matrix(rpy: str) -> np.ndarray:
    roll, pitch, yaw = (float(v) for v in rpy.split())
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _link_rotations(root: ET.Element) -> dict[str, np.ndarray]:
    """Rotation of every link frame against the root link with all joints at zero."""
    parents: dict[str, tuple[str, np.ndarray]] = {}
    for joint in root.findall('joint'):
        parent = joint.find('parent')
        child = joint.find('child')
        if parent is None or child is None:
            continue
        origin = joint.find('origin')
        rpy = '0 0 0' if origin is None else origin.attrib.get('rpy', '0 0 0')
        parents[child.attrib['link']] = (parent.attrib['link'], _rpy_matrix(rpy))

    rotations: dict[str, np.ndarray] = {}

    def resolve(link: str) -> np.ndarray:
        if link not in rotations:
            if link in parents:
                parent, local = parents[link]
                rotations[link] = resolve(parent) @ local
            else:
                rotations[link] = np.eye(3)
        return rotations[link]

    for link in root.findall('link'):
        resolve(link.attrib['name'])
    return rotations


def _vector(element: ET.Element) -> np.ndarray | None:
    raw = element.attrib.get('value') or element.text or ''
    try:
        values = [float(v) for v in raw.split()]
    except ValueError:
        return None
    if len(values) != 3 or not any(values):
        return None
    vector = np.array(values)
    return vector / np.linalg.norm(vector)


def _coefficient(gazebo: ET.Element, *tags: str) -> float | None:
    for tag in tags:
        for element in gazebo.iter(tag):
            raw = element.attrib.get('value') or element.text or ''
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def roller_wheels(root: ET.Element) -> list[RollerWheel]:
    """Links whose <gazebo> friction grips along fdir1 and slides freely across it, or the reverse."""
    rotations = _link_rotations(root)
    wheels: list[RollerWheel] = []
    for gazebo in root.findall('gazebo'):
        link = gazebo.attrib.get('reference')
        if link not in rotations:
            continue
        direction = next(gazebo.iter('fdir1'), None)
        if direction is None:
            continue
        along = _vector(direction)
        mu = _coefficient(gazebo, 'mu', 'mu1')
        mu2 = _coefficient(gazebo, 'mu2')
        if along is None or mu is None or mu2 is None or (mu > 0.0) == (mu2 > 0.0):
            continue
        frame = direction.attrib.get(_GZ_EXPRESSED_IN, link)
        if frame not in rotations:
            continue
        to_link = rotations[link].T @ rotations[frame]
        down = to_link @ np.array([0.0, 0.0, -1.0])
        grip = to_link @ along
        if mu2 > 0.0:
            grip = np.cross(down, grip)
        wheels.append(
            RollerWheel(
                link=link.replace('-', '_'),
                grip=(float(grip[0]), float(grip[1]), float(grip[2])),
                down=(float(down[0]), float(down[1]), float(down[2])),
                friction=max(mu, mu2),
            )
        )
    return wheels


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = 1.0 - c
    return np.array(
        [
            [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
            [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
            [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
        ]
    )


def roller_layout(
    radius: float,
    axle: np.ndarray,
    down: np.ndarray,
    grip: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]]:
    """Per roller its hinge origin, hinge axis and (offset along the axis, radius) of each sphere, all touching the wheel circle."""
    axle = axle / np.linalg.norm(axle)
    down = down - (down @ axle) * axle
    down = down / np.linalg.norm(down)
    forward = np.cross(axle, down)
    engaged = math.pi / _ROLLERS
    along_axle = float(grip @ axle)
    along_forward = float(grip @ forward) * engaged / math.sin(engaged)
    norm = math.hypot(along_axle, along_forward)
    along_axle /= norm
    along_forward /= norm
    bottom_axis = along_axle * axle + along_forward * forward
    pitch = radius * (1.0 - _CENTER_RADIUS_SHARE)
    step = 2.0 * math.pi / (_ROLLERS * _SPHERES)
    spheres: list[tuple[float, float]] = []
    for index in range(_SPHERES):
        angle = (index - (_SPHERES - 1) / 2.0) * step
        offset = pitch * math.tan(angle) / along_forward
        spheres.append((offset, radius - pitch / math.cos(angle)))
    layout = []
    for index in range(_ROLLERS):
        turn = _axis_rotation(axle, 2.0 * math.pi * index / _ROLLERS)
        layout.append((turn @ (pitch * down), turn @ bottom_axis, spheres))
    return layout


def _wheel_radius(colliders: list[mujoco.MjsGeom]) -> float | None:
    radii = [float(geom.size[0]) for geom in colliders if geom.type in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CYLINDER)]
    if not radii or len(radii) != len(colliders):
        return None
    return max(radii)


def _welded(body: mujoco.MjsBody) -> list[mujoco.MjsBody]:
    """Bodies rigidly fixed to body, itself included."""
    top = body
    while not top.joints and top.parent.name != 'world':
        top = top.parent
    group: list[mujoco.MjsBody] = []
    stack = [top]
    while stack:
        current = stack.pop()
        group.append(current)
        stack.extend(child for child in current.bodies if not child.joints)
    return group


def add_rollers(spec: mujoco.MjSpec, wheels: list[RollerWheel]) -> None:
    """Replace each wheel's own collision by free-spinning rollers around its rim."""
    for wheel in wheels:
        body = spec.body(wheel.link)
        if body is None or len(body.joints) != 1 or body.joints[0].type != mujoco.mjtJoint.mjJNT_HINGE:
            _logger.warning(f'{wheel.link}: directional friction ignored, not a link on one revolute joint')
            continue
        colliders = [geom for geom in body.geoms if geom.contype or geom.conaffinity]
        radius = _wheel_radius(colliders)
        grip = np.array(wheel.grip)
        down = np.array(wheel.down)
        axle = np.array(body.joints[0].axis, dtype=float)
        axle /= np.linalg.norm(axle)
        if radius is None or abs(float(grip @ np.cross(axle, down))) < 1e-3:
            _logger.warning(f'{wheel.link}: directional friction ignored, needs sphere or cylinder collision and a grip direction off the axle')
            continue

        for geom in colliders:
            geom.contype = 0
            geom.conaffinity = 0
        roller_mass = body.mass * _MASS_SHARE / _ROLLERS
        body.fullinertia = [value * (1.0 - _MASS_SHARE) for value in body.fullinertia]
        body.inertia = [value * (1.0 - _MASS_SHARE) for value in body.inertia]
        body.mass *= 1.0 - _MASS_SHARE

        shielded = [other.name for other in _welded(body.parent) if any(geom.contype or geom.conaffinity for geom in other.geoms)]
        for index, (origin, axis, spheres) in enumerate(roller_layout(radius, axle, down, grip)):
            name = f'{wheel.link}_roller_{index}'
            roller = body.add_body(name=name, pos=list(origin))
            roller.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(axis))
            for offset, sphere_radius in spheres:
                roller.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[sphere_radius, 0.0, 0.0],
                    pos=list(offset * axis),
                    mass=roller_mass / len(spheres),
                    friction=[wheel.friction, *list(colliders[0].friction)[1:]],
                    contype=0,
                    conaffinity=1,
                    group=colliders[0].group,
                    rgba=list(colliders[0].rgba),
                )
            for other in shielded:
                spec.add_exclude(bodyname1=name, bodyname2=other)


__all__ = ['RollerWheel', 'add_rollers', 'roller_layout', 'roller_wheels']
