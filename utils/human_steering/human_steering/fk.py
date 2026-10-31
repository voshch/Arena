"""FK preview: the RViz body URDF of rviz_utils.hri.urdf, posed by composed semantic angles and projected front and side.

The FK math (forward_kinematics/project_front/project_side) is plain Python, only `preview()` needs rviz_utils.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping

try:
    from rviz_utils.hri.rig import semantic_positions
    from rviz_utils.hri.urdf import body_urdf
except ImportError:  # pragma: no cover - exercised only without a sourced ROS install
    semantic_positions = None
    body_urdf = None

PREVIEW_ID = "fk_preview"

_ZERO3 = (0.0, 0.0, 0.0)
_IDENTITY_R = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@dataclasses.dataclass(frozen=True)
class Joint:
    """One URDF joint: static origin (xyz/rpy) plus rotation axis for revolute/continuous joints."""

    name: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    kind: str


def _mat3_mul(a: tuple, b: tuple) -> tuple:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _mat3_vec(a: tuple, v: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = (sum(a[i][k] * v[k] for k in range(3)) for i in range(3))
    return (x, y, z)


def _vec3_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _rotation_rpy(rpy: tuple[float, float, float]) -> tuple:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _rotation_axis_angle(axis: tuple[float, float, float], angle: float) -> tuple:
    """Rodrigues' rotation formula about an arbitrary axis."""
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-9:
        return _IDENTITY_R
    x, y, z = x / norm, y / norm, z / norm
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return (
        (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    )


def _floats3(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not text:
        return default
    parts = [float(v) for v in text.split()]
    if len(parts) != 3:
        return default
    return (parts[0], parts[1], parts[2])


def parse_urdf(urdf_xml: str) -> tuple[str, dict[str, Joint]]:
    """Parse every <joint> (name/parent/child/origin/axis/type) out of a URDF string.

    Returns (root_link_name, joints_by_name). The root link is whichever link
    is never named as a joint's child.
    """
    root_elem = ET.fromstring(urdf_xml)  # noqa: S314
    joints: dict[str, Joint] = {}
    children: set[str] = set()
    all_links = {link.get("name", "") for link in root_elem.findall("link")}

    for joint_elem in root_elem.findall("joint"):
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            continue
        name = joint_elem.get("name", "")
        origin_elem = joint_elem.find("origin")
        axis_elem = joint_elem.find("axis")
        joint = Joint(
            name=name,
            parent=parent_elem.get("link", ""),
            child=child_elem.get("link", ""),
            xyz=_floats3(origin_elem.get("xyz") if origin_elem is not None else None, _ZERO3),
            rpy=_floats3(origin_elem.get("rpy") if origin_elem is not None else None, _ZERO3),
            axis=_floats3(axis_elem.get("xyz") if axis_elem is not None else None, (1.0, 0.0, 0.0)),
            kind=joint_elem.get("type", "fixed"),
        )
        joints[name] = joint
        children.add(joint.child)

    roots = all_links - children
    root = next(iter(roots)) if roots else next(iter(all_links), "")
    return root, joints


def forward_kinematics(
    root: str,
    joints: Mapping[str, Joint],
    angles: Mapping[str, float],
) -> dict[str, tuple[float, float, float]]:
    """World-frame position of every link. angles keys are URDF (already-suffixed)
    joint names, joints absent from angles (fixed joints, unposed DOF) contribute
    only their static origin."""
    by_parent: dict[str, list[Joint]] = {}
    for joint in joints.values():
        by_parent.setdefault(joint.parent, []).append(joint)

    positions: dict[str, tuple[float, float, float]] = {root: _ZERO3}
    frames: dict[str, tuple[tuple, tuple[float, float, float]]] = {root: (_IDENTITY_R, _ZERO3)}
    stack = [root]
    while stack:
        parent = stack.pop()
        parent_r, parent_t = frames[parent]
        for joint in by_parent.get(parent, ()):
            origin_r = _rotation_rpy(joint.rpy)
            origin_t = _vec3_add(_mat3_vec(parent_r, joint.xyz), parent_t)
            combined_r = _mat3_mul(parent_r, origin_r)
            angle = angles.get(joint.name, 0.0)
            if joint.kind in ("revolute", "continuous") and angle != 0.0:
                combined_r = _mat3_mul(combined_r, _rotation_axis_angle(joint.axis, angle))
            frames[joint.child] = (combined_r, origin_t)
            positions[joint.child] = origin_t
            stack.append(joint.child)
    return positions


def project_front(positions: Mapping[str, tuple[float, float, float]]) -> dict[str, tuple[float, float]]:
    """Front view (facing the ped): horizontal = -y, vertical = z."""
    return {name: (-y, z) for name, (_x, y, z) in positions.items()}


def project_side(positions: Mapping[str, tuple[float, float, float]]) -> dict[str, tuple[float, float]]:
    """Side view: horizontal = x (forward), vertical = z (up)."""
    return {name: (x, z) for name, (x, _y, z) in positions.items()}


@functools.lru_cache(maxsize=4)
def _cached_tree(body_id: str) -> tuple[str, dict[str, Joint]]:
    assert body_urdf is not None
    return parse_urdf(body_urdf(body_id))


Segment = tuple[tuple[float, float], tuple[float, float]]


def _bone_segments(points: Mapping[str, tuple[float, float]], joints: Mapping[str, Joint]) -> list[Segment]:
    return [(points[j.parent], points[j.child]) for j in joints.values() if j.parent in points and j.child in points]


def preview(
    angles: Mapping[str, float],
    body_id: str = PREVIEW_ID,
) -> tuple[list[Segment], list[Segment]]:
    """Front and side bone segments of the RViz body posed by semantic angles over the default stance."""
    if semantic_positions is None:
        raise RuntimeError("rviz_utils.hri.rig is unavailable (source the ROS install)")
    root, joints = _cached_tree(body_id)
    positions = forward_kinematics(root, joints, semantic_positions(angles, "", body_id))
    front, side = project_front(positions), project_side(positions)
    return _bone_segments(front, joints), _bone_segments(side, joints)
