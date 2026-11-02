"""URDF of an Arena body for RViz from the body rig template: a three-axis ball per rig joint, the REP-155 body frames as fixed aliases."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence

import numpy as np
from arena_simulation_setup.shared.body import Template, load_template, quat_to_matrix

AXES = (("rx", (1.0, 0.0, 0.0)), ("ry", (0.0, 1.0, 0.0)), ("rz", (0.0, 0.0, 1.0)))
ROOT_AXES = (("root_x", (1.0, 0.0, 0.0)), ("root_y", (0.0, 1.0, 0.0)), ("root_z", (0.0, 0.0, 1.0)))
REP155_FRAMES = {
    "waist": "spine1",
    "torso": "spine3",
    **{f"{s}_{part}": f"{side}_{part}" for s, side in (("l", "left"), ("r", "right")) for part in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")},
}
HEAD_RADIUS = 0.11
EYES = ((0.12, 0.032, 0.06), (0.12, -0.032, 0.06))
_DIGITS = ("thumb", "index", "middle", "ring", "pinky")
SKIN = (0.86, 0.68, 0.56, 1.0)
CLOTH = (0.32, 0.4, 0.55, 1.0)


def _fmt(values: Sequence[float] | np.ndarray) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _radius(joint: str) -> float:
    if any(digit in joint for digit in _DIGITS):
        return 0.008
    if joint.endswith(("wrist", "foot", "ankle")):
        return 0.024
    if joint.startswith(("spine", "pelvis", "neck")):
        return 0.045
    return 0.033


def _color(joint: str) -> tuple[float, float, float, float]:
    return SKIN if joint.endswith(("wrist", "head", "neck")) or any(digit in joint for digit in _DIGITS) else CLOTH


def _along(direction: np.ndarray) -> tuple[float, float, float]:
    """rpy turning z onto a direction."""
    z = direction / np.linalg.norm(direction)
    x = np.cross(z, (0.0, 0.0, 1.0)) if math.hypot(z[0], z[1]) > 1e-9 else np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return math.atan2(y[2], z[2]), 0.0, math.atan2(x[1], x[0])


def _visual(link: ET.Element, xyz: Sequence[float] | np.ndarray, rpy: Sequence[float], color: Sequence[float], shape: str, **size: float) -> None:
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, shape, attrib={key: f"{value:.6g}" for key, value in size.items()})
    material = ET.SubElement(visual, "material", name="rgba_" + "_".join(f"{c:.2f}" for c in color))
    ET.SubElement(material, "color", rgba=_fmt(color))


def _joint(robot: ET.Element, name: str, kind: str, parent: str, child: str, xyz: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0), axis: Sequence[float] | None = None) -> None:
    joint = ET.SubElement(robot, "joint", name=name, type=kind)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "origin", xyz=_fmt(xyz), rpy="0 0 0")
    if axis is not None:
        ET.SubElement(joint, "axis", xyz=_fmt(axis))
    if kind == "prismatic":
        ET.SubElement(joint, "limit", lower="-2", upper="2", effort="0", velocity="0")


def body_urdf(body_id: str, template: Template | None = None) -> str:
    """The URDF of body `body_id`: root `body_<id>` at the pelvis, links `<joint>_<id>`, joints `<joint>_<rx|ry|rz>_<id>`, the root offset on `root_<x|y|z>_<id>`."""
    template = template if template is not None else load_template()
    robot = ET.Element("robot", name=f"human_{body_id}")
    children: dict[str, list[str]] = {joint: [] for joint in template.parents}
    for joint, parent in template.parents.items():
        if parent is not None:
            children[parent].append(joint)
    ET.SubElement(robot, "link", name=f"body_{body_id}")
    above = f"body_{body_id}"
    for name, axis in ROOT_AXES:
        root = f"{name}_{body_id}"
        ET.SubElement(robot, "link", name=root)
        _joint(robot, root, "prismatic", above, root, axis=axis)
        above = root
    for joint, parent in template.parents.items():
        xyz = template.offsets[joint] if parent is not None else (0.0, 0.0, 0.0)
        above = f"{parent}_{body_id}" if parent is not None else f"root_z_{body_id}"
        for k, (axis_name, axis) in enumerate(AXES[:-1]):
            child = f"{joint}_{axis_name}_{body_id}"
            ET.SubElement(robot, "link", name=child)
            _joint(robot, child, "continuous", above, child, xyz if k == 0 else (0.0, 0.0, 0.0), axis)
            above = child
        link = ET.SubElement(robot, "link", name=f"{joint}_{body_id}")
        _joint(robot, f"{joint}_{AXES[-1][0]}_{body_id}", "continuous", above, f"{joint}_{body_id}", axis=AXES[-1][1])
        for kid in children[joint]:
            offset = np.asarray(template.offsets[kid], dtype=float)
            length = float(np.linalg.norm(offset))
            if length > 1e-6:
                _visual(link, offset / 2.0, _along(offset), _color(joint), "cylinder", radius=_radius(joint), length=length)
        if joint == "head":
            _visual(link, (0.02, 0.0, 0.06), (0.0, 0.0, 0.0), SKIN, "sphere", radius=HEAD_RADIUS)
            for eye in EYES:
                _visual(link, eye, (0.0, 0.0, 0.0), (0.1, 0.1, 0.12, 1.0), "sphere", radius=0.012)
        elif not children[joint]:
            _visual(link, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), _color(joint), "sphere", radius=_radius(joint))
    for frame, joint in REP155_FRAMES.items():
        if joint in template.parents:
            ET.SubElement(robot, "link", name=f"{frame}_{body_id}")
            _joint(robot, f"{frame}_frame_{body_id}", "fixed", f"{joint}_{body_id}", f"{frame}_{body_id}")
    ET.indent(robot)
    return ET.tostring(robot, encoding="unicode")


def ball_angles(rotation: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """(rx, ry, rz) of a local x y z w rotation composed as Rx Ry Rz."""
    m = quat_to_matrix(rotation)
    ry = math.asin(max(-1.0, min(1.0, float(m[0, 2]))))
    return math.atan2(-float(m[1, 2]), float(m[2, 2])), ry, math.atan2(-float(m[0, 1]), float(m[0, 0]))


def body_positions(body_id: str, rotations: Mapping[str, Sequence[float] | np.ndarray], root_offset: Sequence[float] = (0.0, 0.0, 0.0)) -> dict[str, float]:
    """Joint positions of body `body_id` for local x y z w joint rotations and the root offset in m."""
    positions = {f"{name}_{body_id}": float(value) for (name, _), value in zip(ROOT_AXES, root_offset, strict=True)}
    for joint, rotation in rotations.items():
        for (axis_name, _), angle in zip(AXES, ball_angles(rotation), strict=True):
            positions[f"{joint}_{axis_name}_{body_id}"] = angle
    return positions
