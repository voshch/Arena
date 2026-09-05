"""gait.py's joint table must match the revolute joints of human_description/urdf/human-tpl.xacro."""

from __future__ import annotations

import pytest
from task_generator.simulators.human.gait import LIMITS, GaitGenerator
from test_rig_proportions import _xacro_path

# anatomical in gait.py (see its module docstring), not on URDF axes, so their ranges are not comparable
ANATOMICAL = {f"{side}_{axis}_shoulder" for side in "lr" for axis in "ypr"}


def _urdf_joints() -> dict[str, tuple[float, float, str]]:
    xacro = pytest.importorskip("xacro")
    path = _xacro_path()
    if path is None:
        pytest.skip("human_description xacro not found")
    doc = xacro.process_file(str(path), mappings={"id": "X", "height": "1.65"})
    joints: dict[str, tuple[float, float, str]] = {}
    for joint in doc.getElementsByTagName("joint"):
        if joint.getAttribute("type") != "revolute":
            continue
        limit = joint.getElementsByTagName("limit")[0]
        axis = joint.getElementsByTagName("axis")
        joints[joint.getAttribute("name").removesuffix("_X")] = (
            float(limit.getAttribute("lower")),
            float(limit.getAttribute("upper")),
            axis[0].getAttribute("xyz") if axis else "1 0 0",
        )
    return joints


def test_joint_names_match_urdf() -> None:
    assert set(GaitGenerator.JOINT_NAMES) == set(_urdf_joints())


def test_limits_within_urdf() -> None:
    joints = _urdf_joints()
    for (lo, hi), name in zip(LIMITS, GaitGenerator.JOINT_NAMES, strict=True):
        if name in ANATOMICAL:
            continue
        ulo, uhi, _ = joints[name]
        assert ulo - 1e-6 <= lo and hi <= uhi + 1e-6, f"{name} gait {lo, hi} exceeds urdf {ulo, uhi}"


def test_shoulder_axes_mirrored() -> None:
    """The anatomical shoulder convention relies on the URDF mirroring left and right shoulder axes."""
    joints = _urdf_joints()
    for axis in "ypr":
        left = [float(v) for v in joints[f"l_{axis}_shoulder"][2].split()]
        right = [float(v) for v in joints[f"r_{axis}_shoulder"][2].split()]
        assert left == [-v for v in right], f"{axis}_shoulder axes no longer mirrored: {left} vs {right}"
