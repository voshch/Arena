from __future__ import annotations

import math

import numpy as np
import pytest
from arena_people_msgs.msg import Pedestrian
from arena_simulation_setup.shared.body import DEFAULT_STANCE, SEMANTIC_DOFS, forward, load_template, matrix_to_quat, quat_to_matrix, semantic_to_body
from geometry_msgs.msg import Quaternion
from urdf_parser_py import urdf as urdf_parser

from rviz_utils.hri.body_frames import _origin_matrix
from rviz_utils.hri.rig import _rot_axis, ped_positions, semantic_positions
from rviz_utils.hri.urdf import REP155_FRAMES, ball_angles, body_positions, body_urdf

_BODY = "env_0_agent_3"


def _Raxis(axis: np.ndarray, a: float) -> np.ndarray:
    """Rodrigues' rotation formula, spec-independent of rig.py's own implementation."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(a), math.sin(a)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _link_worlds(urdf: str, positions: dict[str, float]) -> dict[str, np.ndarray]:
    robot = urdf_parser.URDF.from_xml_string(urdf)
    root = robot.get_root()
    worlds = {root: np.eye(4)}
    stack = [root]
    while stack:
        parent = stack.pop()
        for joint_name, child in robot.child_map.get(parent, ()):
            joint = robot.joint_map[joint_name]
            m = worlds[parent] @ _origin_matrix(joint.origin)
            q = positions.get(joint_name, 0.0)
            move = np.eye(4)
            if joint.type == "prismatic":
                move[:3, 3] = np.asarray(joint.axis, dtype=float) * q
            elif joint.type in ("revolute", "continuous"):
                move[:3, :3] = _Raxis(np.asarray(joint.axis, dtype=float), q)
            worlds[child] = m @ move
            stack.append(child)
    return worlds


def _random_pose(rng: np.random.Generator) -> dict[str, np.ndarray]:
    template = load_template()
    pose = {}
    for joint in template.parents:
        axis = rng.normal(size=3)
        pose[joint] = matrix_to_quat(_Raxis(axis, float(rng.uniform(-1.2, 1.2))))
    return pose


def test_the_urdf_roots_at_body_and_carries_every_rig_joint_and_the_rep155_frames() -> None:
    robot = urdf_parser.URDF.from_xml_string(body_urdf(_BODY))
    template = load_template()
    assert robot.get_root() == f"body_{_BODY}"
    links = set(robot.link_map)
    assert {f"{joint}_{_BODY}" for joint in template.parents} <= links
    assert {f"{frame}_{_BODY}" for frame in REP155_FRAMES} <= links
    assert f"head_{_BODY}" in links
    assert len(template.parents) == 52


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_urdf_kinematics_reproduce_the_rig_template_for_any_pose(seed: int) -> None:
    rng = np.random.default_rng(seed)
    pose = _random_pose(rng)
    offset = rng.uniform(-0.3, 0.3, size=3)
    template = load_template()
    want, frames = forward(template, pose, offset)
    worlds = _link_worlds(body_urdf(_BODY), body_positions(_BODY, pose, offset))
    pelvis = template.offsets["pelvis"]
    for joint in template.parents:
        got = worlds[f"{joint}_{_BODY}"]
        np.testing.assert_allclose(got[:3, 3], want[joint] - pelvis, atol=1e-9)
        np.testing.assert_allclose(got[:3, :3], frames[joint], atol=1e-9)


def test_every_bone_cylinder_spans_its_parent_and_child_joint() -> None:
    robot = urdf_parser.URDF.from_xml_string(body_urdf(_BODY))
    template = load_template()
    checked = 0
    for joint, parent in template.parents.items():
        if parent is None:
            continue
        offset = np.asarray(template.offsets[joint], dtype=float)
        length = float(np.linalg.norm(offset))
        if length < 1e-6:
            continue
        cylinders = [v for v in robot.link_map[f"{parent}_{_BODY}"].visuals if isinstance(v.geometry, urdf_parser.Cylinder) and abs(v.geometry.length - length) < 1e-5]
        ends = [(_origin_matrix(v.origin) @ np.array((0.0, 0.0, s * length / 2.0, 1.0)))[:3] for v in cylinders for s in (-1.0, 1.0)]
        assert any(np.allclose(a, (0.0, 0.0, 0.0), atol=1e-5) and np.allclose(b, offset, atol=1e-5) for a, b in zip(ends[::2], ends[1::2])), joint
        checked += 1
    assert checked > 40


@pytest.mark.parametrize("seed", range(5))
def test_ball_angles_rebuild_the_rotation(seed: int) -> None:
    rng = np.random.default_rng(seed)
    q = matrix_to_quat(_Raxis(rng.normal(size=3), float(rng.uniform(-2.5, 2.5))))
    a, b, c = ball_angles(q)
    rebuilt = _Raxis(np.array((1.0, 0.0, 0.0)), a) @ _Raxis(np.array((0.0, 1.0, 0.0)), b) @ _Raxis(np.array((0.0, 0.0, 1.0)), c)
    np.testing.assert_allclose(rebuilt, quat_to_matrix(q), atol=1e-9)


def test_ped_positions_read_the_body_pose_off_the_wire() -> None:
    pose = semantic_to_body({"l_elbow": 1.1, "r_curl_index": 0.9}, DEFAULT_STANCE)
    ped = Pedestrian()
    ped.body.joints = list(pose)
    ped.body.rotations = [Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])) for q in pose.values()]
    ped.body.root_offset.z = -0.2
    got = ped_positions(ped, _BODY)
    assert got == pytest.approx(body_positions(_BODY, pose, (0.0, 0.0, -0.2)))
    assert got[f"root_z_{_BODY}"] == pytest.approx(-0.2)


def test_semantic_positions_pose_the_body_over_the_default_stance_without_a_rig() -> None:
    angles = dict.fromkeys(SEMANTIC_DOFS, 0.0) | {"l_knee": -0.7}
    got = semantic_positions(angles, "", _BODY)
    assert got == pytest.approx(body_positions(_BODY, semantic_to_body(angles, DEFAULT_STANCE)))


def test_rot_axis_zero_axis_is_identity() -> None:
    assert np.array_equal(_rot_axis(np.zeros(3), 1.3), np.eye(3))


@pytest.mark.parametrize("axis", [(0.0, 0.0, 2.5), (1.0, -2.0, 0.5), (-3.0, 0.0, 0.0)])
def test_rot_axis_matches_rodrigues_for_non_unit_axis(axis: tuple[float, float, float]) -> None:
    got = _rot_axis(np.array(axis), 0.8)
    assert np.allclose(got, _Raxis(axis, 0.8), atol=1e-12)
    assert np.allclose(got @ np.array(axis), np.array(axis), atol=1e-12)
    assert np.allclose(got @ got.T, np.eye(3), atol=1e-12)
