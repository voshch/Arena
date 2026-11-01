"""The arena_peds wire: joint_state compiled into the body pose, driven joints left behind."""

from __future__ import annotations

import math

import numpy as np
import pytest
from arena_people_msgs.msg import Pedestrian
from arena_simulation_setup.shared.body import BODY_JOINTS, DEFAULT_STANCE, DIGIT_JOINTS, SEMANTIC_DOFS, body_to_semantic, quat_to_matrix
from arena_runtime.humans.animation_manager import BodyLayer, nlerp
from arena_runtime.humans.profile import JOINT_NAMES
from arena_runtime.humans.wire import fill_body


def _ped(angles: dict[str, float]) -> Pedestrian:
    ped = Pedestrian()
    ped.joint_state.name = list(angles)
    ped.joint_state.position = list(angles.values())
    return ped


def test_semantic_angles_become_the_body_pose_and_leave_the_joint_state() -> None:
    angles = dict.fromkeys(JOINT_NAMES, 0.0) | {"l_knee": -0.7, "l_p_shoulder": 0.4, "waist": 0.1}
    ped = _ped(angles)
    fill_body(ped, DEFAULT_STANCE)
    assert tuple(ped.body.joints) == BODY_JOINTS + DIGIT_JOINTS
    assert list(ped.joint_state.name) == []
    rotations = {joint: (q.x, q.y, q.z, q.w) for joint, q in zip(ped.body.joints, ped.body.rotations, strict=True)}
    assert body_to_semantic(rotations) == pytest.approx(dict.fromkeys(SEMANTIC_DOFS, 0.0) | angles, abs=1e-9)


def test_digits_without_dofs_hold_the_neutral_stance() -> None:
    wrist = quat_to_matrix((0.0, 0.0, math.sin(0.2), math.cos(0.2)))
    stance = {**DEFAULT_STANCE, "right_index1": wrist}
    ped = _ped(dict.fromkeys(JOINT_NAMES, 0.0))
    fill_body(ped, stance)
    rotations = dict(zip(ped.body.joints, ped.body.rotations, strict=True))
    q = rotations["right_index1"]
    held = DEFAULT_STANCE["right_wrist"].T @ wrist
    assert quat_to_matrix((q.x, q.y, q.z, q.w)) == pytest.approx(held, abs=1e-9)


def test_digit_dofs_reach_the_finger_joints() -> None:
    ped = _ped(dict.fromkeys(JOINT_NAMES, 0.0) | {"r_curl_index": 1.2, "l_spread": 0.2})
    fill_body(ped, DEFAULT_STANCE)
    rotations = {joint: (q.x, q.y, q.z, q.w) for joint, q in zip(ped.body.joints, ped.body.rotations, strict=True)}
    semantic = body_to_semantic(rotations)
    assert semantic["r_curl_index"] == pytest.approx(1.2, abs=1e-9)
    assert semantic["l_spread"] == pytest.approx(0.2, abs=1e-9)
    assert list(ped.joint_state.name) == []


def test_driven_joints_stay_in_the_joint_state() -> None:
    ped = _ped(dict.fromkeys(JOINT_NAMES, 0.0) | {"l_wheel": 1.5, "joystick_pitch": 0.6})
    fill_body(ped, DEFAULT_STANCE)
    assert dict(zip(ped.joint_state.name, ped.joint_state.position, strict=True)) == {"l_wheel": 1.5, "joystick_pitch": 0.6}


def test_a_ped_without_semantic_angles_keeps_an_empty_body() -> None:
    ped = _ped({"l_wheel": 1.5})
    fill_body(ped, DEFAULT_STANCE)
    assert list(ped.body.joints) == []
    assert list(ped.joint_state.name) == ["l_wheel"]


def test_a_clip_body_group_blends_over_the_compiled_pose() -> None:
    angles = dict.fromkeys(JOINT_NAMES, 0.0)
    plain = _ped(angles)
    fill_body(plain, DEFAULT_STANCE)
    quarter = (0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0))
    layer = BodyLayer({"pelvis": (quarter, 1.0), "spine1": (quarter, 0.5)}, np.array([0.0, 0.0, -0.4]), 1.0)
    ped = _ped(angles)
    fill_body(ped, DEFAULT_STANCE, [layer])
    index = list(ped.body.joints).index
    q = ped.body.rotations[index("pelvis")]
    assert (q.x, q.y, q.z, q.w) == pytest.approx(quarter)
    half = ped.body.rotations[index("spine1")]
    assert half.z == pytest.approx(nlerp((0.0, 0.0, 0.0, 1.0), quarter, 0.5)[2])
    assert ped.body.rotations[index("spine2")] == plain.body.rotations[index("spine2")]
    assert ped.body.root_offset.z == pytest.approx(-0.4)
