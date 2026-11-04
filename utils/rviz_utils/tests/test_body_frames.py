from __future__ import annotations

import math

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from urdf_parser_py import urdf as urdf_parser

from rviz_utils.hri import body_frames
from rviz_utils.hri.body_frames import BodyFrames, _origin_matrix
from rviz_utils.hri.urdf import REP155_FRAMES, body_urdf

_BODY = "env_0_agent_3"
_STAMP = Time(sec=10)


def _quat_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rodrigues(axis: np.ndarray, a: float) -> np.ndarray:
    x, y, z = axis / np.linalg.norm(axis)
    c, s, t = math.cos(a), math.sin(a), 1.0 - math.cos(a)
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _expected(joint: urdf_parser.Joint, position: float) -> np.ndarray:
    m = _origin_matrix(joint.origin)
    motion = np.eye(4)
    if joint.type == "prismatic":
        motion[:3, 3] = np.asarray(joint.axis) * position
    else:
        motion[:3, :3] = _rodrigues(np.asarray(joint.axis, dtype=float), position)
    return m @ motion


def _positions(robot: urdf_parser.URDF, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {joint.name: float(rng.uniform(-1.5, 1.5)) for joint in robot.joints if joint.type != "fixed"}


def _frames(body_id: str = _BODY, max_bodies: int = 32) -> BodyFrames:
    frames = BodyFrames(max_bodies=max_bodies)
    frames.sync({body_id})
    return frames


def test_moving_transform_is_origin_then_motion_about_the_axis():
    robot = urdf_parser.URDF.from_xml_string(body_urdf(_BODY))
    positions = _positions(robot, 1)
    msg = _frames().moving_transforms(_BODY, positions, _STAMP)
    assert msg is not None
    by_child = {tf.child_frame_id: tf for tf in msg.transforms}
    moving = [joint for joint in robot.joints if joint.type != "fixed"]
    assert len(by_child) == len(moving)
    for joint in moving:
        tf = by_child[joint.child]
        assert tf.header.frame_id == joint.parent
        assert tf.header.stamp == _STAMP
        t, r = tf.transform.translation, tf.transform.rotation
        got = np.eye(4)
        got[:3, :3] = _quat_matrix(r.x, r.y, r.z, r.w)
        got[:3, 3] = t.x, t.y, t.z
        np.testing.assert_allclose(got, _expected(joint, positions[joint.name]), atol=1e-9)


def test_moving_transform_honors_a_rotated_origin():
    urdf = (
        '<robot name="r"><link name="a"/><link name="b"/><link name="c"/>'
        '<joint name="hinge" type="continuous"><parent link="a"/><child link="b"/>'
        '<origin xyz="1 2 3" rpy="0.3 -0.4 0.5"/><axis xyz="0 0 1"/></joint>'
        '<joint name="slide" type="prismatic"><parent link="b"/><child link="c"/>'
        '<origin xyz="0 1 0" rpy="-0.2 0.1 0.9"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="0" velocity="0"/></joint>'
        "</robot>"
    )
    robot = urdf_parser.URDF.from_xml_string(urdf)
    body = body_frames._Body(urdf)
    body_frames._animate(body, body_frames._joint_specs(robot), _BODY)
    for name, position in (("hinge", 0.7), ("slide", 0.6)):
        joint = robot.joint_map[name]
        body_frames._pose_moving(body.moving[name], position)
        t, r = body.moving[name].tf.transform.translation, body.moving[name].tf.transform.rotation
        got = np.eye(4)
        got[:3, :3] = _quat_matrix(r.x, r.y, r.z, r.w)
        got[:3, 3] = t.x, t.y, t.z
        np.testing.assert_allclose(got, _expected(joint, position), atol=1e-9)


def test_each_body_gets_its_own_names_in_urdf_and_frames():
    frames = BodyFrames()
    other = "env_1_agent_8"
    frames.sync({_BODY, other})
    for bid in (_BODY, other):
        assert frames.urdf_for(bid) == body_urdf(bid)
    robot = urdf_parser.URDF.from_xml_string(body_urdf(other))
    msg = frames.moving_transforms(other, _positions(robot, 4), _STAMP)
    assert msg is not None
    assert {tf.child_frame_id for tf in msg.transforms} == {joint.child for joint in robot.joints if joint.type != "fixed"}
    assert all(other in tf.child_frame_id and _BODY not in tf.child_frame_id for tf in msg.transforms)


def test_moving_transforms_cover_only_the_joints_given():
    robot = urdf_parser.URDF.from_xml_string(body_urdf(_BODY))
    name, position = next(iter(_positions(robot, 2).items()))
    msg = _frames().moving_transforms(_BODY, {name: position, "not_a_joint": 1.0}, _STAMP)
    assert msg is not None
    assert [tf.child_frame_id for tf in msg.transforms] == [robot.joint_map[name].child]


def test_moving_transforms_keep_to_twenty_hertz_and_restart_after_a_clock_jump_back():
    robot = urdf_parser.URDF.from_xml_string(body_urdf(_BODY))
    positions = _positions(robot, 3)
    frames = _frames()
    assert frames.moving_transforms(_BODY, positions, Time(sec=10)) is not None
    assert frames.moving_transforms(_BODY, positions, Time(sec=10, nanosec=49_000_000)) is None
    assert frames.moving_transforms(_BODY, positions, Time(sec=10, nanosec=50_000_000)) is not None
    assert frames.moving_transforms(_BODY, positions, Time(sec=1)) is not None


def test_static_transforms_hold_the_rep155_frames_and_follow_the_roster():
    frames = _frames()
    first = frames.static_transforms(_STAMP)
    assert first is not None
    assert {tf.child_frame_id for tf in first.transforms} == {f"{frame}_{_BODY}" for frame in REP155_FRAMES}
    assert frames.static_transforms(_STAMP) is None
    frames.sync(set())
    emptied = frames.static_transforms(_STAMP)
    assert emptied is not None
    assert emptied.transforms == []


def test_bodies_over_the_cap_keep_urdf_and_lift_but_wait_for_a_slot(capfd: pytest.CaptureFixture[str]):
    ids = {f"env_0_agent_{i}" for i in range(3)}
    frames = BodyFrames(max_bodies=2)
    frames.sync(ids)
    frames.sync(ids)
    assert capfd.readouterr().err.count("max_bodies=2") == 1
    assert frames.active_ids() == ids
    waiting = [bid for bid in sorted(ids) if frames.moving_transforms(bid, {}, _STAMP) is None]
    assert waiting == ["env_0_agent_2"]
    assert frames.urdf_for("env_0_agent_2") is not None
    assert frames.foot_offset("env_0_agent_2") > 0.0
    frames.sync(ids - {"env_0_agent_0"})
    assert frames.moving_transforms("env_0_agent_2", {}, _STAMP) is not None
    assert frames.urdf_for("env_0_agent_0") is None
    assert frames.foot_offset("env_0_agent_0") == 0.0
