from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("geometry_msgs.msg")

from arena_rclpy_mixins.shared import FrameNamespace
from arena_robots.caps import ArmSpec

from arena_runtime.env.adapters._reach_sampling import reach_frame, reach_target, sample_reach_target

ROBOT = FrameNamespace("env_0/jackal")


@pytest.mark.parametrize(
    ("frame", "resolved"),
    [
        ("base_link", "env_0/jackal/base_link"),
        ("arm/tool0", "env_0/jackal/arm/tool0"),
        ("env_0/jackal/base_link", "env_0/jackal/base_link"),
        ("env_0/jackal_2/base_link", "env_0/jackal/env_0/jackal_2/base_link"),
    ],
)
def test_reach_frames_take_the_robot_prefix_once(frame, resolved):
    assert reach_frame(frame, ROBOT) == resolved


def test_sampled_and_explicit_targets_share_the_prefixed_frame():
    arm = ArmSpec(path=Path("caps/arm.yaml"), raw={"workspace": {"type": "box", "min": [0.1, -0.1, 0.2], "max": [0.3, 0.1, 0.4], "frame": "base_link"}}, name="arm")
    sampled = sample_reach_target(arm, ROBOT, np.random.default_rng(0))
    assert sampled.header.frame_id == reach_frame("base_link", ROBOT) == "env_0/jackal/base_link"


def test_a_map_target_is_re_expressed_in_the_robot_base_frame():
    import math

    import tf2_ros
    from arena_simulation_setup.utils.geometry import Orientation, Pose, Position
    from geometry_msgs.msg import TransformStamped

    buffer = tf2_ros.Buffer()
    base = TransformStamped()
    base.header.frame_id = "map"
    base.child_frame_id = "env_0/jackal/base_link"
    base.transform.translation.x, base.transform.translation.y = 1.0, 2.0
    base.transform.rotation.z, base.transform.rotation.w = math.sin(math.pi / 4), math.cos(math.pi / 4)
    buffer.set_transform_static(base, "test")

    target = Pose(Position(1.0, 3.0, 0.5), Orientation.from_yaw(math.pi / 2))
    msg = reach_target(target, "map", ROBOT, "env_0/jackal/base_link", buffer)
    assert msg.header.frame_id == "env_0/jackal/base_link"
    assert (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z) == pytest.approx((1.0, 0.0, 0.5))
    assert Orientation.from_msg(msg.pose.orientation).to_yaw() == pytest.approx(0.0, abs=1e-9)

    local = reach_target(target, "base_link", ROBOT, "env_0/jackal/base_link", buffer)
    assert local.header.frame_id == "env_0/jackal/base_link"
    assert local.pose.position.y == pytest.approx(3.0)


def test_a_map_target_without_tf_fails_with_the_frames_named():
    import tf2_ros
    from arena_simulation_setup.utils.geometry import Pose, Position

    with pytest.raises(RuntimeError, match="no TF from map to env_0/jackal/base_link"):
        reach_target(Pose(Position(0.0, 0.0, 0.0)), "map", ROBOT, "env_0/jackal/base_link", tf2_ros.Buffer())
