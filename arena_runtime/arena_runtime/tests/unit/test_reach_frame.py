from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("geometry_msgs.msg")

from arena_rclpy_mixins.shared import FrameNamespace
from arena_robots.caps import ArmSpec

from arena_runtime.env.adapters._reach_sampling import reach_frame, sample_reach_target

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
