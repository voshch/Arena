"""Workspace sampling helpers for REACH_POSE task modes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geometry_msgs.msg

if TYPE_CHECKING:
    import numpy.random
    from arena_rclpy_mixins.shared import FrameNamespace
    from arena_robots.caps import ArmSpec


def reach_frame(frame: str, robot_frame: FrameNamespace) -> str:
    """`frame` under the robot's TF prefix, unchanged when it already carries it."""
    return frame if frame.startswith(robot_frame.tf()) else robot_frame.tf(frame)


def sample_reach_target(arm: ArmSpec, robot_frame: FrameNamespace, rng: numpy.random.Generator) -> geometry_msgs.msg.PoseStamped:
    """Uniform-in-box sample over ``arm.workspace`` (cap-file-declared).

    Returns a PoseStamped with frame_id = ``<robot_frame>/<workspace.frame>`` (TF prefix,
    matching robot_state_publisher) and identity orientation."""
    ws = arm.workspace
    if ws is None:
        raise ValueError(f"arm '{arm.name}' has no workspace declaration; cannot sample reach targets")
    if ws.get("type") != "box":
        raise ValueError(f"arm '{arm.name}' workspace.type {ws.get('type')!r} unsupported (only 'box')")
    lo = ws["min"]
    hi = ws["max"]
    if not isinstance(lo, list) or not isinstance(hi, list):
        raise ValueError(f"arm '{arm.name}' workspace min/max must be lists, got {lo!r} and {hi!r}")
    frame = ws.get("frame", "base_link")

    msg = geometry_msgs.msg.PoseStamped()
    msg.header.frame_id = reach_frame(str(frame), robot_frame)
    msg.pose.position.x = float(rng.uniform(lo[0], hi[0]))
    msg.pose.position.y = float(rng.uniform(lo[1], hi[1]))
    msg.pose.position.z = float(rng.uniform(lo[2], hi[2]))
    msg.pose.orientation.w = 1.0
    return msg
