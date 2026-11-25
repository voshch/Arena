"""Workspace sampling helpers for REACH_POSE task modes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geometry_msgs.msg
import rclpy.time
import tf2_ros
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

if TYPE_CHECKING:
    import numpy.random
    from arena_rclpy_mixins.shared import FrameNamespace
    from arena_robots.caps import ArmSpec


MAP_FRAME = "map"


def reach_frame(frame: str, robot_frame: FrameNamespace) -> str:
    """`frame` under the robot's TF prefix, unchanged when it already carries it."""
    return frame if frame.startswith(robot_frame.tf()) else robot_frame.tf(frame)


def reach_target(target: Pose, frame: str, robot_frame: FrameNamespace, base_frame: str, tf_buffer: tf2_ros.Buffer) -> geometry_msgs.msg.PoseStamped:
    """`target` given in `frame` as a pose the arm planner resolves, one in the env's `map` re-expressed in the robot's base frame through TF."""
    msg = geometry_msgs.msg.PoseStamped()
    if frame != MAP_FRAME:
        msg.header.frame_id = reach_frame(frame, robot_frame)
        msg.pose = target.to_msg()
        return msg
    try:
        t = tf_buffer.lookup_transform(MAP_FRAME, base_frame, rclpy.time.Time())
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        raise RuntimeError(f"no TF from {MAP_FRAME} to {base_frame} for a map reach target: {e}") from e
    tr, rot = t.transform.translation, t.transform.rotation
    base = Pose(position=Position(x=tr.x, y=tr.y, z=tr.z), orientation=Orientation(w=rot.w, x=rot.x, y=rot.y, z=rot.z))
    msg.header.frame_id = base_frame
    msg.pose = (base.inverse() * target).to_msg()
    return msg


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
