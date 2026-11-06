"""Rerun entity-path and transform-frame conventions for the arena_viz bridge."""

from __future__ import annotations

import rerun as rr
from arena_runtime_msgs.msg import RobotDescriptor

#: Implicit transform frame of the recording root, the target frame of views on `/`; one recording per env.
ENV_ORIGIN = "tf#/"


def env_root(env_id: int) -> str:
    return f"env_{env_id}"


def display_path(env_id: int, robot: RobotDescriptor | None, display_name: str) -> str:
    safe = display_name.replace(" ", "_").replace("/", "_")
    if robot is None:
        return f"{env_root(env_id)}/{safe}"
    return f"{env_root(env_id)}/robots/{robot.name}/{safe}"


def tf_path(env_id: int, frame: str) -> str:
    """The entity that holds the transform into `frame`, one per frame as rerun requires."""
    return f"{env_root(env_id)}/tf/{frame_name(frame)}"


def root_path(env_id: int, frame: str) -> str:
    """The entity that ties the root `frame` to the env's origin, apart from the one that would hold its parent."""
    return f"{env_root(env_id)}/tf_roots/{frame_name(frame)}"


def frame_name(frame_id: str) -> str:
    """The rerun transform frame of a ROS frame id: the id itself without a leading slash, which tf2 ignores too."""
    return frame_id.lstrip("/")


def in_frame(frame_id: str) -> list[rr.CoordinateFrame]:
    """Components that place an entity's data in the ROS frame `frame_id`, none for an empty id (the env root)."""
    return [rr.CoordinateFrame(frame_name(frame_id))] if frame_name(frame_id) else []
