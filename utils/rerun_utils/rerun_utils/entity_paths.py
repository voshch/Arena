"""Rerun entity-path conventions for the arena_viz bridge."""

from __future__ import annotations

from arena_runtime_msgs.msg import RobotDescriptor


def env_root(env_id: int) -> str:
    return f"env_{env_id}"


def display_path(env_id: int, robot: RobotDescriptor | None, display_name: str) -> str:
    safe = display_name.replace(" ", "_").replace("/", "_")
    if robot is None:
        return f"{env_root(env_id)}/{safe}"
    return f"{env_root(env_id)}/robots/{robot.name}/{safe}"


def tf_path(env_id: int, frame: str) -> str:
    return f"{env_root(env_id)}/tf/{frame}"
