"""Coordination seam between robot spawn, the step loop, and pluggable subsystems."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import mujoco
    import rclpy.node

    from arena_mujoco.services.SpawnUrdf import RobotEntry

    PreCompileHook = Callable[[int, 'mujoco.MjSpec', 'mujoco.MjsBody', 'RobotEntry', 'list[str]'], None]
    PostSpawnHook = Callable[[int, str, 'RobotEntry', 'list[str]'], None]
    DespawnHook = Callable[[int, 'RobotEntry'], None]
    Pump = Callable[[], None]


_node: rclpy.node.Node | None = None
_pre_compile: list[PreCompileHook] = []
_post_spawn: list[PostSpawnHook] = []
_despawn: list[DespawnHook] = []
_pumps: list[Pump] = []


def set_node(node: rclpy.node.Node) -> None:
    """Install the process-wide ROS node, called once during server startup."""
    global _node
    _node = node


def get_node() -> rclpy.node.Node:
    """Return the installed ROS node, raising RuntimeError if it was never set."""
    if _node is None:
        raise RuntimeError('hooks node not initialized, call set_node first')
    return _node


def register_pre_compile(fn: PreCompileHook) -> None:
    """Register a hook fired after attach but before recompile so it can extend the spec."""
    _pre_compile.append(fn)


def register_post_spawn(fn: PostSpawnHook) -> None:
    """Register a hook fired after recompile so it can create ROS pubs and subs."""
    _post_spawn.append(fn)


def register_despawn(fn: DespawnHook) -> None:
    """Register a hook fired when a robot is deleted, to release its ROS endpoints."""
    _despawn.append(fn)


def register_pump(fn: Pump) -> None:
    """Register a per-tick pump fired once after each server step."""
    _pumps.append(fn)


def fire_pre_compile(
    env_id: int,
    spec: mujoco.MjSpec,
    robot_root_body: mujoco.MjsBody,
    robot_entry: RobotEntry,
    joint_names: list[str],
) -> None:
    """Run every pre-compile hook in registration order."""
    for fn in _pre_compile:
        fn(env_id, spec, robot_root_body, robot_entry, joint_names)


def fire_post_spawn(
    env_id: int,
    name: str,
    robot_entry: RobotEntry,
    joint_names: list[str],
) -> None:
    """Run every post-spawn hook in registration order."""
    for fn in _post_spawn:
        fn(env_id, name, robot_entry, joint_names)


def fire_despawn(env_id: int, robot_entry: RobotEntry) -> None:
    """Run every despawn hook in registration order."""
    for fn in _despawn:
        fn(env_id, robot_entry)


def run_pumps() -> None:
    """Run every registered pump once, in registration order."""
    for fn in _pumps:
        fn()


__all__ = [
    'fire_despawn',
    'fire_post_spawn',
    'fire_pre_compile',
    'get_node',
    'register_despawn',
    'register_post_spawn',
    'register_pre_compile',
    'register_pump',
    'run_pumps',
    'set_node',
]
