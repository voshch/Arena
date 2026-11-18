from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import trajectory_msgs.msg
from arena_rclpy_mixins.Time import Time

from arena_runtime.env.adapters import ADAPTERS, Adapter

_log = logging.getLogger(__name__)

_PARK_SECONDS = 3
_PARK_SETTLE_SECONDS = 1.0

if TYPE_CHECKING:
    from arena_robots.caps import ArmSpec

    from arena_runtime.env.robot_manager.robot_manager import RobotManager


@ADAPTERS["arm"].register("moveit")
def _load_moveit() -> type[Adapter]:
    from .moveit import MoveItArmAdapter

    return MoveItArmAdapter


@ADAPTERS["arm"].register("none")
def _load_none() -> type[Adapter]:
    from .none import NoneArmAdapter

    return NoneArmAdapter


def park_positions(arm: ArmSpec) -> list[float]:
    """Per-chain-joint target: the 'stow' named pose, zeros (spawn pose) for joints it omits."""
    pose = arm.named_poses.get("stow", {})
    return [float(pose.get(joint, 0.0)) for joint in arm.chain]


def park_seconds(arm: ArmSpec) -> float:
    """The larger of the default park time and the slowest chain joint's position span over its max velocity."""
    seconds = float(_PARK_SECONDS)
    for joint in arm.chain:
        limits = arm.joint_limits.get(joint, {})
        low, high, speed = limits.get("min_position"), limits.get("max_position"), limits.get("max_velocity")
        if isinstance(low, int | float) and isinstance(high, int | float) and isinstance(speed, int | float) and speed > 0:
            seconds = max(seconds, (high - low) / speed)
    return seconds


async def park_arms(robot: RobotManager) -> list:
    """Command every arm instance to its park pose, once per bring-up (joints droop
    while their controllers step up), and wait in sim time until the arms settled.
    Returns the publishers so callers keep them alive."""
    pubs = []
    durations = []
    caps = robot.caps
    for arm in (caps.arm or {}).values():
        point = trajectory_msgs.msg.JointTrajectoryPoint()
        point.positions = park_positions(arm)
        seconds = park_seconds(arm)
        durations.append(seconds)
        point.time_from_start.sec = int(seconds)
        point.time_from_start.nanosec = int((seconds - int(seconds)) * 1e9)
        traj = trajectory_msgs.msg.JointTrajectory(joint_names=list(arm.chain), points=[point])
        topic = str(robot.namespace(arm.controller, "joint_trajectory"))
        pub = robot.node.create_publisher(trajectory_msgs.msg.JointTrajectory, topic, 10)
        # publishing before the JTC subscription matches drops the message
        deadline = time.monotonic() + 30.0
        while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        if pub.get_subscription_count() == 0:
            _log.warning("parking arm %r: no subscriber on %s, trajectory may be dropped", arm.name, topic)
        pub.publish(traj)
        pubs.append(pub)
    if pubs and not await robot.node.await_sim_time(robot.node.sim_time + Time.from_float(max(durations) + _PARK_SETTLE_SECONDS), freeze_timeout=5.0):
        _log.warning("parking arms of %r: sim clock froze before the arms settled", robot.name)
    return pubs
