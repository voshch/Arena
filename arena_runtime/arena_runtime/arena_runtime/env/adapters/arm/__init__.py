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


async def park_arms(robot: RobotManager) -> list:
    """Command every arm instance to its park pose, once per bring-up (joints droop
    while their controllers step up), and wait in sim time until the arms settled.
    Returns the publishers so callers keep them alive."""
    pubs = []
    caps = robot.robot_view.effective_caps(robot.robot.resolved_request, frames=robot.robot.frames)
    for arm in (caps.arm or {}).values():
        point = trajectory_msgs.msg.JointTrajectoryPoint()
        point.positions = park_positions(arm)
        point.time_from_start.sec = _PARK_SECONDS
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
    if pubs and not await robot.node.await_sim_time(robot.node.sim_time + Time.from_float(_PARK_SECONDS + _PARK_SETTLE_SECONDS), freeze_timeout=5.0):
        _log.warning("parking arms of %r: sim clock froze before the arms settled", robot.name)
    return pubs
