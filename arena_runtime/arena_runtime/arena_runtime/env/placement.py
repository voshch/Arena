from __future__ import annotations

import math
import typing

from arena_runtime.shared import Orientation, Pose

if typing.TYPE_CHECKING:
    from arena_runtime.env.robot_manager import RobotsManager
    from arena_runtime.env.world_manager.world_manager_ros import WorldManagerROS


def random_obstacle_pose(world_manager: WorldManagerROS, safe_dist: float = 1.0, level_id: str = "") -> Pose:
    """A random free pose on the current map."""
    points = world_manager.get_positions_on_map(n=1, safe_dist=safe_dist, level_id=level_id)
    rng = world_manager.node.conf.General.RNG.stream("placement", "obstacle")
    return Pose(points[0], orientation=Orientation.from_yaw(2 * math.pi * float(rng.random())))


def random_robot_pose(world_manager: WorldManagerROS, robots_manager: RobotsManager, safe_dist: float = 0.0, level_id: str = "") -> Pose:
    """A random free pose on the current map, clear for safe_dist and for the biggest robot."""
    clearance = max((safe_dist, *(robot.safe_distance for robot in robots_manager.managers.values())))
    points = world_manager.get_positions_on_map(n=1, safe_dist=clearance, level_id=level_id)
    rng = world_manager.node.conf.General.RNG.stream("placement", "robot")
    return Pose(points[0], orientation=Orientation.from_yaw(2 * math.pi * float(rng.random())))
