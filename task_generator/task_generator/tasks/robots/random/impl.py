import math
from collections.abc import Sequence

from arena_runtime.env.task_api import GoToPhase, ReachPhase, TaskPhase, TaskRequest
from arena_runtime.env.task_layer import ResetPlan, RobotReset
from arena_runtime.shared import Orientation, Pose, Position, PositionRadius

from task_generator.tasks.robots import TM_Robots


def goal_clearance(goals: Sequence[Position], tolerance: float) -> list[PositionRadius]:
    """Zones a start must stay out of so that no robot begins inside a goal's tolerance."""
    return [PositionRadius(x=goal.x, y=goal.y, radius=tolerance) for goal in goals]


class TM_Random(TM_Robots):
    """Cap-blind random-action task mode."""

    async def reset(self) -> ResetPlan:
        plan = await super().reset()

        biggest_robot = max((robot.safe_distance for robot in self._ctx.robots.values()), default=0)
        rng = self._ctx.conf.RNG.stream("robots", "random")
        n = len(self._ctx.robots)

        goal_positions = self._ctx.world_manager.get_positions_on_map(n=n, safe_dist=0)
        tolerance = self._ctx.conf.GOAL_TOLERANCE_RADIUS.value
        start_positions = self._ctx.world_manager.get_positions_on_map(n=n, safe_dist=biggest_robot, forbidden_zones=goal_clearance(goal_positions, tolerance))

        starts = [Pose(p, Orientation.from_yaw(2 * math.pi * rng.random())) for p in start_positions]
        goals = [Pose(p, Orientation.from_yaw(2 * math.pi * rng.random())) for p in goal_positions]

        for robot, start, goal in zip(self._ctx.robots.values(), starts, goals, strict=True):
            phases: list[TaskPhase] = [
                GoToPhase(pose=goal),
                ReachPhase(named_target="ready", planning_time=2.0),
                ReachPhase(random=True, planning_time=2.0),
                ReachPhase(named_target="stow", planning_time=2.0),
            ]
            plan.robots[robot.name] = RobotReset(start_pose=start, task=TaskRequest(phases=phases))
        return plan
