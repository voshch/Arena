import math

from task_generator.shared import Orientation, Pose
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.request import GoToPhase, ReachPhase, TaskPhase, TaskRequest


class TM_Random(TM_Robots):
    """Cap-blind random-action task mode."""

    async def reset(self) -> None:
        await super().reset()

        biggest_robot = max((robot.safe_distance for robot in self._ctx.robots.values()), default=0)
        rng = self.node.conf.General.RNG.stream("robots", "random")
        n = len(self._ctx.robots)

        goal_positions = self._ctx.world_manager.get_positions_on_map(n=n, safe_dist=0)
        start_positions = self._ctx.world_manager.get_positions_on_map(n=n, safe_dist=biggest_robot)

        starts = [Pose(p, Orientation.from_yaw(2 * math.pi * rng.random())) for p in start_positions]
        goals = [Pose(p, Orientation.from_yaw(2 * math.pi * rng.random())) for p in goal_positions]

        for robot, start, goal in zip(self._ctx.robots.values(), starts, goals, strict=True):
            self._start_poses[robot.name] = start
            phases: list[TaskPhase] = [
                GoToPhase(pose=goal),
                ReachPhase(named_target="ready", planning_time=2.0),
                ReachPhase(random=True, planning_time=2.0),
                ReachPhase(named_target="stow", planning_time=2.0),
            ]
            await robot.submit_task(TaskRequest(phases=phases))
