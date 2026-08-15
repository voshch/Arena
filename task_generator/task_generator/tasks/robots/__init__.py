import asyncio

from task_generator.shared import Pose
from task_generator.shared import Robot as RobotEntity
from task_generator.tasks.mode import TaskMode
from task_generator.tasks.robots._placement import random_placement
from task_generator.tasks.robots.request import GoToPhase, TaskRequest

from . import characterization, demo, explore, guided, random, scenario, stationary


class TM_Robots(TaskMode):
    """
    Task mode for controlling one or multiple robots.

    Attributes:
        _ctx (TaskContext): Shared task context.
    """

    _last_reset: int
    _start_poses: dict[str, Pose]

    @property
    def start_poses(self) -> dict[str, Pose]:
        return self._start_poses

    async def reset(self, **kwargs: object) -> None:
        self._last_reset = self.node.sim_time.sec
        self._start_poses = {}

    async def teardown(self) -> None:
        """Release everything this mode drives; called when the mode is replaced or the task ends."""

    async def set_position(self, pose: Pose):
        """Handle an external pose-estimate override for the robots in this mode.

        Default: teleport only the most recently added robot in scope to ``pose``.
        """
        robots = self._ctx.robots
        if robots:
            await next(reversed(robots.values())).move(pose)

    async def set_goal(self, pose: Pose):
        """Dispatch a single-phase GOTO request targeting ``pose`` on every robot."""
        for robot_manager in self._ctx.robots.values():
            await robot_manager.submit_task(TaskRequest(phases=[GoToPhase(pose=pose)]))

    async def extend(
        self,
        model: str,
        name: str | None = None,
        pose: Pose | None = None,
        args: dict[str, str] | None = None,
    ) -> str:
        from arena_robots.SetupFile import Config as RobotSetupConfig

        resolved_pose = pose if pose is not None else await random_placement(self._ctx)
        assigned_name = name or self.node._robots_manager.next_name(model)
        config_dict: dict[str, object] = {'robot': model, 'name': assigned_name, 'pos': resolved_pose.to_2d(), **(args or {})}
        (config,) = RobotSetupConfig.parse(config_dict)
        robot = RobotEntity.from_setup(config, node=self.node)
        self.node._robots_manager.add_pending(assigned_name, robot)
        return assigned_name

    @property
    async def done(self) -> bool:
        if (self.node.sim_time.sec - self._last_reset) > self.node.conf.Robot.TIMEOUT.value:
            return True

        if not self._ctx.robots:
            return False
        if not all(await asyncio.gather(*(robot_manager.is_done for robot_manager in self._ctx.robots.values()))):
            return False
        return True


__all__ = ["TM_Robots", "characterization", "demo", "explore", "guided", "random", "scenario", "stationary"]
