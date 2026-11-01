import asyncio

from arena_runtime.env.request import GoToPhase, TaskRequest
from arena_runtime.shared import Pose

from task_generator.tasks.mode import TaskMode

from . import characterization, demo, explore, guided, random, scenario, stationary


class TM_Robots(TaskMode):
    """
    Task mode for controlling one or multiple robots.

    Attributes:
        _ctx (TaskContext): Shared task context.
    """

    _start_poses: dict[str, Pose]

    @property
    def start_poses(self) -> dict[str, Pose]:
        return self._start_poses

    async def reset(self) -> None:
        self._start_poses = {}

    async def teardown(self) -> None:
        """Release everything this mode drives; called when the mode is replaced or the task ends."""

    def goal_editing_robots(self) -> frozenset[str]:
        """Names of robots whose goals this mode already exposes an editor for."""
        return frozenset()

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

    @property
    async def done(self) -> bool:
        if not self._ctx.robots:
            return False
        if not all(await asyncio.gather(*(robot_manager.is_done for robot_manager in self._ctx.robots.values()))):
            return False
        return True


__all__ = ["TM_Robots", "characterization", "demo", "explore", "guided", "random", "scenario", "stationary"]
