import asyncio
from collections.abc import Collection, Mapping

from arena_runtime.env.task_api import GoToPhase, PhaseState, TaskRequest
from arena_runtime.env.task_layer import ResetPlan
from arena_runtime.shared import Pose

from task_generator.tasks.mode import TaskMode

from . import characterization, demo, explore, guided, random, scenario, stationary


class TM_Robots(TaskMode):
    """
    Task mode for controlling one or multiple robots.

    Attributes:
        _ctx (TaskContext): Shared task context.
    """

    async def reset(self) -> ResetPlan:
        """The starts and tasks this mode wants for the next episode, computed with the sim paused."""
        return ResetPlan()

    async def tick(self, ended: Mapping[str, PhaseState], assigned: Collection[str]) -> None:
        """Runs every sim tick of a running episode, ended maps a robot to the phase that ended on this tick and assigned holds the robots given a task from outside since the last tick."""

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
