import asyncio
import uuid

from task_generator.shared import Pose
from task_generator.shared import Robot as RobotEntity
from task_generator.tasks.mode import TaskMode
from task_generator.tasks.robots._placement import random_placement
from task_generator.tasks.robots.request import GoToPhase, TaskRequest

from . import demo, explore, guided, random, scenario, vla


class TM_Robots(TaskMode):
    """
    Task mode for controlling one or multiple robots.

    Args:
        **kwargs: Additional keyword arguments.

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

    async def set_position(self, pose: Pose):
        """Teleport every robot to ``pose``."""
        for robot_manager in self._ctx.robots.values():
            await robot_manager.move(pose)

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
        resolved_pose = pose if pose is not None else await random_placement(self._ctx)
        assigned_name = name or f"{model}_{uuid.uuid4().hex[:6]}"
        value: dict[str, object] = dict(args or {})
        value['model'] = model
        value['name'] = assigned_name
        value['pos'] = resolved_pose.to_2d()
        robot = RobotEntity.parse(value, node=self.node)
        self.node._robots_manager.add_pending(assigned_name, robot)
        return assigned_name

    @property
    async def done(self) -> bool:
        """
        Check if all robots have completed their tasks.

        Returns:
            bool: True if all robots are done, False otherwise.

        """
        if (self.node.sim_time.sec - self._last_reset) > self.node.conf.Robot.TIMEOUT.value:
            return True

        if not self._ctx.robots:
            return False
        if not all(await asyncio.gather(*(robot_manager.is_done for robot_manager in self._ctx.robots.values()))):
            return False
        return True


__all__ = ["TM_Robots", "demo", "explore", "guided", "random", "scenario", "vla"]
