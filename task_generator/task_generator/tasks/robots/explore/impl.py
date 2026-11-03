import math
import typing
from collections.abc import Collection, Mapping

from arena_rclpy_mixins.Time import Time
from arena_runtime.env.task_api import GoToPhase, PhaseState, TaskRequest
from arena_runtime.env.task_layer import ResetPlan
from arena_runtime.shared import Orientation, Pose

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.robots.random.impl import TM_Random


class TM_Explore(TM_Random):
    """Explore task mode: each robot roams to random reachable poses."""

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        super().__init__(**kwargs)
        self._timeouts: dict[str, Time] = {}
        self._assigned: set[str] = set()

    async def reset(self) -> ResetPlan:
        plan = await super().reset()
        self._timeouts = {}
        self._assigned = set()
        for name in self._ctx.robots.keys():
            self._reset_timeout(name)
        return plan

    async def tick(self, ended: Mapping[str, PhaseState], assigned: Collection[str]) -> None:
        """Give a robot whose task finished a fresh goal and teleport timed-out robots, except robots still holding a task assigned from outside."""
        self._assigned.update(assigned)
        for robot in list(self._assigned):
            manager = self._ctx.robots.get(robot)
            if manager is None or await manager.is_done:
                self._assigned.discard(robot)
        rng = self._ctx.conf.RNG.stream("robots", "explore")
        for robot, manager in self._ctx.robots.items():
            if robot in self._assigned:
                continue
            if await manager.is_done:
                waypoint = self._ctx.world_manager.get_position_on_map(safe_dist=manager.safe_distance, forbid=False)
                await self._set_goal(robot, Pose(waypoint, Orientation.from_yaw(rng.random() * 2 * math.pi)))

            elif (self.node.sim_time.sec - self._timeouts.get(robot, Time()).sec) >= self._ctx.conf.TIMEOUT.value:
                waypoint = self._ctx.world_manager.get_position_on_map(safe_dist=manager.safe_distance, forbid=False)
                await self._set_position(robot, Pose(waypoint, Orientation.from_yaw(rng.random() * 2 * math.pi)))

    @property
    async def done(self) -> bool:
        return False

    def _reset_timeout(self, robot: str):
        self._timeouts[robot] = self.node.sim_time

    async def _set_position(self, name: str, pose: Pose):
        self._reset_timeout(name)
        await self._ctx.robots[name].move(pose)

    async def _set_goal(self, name: str, pose: Pose):
        self._reset_timeout(name)
        await self._ctx.robots[name].submit_task(TaskRequest(phases=[GoToPhase(pose=pose)]))
