import math

from arena_rclpy_mixins.Time import Time

from task_generator.shared import Orientation, Pose
from task_generator.tasks.robots.random.impl import TM_Random
from task_generator.tasks.robots.request import GoToPhase, TaskRequest


class TM_Explore(TM_Random):
    """Explore task mode: each robot roams to random reachable poses."""

    _timeouts: dict[str, Time]

    async def reset(self) -> None:
        await super().reset()
        self._timeouts = {}
        for name in self._ctx.robots.keys():
            self._reset_timeout(name)

    @property
    async def done(self) -> bool:
        """Roll over completed robots to fresh goals; teleport timed-out robots."""
        rng = self.node.conf.General.RNG.stream("robots", "explore")
        for robot, manager in self._ctx.robots.items():
            if await manager.is_done:
                waypoint = self._ctx.world_manager.get_position_on_map(safe_dist=manager.safe_distance, forbid=False)
                await self._set_goal(robot, Pose(waypoint, Orientation.from_yaw(rng.random() * 2 * math.pi)))

            elif (self.node.sim_time.sec - self._timeouts.get(robot, Time()).sec) >= self.node.conf.Robot.TIMEOUT.value:
                waypoint = self._ctx.world_manager.get_position_on_map(safe_dist=manager.safe_distance, forbid=False)
                await self._set_position(robot, Pose(waypoint, Orientation.from_yaw(rng.random() * 2 * math.pi)))

        return False

    def _reset_timeout(self, robot: str):
        self._timeouts[robot] = self.node.sim_time

    async def _set_position(self, name: str, pose: Pose):
        self._reset_timeout(name)
        await self._ctx.robots[name].move(pose)

    async def _set_goal(self, name: str, pose: Pose):
        self._reset_timeout(name)
        await self._ctx.robots[name].submit_task(TaskRequest(phases=[GoToPhase(pose=pose)]))

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
