import math
import typing

from arena_runtime.env.task_layer import ResetPlan, RobotReset
from arena_runtime.shared import Orientation, Pose, Position

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.stationary import StationaryParams


class TM_Stationary(TM_Robots):
    """Stationary task mode: robot stays parked without goal dispatch."""

    _params: StationaryParams

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        TM_Robots.__init__(self, **kwargs)

        self._params = StationaryParams(self.node, namespace=self.namespace())

    async def reset(self) -> ResetPlan:
        plan = await super().reset()

        pos_x, pos_y = self._params.POS_X.value, self._params.POS_Y.value

        robots = list(self._ctx.robots.values())
        if math.isfinite(pos_x) and math.isfinite(pos_y):
            override_pose = Pose(
                position=Position(x=pos_x, y=pos_y, z=0.0),
                orientation=Orientation.from_yaw(self._params.POS_THETA.value),
            )
            starts = [override_pose] * len(robots)
        else:
            rng = self._ctx.conf.RNG.stream("robots", "stationary")
            biggest_robot = max((robot.safe_distance for robot in robots), default=0)
            starts = [Pose(p, Orientation.from_yaw(2 * math.pi * rng.random())) for p in self._ctx.world_manager.get_positions_on_map(n=len(robots), safe_dist=biggest_robot)]

        for robot, start in zip(robots, starts, strict=True):
            plan.robots[robot.name] = RobotReset(start_pose=start)
        return plan

    @property
    async def done(self) -> bool:
        return False

    async def set_position(self, pose: Pose):
        for robot in self._ctx.robots.values():
            await robot.move(pose)

    async def set_goal(self, pose: Pose):
        del pose
        return None
