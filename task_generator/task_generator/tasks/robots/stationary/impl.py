import math

from arena_rclpy_mixins.ROSParamServer import ROSParamT

from task_generator.shared import Orientation, Pose, Position
from task_generator.tasks.robots import TM_Robots


class TM_Stationary(TM_Robots):
    """Stationary task mode: robot stays parked at start pose without goal dispatch."""

    _pos_x: ROSParamT[float]
    _pos_y: ROSParamT[float]
    _pos_theta: ROSParamT[float]

    def __init__(self, **kwargs: object) -> None:
        TM_Robots.__init__(self, **kwargs)

        self._pos_x = self.node.ROSParam[float](self.namespace('pos_x'), math.nan)
        self._pos_y = self.node.ROSParam[float](self.namespace('pos_y'), math.nan)
        self._pos_theta = self.node.ROSParam[float](self.namespace('pos_theta'), 0.0)

    async def reset(self) -> None:
        await super().reset()

        pos_x, pos_y = self._pos_x.value, self._pos_y.value

        override_pose: Pose | None = None
        if math.isfinite(pos_x) and math.isfinite(pos_y):
            override_pose = Pose(
                position=Position(x=pos_x, y=pos_y, z=0.0),
                orientation=Orientation.from_yaw(self._pos_theta.value),
            )

        for robot in self._ctx.robots.values():
            self._start_poses[robot.name] = override_pose if override_pose is not None else robot.start_pos

    @property
    async def done(self) -> bool:
        return False

    async def set_position(self, pose: Pose):
        for robot in self._ctx.robots.values():
            await robot.move(pose)

    async def set_goal(self, pose: Pose):
        del pose
        return None
