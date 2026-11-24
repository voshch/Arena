import math
import typing

import shapely
from arena_runtime.env.task_api import GoToPhase, TaskRequest
from arena_runtime.env.task_layer import ResetPlan, RobotReset
from arena_runtime.shared import Orientation, Pose, Position

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.hold import HoldParams
from task_generator.tasks.robots.hold_area import hold_within
from task_generator.tasks.robots.random.impl import goal_clearance
from task_generator.tasks.robots.sampling_area import sampling_area


class TM_Hold(TM_Robots):
    """Hold task mode: each robot drives to a spot the seed draws from a zone or around a center, and must stay in the target disc around it for hold_s seconds."""

    _params: HoldParams

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        TM_Robots.__init__(self, **kwargs)

        self._params = HoldParams(self.node, namespace=self.namespace())

    def _region(self) -> shapely.Geometry:
        """The polygon the spots are drawn from: the named zone, or the disc around the center."""
        region = sampling_area(
            prefix="task.hold.",
            world=self._ctx.world_manager,
            zone=self._params.ZONE.value,
            center_x=self._params.CENTER_X.value,
            center_y=self._params.CENTER_Y.value,
            radius=self._params.RADIUS.value,
        )
        if region is None:
            raise ValueError("task.hold: set task.hold.zone or both task.hold.center_x and task.hold.center_y")
        return region

    def _hold_phase(self, goal: Pose) -> GoToPhase:
        """Dispatches to the goal, and counts the hold from the moment the robot is within the goal's target disc."""
        return hold_within(goal, self._ctx.conf.GOAL_TOLERANCE_RADIUS.value, self._params.HOLD_S.value)

    async def reset(self) -> ResetPlan:
        plan = await super().reset()

        robots = list(self._ctx.robots.values())
        if not robots:
            return plan
        region = self._region()
        world = self._ctx.world_manager
        biggest_robot = max(robot.safe_distance for robot in robots)
        tolerance = self._ctx.conf.GOAL_TOLERANCE_RADIUS.value
        rng = self._ctx.conf.RNG.stream("robots", "hold")

        spots: list[Position] = world.get_positions_on_map(n=len(robots), safe_dist=biggest_robot, polygon=region)
        starts = world.get_positions_on_map(n=len(robots), safe_dist=biggest_robot, forbidden_zones=goal_clearance(spots, tolerance))

        for robot, start, spot in zip(robots, starts, spots, strict=True):
            start_pose = Pose(start, Orientation.from_yaw(2 * math.pi * rng.random()))
            goal = Pose(spot, Orientation.from_yaw(2 * math.pi * rng.random()))
            plan.robots[robot.name] = RobotReset(start_pose=start_pose, task=TaskRequest(phases=[self._hold_phase(goal)]))
        return plan
