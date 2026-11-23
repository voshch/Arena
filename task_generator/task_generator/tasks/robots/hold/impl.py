import math
import typing

import shapely
from arena_runtime.env.task_api import GoToPhase, TaskRequest
from arena_runtime.env.task_layer import ResetPlan, RobotReset
from arena_runtime.shared import Orientation, Pose

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.hold import HoldParams
from task_generator.tasks.robots.random.impl import goal_clearance


class TM_Hold(TM_Robots):
    """Hold task mode: each robot drives to a spot the seed draws inside a zone or around a center, and holds there for hold_s seconds."""

    _params: HoldParams

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        TM_Robots.__init__(self, **kwargs)

        self._params = HoldParams(self.node, namespace=self.namespace())

    def _region(self) -> shapely.Polygon:
        """The polygon the hold spots are drawn from: the named zone, or the disc around the center."""
        zone = self._params.ZONE.value
        center_x, center_y = self._params.CENTER_X.value, self._params.CENTER_Y.value
        has_center = math.isfinite(center_x) and math.isfinite(center_y)
        if zone and has_center:
            raise ValueError("task.hold: set either task.hold.zone or task.hold.center_x and center_y, not both")
        if zone:
            vertices = self._ctx.world_manager.world_compacted().lookup_zone_polygon(zone)
            if vertices is None:
                raise ValueError(f"task.hold.zone {zone!r} is not a zone of the loaded world")
            return shapely.Polygon([(vertex.x, vertex.y) for vertex in vertices])
        if has_center:
            return shapely.Point(center_x, center_y).buffer(self._params.RADIUS.value)
        raise ValueError("task.hold: set task.hold.zone or both task.hold.center_x and task.hold.center_y")

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

        spots = world.get_positions_on_map(n=len(robots), safe_dist=biggest_robot, polygon=region)
        starts = world.get_positions_on_map(n=len(robots), safe_dist=biggest_robot, forbidden_zones=goal_clearance(spots, tolerance))

        for robot, start, spot in zip(robots, starts, spots, strict=True):
            start_pose = Pose(start, Orientation.from_yaw(2 * math.pi * rng.random()))
            goal = Pose(spot, Orientation.from_yaw(2 * math.pi * rng.random()))
            plan.robots[robot.name] = RobotReset(start_pose=start_pose, task=TaskRequest(phases=[GoToPhase(pose=goal, hold_time=self._params.HOLD_S.value)]))
        return plan
