import math
import typing
from collections.abc import Sequence

import shapely
from arena_runtime.env.task_api import GoToPhase, ReachPhase, TaskPhase, TaskRequest
from arena_runtime.env.task_layer import ResetPlan, RobotReset
from arena_runtime.shared import Orientation, Pose, Position, PositionRadius

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.hold_area import hold_within
from task_generator.tasks.robots.random import RandomParams
from task_generator.tasks.robots.sampling_area import sampling_area


def goal_clearance(goals: Sequence[Position], tolerance: float) -> list[PositionRadius]:
    """Zones a start must stay out of so that no robot begins inside a goal's tolerance."""
    return [PositionRadius(x=goal.x, y=goal.y, radius=tolerance) for goal in goals]


class TM_RandomBase(TM_Robots):
    """Random task mode with the whole map as goal and start area and a plain move to each goal. Subclasses override the areas and the move."""

    def _goal_area(self) -> shapely.Geometry | None:
        """The polygon goals are drawn from, None for the whole map."""
        return None

    def _start_area(self) -> shapely.Geometry | None:
        """The polygon starts are drawn from, None for the whole map."""
        return None

    def _goal_phase(self, goal: Pose) -> GoToPhase:
        """The move to a robot's goal, the first phase of its task."""
        return GoToPhase(pose=goal)

    async def reset(self) -> ResetPlan:
        plan = await super().reset()

        biggest_robot = max((robot.safe_distance for robot in self._ctx.robots.values()), default=0)
        rng = self._ctx.conf.RNG.stream("robots", "random")
        n = len(self._ctx.robots)

        goal_positions = self._place(n, what="goal", safe_dist=0, polygon=self._goal_area())
        tolerance = self._ctx.conf.GOAL_TOLERANCE_RADIUS.value
        start_positions = self._place(n, what="start", safe_dist=biggest_robot, polygon=self._start_area(), forbidden_zones=goal_clearance(goal_positions, tolerance))

        starts = [Pose(p, Orientation.from_yaw(2 * math.pi * rng.random())) for p in start_positions]
        goals = [Pose(p, Orientation.from_yaw(2 * math.pi * rng.random())) for p in goal_positions]

        for robot, start, goal in zip(self._ctx.robots.values(), starts, goals, strict=True):
            phases: list[TaskPhase] = [
                self._goal_phase(goal),
                ReachPhase(named_target="ready", planning_time=2.0),
                ReachPhase(random=True, planning_time=2.0),
                ReachPhase(named_target="stow", planning_time=2.0),
            ]
            plan.robots[robot.name] = RobotReset(start_pose=start, task=TaskRequest(phases=phases))
        return plan

    def _place(self, n: int, *, what: str, safe_dist: float, polygon: shapely.Geometry | None, forbidden_zones: list[PositionRadius] | None = None) -> list[Position]:
        """n positions inside polygon; when they do not fit, the error names the parameters that set the area."""
        try:
            return self._ctx.world_manager.get_positions_on_map(n=n, safe_dist=safe_dist, polygon=polygon, forbidden_zones=forbidden_zones)
        except RuntimeError as error:
            raise RuntimeError(f"task.robots_random: fewer than {n} {what} positions fit in the {what} area (task.robots_random.{what}_zone, or task.robots_random.{what}_center_x, {what}_center_y and {what}_radius): {error}") from error


class TM_Random(TM_RandomBase):
    """Random task mode: goals and starts come from the whole map, or from a declared zone or disc, and each robot may hold in its goal's target disc."""

    _params: RandomParams

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        super().__init__(**kwargs)

        self._params = RandomParams(self.node, namespace=self.namespace())

    def _goal_area(self) -> shapely.Geometry | None:
        return sampling_area(
            prefix="task.robots_random.goal_",
            world=self._ctx.world_manager,
            zone=self._params.GOAL_ZONE.value,
            center_x=self._params.GOAL_CENTER_X.value,
            center_y=self._params.GOAL_CENTER_Y.value,
            radius=self._params.GOAL_RADIUS.value,
        )

    def _start_area(self) -> shapely.Geometry | None:
        return sampling_area(
            prefix="task.robots_random.start_",
            world=self._ctx.world_manager,
            zone=self._params.START_ZONE.value,
            center_x=self._params.START_CENTER_X.value,
            center_y=self._params.START_CENTER_Y.value,
            radius=self._params.START_RADIUS.value,
        )

    def _goal_phase(self, goal: Pose) -> GoToPhase:
        """The move to the goal; with hold_s above zero it ends once the robot has stayed in the goal's target disc for hold_s seconds."""
        seconds = self._params.HOLD_S.value
        if seconds <= 0:
            return super()._goal_phase(goal)
        return hold_within(goal, self._ctx.conf.GOAL_TOLERANCE_RADIUS.value, seconds)
