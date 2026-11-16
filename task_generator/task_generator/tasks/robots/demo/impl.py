import math
import typing

from arena_runtime.env.task_api import GoToPhase, PlayGesturePhase, TaskPhase, TaskRequest
from arena_runtime.env.task_layer import ResetPlan, RobotReset
from arena_runtime.shared import Orientation, Pose, Position, PositionRadius

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.demo import DemoOrientation, DemoParams


def _vertices(center: Position, n: int, radius: float, orientation: DemoOrientation) -> list[Pose]:
    if n < 3:
        raise ValueError(f"TM_Demo.VERTICES must be >= 3; got {n}")
    if radius <= 0:
        raise ValueError(f"TM_Demo.RADIUS must be > 0; got {radius}")
    if not isinstance(orientation, DemoOrientation):
        raise ValueError(f"TM_Demo.ORIENTATION must be one of {[o.value for o in DemoOrientation]}; got {orientation!r}")
    phase = -math.pi / 2 + math.pi / n
    pts = [(center.x + radius * math.cos(phase + 2 * math.pi * i / n), center.y + radius * math.sin(phase + 2 * math.pi * i / n)) for i in range(n)]
    out: list[Pose] = []
    for i, (x, y) in enumerate(pts):
        nx, ny = pts[(i + 1) % n]
        if orientation is DemoOrientation.TANGENT:
            yaw = math.atan2(ny - y, nx - x)
        elif orientation is DemoOrientation.RADIAL_IN:
            yaw = math.atan2(center.y - y, center.x - x)
        else:
            yaw = math.atan2(y - center.y, x - center.x)
        out.append(Pose(Position(x, y, 0.0), Orientation.from_yaw(yaw)))
    return out


def _pick_gesture(value: str) -> str | None:
    return None if value == "<random>" else value


class TM_Demo(TM_Robots):
    """Drive a regular N-vertex polygon; gesture at every vertex."""

    _params: DemoParams

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        super().__init__(**kwargs)
        self._params = DemoParams(self.node, namespace=self.namespace())

    async def reset(self) -> ResetPlan:
        plan = await super().reset()
        gesture_name = self._params.GESTURE.value
        radius = self._params.RADIUS.value
        orientation = self._params.ORIENTATION.value
        n = self._params.VERTICES.value
        robots = list(self._ctx.robots.values())
        biggest_robot = max((r.safe_distance for r in robots), default=0)
        centers = self._ctx.world_manager.get_positions_on_map(
            n=len(robots),
            safe_dist=radius + biggest_robot,
        )
        if len(centers) < len(robots):
            self._logger.warn(
                f"TM_Demo: only {len(centers)} safe centers for {len(robots)} robots; unassigned robots will idle this episode.",
                once=True,
            )
        for robot, center in zip(robots, centers, strict=False):
            vertices = _vertices(center, n, radius, orientation)
            v0 = vertices[0]
            spawn_yaw = math.atan2(v0.position.y - center.y, v0.position.x - center.x)
            center_pose = Pose(Position(center.x, center.y, 0.0), Orientation.from_yaw(spawn_yaw))
            phases: list[TaskPhase] = []
            for v in vertices:
                phases.append(GoToPhase(pose=v))
                phases.append(PlayGesturePhase(gesture=_pick_gesture(gesture_name)))
            phases.append(GoToPhase(pose=center_pose))
            plan.robots[robot.name] = RobotReset(start_pose=center_pose, task=TaskRequest(phases=phases))
            plan.forbidden_zones.extend(
                [
                    PositionRadius(x=center.x, y=center.y, radius=biggest_robot),
                    *(PositionRadius(x=v.position.x, y=v.position.y, radius=biggest_robot) for v in vertices),
                ]
            )
        return plan
