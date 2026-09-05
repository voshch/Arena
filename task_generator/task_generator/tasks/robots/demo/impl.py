import enum
import math

from arena_rclpy_mixins.declarations import declare_double, declare_enum, declare_int
from arena_rclpy_mixins.ROSParamServer import ROSParamT

from task_generator.shared import Orientation, Pose, Position, PositionRadius
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.request import GoToPhase, PlayGesturePhase, TaskPhase, TaskRequest

_RANDOM_ALIASES = frozenset({"", "random", "<random>"})


def _parse_gesture(value: str) -> str:
    from arena_simulation_setup import ASS_DIR

    if value in _RANDOM_ALIASES:
        return "<random>"
    available = {p.stem for p in (ASS_DIR / "configs" / "gestures").glob("*.yaml")}
    if value not in available:
        raise ValueError(f"unknown gesture {value!r}; available: {sorted(available | {'<random>'})}")
    return value


class _Orientation(enum.Enum):
    TANGENT = "tangent"
    RADIAL_IN = "radial_in"
    RADIAL_OUT = "radial_out"


def _vertices(center: Position, n: int, radius: float, orientation: _Orientation) -> list[Pose]:
    if n < 3:
        raise ValueError(f"TM_Demo.VERTICES must be >= 3; got {n}")
    if radius <= 0:
        raise ValueError(f"TM_Demo.RADIUS must be > 0; got {radius}")
    if not isinstance(orientation, _Orientation):
        raise ValueError(f"TM_Demo.ORIENTATION must be one of {[o.value for o in _Orientation]}; got {orientation!r}")
    phase = -math.pi / 2 + math.pi / n
    pts = [(center.x + radius * math.cos(phase + 2 * math.pi * i / n), center.y + radius * math.sin(phase + 2 * math.pi * i / n)) for i in range(n)]
    out: list[Pose] = []
    for i, (x, y) in enumerate(pts):
        nx, ny = pts[(i + 1) % n]
        if orientation is _Orientation.TANGENT:
            yaw = math.atan2(ny - y, nx - x)
        elif orientation is _Orientation.RADIAL_IN:
            yaw = math.atan2(center.y - y, center.x - x)
        else:
            yaw = math.atan2(y - center.y, x - center.x)
        out.append(Pose(Position(x, y, 0.0), Orientation.from_yaw(yaw)))
    return out


def _pick_gesture(value: str) -> str | None:
    return None if value == "<random>" else value


def _available_gestures() -> list[str]:
    from arena_simulation_setup import ASS_DIR

    return sorted({p.stem for p in (ASS_DIR / "configs" / "gestures").glob("*.yaml")})


class TM_Demo(TM_Robots):
    """Drive a regular N-vertex polygon; gesture at every vertex."""

    _vertices_p: ROSParamT[int]
    _radius_p: ROSParamT[float]
    _gesture_p: ROSParamT[str]
    _orientation_p: ROSParamT[_Orientation]

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        declare_int(self.node, str(self.namespace("vertices")), 4, label="Vertices", lo=3, hi=20)
        declare_double(self.node, str(self.namespace("radius")), 1.5, label="Radius (m)", lo=0.5, hi=10.0)
        declare_enum(
            self.node,
            str(self.namespace("gesture")),
            "<random>",
            choices=["<random>", *_available_gestures()],
            label="Gesture",
        )
        declare_enum(
            self.node,
            str(self.namespace("orientation")),
            _Orientation.RADIAL_IN.value,
            choices=[o.value for o in _Orientation],
            label="Orientation",
        )
        self._vertices_p = self.node.ROSParam[int](self.namespace("vertices"))
        self._radius_p = self.node.ROSParam[float](self.namespace("radius"))
        self._gesture_p = self.node.ROSParam[str](self.namespace("gesture"), parse=_parse_gesture)
        self._orientation_p = self.node.ROSParam[_Orientation](self.namespace("orientation"), parse=_Orientation)

    async def reset(self) -> None:
        await super().reset()
        gesture_name = self._gesture_p.value
        radius = self._radius_p.value
        orientation = self._orientation_p.value
        n = self._vertices_p.value
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
            self._start_poses[robot.name] = center_pose
            phases: list[TaskPhase] = []
            for v in vertices:
                phases.append(GoToPhase(pose=v))
                phases.append(PlayGesturePhase(gesture=_pick_gesture(gesture_name)))
            phases.append(GoToPhase(pose=center_pose))
            await robot.submit_task(TaskRequest(phases=phases))
            self._ctx.world_manager.forbid(
                [
                    PositionRadius(center.x, center.y, biggest_robot),
                    *(PositionRadius(v.position.x, v.position.y, biggest_robot) for v in vertices),
                ]
            )
