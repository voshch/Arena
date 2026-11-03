import typing
from collections.abc import Awaitable, Callable

import geometry_msgs.msg as geometry_msgs
from arena_runtime.env.task_api import GoToPhase, Menu, Realizer, TaskPhase, TaskRequest, planar_marker, static_marker, visual
from arena_runtime.env.task_layer import ResetPlan
from arena_runtime.shared import Pose
from visualization_msgs.msg import Marker

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.robots.guided import chain
from task_generator.tasks.robots.random.impl import TM_Random

_SHARED_CHAIN_COLOR = (0.1, 0.4, 1.0)
_LINE_WIDTH = 0.05


class TM_Guided(TM_Random):
    """Guided-waypoints task mode; emits a single multi-phase TaskRequest per robot."""

    PARAM_WAYPOINTS = "guided_waypoints"

    _waypoints: list[Pose]
    _starts: dict[str, Pose]

    async def reset(self) -> ResetPlan:
        plan = await super().reset()
        self._waypoints = []
        self._starts = {name: entry.start_pose for name, entry in plan.robots.items()}
        self._show_chain()
        for entry in plan.robots.values():
            entry.task = self._request(entry.start_pose)
        return plan

    @property
    async def done(self) -> bool:
        """Guided episodes end only through the rviz reset or the episode timeout."""
        return False

    def goal_editing_robots(self) -> frozenset[str]:
        return frozenset(self._ctx.robots)

    async def set_position(self, pose: Pose):
        del pose
        self._waypoints = []
        for robot in self._ctx.robots.values():
            await robot.move(robot.start_pos)
        await self._publish_chain()

    async def set_goal(self, pose: Pose):
        """Append a waypoint and re-submit the full sequence."""
        await self._edit([*self._waypoints, pose])

    async def _edit(self, waypoints: list[Pose]) -> None:
        self._waypoints = waypoints
        await self._publish_chain()

    def _show_chain(self) -> None:
        self.node.rosparam[list[list[float]]].set(self.PARAM_WAYPOINTS, [[wp.position.x, wp.position.y, wp.orientation.to_yaw()] for wp in self._waypoints])
        self._rebuild_markers()

    def _request(self, start: Pose) -> TaskRequest:
        phases: list[TaskPhase] = [GoToPhase(pose=wp) for wp in self._waypoints] or [GoToPhase(pose=start)]
        return TaskRequest(phases=phases)

    async def _publish_chain(self, from_start: bool = False) -> None:
        self._show_chain()
        for robot in self._ctx.robots.values():
            start = self._starts.get(robot.name, robot.start_pos)
            await robot.submit_task(self._request(start), start if from_start else None)

    async def _reset_waypoints(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self._waypoints = []
        await self._publish_chain(from_start=True)

    @property
    def _scope(self) -> str:
        return chain.scope_prefix(list(self._ctx.robots))

    def _rebuild_markers(self) -> None:
        interactive = self._ctx.interactive
        assert interactive is not None, "guided needs the env's interactive context"
        hub = interactive.markers
        scope = self._scope
        hub.erase_prefix(scope)
        if not self._waypoints:
            return

        realizer = self._ctx.realizer
        clear_menu = Menu("Clear chain", self._clear_chain)
        robots = list(self._ctx.robots)
        r, g, b, _ = interactive.robot_colors.rgba(robots[0], 1.0) if len(robots) == 1 else (*_SHARED_CHAIN_COLOR, 1.0)

        for index, waypoint in enumerate(self._waypoints):
            hub.put(
                planar_marker(
                    chain.waypoint_name(scope, index),
                    realizer.realize(waypoint).to_msg(),
                    description=str(index + 1),
                    scale=0.8,
                    visuals=[
                        visual(Marker.CYLINDER, (0.4, 0.4, 0.02), (r, g, b, 0.7)),
                        visual(Marker.ARROW, (0.5, 0.08, 0.08), (r, g, b, 0.7)),
                    ],
                ),
                on_pose=self._waypoint_mover(index, realizer),
                menu=[
                    Menu("Delete", self._waypoint_deleter(index)),
                    Menu("Insert after", self._waypoint_inserter(index)),
                    clear_menu,
                ],
            )

        if len(self._waypoints) > 1:
            line = visual(Marker.LINE_STRIP, (_LINE_WIDTH, 0.0, 0.0), (r, g, b, 0.5))
            line.points = [realizer.realize(waypoint).position.to_msg() for waypoint in self._waypoints]
            anchor = geometry_msgs.Pose()
            anchor.orientation.w = 1.0
            hub.put(static_marker(chain.line_name(scope), anchor, [line]), menu=[clear_menu])

    def _waypoint_mover(self, index: int, realizer: Realizer) -> Callable[[geometry_msgs.Pose], Awaitable[None]]:
        async def move(msg: geometry_msgs.Pose) -> None:
            await self._edit(chain.replace(self._waypoints, index, realizer.ezilear(Pose.from_msg(msg))))

        return move

    def _waypoint_deleter(self, index: int) -> Callable[[], Awaitable[None]]:
        async def delete() -> None:
            await self._edit(chain.delete(self._waypoints, index))

        return delete

    def _waypoint_inserter(self, index: int) -> Callable[[], Awaitable[None]]:
        async def insert() -> None:
            await self._edit(chain.insert_after(self._waypoints, index))

        return insert

    async def _clear_chain(self) -> None:
        await self._reset_waypoints()

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        super().__init__(**kwargs)
        self._waypoints = []
        self._starts = {}
