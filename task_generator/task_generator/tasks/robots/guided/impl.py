from collections.abc import Awaitable, Callable

import geometry_msgs.msg as geometry_msgs
from visualization_msgs.msg import Marker

from task_generator.interactive.hub import Menu, planar_marker, static_marker, visual
from task_generator.manager.realizer import Realizer
from task_generator.shared import Pose
from task_generator.tasks import TaskContext
from task_generator.tasks.robots.guided import chain
from task_generator.tasks.robots.random.impl import TM_Random
from task_generator.tasks.robots.request import GoToPhase, TaskPhase, TaskRequest

_SHARED_CHAIN_COLOR = (0.1, 0.4, 1.0)
_LINE_WIDTH = 0.05


class TM_Guided(TM_Random):
    """Guided-waypoints task mode; emits a single multi-phase TaskRequest per robot."""

    PARAM_WAYPOINTS = "guided_waypoints"

    _waypoints: list[Pose]

    async def reset(self) -> None:
        await super().reset()
        await self._reset_waypoints()

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

    async def _publish_chain(self, from_start: bool = False) -> None:
        self.node.rosparam[list[list[float]]].set(self.PARAM_WAYPOINTS, [[wp.position.x, wp.position.y, wp.orientation.to_yaw()] for wp in self._waypoints])
        self._rebuild_markers()

        for robot in self._ctx.robots.values():
            phases: list[TaskPhase] = [GoToPhase(pose=wp) for wp in self._waypoints] or [GoToPhase(pose=robot.start_pos)]
            await robot.submit_task(TaskRequest(phases=phases), robot.start_pos if from_start else None)

    async def _reset_waypoints(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self._waypoints = []

        for robot in self._ctx.robots.values():
            self._start_poses[robot.name] = robot.start_pos

        await self._publish_chain(from_start=True)

    @property
    def _scope(self) -> str:
        return chain.scope_prefix(list(self._ctx.robots))

    def _rebuild_markers(self) -> None:
        hub = self.node.markers
        scope = self._scope
        hub.erase_prefix(scope)
        if not self._waypoints:
            return

        realizer = self.node._realizer
        clear_menu = Menu("Clear chain", self._clear_chain)
        robots = list(self._ctx.robots)
        r, g, b, _ = self.node.robot_colors.rgba(robots[0], 1.0) if len(robots) == 1 else (*_SHARED_CHAIN_COLOR, 1.0)

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
            for waypoint in self._waypoints:
                line.points.append(realizer.realize(waypoint).position.to_msg())
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

    def __init__(self, **kwargs: TaskContext) -> None:
        super().__init__(**kwargs)
        self._waypoints = []
