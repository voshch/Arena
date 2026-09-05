from task_generator.shared import Pose
from task_generator.tasks import TaskContext
from task_generator.tasks.robots.random.impl import TM_Random
from task_generator.tasks.robots.request import GoToPhase, TaskPhase, TaskRequest


class TM_Guided(TM_Random):
    """Guided-waypoints task mode; emits a single multi-phase TaskRequest per robot."""

    PARAM_WAYPOINTS = "guided_waypoints"

    _waypoints: list[Pose]

    async def reset(self) -> None:
        await super().reset()
        await self._reset_waypoints()

    async def set_position(self, pose: Pose):
        del pose
        self._waypoints = []
        for robot in self._ctx.robots.values():
            await robot.move(robot.start_pos)
            await robot.submit_task(TaskRequest(phases=[GoToPhase(pose=robot.start_pos)]))
        self.node.rosparam[list[list[float]]].set(self.PARAM_WAYPOINTS, [])

    async def set_goal(self, pose: Pose):
        """Append a waypoint and re-submit the full sequence."""
        self._waypoints.append(pose)
        self.node.rosparam[list[list[float]]].set(self.PARAM_WAYPOINTS, [[wp.position.x, wp.position.y, wp.orientation.to_yaw()] for wp in self._waypoints])

        phases: list[TaskPhase] = [GoToPhase(pose=wp) for wp in self._waypoints]
        request = TaskRequest(phases=phases)
        for robot in self._ctx.robots.values():
            await robot.submit_task(request)

    async def _reset_waypoints(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self._waypoints = []

        for robot in self._ctx.robots.values():
            self._start_poses[robot.name] = robot.start_pos
            await robot.submit_task(TaskRequest(phases=[GoToPhase(pose=robot.start_pos)]))

        self.node.rosparam[list[list[float]]].set(self.PARAM_WAYPOINTS, [])

    def __init__(self, **kwargs: TaskContext) -> None:
        super().__init__(**kwargs)
        self._waypoints = []
