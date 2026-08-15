"""External-planner adapter: hand-off to third-party navigation stacks."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from arena_robots.bringup.mobile.external import ExternalBringup
from arena_robots.clients.goto_pose import GotoPoseClient
from arena_robots.task_kinds import TaskKind
from arena_robots_msgs.action import GotoPose

from task_generator.manager.world_manager.shims import requires_map_server
from task_generator.tasks.robots.adapters import AdapterMeta
from task_generator.tasks.robots.adapters.mobile import MobileAdapter
from task_generator.tasks.robots.request import GoToPhase, TaskPhase

if TYPE_CHECKING:
    import geometry_msgs.msg

    from task_generator.manager.robot_manager.robot_manager import RobotManager
    from task_generator.tasks.robots.adapters import ResetContext


@AdapterMeta.attach(
    accepts={TaskKind.GOTO_POSE},
    bringup=ExternalBringup,
    client=GotoPoseClient,
    cap="mobile",
)
@requires_map_server
class ExternalAdapter(MobileAdapter):
    kind: ClassVar[str] = "external"

    def is_phase_done(self, phase: TaskPhase, robot: RobotManager) -> bool | None:
        return None

    async def dispatch_phase(
        self,
        phase: TaskPhase,
        robot: RobotManager,
    ) -> None:
        assert isinstance(phase, GoToPhase), f"ExternalAdapter only accepts GOTO_POSE phases; got {type(phase).__name__} (kind={phase.kind!r})"
        robot._goal_pos = phase.pose  # pylint: disable=protected-access
        if self.client.is_done() is False:
            self.client.cancel()
        goal = GotoPose.Goal()
        goal.target = self._phase_to_pose_stamped(phase, robot)
        goal.pose_tolerance, goal.yaw_tolerance = self._resolve_tolerances(phase, robot)
        await self.client.send_goal(goal)

    async def on_reset(self, robot: RobotManager, ctx: ResetContext) -> None:
        if self.client.is_done() is False:
            self.client.cancel()
        await super().on_reset(robot, ctx)

    def _phase_to_pose_stamped(
        self,
        phase: GoToPhase,
        robot: RobotManager,
    ) -> geometry_msgs.msg.PoseStamped:
        import geometry_msgs.msg

        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = robot.node.sim_time.to_msg()
        msg.pose = phase.pose.to_msg()
        return msg


__all__ = ["ExternalAdapter"]
