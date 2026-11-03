"""External-planner adapter: hand-off to third-party navigation stacks."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from arena_robots.bringup.mobile.external import ExternalBringup
from arena_robots.clients.goto_pose import GotoPoseClient
from arena_robots.task_kinds import TaskKind
from arena_robots_msgs.action import GotoPose

from arena_runtime.env.adapters import AdapterMeta
from arena_runtime.env.adapters.mobile import MobileAdapter
from arena_runtime.env.request import GoToPhase, TaskPhase
from arena_runtime.env.world_manager.shims import requires_map_server

if TYPE_CHECKING:
    import geometry_msgs.msg

    from arena_runtime.env.adapters import ResetContext
    from arena_runtime.env.robot_manager.robot_manager import RobotManager


@AdapterMeta.attach(
    accepts={TaskKind.GOTO_POSE},
    bringup=ExternalBringup,
    client=GotoPoseClient,
    cap="mobile",
)
@requires_map_server
class ExternalAdapter(MobileAdapter):
    kind: ClassVar[str] = "external"

    async def dispatch_phase(
        self,
        phase: TaskPhase,
        robot: RobotManager,
    ) -> None:
        assert isinstance(phase, GoToPhase), f"ExternalAdapter only accepts GOTO_POSE phases; got {type(phase).__name__} (kind={phase.kind!r})"
        assert phase.pose is not None
        robot._goal_pos = phase.pose  # pylint: disable=protected-access
        if self.client.is_done() is False:
            self.client.cancel()
        await self._send_goal(phase, robot)

    async def on_reset(self, robot: RobotManager, ctx: ResetContext) -> None:
        if self.client.is_done() is False:
            self.client.cancel()
        await super().on_reset(robot, ctx)
        phase = robot.active_dispatch_phase()
        if isinstance(phase, GoToPhase):
            await self._send_goal(phase, robot)

    async def _send_goal(self, phase: GoToPhase, robot: RobotManager) -> None:
        goal = GotoPose.Goal()
        goal.target = self._phase_to_pose_stamped(phase, robot)
        goal.pose_tolerance, goal.yaw_tolerance = self._resolve_tolerances(phase, robot)
        await self.client.send_goal(goal)

    def _phase_to_pose_stamped(
        self,
        phase: GoToPhase,
        robot: RobotManager,
    ) -> geometry_msgs.msg.PoseStamped:
        import geometry_msgs.msg

        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = robot.node.sim_time.to_msg()
        assert phase.pose is not None
        msg.pose = phase.pose.to_msg()
        return msg


__all__ = ["ExternalAdapter"]
