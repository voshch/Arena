"""rosnav_rl mobile adapter: runs the policy in-process, no nav2."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar

from arena_robots.bringup.mobile.rosnav_rl import RosnavRlBringup
from arena_robots.clients.goto_pose import GotoPoseClient
from arena_robots.task_kinds import TaskKind
from arena_robots_msgs.action import GotoPose
from arena_viz.kinds import DisplayKind

from arena_runtime.env.adapters import AdapterDisplayHint, AdapterMeta
from arena_runtime.env.adapters.mobile import MobileAdapter
from arena_runtime.env.request import GoToPhase, TaskPhase
from arena_runtime.env.world_manager.shims import requires_map_server

if TYPE_CHECKING:
    import geometry_msgs.msg

    from arena_runtime.env.adapters import ResetContext
    from arena_runtime.env.robot_manager.robot_manager import RobotManager


@AdapterMeta.attach(
    accepts={TaskKind.GOTO_POSE},
    bringup=RosnavRlBringup,
    client=GotoPoseClient,
    cap="mobile",
    displays=[
        AdapterDisplayHint(
            name="Subgoal",
            topic="{ns}/subgoal",
            topic_type="geometry_msgs/PoseStamped",
            kind=DisplayKind.POSE,
        ),
    ],
)
@requires_map_server
class RosnavRlAdapter(MobileAdapter):
    kind: ClassVar[str] = "rosnav_rl"
    bringup: RosnavRlBringup

    async def dispatch_phase(
        self,
        phase: TaskPhase,
        robot: RobotManager,
    ) -> None:
        assert isinstance(phase, GoToPhase), f"RosnavRlAdapter only accepts GOTO_POSE phases; got {type(phase).__name__} (kind={phase.kind!r})"
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

    async def wait_until_ready(
        self,
        robot: RobotManager,
        node_paths: set[str],
    ) -> None:
        inference_node_path = str(robot.namespace(self.bringup.inference_node_name))
        while inference_node_path not in node_paths:
            await asyncio.sleep(0.01)
        await super().wait_until_ready(robot, node_paths)

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


__all__ = ["RosnavRlAdapter"]
