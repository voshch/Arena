"""No-op arm adapter (arm: none, arm cap)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from arena_robots.bringup.arm.none import NoneArmBringup
from arena_robots.clients.reach_pose import ReachPoseClient
from arena_robots.task_kinds import TaskKind
from arena_robots_msgs.action import ReachPose

from arena_runtime.env.adapters import Adapter, AdapterMeta
from arena_runtime.env.adapters._reach_sampling import reach_frame
from arena_runtime.env.adapters.arm import park_arms
from arena_runtime.env.request import ReachPhase, TaskPhase

if TYPE_CHECKING:
    from arena_runtime.env.robot_manager.robot_manager import RobotManager


@AdapterMeta.attach(
    accepts={TaskKind.REACH_POSE},
    bringup=NoneArmBringup,
    client=ReachPoseClient,
    cap="arm",
)
class NoneArmAdapter(Adapter):
    kind: ClassVar[str] = "none"

    async def dispatch_phase(
        self,
        phase: TaskPhase,
        robot: RobotManager,
    ) -> None:
        assert isinstance(phase, ReachPhase), f"NoneArmAdapter only accepts REACH_POSE phases; got {type(phase).__name__} (kind={phase.kind!r})"
        goal = ReachPose.Goal()
        if phase.target is not None:
            goal.target.header.frame_id = reach_frame(phase.frame or robot.base_frame, robot.frame)
            goal.target.header.stamp = robot.node.sim_time.to_msg()
            goal.target.pose = phase.target.to_msg()
        if phase.named_target is not None:
            goal.named_target = phase.named_target
        goal.position_tolerance = float(phase.position_tolerance or 0.0)
        goal.orientation_tolerance = float(phase.orientation_tolerance or 0.0)
        goal.planning_time = float(phase.planning_time or 0.0)
        await self.client.send_goal(goal)

    async def on_controllers_active(self, robot: RobotManager) -> None:
        self._park_pubs = await park_arms(robot)
