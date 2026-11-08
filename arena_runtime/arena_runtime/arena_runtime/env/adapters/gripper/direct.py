"""Direct gripper adapter (gripper: direct, gripper cap)."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, ClassVar

import control_msgs.action
import control_msgs.msg
from arena_robots.actuators import clamp_gripper, gripper_topic
from arena_robots.bringup.gripper.direct import DirectGripperBringup
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from arena_runtime.env.adapters import Adapter, AdapterMeta

if TYPE_CHECKING:
    from arena_rclpy_mixins.Async import ActionClientWrapper
    from arena_robots.caps import GripperSpec
    from rclpy.publisher import Publisher

    from arena_runtime.env.request import TaskPhase
    from arena_runtime.env.robot_manager.robot_manager import RobotManager

_log = logging.getLogger(__name__)

_SEND_TIMEOUT_S = 10.0
_LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


@AdapterMeta.attach(accepts=(), bringup=DirectGripperBringup, clients={}, cap="gripper")
class DirectGripperAdapter(Adapter):
    """Forwards GripperCommand messages on <ns>/gripper/<instance>/command, clamped to the open and close positions, to the gripper's GripperCommand action."""

    kind: ClassVar[str] = "direct"

    def __init__(self, robot_manager: RobotManager, **bringup_kwargs: object) -> None:
        super().__init__(robot_manager, **bringup_kwargs)
        caps = robot_manager.robot_view.effective_caps(robot_manager.robot.resolved_request, frames=robot_manager.robot.frames)
        namespace = str(robot_manager.namespace)
        node = robot_manager.node
        self._actions: dict[str, ActionClientWrapper] = {}
        self._applied: dict[str, Publisher] = {}
        self._subscriptions = []
        for name, spec in (caps.gripper or {}).items():
            self._actions[name] = node.create_action_client_wrapper(control_msgs.action.GripperCommand, str(robot_manager.namespace(spec.controller, "gripper_cmd")))
            self._applied[name] = node.create_publisher(control_msgs.msg.GripperCommand, gripper_topic(namespace, name, "applied"), _LATCHED)
            self._subscriptions.append(node.create_subscription(control_msgs.msg.GripperCommand, gripper_topic(namespace, name), self._handler(name, spec), 10))

    def _handler(self, name: str, spec: GripperSpec) -> Callable[[control_msgs.msg.GripperCommand], Awaitable[None]]:
        async def on_command(msg: control_msgs.msg.GripperCommand) -> None:
            try:
                await self.set_gripper(name, spec, msg.position, msg.max_effort)
            except Exception:
                _log.exception("gripper %r command rejected", name)

        return on_command

    async def set_gripper(self, name: str, spec: GripperSpec, position: float, max_effort: float) -> tuple[float, float]:
        applied, effort = clamp_gripper(spec, position, max_effort)
        goal = control_msgs.action.GripperCommand.Goal()
        goal.command.position = applied
        goal.command.max_effort = effort
        self._applied[name].publish(control_msgs.msg.GripperCommand(position=applied, max_effort=effort))
        await self._actions[name].send_goal_timeout(goal, _SEND_TIMEOUT_S)
        return applied, effort

    async def dispatch_phase(self, phase: TaskPhase, robot: RobotManager) -> None:
        raise RuntimeError(f"DirectGripperAdapter accepts no task phases; got {type(phase).__name__}")
