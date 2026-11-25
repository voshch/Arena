"""Direct gripper adapter (gripper: direct, gripper cap)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, ClassVar

import control_msgs.action
import control_msgs.msg
from arena_robots.actuators import clamp_gripper, gripper_topic
from arena_robots.bringup.gripper.direct import DirectGripperBringup
from arena_robots.task_kinds import TaskKind
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from arena_runtime.env.adapters import Adapter, AdapterMeta
from arena_runtime.env.adapters.gripper._client import STATUS_SUCCEEDED, GraspGoal, GripperGraspClient
from arena_runtime.env.request import GraspPhase

if TYPE_CHECKING:
    from arena_rclpy_mixins.Async import ActionClientWrapper
    from arena_robots.caps import ArmSpec, GripperSpec
    from rclpy.publisher import Publisher

    from arena_runtime.env.request import TaskPhase
    from arena_runtime.env.robot_manager.robot_manager import RobotManager

_log = logging.getLogger(__name__)

_SEND_TIMEOUT_S = 10.0
_LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


@AdapterMeta.attach(accepts={TaskKind.GRASP}, bringup=DirectGripperBringup, clients={}, cap="gripper")
class DirectGripperAdapter(Adapter):
    """Forwards GripperCommand messages on <ns>/gripper/<instance>/command, clamped to the open and close positions, to the gripper's GripperCommand action, and runs grasp phases as open and close goals on it."""

    kind: ClassVar[str] = "direct"

    def __init__(self, robot_manager: RobotManager, **bringup_kwargs: object) -> None:
        super().__init__(robot_manager, **bringup_kwargs)
        caps = robot_manager.caps
        namespace = str(robot_manager.namespace)
        node = robot_manager.node
        self._specs: dict[str, GripperSpec] = dict(caps.gripper or {})
        self._arms: dict[str, ArmSpec] = dict(caps.arm or {})
        self._actions: dict[str, ActionClientWrapper] = {}
        self._applied: dict[str, Publisher] = {}
        self._subscriptions = []
        self._attach_task: asyncio.Task[None] | None = None
        for name, spec in self._specs.items():
            self._actions[name] = node.create_action_client_wrapper(control_msgs.action.GripperCommand, str(robot_manager.namespace(spec.controller, "gripper_cmd")))
            self._applied[name] = node.create_publisher(control_msgs.msg.GripperCommand, gripper_topic(namespace, name, "applied"), _LATCHED)
            self._subscriptions.append(node.create_subscription(control_msgs.msg.GripperCommand, gripper_topic(namespace, name), self._handler(name, spec), 10))
        self._clients[TaskKind.GRASP] = GripperGraspClient(
            robot_manager.robot_view,
            namespace,
            node=node,
            tf_buffer=robot_manager.tf_buffer,
            actions=self._actions,
            endpoints={name: str(robot_manager.namespace(spec.controller, "gripper_cmd")) for name, spec in self._specs.items()},
        )

    def _handler(self, name: str, spec: GripperSpec) -> Callable[[control_msgs.msg.GripperCommand], Awaitable[None]]:
        async def on_command(msg: control_msgs.msg.GripperCommand) -> None:
            try:
                await self.set_gripper(name, spec, msg.position, msg.max_effort)
            except Exception:
                _log.exception("gripper %r command rejected", name)

        return on_command

    async def set_gripper(self, name: str, spec: GripperSpec, position: float, max_effort: float) -> tuple[float, float]:
        goal = self._command(name, spec, position, max_effort)
        await self._actions[name].send_goal_timeout(goal, _SEND_TIMEOUT_S)
        return goal.command.position, goal.command.max_effort

    def _command(self, name: str, spec: GripperSpec, position: float, max_effort: float) -> control_msgs.action.GripperCommand.Goal:
        applied, effort = clamp_gripper(spec, position, max_effort)
        goal = control_msgs.action.GripperCommand.Goal()
        goal.command.position = applied
        goal.command.max_effort = effort
        self._applied[name].publish(control_msgs.msg.GripperCommand(position=applied, max_effort=effort))
        return goal

    def _instance(self, requested: str, robot: RobotManager) -> tuple[str, GripperSpec]:
        if requested:
            if requested not in self._specs:
                raise ValueError(f"grasp instance {requested!r} not found on robot {robot.name!r}; available: {sorted(self._specs)}")
            return requested, self._specs[requested]
        if len(self._specs) != 1:
            raise ValueError(f"grasp needs an explicit gripper instance on robot {robot.name!r}; available: {sorted(self._specs)}")
        ((name, spec),) = self._specs.items()
        return name, spec

    async def dispatch_phase(self, phase: TaskPhase, robot: RobotManager) -> None:
        if not isinstance(phase, GraspPhase):
            raise RuntimeError(f"DirectGripperAdapter accepts only grasp phases; got {type(phase).__name__}")
        name, spec = self._instance(phase.instance, robot)
        if self._attach_task is not None:
            self._attach_task.cancel()
            self._attach_task = None
        if phase.action == "open":
            await robot.environment_manager.grasp_detach(robot.robot, spec.joint)
        goal = self._command(name, spec, spec.positions[phase.action], 0.0)
        await self.client_for(TaskKind.GRASP).send_goal(GraspGoal(instance=name, command=goal))
        if phase.action == "close" and phase.object:
            self._attach_task = asyncio.create_task(self._attach_when_closed(robot, spec.joint, phase.object, self._hand(spec, robot)))

    def _hand(self, spec: GripperSpec, robot: RobotManager) -> str:
        """TF frame of the tip of the gripper's arm, the only arm's without a named one, empty without an arm."""
        arm = self._arms.get(spec.arm) if spec.arm is not None else None
        if arm is None and len(self._arms) == 1:
            ((_, arm),) = self._arms.items()
        return "" if arm is None else robot.robot.frame.tf(arm.tip_link)

    async def _attach_when_closed(self, robot: RobotManager, joint: str, obj: str, hand: str) -> None:
        client = self.client_for(TaskKind.GRASP)
        assert isinstance(client, GripperGraspClient)
        try:
            await client.await_result()
            if client.status != STATUS_SUCCEEDED:
                return
            if not await robot.environment_manager.grasp_attach(robot.robot, joint, obj, hand):
                robot.fail_phase(f"gripper joint {joint!r} holds no {obj!r} after closing")
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.exception("attaching %r to gripper joint %r of robot %r failed", obj, joint, robot.name)
