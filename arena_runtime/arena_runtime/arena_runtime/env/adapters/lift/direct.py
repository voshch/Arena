"""Direct lift adapter (lift: direct, lift cap)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import std_msgs.msg
from arena_robots.actuators import clamp_lift, lift_limits, lift_topic
from arena_robots.bringup.lift.direct import DirectLiftBringup
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from arena_runtime.env.adapters import Adapter, AdapterMeta

if TYPE_CHECKING:
    from arena_robots.caps import LiftSpec
    from rclpy.publisher import Publisher

    from arena_runtime.env.request import TaskPhase
    from arena_runtime.env.robot_manager.robot_manager import RobotManager

_log = logging.getLogger(__name__)

_LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


@AdapterMeta.attach(accepts=(), bringup=DirectLiftBringup, clients={}, cap="lift")
class DirectLiftAdapter(Adapter):
    """Forwards Float64 heights on <ns>/lift/<instance>/command, clamped to the joint range, to the lift's position controller."""

    kind: ClassVar[str] = "direct"

    def __init__(self, robot_manager: RobotManager, **bringup_kwargs: object) -> None:
        super().__init__(robot_manager, **bringup_kwargs)
        caps = robot_manager.robot_view.effective_caps(robot_manager.robot.resolved_request, frames=robot_manager.robot.frames)
        namespace = str(robot_manager.namespace)
        node = robot_manager.node
        self._commands: dict[str, Publisher] = {}
        self._applied: dict[str, Publisher] = {}
        self._subscriptions = []
        for name, spec in (caps.lift or {}).items():
            if lift_limits(spec) is None:
                _log.warning("lift %r of robot %r declares no limits, commands are not clamped", name, robot_manager.name)
            self._commands[name] = node.create_publisher(std_msgs.msg.Float64MultiArray, str(robot_manager.namespace(spec.controller, "commands")), 10)
            self._applied[name] = node.create_publisher(std_msgs.msg.Float64, lift_topic(namespace, name, "applied"), _LATCHED)
            self._subscriptions.append(node.create_subscription(std_msgs.msg.Float64, lift_topic(namespace, name), self._handler(name, spec), 10))

    def _handler(self, name: str, spec: LiftSpec) -> Callable[[std_msgs.msg.Float64], None]:
        def on_command(msg: std_msgs.msg.Float64) -> None:
            self.set_height(name, spec, msg.data)

        return on_command

    def set_height(self, name: str, spec: LiftSpec, height: float) -> float:
        applied = clamp_lift(spec, height)
        command = std_msgs.msg.Float64MultiArray()
        command.data = [applied]
        self._commands[name].publish(command)
        self._applied[name].publish(std_msgs.msg.Float64(data=applied))
        return applied

    async def dispatch_phase(self, phase: TaskPhase, robot: RobotManager) -> None:
        raise RuntimeError(f"DirectLiftAdapter accepts no task phases; got {type(phase).__name__}")
