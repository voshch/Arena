"""Fire-once, best-effort hard-channel registration for sim-time producers."""

from __future__ import annotations

from arena_runtime_msgs.msg import LockstepChannel, LockstepRegistration
from arena_runtime_msgs.srv import LockstepRegister
from rclpy.clock import Clock
from rclpy.node import Node
from rclpy.task import Future as RclFuture

REGISTER_SERVICE = "/arena/sim_lifecycle/lockstep/register"
_POLL_S = 0.5
_WAIT_S = 3.0


def register_hard_channel(node: Node, *, name: str, topic: str, msg_type: str, period_s: float, env: str) -> None:
    """Register one hard channel once the register service is up, or give up after a short wait."""
    client = node.create_client(LockstepRegister, REGISTER_SERVICE)
    steady = Clock()
    waited = 0.0

    def _done(future: RclFuture) -> None:
        try:
            response = future.result()
        except Exception as exc:
            node.get_logger().warning(f"lockstep channel registration call failed: {exc}")
            return
        if not response.success:
            node.get_logger().warning(f"lockstep channel registration rejected: {response.error_msg}")

    def _poll() -> None:
        nonlocal waited
        if not client.service_is_ready():
            waited += _POLL_S
            if waited >= _WAIT_S:
                node.get_logger().info("lockstep register service not available, skipping channel registration")
                timer.cancel()
            return
        timer.cancel()
        request = LockstepRegister.Request()
        request.registration = LockstepRegistration(
            caller=node.get_fully_qualified_name(),
            env=env,
            channels=[LockstepChannel(name=name, topic=topic, type=msg_type, period_s=period_s, hard=True)],
        )
        client.call_async(request).add_done_callback(_done)

    timer = node.create_timer(_POLL_S, _poll, clock=steady)
