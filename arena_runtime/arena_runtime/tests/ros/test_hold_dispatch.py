from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

try:
    import geometry_msgs.msg
    import rclpy.logging
    import tf2_ros
    from arena_rclpy_mixins.shared import FrameNamespace
    from arena_simulation_setup.shared.judge import Sample
    from arena_simulation_setup.shared.task import GoToPhase

    from arena_runtime.env.realizer import Realizer
    from arena_runtime.env.robot_manager.robot_manager import RobotManager
    from arena_runtime.env.robot_manager.task_runner import TaskRunner
    from arena_runtime.env.request import TaskKind
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

FRAME = "probe"
PREVIOUS_EPISODE = (9.0, 9.0, 0.7071, 0.7071)
START = (2.0, 3.0, 1.0, 0.0)
UNTIL = "3.0,10.0 within 1.0 of passer"


class _Adapter:
    kind = "nav2"
    signal = None

    def __init__(self) -> None:
        self.dispatched: list[GoToPhase] = []

    async def dispatch_phase(self, phase: GoToPhase, robot: RobotManager) -> None:
        self.dispatched.append(phase)

    def client_for(self, kind: TaskKind) -> SimpleNamespace:
        return SimpleNamespace(is_done=lambda: False, status=None, reason=None)

    def on_phase_tick(self, phase: GoToPhase, robot: RobotManager) -> None:
        pass

    async def on_reset(self, robot: RobotManager, ctx: SimpleNamespace) -> None:
        pass


def _set_tf(manager: RobotManager, x: float, y: float, qz: float, qw: float) -> None:
    transform = geometry_msgs.msg.TransformStamped()
    transform.header.stamp.sec = 1
    transform.header.frame_id = "map"
    transform.child_frame_id = manager.frame("base_link").raw()
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.rotation.z = qz
    transform.transform.rotation.w = qw
    manager.node.tf_buffer.set_transform(transform, "test")


def _manager() -> tuple[RobotManager, _Adapter]:
    adapter = _Adapter()
    manager = RobotManager.__new__(RobotManager)
    manager._adapter_instances = [adapter]
    manager._adapters = {TaskKind.GOTO_POSE: adapter}
    manager._environment_manager = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix="env_0"))
    manager._robot = SimpleNamespace(name=FRAME, frame=FrameNamespace(FRAME))
    manager._config = SimpleNamespace(model_params=SimpleNamespace(base_frame="base_link"))
    manager._runner = TaskRunner(FRAME)
    manager._publish_goal_task = None
    manager._phase_failure = None
    manager._cabin_to_clear = ""
    manager._NodeInterface__node = SimpleNamespace(tf_buffer=tf2_ros.Buffer(), on_task_submitted=lambda _: None, get_logger=lambda: rclpy.logging.get_logger("test_hold_dispatch"))
    manager.begin_episode()
    return manager, adapter


def test_hold_submitted_before_the_teleport_targets_the_landed_pose() -> None:
    manager, adapter = _manager()
    _set_tf(manager, *PREVIOUS_EPISODE)

    async def episode() -> None:
        manager._runner.submit([GoToPhase(until=UNTIL, tolerance_radius=1.0, tolerance_angle=0.5)])
        await manager._dispatch(0)
        assert adapter.dispatched == []
        _set_tf(manager, *START)
        await manager.reset(SimpleNamespace(start_pose=None))
        await manager.tick(Sample(t=0.0, robots={FRAME: (2.0, 3.0, 3.14159)}), {})

    asyncio.run(episode())
    assert len(adapter.dispatched) == 1
    pose = adapter.dispatched[0].pose
    assert (pose.position.x, pose.position.y) == pytest.approx((2.0, 3.0))
    assert abs(pose.orientation.to_yaw()) == pytest.approx(math.pi, abs=1e-3)


def test_hold_stays_with_the_until_open_once_dispatched_on_a_landed_robot() -> None:
    manager, adapter = _manager()
    _set_tf(manager, *START)

    async def episode() -> None:
        manager._runner.submit([GoToPhase(until=UNTIL, tolerance_radius=1.0, tolerance_angle=0.5)])
        await manager.reset(SimpleNamespace(start_pose=None))
        await manager.tick(Sample(t=0.0, robots={FRAME: (2.0, 3.0, 3.14159)}), {})

    asyncio.run(episode())
    assert manager._runner.active_state.monitor.held
