from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

try:
    import geometry_msgs.msg
    import rclpy.logging
    import tf2_ros
    from arena_rclpy_mixins.shared import FrameNamespace
    from arena_simulation_setup.shared.judge import Sample
    from arena_simulation_setup.shared.task import GoToPhase
    from arena_simulation_setup.utils.geometry import Pose

    from arena_runtime.env.realizer import Realizer
    from arena_runtime.env.robot_manager.robot_manager import RobotManager
    from arena_runtime.env.robot_manager.task_runner import TaskRunner
    from arena_runtime.env.request import TaskKind
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

FRAME = "probe"
STATUS_CANCELED = 5


class _Adapter:
    kind = "nav2"
    signal = None

    def __init__(self, status: int | None = None) -> None:
        self.dispatched: list[GoToPhase] = []
        self.ticked: list[GoToPhase] = []
        self.status = status

    async def dispatch_phase(self, phase: GoToPhase, robot: RobotManager) -> None:
        self.dispatched.append(phase)

    def client_for(self, kind: TaskKind) -> SimpleNamespace:
        return SimpleNamespace(is_done=lambda: True, status=self.status, reason="stale")

    def on_phase_tick(self, phase: GoToPhase, robot: RobotManager) -> None:
        self.ticked.append(phase)

    async def on_reset(self, robot: RobotManager, ctx: SimpleNamespace) -> None:
        pass


def _manager(adapter: _Adapter) -> RobotManager:
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
    manager._NodeInterface__node = SimpleNamespace(tf_buffer=tf2_ros.Buffer(), on_task_submitted=lambda _: None, get_logger=lambda: rclpy.logging.get_logger("test_judge_only_dispatch"))
    manager.begin_episode()
    transform = geometry_msgs.msg.TransformStamped()
    transform.header.stamp.sec = 1
    transform.header.frame_id = "map"
    transform.child_frame_id = manager.frame("base_link").raw()
    transform.transform.rotation.w = 1.0
    manager.node.tf_buffer.set_transform(transform, "test")
    return manager


def _judge_only() -> GoToPhase:
    return GoToPhase(pose=Pose.parse([5.0, 0.0, 0.0]), tolerance_radius=1.0, tolerance_angle=6.3, dispatch=False)


def _run(manager: RobotManager, *robot_xs: float) -> None:
    async def episode() -> None:
        manager._runner.submit([_judge_only()])
        await manager.reset(SimpleNamespace(start_pose=None))
        for t, x in enumerate(robot_xs):
            await manager.tick(Sample(t=float(t), robots={FRAME: (x, 0.0, 0.0)}), {})

    asyncio.run(episode())


def test_a_judge_only_goto_never_reaches_the_adapter() -> None:
    adapter = _Adapter()
    manager = _manager(adapter)
    _run(manager, 0.0, 1.0)
    assert adapter.dispatched == []
    assert adapter.ticked == []
    assert manager._runner.phases[0].dispatched
    assert manager.active_dispatch_phase() is None


def test_a_judge_only_goto_gives_progress_and_rviz_no_goal() -> None:
    manager = _manager(_Adapter())
    _run(manager, 0.0, 1.0)
    assert manager._runner.active_state is not None
    assert manager.goal is None


def test_a_dispatched_goto_gives_progress_and_rviz_its_pose() -> None:
    manager = _manager(_Adapter())

    async def episode() -> None:
        manager._runner.submit([GoToPhase(pose=Pose.parse([5.0, 0.0, 0.0]), tolerance_radius=1.0, tolerance_angle=6.3)])
        await manager.reset(SimpleNamespace(start_pose=None))
        await manager.tick(Sample(t=0.0, robots={FRAME: (0.0, 0.0, 0.0)}), {})

    asyncio.run(episode())
    assert manager.goal is not None
    assert (manager.goal.position.x, manager.goal.position.y) == (5.0, 0.0)


def test_a_judge_only_goto_is_met_by_the_judge_whatever_the_planner_reports() -> None:
    manager = _manager(_Adapter(status=STATUS_CANCELED))
    _run(manager, 0.0, 4.5)
    assert manager._runner.phases[0].outcome == "met"
