from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import shapely

try:
    import geometry_msgs.msg
    import rclpy.logging
    import rclpy.task
    import tf2_ros
    from arena_rclpy_mixins.shared import FrameNamespace
    from arena_runtime.env.adapters import Adapter
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.env.request import TaskKind
    from arena_runtime.env.robot_manager.robot_manager import RobotManager
    from arena_runtime.env.robot_manager.task_runner import TaskRunner
    from arena_simulation_setup.shared.conditions import EpisodeCondition
    from arena_simulation_setup.shared.judge import Sample
    from arena_simulation_setup.shared.task import LoopPhase, ReachPhase
    from arena_simulation_setup.utils.geometry import Pose
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

FRAME = "probe"
STATUS_CANCELED = 5
HALL = {"hall": shapely.Polygon([(10, 0), (10, 4), (14, 4), (14, 0)])}
OUTSIDE = (1.0, 1.0, 0.0)
INSIDE = (12.0, 1.0, 0.0)


class _Client:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.running = False
        self.status: int | None = None
        self.reason: str | None = None

    def is_done(self) -> bool:
        return not self.running

    def cancel(self) -> rclpy.task.Future:
        self.events.append("cancel")
        self.running = False
        self.status = STATUS_CANCELED
        answer = rclpy.task.Future()
        answer.set_result(None)
        return answer


async def _await_ros(future: rclpy.task.Future) -> object:
    return future.result()


class _Adapter:
    kind = "moveit"
    signal = None
    cancel_and_wait = Adapter.cancel_and_wait

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self._clients = {TaskKind.REACH_POSE: _Client(events)}
        self.rm = SimpleNamespace(node=SimpleNamespace(await_ros=_await_ros))

    async def dispatch_phase(self, phase: ReachPhase, robot: RobotManager) -> None:
        self.events.append("dispatch")
        self._clients[TaskKind.REACH_POSE].running = True
        self._clients[TaskKind.REACH_POSE].status = None

    def client_for(self, kind: TaskKind) -> _Client:
        return self._clients[kind]

    def on_phase_tick(self, phase: ReachPhase, robot: RobotManager) -> None:
        pass

    async def on_reset(self, robot: RobotManager, ctx: SimpleNamespace) -> None:
        pass


def _manager(events: list[str]) -> RobotManager:
    adapter = _Adapter(events)
    manager = RobotManager.__new__(RobotManager)
    manager._adapter_instances = [adapter]
    manager._adapters = {TaskKind.REACH_POSE: adapter}
    manager._environment_manager = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix="env_0"))
    manager._robot = SimpleNamespace(name=FRAME, frame=FrameNamespace(FRAME))
    manager._config = SimpleNamespace(model_params=SimpleNamespace(base_frame="base_link"))
    manager._runner = TaskRunner(FRAME)
    manager._publish_goal_task = None
    manager._phase_failure = None
    manager._cabin_to_clear = ""
    manager._abort_episode = lambda reason: events.append("abort")
    manager._NodeInterface__node = SimpleNamespace(tf_buffer=tf2_ros.Buffer(), on_task_submitted=lambda _: None, get_logger=lambda: rclpy.logging.get_logger("test_early_end_cancel"))
    manager.begin_episode()
    transform = geometry_msgs.msg.TransformStamped()
    transform.header.stamp.sec = 1
    transform.header.frame_id = "map"
    transform.child_frame_id = manager.frame("base_link").raw()
    transform.transform.rotation.w = 1.0
    manager.node.tf_buffer.set_transform(transform, "test")
    return manager


def _reach(**kwargs: object) -> ReachPhase:
    return ReachPhase(target=Pose.parse([0.4, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]), **kwargs)


def _run(manager: RobotManager, phases: list, *positions: tuple[float, float, float]) -> None:
    async def episode() -> None:
        manager._runner.submit(phases)
        await manager.reset(SimpleNamespace(start_pose=None))
        for t, position in enumerate(positions):
            await manager.tick(Sample(t=float(t), robots={FRAME: position}), HALL)

    asyncio.run(episode())


def test_a_loop_exit_cancels_the_running_reach_before_the_next_phase_dispatches() -> None:
    events: list[str] = []
    manager = _manager(events)
    _run(manager, [LoopPhase(phases=[_reach(), _reach()], until="robot in hall"), _reach()], OUTSIDE, INSIDE)
    assert events == ["dispatch", "cancel", "dispatch"]
    assert manager._runner.phases[0].outcome == "dropped"


def test_an_episode_abort_cancels_the_running_reach_before_the_abort_fires() -> None:
    events: list[str] = []
    manager = _manager(events)
    guard = EpisodeCondition(op="never", p="robot in hall", on_failure="abort_episode")
    _run(manager, [_reach(conditions=[guard]), _reach()], OUTSIDE, INSIDE)
    assert events == ["dispatch", "cancel", "abort"]


def test_a_reach_met_by_stop_stays_met_when_its_cancel_reports_canceled() -> None:
    events: list[str] = []
    manager = _manager(events)
    _run(manager, [_reach(stop="robot in hall"), _reach()], OUTSIDE, INSIDE)
    assert events == ["dispatch", "cancel", "dispatch"]
    assert manager._runner.phases[0].outcome == "met"
    assert manager._runner.phases[0].reason == ""
