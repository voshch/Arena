from __future__ import annotations

import asyncio
import time
import types
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _ros_gate() -> None:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


class _Task:
    abort_reason: str | None = None

    def abort_episode(self, reason: str) -> None:
        self.abort_reason = reason


def _runner_with_open_episode_condition():
    import shapely
    from arena_simulation_setup.shared.conditions import EpisodeCondition
    from arena_simulation_setup.shared.judge import Sample
    from arena_simulation_setup.shared.task import GoToPhase
    from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

    from task_generator.manager.robot_manager.task_runner import TaskRunner

    runner = TaskRunner("jackal_0")
    runner.set_episode_conditions([EpisodeCondition(op="eventually", p="robot in kitchen")])
    runner.submit([GoToPhase(pose=Pose(Position(12.0, 1.0), Orientation.identity()), tolerance_radius=0.5, tolerance_angle=0.0, hold_time=0.0)])
    zones = {"kitchen": shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)])}
    runner.step(Sample(t=0.0, robots={"jackal_0": (12.0, 3.0, 0.0)}, field=lambda e, f: None), zones, None, (0, None))
    return runner


def _node_stub(publisher, runner, loop: asyncio.AbstractEventLoop):
    import rclpy.time

    from task_generator.manager.realizer import Realizer
    from task_generator.node import EpisodeRuntime, TaskGenerator

    stub = type("Stub", (), {})()
    stub._task = _Task()
    stub._episodes = EpisodeRuntime()
    stub._episodes.pending_outcomes[stub._episodes.current.episode_id] = loop.create_future()
    stub._robots_manager = SimpleNamespace(managers={"jackal_0": SimpleNamespace(name="jackal_0", runner=runner, finish_episode=runner.finish_episode)})
    stub._realizer = Realizer(Realizer._Configuration(prefix="env_0"))
    stub._simulator = SimpleNamespace(semantics_snapshot=lambda: [])
    stub._world_manager = SimpleNamespace(loaded_world="map_empty", world=SimpleNamespace(levels={}), publish_world_markers=lambda snapshots: None)
    stub._env_id = 0
    stub.sim_time = rclpy.time.Time()
    stub._semantics_dirty = True
    stub._pub_state_semantics = publisher
    stub.get_logger = lambda: SimpleNamespace(warn=lambda *a, **kw: None)
    stub._robot_fields = TaskGenerator._robot_fields
    for name in ("_close_condition_scopes", "_publish_semantics_snapshot", "_zone_semantic_states", "_robot_semantic_states", "_iter_zones", "_semantic_entity_state_msg"):
        setattr(stub, name, types.MethodType(getattr(TaskGenerator, name), stub))
    return stub


def test_fail_episode_judges_open_scopes_and_publishes_the_snapshot_at_once(rclpy_context) -> None:
    import rclpy
    import task_generator_msgs.action
    import task_generator_msgs.msg

    from task_generator.node import TaskGenerator

    host = rclpy.create_node("episode_end_scopes_host")
    received: list[task_generator_msgs.msg.SemanticSnapshot] = []
    topic = "/episode_end_scopes/state/semantics"
    host.create_subscription(task_generator_msgs.msg.SemanticSnapshot, topic, received.append, 10)
    publisher = host.create_publisher(task_generator_msgs.msg.SemanticSnapshot, topic, 10)
    loop = asyncio.new_event_loop()
    try:
        deadline = time.monotonic() + 5.0
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(host, timeout_sec=0.05)
        runner = _runner_with_open_episode_condition()
        stub = _node_stub(publisher, runner, loop)

        TaskGenerator.fail_episode(stub, "no progress")

        assert stub._task.abort_reason == "no progress"
        assert runner.violated == ["e0"]
        assert stub._semantics_dirty is False
        assert stub._episodes.pending_outcomes[stub._episodes.current.episode_id].result() == (task_generator_msgs.action.RunEpisode.Result.FAILED, "no progress")
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(host, timeout_sec=0.05)
        robots = [e for e in received[0].entities if e.kind == "robot"]
        assert [e.entity for e in robots] == ["env_0/jackal_0"]
        assert dict(zip(robots[0].discrete_names, robots[0].discrete_values, strict=True)) == {"phase": "0", "met": "", "failed": "", "dropped": "", "violated": "e0"}

        TaskGenerator.fail_episode(stub, "again")
        assert stub._task.abort_reason == "no progress"
        assert runner.violated == ["e0"]
    finally:
        loop.close()
        host.destroy_node()
