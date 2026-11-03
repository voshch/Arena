from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace


def _runner_with_open_episode_condition():
    import shapely
    from arena_simulation_setup.shared.conditions import EpisodeCondition
    from arena_simulation_setup.shared.judge import Sample
    from arena_simulation_setup.shared.task import GoToPhase
    from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

    from arena_runtime.env.robot_manager.task_runner import TaskRunner

    runner = TaskRunner("jackal_0")
    runner.set_episode_conditions([EpisodeCondition(op="eventually", p="robot in kitchen")])
    runner.submit([GoToPhase(pose=Pose(Position(12.0, 1.0), Orientation.identity()), tolerance_radius=0.5, tolerance_angle=0.0, hold_time=0.0)])
    zones = {"kitchen": shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)])}
    runner.step(Sample(t=0.0, robots={"jackal_0": (12.0, 3.0, 0.0)}, field=lambda e, f: None), zones, None, (0, None))
    return runner


def _node_stub(publisher, runner, loop: asyncio.AbstractEventLoop):
    import rclpy.time

    from arena_runtime.env.realizer import Realizer
    from arena_runtime.env.episode_node import EpisodeRuntime, EpisodeNode

    stub = object.__new__(type("Stub", (EpisodeNode,), {"sim_time": rclpy.time.Time()}))
    stub._task = SimpleNamespace()
    stub._abort_reason = None
    stub._force_reset = False
    stub._episodes = EpisodeRuntime()
    stub._episodes.pending_outcomes[stub._episodes.current.episode_id] = loop.create_future()
    stub._robots_manager = SimpleNamespace(managers={"jackal_0": SimpleNamespace(name="jackal_0", runner=runner, finish_episode=runner.finish_episode)})
    stub._realizer = Realizer(Realizer._Configuration(prefix="env_0"))
    stub._simulator = SimpleNamespace(semantics_snapshot=lambda: [])
    stub._world_manager = SimpleNamespace(loaded_world="map_empty", world=SimpleNamespace(levels={}), publish_world_markers=lambda snapshots: None)
    stub._env_id = 0
    stub._semantics_dirty = True
    stub._stimuli_fired = {}
    stub._pub_state_semantics = publisher
    stub.get_logger = lambda: SimpleNamespace(warn=lambda *a, **kw: None)
    return stub


def test_fail_episode_judges_open_scopes_and_publishes_the_snapshot_at_once(rclpy_context) -> None:
    import rclpy
    import arena_runtime_msgs.action
    import arena_runtime_msgs.msg

    from arena_runtime.env.episode_node import EpisodeNode

    host = rclpy.create_node("episode_end_scopes_host")
    received: list[arena_runtime_msgs.msg.SemanticSnapshot] = []
    topic = "/episode_end_scopes/state/semantics"
    host.create_subscription(arena_runtime_msgs.msg.SemanticSnapshot, topic, received.append, 10)
    publisher = host.create_publisher(arena_runtime_msgs.msg.SemanticSnapshot, topic, 10)
    loop = asyncio.new_event_loop()
    try:
        deadline = time.monotonic() + 5.0
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(host, timeout_sec=0.05)
        runner = _runner_with_open_episode_condition()
        stub = _node_stub(publisher, runner, loop)

        EpisodeNode.fail_episode(stub, "no progress")

        assert stub._abort_reason == "no progress"
        assert runner.violated == ["e0"]
        assert stub._semantics_dirty is False
        assert stub._episodes.pending_outcomes[stub._episodes.current.episode_id].result() == (arena_runtime_msgs.action.RunEpisode.Result.FAILED, "no progress")
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(host, timeout_sec=0.05)
        robots = [e for e in received[0].entities if e.kind == "robot"]
        assert [e.entity for e in robots] == ["env_0/jackal_0"]
        assert dict(zip(robots[0].discrete_names, robots[0].discrete_values, strict=True)) == {"phase": "0", "met": "", "failed": "", "dropped": "", "violated": "e0"}

        EpisodeNode.fail_episode(stub, "again")
        assert stub._abort_reason == "no progress"
        assert runner.violated == ["e0"]
    finally:
        loop.close()
        host.destroy_node()


def _done_stub(*, force_reset: bool = False, elapsed: int = 0, timeout: float = 100.0, modes_done: bool = False) -> object:
    from arena_runtime.env.episode_node import EpisodeNode

    async def done() -> bool:
        return modes_done

    stub = object.__new__(type("Stub", (EpisodeNode,), {"sim_time": SimpleNamespace(sec=elapsed), "conf": SimpleNamespace(Robot=SimpleNamespace(TIMEOUT=SimpleNamespace(value=timeout)))}))
    stub._task = SimpleNamespace(done=done())
    stub._force_reset = force_reset
    stub._abort_reason = None
    stub._reset_sim_time = 0
    return stub


def test_a_forced_reset_ends_the_episode_without_a_reason() -> None:
    stub = _done_stub(force_reset=True)
    assert asyncio.run(stub._episode_done()) is True
    assert stub._abort_reason is None
    stub._task.done.close()


def test_the_timeout_ends_the_episode_as_aborted() -> None:
    stub = _done_stub(elapsed=101)
    assert asyncio.run(stub._episode_done()) is True
    assert stub._abort_reason == "timeout"
    stub._task.done.close()


def test_otherwise_the_bound_modes_decide() -> None:
    assert asyncio.run(_done_stub(modes_done=True)._episode_done()) is True
    assert asyncio.run(_done_stub(modes_done=False)._episode_done()) is False
