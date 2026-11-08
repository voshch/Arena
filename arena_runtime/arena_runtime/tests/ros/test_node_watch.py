"""Hung node detection: the miss tally, and real node processes frozen, killed and left running."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import threading
import typing


NS = "/node_watch_probe"
TIMEOUT = 1.0
_CHILD = """
import rclpy, sys
rclpy.init()
node = rclpy.create_node(sys.argv[1], namespace=sys.argv[2])
rclpy.spin(node)
"""


_LIFECYCLE_CHILD = """
import rclpy, sys
from rclpy.lifecycle import LifecycleNode
rclpy.init()
node = LifecycleNode(sys.argv[1], namespace=sys.argv[2])
if sys.argv[3] == "activate":
    node.trigger_configure()
    node.trigger_activate()
rclpy.spin(node)
"""


def _tally(limit: int = 3) -> typing.Any:
    from arena_runtime.env.node_watch import MissTally

    return MissTally(limit)


def test_a_node_past_the_limit_is_returned_once_its_misses_add_up() -> None:
    tally = _tally()
    assert tally.record({"a", "b", "c"}) is None
    assert [tally.record({"b", "c"}) for _ in range(3)] == [None, None, "a"]


def test_an_answer_clears_the_misses() -> None:
    tally = _tally()
    tally.record({"a", "b", "c"})
    assert [tally.record({"b", "c"}), tally.record({"b", "c"}), tally.record({"a", "b", "c"}), tally.record({"b", "c"})] == [None, None, None, None]


def test_a_round_most_nodes_missed_counts_for_nobody() -> None:
    tally = _tally(1)
    tally.record({"a", "b", "c"})
    assert tally.record({"c"}) is None
    assert tally.record(set()) is None


def test_a_node_that_never_answered_is_not_tracked() -> None:
    tally = _tally(1)
    tally.record({"a", "b"})
    assert tally.record({"a", "b"}) is None


def test_nodes_gone_between_episodes_are_forgotten() -> None:
    tally = _tally(1)
    tally.record({"a", "b", "c"})
    tally.forget_absent({"b", "c"})
    assert tally.record({"b", "c"}) is None


def _spawn(*names: str) -> dict[str, subprocess.Popen]:
    return {name: subprocess.Popen([sys.executable, "-c", _CHILD, name, NS]) for name in names}


def _spawn_lifecycle(name: str, start: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", _LIFECYCLE_CHILD, name, NS, start])


async def _ping(pinger: typing.Any, *, new_episode: bool = False) -> str | None:
    return await pinger.round(TIMEOUT, new_episode=new_episode)


async def _until_all_answer(pinger: typing.Any, count: int) -> None:
    for _ in range(150):
        await _ping(pinger)
        await asyncio.sleep(0.2)
        if len(pinger._tally._misses) == count:
            return
    raise AssertionError(f"only {sorted(pinger._tally._misses)} answered")


def _with_pinger(scenario: typing.Callable[[typing.Any, dict[str, subprocess.Popen]], typing.Coroutine[typing.Any, typing.Any, None]]) -> None:
    from arena_rclpy_mixins.Async import AsyncNode
    from rclpy.executors import SingleThreadedExecutor

    from arena_runtime.env.node_watch import NodePinger

    async def run() -> None:
        node = AsyncNode("node_watch_host", namespace=NS)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        threading.Thread(target=executor.spin, daemon=True).start()
        children = _spawn("healthy_a", "healthy_b", "healthy_c", "victim")
        try:
            await scenario(NodePinger(node, NS, 3), children)
        finally:
            for child in children.values():
                child.kill()
                child.wait()
            executor.shutdown()
            node.destroy_node()

    asyncio.run(run())


def test_a_frozen_process_is_named(rclpy_context: object) -> None:
    async def scenario(pinger: typing.Any, children: dict[str, subprocess.Popen]) -> None:
        await _until_all_answer(pinger, 4)
        os.kill(children["victim"].pid, signal.SIGSTOP)
        assert [await _ping(pinger) for _ in range(3)] == [None, None, f"{NS}/victim"]

    _with_pinger(scenario)


def test_a_node_that_dies_mid_episode_is_named(rclpy_context: object) -> None:
    async def scenario(pinger: typing.Any, children: dict[str, subprocess.Popen]) -> None:
        await _until_all_answer(pinger, 4)
        children["victim"].kill()
        children["victim"].wait()
        results = [await _ping(pinger) for _ in range(3)]
        assert results == [None, None, f"{NS}/victim"]

    _with_pinger(scenario)


def test_healthy_nodes_never_fault(rclpy_context: object) -> None:
    async def scenario(pinger: typing.Any, children: dict[str, subprocess.Popen]) -> None:
        await _until_all_answer(pinger, 4)
        assert [await _ping(pinger) for _ in range(6)] == [None] * 6

    _with_pinger(scenario)


def test_a_deactivated_lifecycle_node_is_named(rclpy_context: object) -> None:
    from lifecycle_msgs.msg import Transition
    from lifecycle_msgs.srv import ChangeState

    async def scenario(pinger: typing.Any, children: dict[str, subprocess.Popen]) -> None:
        children["planner"] = _spawn_lifecycle("planner", "activate")
        await _until_all_answer(pinger, 5)
        client = pinger._node.create_client_wrapper(ChangeState, f"{NS}/planner/change_state", timeout=5.0)
        request = ChangeState.Request()
        request.transition.id = Transition.TRANSITION_DEACTIVATE
        assert (await client.call_timeout(request)).success
        assert [await _ping(pinger) for _ in range(3)] == [None, None, f"{NS}/planner"]

    _with_pinger(scenario)


def test_a_lifecycle_node_that_never_went_active_is_not_tracked(rclpy_context: object) -> None:
    async def scenario(pinger: typing.Any, children: dict[str, subprocess.Popen]) -> None:
        children["idle"] = _spawn_lifecycle("idle", "stay")
        await _until_all_answer(pinger, 4)
        assert [await _ping(pinger) for _ in range(4)] == [None] * 4
        assert f"{NS}/idle" not in pinger._tally._misses

    _with_pinger(scenario)


def _episode_stub(pinger: typing.Any, *, auto_reset: bool, period: float = TIMEOUT) -> typing.Any:
    from types import SimpleNamespace

    from arena_runtime.env.episode_node import EpisodeNode, EpisodeRuntime

    conf = SimpleNamespace(Robot=SimpleNamespace(NODE_PING_PERIOD=SimpleNamespace(value=period), NODE_PING_MISSES=SimpleNamespace(value=3)))
    logged: list[str] = []
    logger = SimpleNamespace(warn=logged.append, error=logged.append)
    stub = object.__new__(type("Stub", (EpisodeNode,), {"conf": conf, "rosparam": {bool: SimpleNamespace(get_unsafe=lambda name: auto_reset)}, "get_logger": lambda self: logger}))
    stub.logged = logged
    stub._shutting_down = False
    stub._node_watch_task = None
    stub._node_pinger = pinger
    stub._node_ping_due = 0.0
    stub._node_ping_episode = None
    stub._episodes = EpisodeRuntime()
    stub._task = SimpleNamespace()
    stub._abort_reason = None
    stub._force_reset = False
    stub._robots_manager = None
    return stub


async def _watch_until(stub: typing.Any, done: typing.Callable[[], bool], what: str) -> None:
    for _ in range(300):
        stub._watch_nodes()
        await asyncio.sleep(0.1)
        if done():
            return
    raise AssertionError(f"timed out waiting for {what}")


def test_the_watchdog_faults_a_running_episode_on_a_frozen_node(rclpy_context: object) -> None:
    import arena_runtime_msgs.action

    async def scenario(pinger: typing.Any, children: dict[str, subprocess.Popen]) -> None:
        stub = _episode_stub(pinger, auto_reset=False)
        outcome = asyncio.get_running_loop().create_future()
        stub._episodes.pending_outcomes[stub._episodes.current.episode_id] = outcome
        await _watch_until(stub, lambda: len(pinger._tally._misses) == 4, "every node to answer")
        os.kill(children["victim"].pid, signal.SIGSTOP)
        await _watch_until(stub, outcome.done, "the episode to end")
        reason = f"node {NS}/victim stopped answering or left its active state"
        assert outcome.result() == (arena_runtime_msgs.action.RunEpisode.Result.FATAL, reason)
        assert stub._abort_reason == reason

    _with_pinger(scenario)


def test_the_watchdog_only_logs_a_frozen_node_in_standalone_mode(rclpy_context: object) -> None:
    async def scenario(pinger: typing.Any, children: dict[str, subprocess.Popen]) -> None:
        stub = _episode_stub(pinger, auto_reset=True)
        outcome = asyncio.get_running_loop().create_future()
        stub._episodes.pending_outcomes[stub._episodes.current.episode_id] = outcome
        await _watch_until(stub, lambda: len(pinger._tally._misses) == 4, "every node to answer")
        os.kill(children["victim"].pid, signal.SIGSTOP)
        await _watch_until(stub, lambda: bool(stub.logged), "the hung node to be logged")
        assert stub.logged == [f"node {NS}/victim stopped answering or left its active state"]
        assert not outcome.done()

    _with_pinger(scenario)


def test_the_watchdog_pings_nobody_between_episodes(rclpy_context: object) -> None:
    async def scenario(pinger: typing.Any, children: dict[str, subprocess.Popen]) -> None:
        stub = _episode_stub(pinger, auto_reset=False)
        stub._watch_nodes()
        assert stub._node_watch_task is None

    _with_pinger(scenario)


def test_the_watchdog_starts_no_round_while_one_is_in_flight(rclpy_context: object) -> None:
    async def scenario(pinger: typing.Any, children: dict[str, subprocess.Popen]) -> None:
        stub = _episode_stub(pinger, auto_reset=False, period=0.0)
        stub._episodes.pending_outcomes[stub._episodes.current.episode_id] = asyncio.get_running_loop().create_future()
        stub._watch_nodes()
        first = stub._node_watch_task
        stub._watch_nodes()
        assert stub._node_watch_task is first
        await first

    _with_pinger(scenario)


def test_the_watchdog_pings_at_most_once_a_period(rclpy_context: object) -> None:
    from arena_runtime.env.node_watch import NodePinger

    async def scenario(pinger: typing.Any, children: dict[str, subprocess.Popen]) -> None:
        stub = _episode_stub(NodePinger(pinger._node, "/node_watch_empty", 3), auto_reset=False, period=30.0)
        stub._episodes.pending_outcomes[stub._episodes.current.episode_id] = asyncio.get_running_loop().create_future()
        stub._watch_nodes()
        first = stub._node_watch_task
        await first
        stub._watch_nodes()
        assert stub._node_watch_task is first

    _with_pinger(scenario)
