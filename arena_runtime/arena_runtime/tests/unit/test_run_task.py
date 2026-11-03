"""Mid-episode task assignment: the RunTask goals the env tracks off the robots' runners, and the SubmitTask service next to them."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from types import SimpleNamespace

import attrs
import numpy as np
import pytest
import shapely
from arena_simulation_setup.shared.judge import Sample, WallGrid

from arena_runtime.env.robot_manager.task_runner import TaskRunner

pytest.importorskip("rclpy")
pytest.importorskip("arena_runtime_msgs.action")

ZONES: dict = {"kitchen": shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)])}
OK = (0, None)
FAILED = (2, "aborted")
OPEN_FLOOR = WallGrid(cells=np.zeros((400, 400), dtype=bool), origin=(0.0, 0.0), resolution=0.05)
TWO_GOTOS = "[{goto: [1.0, 1.0, 0.0]}, {goto: [3.0, 3.0, 0.0]}]"
ONE_GOTO = "[{goto: [5.0, 5.0, 0.0]}]"


def at(x: float, y: float) -> Sample:
    return Sample(t=0.0, robots={"r": (x, y, 0.0)}, peds={}, field=lambda e, f: None, clips={}, walls=OPEN_FLOOR)


class _Manager:
    """A robot manager's task side: a real runner behind the submit and stop calls."""

    def __init__(self) -> None:
        self.runner = TaskRunner("r")
        self.stopped = 0

    async def submit_task(self, request: object, start: object = None, pedestrians: object = ()) -> None:
        phases = [attrs.evolve(phase, tolerance_radius=0.5, tolerance_angle=0.0, hold_time=0.0) for phase in request.phases]
        self.runner.submit(phases, request.conditions)

    def stop_task(self) -> None:
        self.stopped += 1
        self.runner.stop()

    def reach(self, x: float, y: float, failure: tuple = OK) -> None:
        self.runner.step(at(x, y), ZONES, None, failure)


class _Handle:
    """The server side of one RunTask goal."""

    def __init__(self, robot: str, task_yaml: str, goal: int = 1) -> None:
        self.request = SimpleNamespace(robot=robot, task_yaml=task_yaml)
        self.goal_id = SimpleNamespace(uuid=[goal] * 16)
        self.is_cancel_requested = False
        self.feedback: list[tuple[int, str]] = []
        self.status = ""

    def publish_feedback(self, feedback: object) -> None:
        self.feedback.append((feedback.phase_index, feedback.outcome))

    def succeed(self) -> None:
        self.status = "succeeded"

    def abort(self) -> None:
        self.status = "aborted"

    def canceled(self) -> None:
        self.status = "canceled"


def _node(initialized: bool = True) -> tuple[object, _Manager]:
    from arena_runtime.env.episode_node import EpisodeNode, EpisodeRuntime
    from arena_runtime.env.realizer import Realizer

    manager = _Manager()
    cls = type("Stub", (EpisodeNode,), {"rosparam": {bool: SimpleNamespace(get=lambda name, default: initialized)}, "event_loop": property(lambda self: asyncio.get_running_loop())})
    node = object.__new__(cls)
    node._robots_manager = SimpleNamespace(managers={"r": manager})
    node._reset_lock = asyncio.Lock()
    node._realizer = Realizer()
    node._episodes = EpisodeRuntime()
    node._run_tasks = {}
    node._assigned = set()
    node._shutting_down = False
    node._timeline = []
    node._sample = SimpleNamespace(t=0.0)
    return node, manager


async def _started(node: object, handle: _Handle) -> asyncio.Task:
    goal = asyncio.create_task(node._run_task(handle))
    deadline = asyncio.get_running_loop().time() + 5.0
    while handle.request.robot not in node._run_tasks and not goal.done():
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0)
    return goal


def test_a_goal_ends_met_when_all_its_phases_are_met_and_feedback_names_each_phase_as_it_ends() -> None:
    async def scenario() -> None:
        node, manager = _node()
        handle = _Handle("r", TWO_GOTOS)
        goal = await _started(node, handle)

        manager.reach(1.0, 1.0)
        node._advance_run_tasks()
        assert handle.feedback == [(0, "met")]
        assert not goal.done()

        manager.reach(3.0, 3.0)
        node._advance_run_tasks()
        result = await goal
        assert handle.feedback == [(0, "met"), (1, "met")]
        assert (result.outcome, list(result.phase_indices), result.error_msg, handle.status) == ("met", [0, 1], "", "succeeded")
        assert node._run_tasks == {}

    asyncio.run(scenario())


def test_a_goal_with_a_failed_phase_runs_its_remaining_phases_and_ends_failed() -> None:
    async def scenario() -> None:
        node, manager = _node()
        handle = _Handle("r", TWO_GOTOS)
        goal = await _started(node, handle)

        manager.reach(1.0, 1.0, FAILED)
        node._advance_run_tasks()
        assert handle.feedback == [(0, "failed")]
        assert not goal.done()

        manager.reach(3.0, 3.0)
        node._advance_run_tasks()
        result = await goal
        assert result.outcome == "failed"
        assert handle.feedback == [(0, "failed"), (1, "met")]
        assert handle.status == "succeeded"

    asyncio.run(scenario())


def test_a_later_goal_for_the_robot_replaces_the_running_one_which_ends_dropped() -> None:
    async def scenario() -> None:
        node, manager = _node()
        first = _Handle("r", TWO_GOTOS, goal=1)
        second = _Handle("r", ONE_GOTO, goal=2)
        first_goal = await _started(node, first)
        second_goal = asyncio.create_task(node._run_task(second))

        result = await first_goal
        assert result.outcome == "dropped"
        assert first.status == "aborted"
        assert manager.runner.outcomes("dropped") == [0, 1]

        for _ in range(3):
            await asyncio.sleep(0)
        assert list(node._run_tasks["r"].phases) == [2]
        manager.reach(5.0, 5.0)
        node._advance_run_tasks()
        assert (await second_goal).outcome == "met"

    asyncio.run(scenario())


def test_a_goal_ends_dropped_when_the_episode_resets_under_it() -> None:
    from arena_runtime.env.task_layer import EpisodeRecord

    async def scenario() -> None:
        node, manager = _node()
        handle = _Handle("r", TWO_GOTOS)
        goal = await _started(node, handle)

        manager.runner.begin_episode()
        node._episodes.current = EpisodeRecord(episode_id=1)
        node._advance_run_tasks()
        result = await goal
        assert result.outcome == "dropped"
        assert handle.feedback == [(0, "dropped"), (1, "dropped")]

    asyncio.run(scenario())


def test_cancelling_a_goal_stops_the_robot_and_ends_it_dropped() -> None:
    from rclpy.action import CancelResponse

    async def scenario() -> None:
        node, manager = _node()
        handle = _Handle("r", TWO_GOTOS)
        goal = await _started(node, handle)

        handle.is_cancel_requested = True
        assert node._run_task_cancel_callback(handle) == CancelResponse.ACCEPT
        result = await goal
        assert (result.outcome, handle.status) == ("dropped", "canceled")
        assert manager.stopped == 1
        assert manager.runner.outcomes("dropped") == [0, 1]
        assert node._run_tasks == {}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("robot", "task_yaml", "initialized", "error"),
    [
        ("ghost", TWO_GOTOS, True, "unknown robot 'ghost'"),
        ("r", "{unclosed", True, ""),
        ("r", TWO_GOTOS, False, "task generator not initialized"),
    ],
)
def test_refused_input_aborts_the_goal_with_the_reason(robot: str, task_yaml: str, initialized: bool, error: str) -> None:
    async def scenario() -> None:
        node, manager = _node(initialized)
        handle = _Handle(robot, task_yaml)
        result = await node._run_task(handle)
        assert handle.status == "aborted"
        assert error in result.error_msg
        assert result.error_msg
        assert result.outcome == ""
        assert node._run_tasks == {}
        assert manager.runner.phases == []

    asyncio.run(scenario())


def test_the_submit_task_service_assigns_through_the_same_method_as_the_action() -> None:
    import arena_runtime_msgs.srv

    async def scenario() -> None:
        node, manager = _node()
        Request, Response = arena_runtime_msgs.srv.SubmitTask.Request, arena_runtime_msgs.srv.SubmitTask.Response

        response = await node._cb_submit_task(Request(robot="r", task_yaml=TWO_GOTOS), Response())
        assert (response.success, list(response.phase_indices)) == (True, [0, 1])
        assert node._assigned == {"r"}

        response = await node._cb_submit_task(Request(robot="r", task_yaml="[]"), Response())
        assert (response.success, list(response.phase_indices), manager.stopped) == (True, [], 1)
        assert manager.runner.outcomes("dropped") == [0, 1]

        response = await node._cb_submit_task(Request(robot="ghost", task_yaml=TWO_GOTOS), Response())
        assert not response.success
        assert "unknown robot 'ghost'" in response.error_msg

    asyncio.run(scenario())


def _runner(*states: object) -> TaskRunner:
    return TaskRunner("r", phases=list(states))


def _state(index: int, outcome: str, *, request: int = 0, reason: str = "") -> object:
    from arena_runtime.env.robot_manager.task_runner import PhaseState

    return PhaseState(index=index, phase=None, monitor=None, outcome=outcome, reason=reason, request=request)


def test_a_loop_exit_dropping_the_rest_of_the_iteration_still_ends_the_request_met() -> None:
    assert _runner(_state(0, "met", request=1), _state(1, "met", request=1), _state(2, "dropped", request=1)).request_outcome(1) == "met"


def test_a_request_stays_open_while_a_later_loop_iteration_is_pending() -> None:
    assert _runner(_state(0, "met", request=1), _state(1, "pending", request=1)).request_outcome(1) is None


def test_phases_dropped_by_a_later_task_or_a_stop_end_the_request_dropped() -> None:
    assert _runner(_state(0, "met", request=1), _state(1, "dropped", request=1, reason="replaced"), _state(2, "pending", request=2)).request_outcome(1) == "dropped"
    assert _runner(_state(0, "dropped", request=1, reason="stopped")).request_outcome(1) == "dropped"


def test_phases_dropped_by_a_failure_policy_end_the_request_failed() -> None:
    assert _runner(_state(0, "met", request=1), _state(1, "dropped", request=1, reason="policy")).request_outcome(1) == "failed"


def test_phases_of_other_requests_do_not_count() -> None:
    assert _runner(_state(0, "failed", request=0), _state(1, "met", request=1)).request_outcome(1) == "met"


def test_the_task_is_ticked_only_while_the_episode_runs_and_the_reset_lock_is_free() -> None:
    from arena_runtime_msgs.action import RunEpisode

    from arena_runtime.env.task_layer import EpisodeRecord
    from arena_runtime.testing.env_host import FakeTask

    async def scenario() -> None:
        node, _ = _node()
        node._task = FakeTask()
        ended = {"r": _state(0, "met")}

        await node._on_tick(ended)
        assert node._task.ticks == []

        node._episodes.current = EpisodeRecord(episode_id=1, outcome_state=RunEpisode.Result.RUNNING)
        async with node._reset_lock:
            await node._on_tick(ended)
        assert node._task.ticks == []

        await node._on_tick(ended)
        assert node._task.ticks == [(ended, set())]

        node._episodes.current.outcome_state = RunEpisode.Result.SUCCESS
        await node._on_tick(ended)
        assert len(node._task.ticks) == 1

    asyncio.run(scenario())


def test_robots_assigned_from_outside_are_handed_to_the_next_task_tick_once() -> None:
    import arena_runtime_msgs.srv
    from arena_runtime_msgs.action import RunEpisode

    from arena_runtime.env.task_layer import EpisodeRecord
    from arena_runtime.testing.env_host import FakeTask

    async def scenario() -> None:
        node, _ = _node()
        node._task = FakeTask()
        node._episodes.current = EpisodeRecord(episode_id=1, outcome_state=RunEpisode.Result.RUNNING)
        Request, Response = arena_runtime_msgs.srv.SubmitTask.Request, arena_runtime_msgs.srv.SubmitTask.Response

        await node._cb_submit_task(Request(robot="r", task_yaml=TWO_GOTOS), Response())
        await node._on_tick({})
        await node._on_tick({})
        assert [assigned for _, assigned in node._task.ticks] == [{"r"}, set()]

        await node._run_task(_Handle("r", "[]"))
        await node._on_tick({})
        assert node._task.ticks[-1][1] == {"r"}

    asyncio.run(scenario())


def test_goals_keep_advancing_on_ticks_the_task_does_not_run_on() -> None:
    from arena_runtime.testing.env_host import FakeTask

    async def scenario() -> None:
        node, manager = _node()
        node._task = FakeTask()
        handle = _Handle("r", ONE_GOTO)
        goal = await _started(node, handle)

        manager.reach(5.0, 5.0)
        await node._on_tick({})
        assert (await goal).outcome == "met"
        assert node._task.ticks == []

    asyncio.run(scenario())


class _Rig:
    """A real node serving the RunTask action with the env node's callbacks over a stub manager, and a client node."""

    def __init__(self, host: object, client: object, manager: _Manager) -> None:
        self.host = host
        self.client = client
        self.manager = manager

    async def send(self, task_yaml: str) -> object:
        from arena_runtime.testing.env_host import register_as_ros_future

        goal = await asyncio.wrap_future(register_as_ros_future(self.client.send_goal_async(_goal(task_yaml))))
        assert goal.accepted
        return goal

    async def cancel(self, goal: object) -> None:
        from arena_runtime.testing.env_host import register_as_ros_future

        await asyncio.wrap_future(register_as_ros_future(goal.cancel_goal_async()))

    async def result(self, goal: object) -> object:
        from arena_runtime.testing.env_host import register_as_ros_future

        return await asyncio.wait_for(asyncio.wrap_future(register_as_ros_future(goal.get_result_async())), 10.0)

    async def until(self, predicate: object, what: str) -> None:
        deadline = asyncio.get_running_loop().time() + 10.0
        while not predicate():
            assert asyncio.get_running_loop().time() < deadline, f"timed out waiting for {what}"
            await asyncio.sleep(0.02)


def _goal(task_yaml: str) -> object:
    from arena_runtime_msgs.action import RunTask

    return RunTask.Goal(robot="r", task_yaml=task_yaml)


@contextlib.asynccontextmanager
async def _rig() -> AsyncIterator[_Rig]:
    import rclpy
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.spin import create_executor
    from rclpy.action import ActionClient, ActionServer
    from rclpy.callback_groups import ReentrantCallbackGroup
    from arena_runtime_msgs.action import RunTask

    from arena_runtime.env.episode_node import EpisodeNode, EpisodeRuntime
    from arena_runtime.env.realizer import Realizer

    manager = _Manager()

    class Host(AsyncNode):
        episode_id = EpisodeNode.episode_id
        _run_task_goal_callback = EpisodeNode._run_task_goal_callback
        _run_task_cancel_callback = EpisodeNode._run_task_cancel_callback
        _run_task_execute_callback = EpisodeNode._run_task_execute_callback
        _run_task = EpisodeNode._run_task
        _until_cancel_requested = EpisodeNode._until_cancel_requested
        _stop_run_task = EpisodeNode._stop_run_task
        _advance_run_tasks = EpisodeNode._advance_run_tasks
        _advance_run_task = EpisodeNode._advance_run_task
        _assign_task = EpisodeNode._assign_task
        _ezilear_phases = EpisodeNode._ezilear_phases

    host = Host("run_task_host")
    host.rosparam = {bool: SimpleNamespace(get=lambda name, default: True)}
    host._robots_manager = SimpleNamespace(managers={"r": manager})
    host._reset_lock = asyncio.Lock()
    host._realizer = Realizer()
    host._episodes = EpisodeRuntime()
    host._run_tasks = {}
    host._assigned = set()
    host._shutting_down = False
    server = ActionServer(host, RunTask, "run_task", execute_callback=host._run_task_execute_callback, goal_callback=host._run_task_goal_callback, cancel_callback=host._run_task_cancel_callback, callback_group=ReentrantCallbackGroup())
    probe = rclpy.create_node("run_task_probe")
    client = ActionClient(probe, RunTask, "run_task")
    executor = create_executor()
    executor.add_node(host)
    executor.add_node(probe)
    spin = asyncio.get_running_loop().run_in_executor(None, executor.spin)
    try:
        assert await asyncio.to_thread(client.wait_for_server, 10.0)
        yield _Rig(host, client, manager)
    finally:
        executor.shutdown()
        await spin
        server.destroy()
        client.destroy()
        probe.destroy_node()
        host.destroy_node()


def test_cancelling_a_goal_through_the_action_server_stops_the_robot_and_ends_it_canceled() -> None:
    from action_msgs.msg import GoalStatus

    async def scenario() -> None:
        async with _rig() as rig:
            goal = await rig.send(TWO_GOTOS)
            await rig.until(lambda: "r" in rig.host._run_tasks, "the goal being tracked")

            await rig.cancel(goal)
            result = await rig.result(goal)
            assert result.status == GoalStatus.STATUS_CANCELED
            assert result.result.outcome == "dropped"
            assert rig.manager.stopped == 1
            assert rig.manager.runner.outcomes("dropped") == [0, 1]
            assert rig.host._run_tasks == {}

    asyncio.run(scenario())


def test_a_cancel_arriving_while_the_assign_waits_is_honored_once_it_returns() -> None:
    from action_msgs.msg import GoalStatus

    async def scenario() -> None:
        async with _rig() as rig:
            async with rig.host._reset_lock:
                goal = await rig.send(TWO_GOTOS)
                await rig.cancel(goal)
                assert rig.host._run_tasks == {}
            result = await rig.result(goal)
            assert result.status == GoalStatus.STATUS_CANCELED
            assert rig.manager.stopped == 1
            assert rig.manager.runner.outcomes("dropped") == [0, 1]
            assert rig.host._run_tasks == {}

    asyncio.run(scenario())


def test_a_cancel_after_the_request_finished_only_resolves_the_goal() -> None:
    from action_msgs.msg import GoalStatus

    async def scenario() -> None:
        async with _rig() as rig:
            goal = await rig.send(ONE_GOTO)
            await rig.until(lambda: "r" in rig.host._run_tasks, "the goal being tracked")
            rig.manager.reach(5.0, 5.0)

            await rig.cancel(goal)
            result = await rig.result(goal)
            assert result.status == GoalStatus.STATUS_CANCELED
            assert result.result.outcome == "met"
            assert rig.manager.stopped == 0

    asyncio.run(scenario())


def test_a_goal_replaced_by_a_later_one_aborts_and_the_later_one_succeeds_when_met() -> None:
    from action_msgs.msg import GoalStatus

    async def scenario() -> None:
        async with _rig() as rig:
            first = await rig.send(TWO_GOTOS)
            await rig.until(lambda: "r" in rig.host._run_tasks, "the first goal being tracked")
            second = await rig.send(ONE_GOTO)

            replaced = await rig.result(first)
            assert (replaced.status, replaced.result.outcome) == (GoalStatus.STATUS_ABORTED, "dropped")

            await rig.until(lambda: "r" in rig.host._run_tasks and rig.host._run_tasks["r"].phases == [2], "the second goal being tracked")
            rig.manager.reach(5.0, 5.0)
            rig.host._advance_run_tasks()
            met = await rig.result(second)
            assert (met.status, met.result.outcome) == (GoalStatus.STATUS_SUCCEEDED, "met")

    asyncio.run(scenario())
