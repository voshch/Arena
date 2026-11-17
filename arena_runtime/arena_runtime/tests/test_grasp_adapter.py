from __future__ import annotations

import asyncio
from types import SimpleNamespace

import control_msgs.action
import pytest
from action_msgs.msg import GoalStatus
from arena_rclpy_mixins.shared import FrameNamespace, Namespace
from arena_robots.Robot import RobotIdentifier
from arena_robots.task_kinds import TaskKind
from rclpy.task import Future

from arena_runtime.env.adapters import ADAPTERS
from arena_runtime.env.request import GraspPhase, GoToPhase, kind_of

CONTROLLER_ACTION = "/env_0/tiago/gripper_controller/gripper_cmd"


class _Publisher:
    def __init__(self) -> None:
        self.sent: list = []

    def publish(self, msg) -> None:
        self.sent.append(msg)


class _Handle:
    def __init__(self) -> None:
        self.result = Future()
        self.cancelled = False

    def get_result_async(self) -> Future:
        return self.result

    def cancel_goal_async(self) -> None:
        self.cancelled = True


class _ActionClient:
    def __init__(self) -> None:
        self.goals: list = []
        self.handles: list[_Handle] = []
        self.ready = True

    async def send_goal_timeout(self, goal, timeout_sec=None):
        self.goals.append(goal)
        self.handles.append(_Handle())
        return self.handles[-1]

    async def ensure(self, timeout_sec: float | None = None) -> bool:
        return self.ready


class _Simulator:
    def __init__(self) -> None:
        self.reachable: set[str] = {"cup_1"}
        self.hand = "env_0/tiago/arm_tool_link"
        self.held: dict[str, str] = {}
        self.goals_when_detached: list[int] = []
        self.goals_sent = lambda: 0

    async def grasp_attach(self, robot, joint: str, obj: str, hand: str) -> bool:
        if obj not in self.reachable or hand != self.hand:
            return False
        self.held[joint] = obj
        return True

    async def grasp_detach(self, robot, joint: str) -> None:
        self.goals_when_detached.append(self.goals_sent())
        self.held.pop(joint, None)


class _Node:
    def __init__(self) -> None:
        self.publishers: dict[str, _Publisher] = {}
        self.action_clients: dict[str, _ActionClient] = {}

    async def await_ros(self, future: Future):
        while not future.done():
            await asyncio.sleep(0)
        return future.result()

    def create_publisher(self, msg_type, topic: str, qos) -> _Publisher:
        self.publishers[topic] = _Publisher()
        return self.publishers[topic]

    def create_subscription(self, msg_type, topic: str, callback, qos) -> object:
        return callback

    def create_action_client_wrapper(self, action_type, name: str) -> _ActionClient:
        self.action_clients[name] = _ActionClient()
        return self.action_clients[name]


def _build(kind: str = "direct"):
    node = _Node()
    simulator = _Simulator()
    failed: list[str] = []
    manager = SimpleNamespace(
        name="tiago",
        failed=failed,
        fail_phase=failed.append,
        node=node,
        environment_manager=simulator,
        namespace=Namespace("/env_0/tiago"),
        robot_view=RobotIdentifier("tiago").resolve_sync(),
        robot=SimpleNamespace(resolved_request={}, frames=None, frame=FrameNamespace("env_0/tiago")),
        tf_buffer=None,
    )
    adapter = ADAPTERS["gripper"].get(kind)(robot_manager=manager)
    simulator.goals_sent = lambda: len(node.action_clients[CONTROLLER_ACTION].goals)
    return adapter, manager, node


def _joint(manager) -> str:
    caps = manager.robot_view.effective_caps(manager.robot.resolved_request, frames=manager.robot.frames)
    return caps.gripper["gripper"].joint


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def _finish(handle: _Handle, status: int, **result) -> None:
    handle.result.set_result(SimpleNamespace(status=status, result=control_msgs.action.GripperCommand.Result(**result)))


def test_grasp_kind_routes_to_the_direct_gripper_adapter():
    adapter, _, _ = _build()
    assert adapter.accepts == frozenset({TaskKind.GRASP})
    assert kind_of(GraspPhase(action="close")) is TaskKind.GRASP
    assert adapter.client_for(TaskKind.GRASP).task_kind is TaskKind.GRASP


def test_none_adapter_refuses_grasp_phases():
    adapter, manager, node = _build("none")
    assert TaskKind.GRASP not in adapter.accepts
    with pytest.raises(RuntimeError, match="accepts no task phases"):
        asyncio.run(adapter.dispatch_phase(GraspPhase(action="close"), manager))
    assert node.action_clients == {}


@pytest.mark.parametrize(("action", "position"), [("close", 0.0), ("open", 0.044)])
def test_grasp_sends_the_specs_position_to_the_controller(action: str, position: float):
    adapter, manager, node = _build()
    asyncio.run(adapter.dispatch_phase(GraspPhase(action=action), manager))
    [goal] = node.action_clients[CONTROLLER_ACTION].goals
    assert (goal.command.position, goal.command.max_effort) == (position, 16.0)
    [applied] = node.publishers["/env_0/tiago/gripper/gripper/applied"].sent
    assert (applied.position, applied.max_effort) == (position, 16.0)
    assert adapter.client_for(TaskKind.GRASP).is_done() is False


def test_grasp_phase_is_done_when_the_action_returns():
    adapter, manager, node = _build()
    asyncio.run(adapter.dispatch_phase(GraspPhase(action="close", object="cup_1"), manager))
    client = adapter.client_for(TaskKind.GRASP)
    assert (client.is_done(), client.status) == (False, None)
    _finish(node.action_clients[CONTROLLER_ACTION].handles[0], GoalStatus.STATUS_SUCCEEDED, reached_goal=True)
    assert (client.is_done(), client.status, client.reason) == (True, 0, None)


def test_a_close_that_stalls_on_an_object_counts_as_done():
    adapter, manager, node = _build()
    asyncio.run(adapter.dispatch_phase(GraspPhase(action="close", object="cup_1"), manager))
    _finish(node.action_clients[CONTROLLER_ACTION].handles[0], GoalStatus.STATUS_ABORTED, stalled=True)
    client = adapter.client_for(TaskKind.GRASP)
    assert (client.is_done(), client.status) == (True, 0)


def test_an_aborted_action_without_a_stall_fails_the_phase():
    adapter, manager, node = _build()
    asyncio.run(adapter.dispatch_phase(GraspPhase(action="close"), manager))
    _finish(node.action_clients[CONTROLLER_ACTION].handles[0], GoalStatus.STATUS_ABORTED)
    client = adapter.client_for(TaskKind.GRASP)
    assert client.is_done() is True
    assert client.status == 2
    assert "status" in (client.reason or "")


def test_a_new_grasp_resets_the_client_and_ignores_the_previous_result():
    adapter, manager, node = _build()
    asyncio.run(adapter.dispatch_phase(GraspPhase(action="close"), manager))
    asyncio.run(adapter.dispatch_phase(GraspPhase(action="open"), manager))
    first, second = node.action_clients[CONTROLLER_ACTION].handles
    _finish(first, GoalStatus.STATUS_SUCCEEDED, reached_goal=True)
    client = adapter.client_for(TaskKind.GRASP)
    assert (client.is_done(), client.status) == (False, None)
    _finish(second, GoalStatus.STATUS_SUCCEEDED, reached_goal=True)
    assert (client.is_done(), client.status) == (True, 0)


def test_cancel_goals_cancels_the_running_grasp():
    adapter, manager, node = _build()
    asyncio.run(adapter.dispatch_phase(GraspPhase(action="close"), manager))
    adapter.cancel_goals()
    assert node.action_clients[CONTROLLER_ACTION].handles[0].cancelled is True


def test_grasp_names_an_unknown_instance():
    adapter, manager, node = _build()
    with pytest.raises(ValueError, match="left"):
        asyncio.run(adapter.dispatch_phase(GraspPhase(action="close", instance="left"), manager))
    assert node.action_clients[CONTROLLER_ACTION].goals == []


def test_direct_adapter_accepts_only_grasp_phases():
    adapter, manager, _ = _build()
    with pytest.raises(RuntimeError, match="only grasp phases"):
        asyncio.run(adapter.dispatch_phase(GoToPhase(), manager))


def test_readiness_does_not_wait_on_the_controller_server():
    adapter, manager, node = _build()
    node.action_clients[CONTROLLER_ACTION].ready = False
    asyncio.run(asyncio.wait_for(adapter.wait_until_ready(manager, set()), timeout=1.0))


def test_grasp_names_the_controller_server_that_is_down():
    adapter, manager, node = _build()
    node.action_clients[CONTROLLER_ACTION].ready = False
    with pytest.raises(RuntimeError, match=CONTROLLER_ACTION):
        asyncio.run(adapter.dispatch_phase(GraspPhase(action="close"), manager))
    assert node.action_clients[CONTROLLER_ACTION].goals == []


def test_a_close_on_an_object_attaches_it_once_the_goal_succeeds():
    async def run():
        adapter, manager, node = _build()
        await adapter.dispatch_phase(GraspPhase(action="close", object="cup_1"), manager)
        await _settle()
        assert manager.environment_manager.held == {}
        _finish(node.action_clients[CONTROLLER_ACTION].handles[0], GoalStatus.STATUS_SUCCEEDED, reached_goal=True)
        await _settle()
        assert manager.environment_manager.held == {_joint(manager): "cup_1"}

    asyncio.run(run())


def test_a_close_that_stalls_on_an_object_attaches_it():
    async def run():
        adapter, manager, node = _build()
        await adapter.dispatch_phase(GraspPhase(action="close", object="cup_1"), manager)
        _finish(node.action_clients[CONTROLLER_ACTION].handles[0], GoalStatus.STATUS_ABORTED, stalled=True)
        await _settle()
        assert manager.environment_manager.held == {_joint(manager): "cup_1"}

    asyncio.run(run())


def test_an_aborted_close_attaches_nothing():
    async def run():
        adapter, manager, node = _build()
        await adapter.dispatch_phase(GraspPhase(action="close", object="cup_1"), manager)
        _finish(node.action_clients[CONTROLLER_ACTION].handles[0], GoalStatus.STATUS_ABORTED)
        await _settle()
        assert manager.environment_manager.held == {}

    asyncio.run(run())


def test_a_close_on_an_object_out_of_reach_holds_nothing():
    async def run():
        adapter, manager, node = _build()
        manager.environment_manager.reachable.clear()
        await adapter.dispatch_phase(GraspPhase(action="close", object="cup_1"), manager)
        _finish(node.action_clients[CONTROLLER_ACTION].handles[0], GoalStatus.STATUS_SUCCEEDED, reached_goal=True)
        await _settle()
        assert manager.environment_manager.held == {}
        assert manager.failed == ["gripper joint 'gripper_right_finger_joint' holds no 'cup_1' after closing"]

    asyncio.run(run())


def test_a_close_without_an_object_attaches_nothing():
    async def run():
        adapter, manager, node = _build()
        await adapter.dispatch_phase(GraspPhase(action="close"), manager)
        _finish(node.action_clients[CONTROLLER_ACTION].handles[0], GoalStatus.STATUS_SUCCEEDED, reached_goal=True)
        await _settle()
        assert manager.environment_manager.held == {}

    asyncio.run(run())


def test_an_open_releases_the_held_object_before_the_goal_goes_out():
    async def run():
        adapter, manager, node = _build()
        manager.environment_manager.held[_joint(manager)] = "cup_1"
        await adapter.dispatch_phase(GraspPhase(action="open"), manager)
        assert manager.environment_manager.held == {}
        assert manager.environment_manager.goals_when_detached == [0]
        assert len(node.action_clients[CONTROLLER_ACTION].goals) == 1

    asyncio.run(run())


def test_a_new_grasp_drops_the_attach_waiting_on_the_previous_close():
    async def run():
        adapter, manager, node = _build()
        await adapter.dispatch_phase(GraspPhase(action="close", object="cup_1"), manager)
        await adapter.dispatch_phase(GraspPhase(action="open"), manager)
        first, _ = node.action_clients[CONTROLLER_ACTION].handles
        _finish(first, GoalStatus.STATUS_SUCCEEDED, reached_goal=True)
        await _settle()
        assert manager.environment_manager.held == {}

    asyncio.run(run())
