from __future__ import annotations

import asyncio
from types import SimpleNamespace

import control_msgs.action
import control_msgs.msg
import pytest
import std_msgs.msg
from arena_rclpy_mixins.shared import Namespace
from arena_robots.Robot import RobotIdentifier

from arena_runtime.env.adapters import ADAPTERS


class _Publisher:
    def __init__(self) -> None:
        self.sent: list = []

    def publish(self, msg) -> None:
        self.sent.append(msg)


class _ActionClient:
    def __init__(self) -> None:
        self.goals: list = []

    async def send_goal_timeout(self, goal, timeout_sec=None):
        self.goals.append(goal)


class _Node:
    def __init__(self) -> None:
        self.publishers: dict[str, _Publisher] = {}
        self.subscriptions: dict[str, object] = {}
        self.action_clients: dict[str, _ActionClient] = {}

    def create_publisher(self, msg_type, topic: str, qos) -> _Publisher:
        self.publishers[topic] = _Publisher()
        return self.publishers[topic]

    def create_subscription(self, msg_type, topic: str, callback, qos) -> object:
        self.subscriptions[topic] = callback
        return callback

    def create_action_client_wrapper(self, action_type, name: str) -> _ActionClient:
        self.action_clients[name] = _ActionClient()
        return self.action_clients[name]


def _manager(node: _Node) -> SimpleNamespace:
    return SimpleNamespace(
        name="tiago",
        node=node,
        namespace=Namespace("/env_0/tiago"),
        robot_view=RobotIdentifier("tiago").resolve_sync(),
        robot=SimpleNamespace(resolved_request={}, frames=None),
    )


def _build(cap: str, kind: str) -> tuple[object, _Node]:
    node = _Node()
    return ADAPTERS[cap].get(kind)(robot_manager=_manager(node)), node


@pytest.mark.parametrize("cap", ["lift", "gripper"])
def test_cap_registers_direct_and_none(cap: str):
    assert sorted(ADAPTERS[cap].keys()) == ["direct", "none"]


@pytest.mark.parametrize(("cap", "kind"), [(cap, kind) for cap in ("lift", "gripper") for kind in ("direct", "none")])
def test_adapter_is_selectable_for_a_robot_advertising_the_cap(cap: str, kind: str):
    cls = ADAPTERS[cap].get(kind)
    available = RobotIdentifier("tiago").resolve_sync().effective_caps({}).available
    assert cls.kind == kind
    assert cls._meta().cap == cap
    assert cls._meta().accepts == frozenset()
    assert cls._meta().bringup._bringup_meta.requires <= available


@pytest.mark.parametrize("cap", ["lift", "gripper"])
def test_none_binds_nothing(cap: str):
    adapter, node = _build(cap, "none")
    assert (node.publishers, node.subscriptions, node.action_clients) == ({}, {}, {})
    assert adapter.accepts == frozenset()


def test_direct_lift_forwards_a_clamped_height_to_the_controller():
    adapter, node = _build("lift", "direct")
    assert list(node.subscriptions) == ["/env_0/tiago/lift/torso/command"]
    node.subscriptions["/env_0/tiago/lift/torso/command"](std_msgs.msg.Float64(data=2.0))
    [command] = node.publishers["/env_0/tiago/torso_controller/commands"].sent
    assert list(command.data) == [0.35]
    [applied] = node.publishers["/env_0/tiago/lift/torso/applied"].sent
    assert applied.data == 0.35


def test_direct_lift_passes_a_height_inside_the_limits():
    _, node = _build("lift", "direct")
    node.subscriptions["/env_0/tiago/lift/torso/command"](std_msgs.msg.Float64(data=0.1))
    [command] = node.publishers["/env_0/tiago/torso_controller/commands"].sent
    assert list(command.data) == [0.1]


def test_direct_gripper_sends_a_clamped_goal_to_the_controller_action():
    adapter, node = _build("gripper", "direct")
    topic = "/env_0/tiago/gripper/gripper/command"
    assert list(node.subscriptions) == [topic]
    asyncio.run(node.subscriptions[topic](control_msgs.msg.GripperCommand(position=0.5, max_effort=100.0)))
    [goal] = node.action_clients["/env_0/tiago/gripper_controller/gripper_cmd"].goals
    assert isinstance(goal, control_msgs.action.GripperCommand.Goal)
    assert (goal.command.position, goal.command.max_effort) == (0.044, 16.0)
    [applied] = node.publishers["/env_0/tiago/gripper/gripper/applied"].sent
    assert (applied.position, applied.max_effort) == (0.044, 16.0)


def test_direct_gripper_survives_a_rejected_command():
    _, node = _build("gripper", "direct")
    asyncio.run(node.subscriptions["/env_0/tiago/gripper/gripper/command"](control_msgs.msg.GripperCommand(position=float("nan"), max_effort=1.0)))
    assert node.action_clients["/env_0/tiago/gripper_controller/gripper_cmd"].goals == []
