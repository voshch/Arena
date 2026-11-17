"""Grasp client: tracks the GripperCommand goal of the active grasp phase."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import attrs
import control_msgs.action
from action_msgs.msg import GoalStatus
from arena_robots.clients import Client
from arena_robots.task_kinds import TaskKind

if TYPE_CHECKING:
    import tf2_ros
    from arena_rclpy_mixins.Async import ActionClientWrapper, AsyncNode
    from arena_robots.Robot import RobotView
    from rclpy.action.client import ClientGoalHandle
    from rclpy.task import Future

STATUS_SUCCEEDED = 0
STATUS_ABORTED = 2
_SERVER_TIMEOUT_S = 10.0


@attrs.frozen
class GraspGoal:
    """A GripperCommand goal for one gripper instance."""

    instance: str
    command: control_msgs.action.GripperCommand.Goal


class GripperGraspClient(Client):
    task_kind = TaskKind.GRASP

    def __init__(self, robot: RobotView, namespace: str, *, node: AsyncNode, tf_buffer: tf2_ros.Buffer, actions: Mapping[str, ActionClientWrapper], endpoints: Mapping[str, str]) -> None:
        super().__init__(robot, namespace, node=node, tf_buffer=tf_buffer)
        self._actions = actions
        self._endpoints = endpoints
        self._handle: ClientGoalHandle | None = None
        self._result_future: Future | None = None
        self._done: bool | None = None
        self._status: int | None = None
        self._reason: str | None = None

    def action_endpoint(self) -> str:
        return ", ".join(self._endpoints.values())

    async def wait_ready(self) -> None:
        return None

    async def send_goal(self, goal: object) -> object:
        assert isinstance(goal, GraspGoal)
        if not await self._actions[goal.instance].ensure(timeout_sec=_SERVER_TIMEOUT_S):
            raise RuntimeError(f"gripper {goal.instance!r}: action server {self._endpoints[goal.instance]} not up after {_SERVER_TIMEOUT_S:.0f}s")
        self._done = False
        self._status = None
        self._reason = None
        self._result_future = None
        handle = await self._actions[goal.instance].send_goal_timeout(goal.command, 10.0)
        if handle is None:
            raise RuntimeError(f"gripper {goal.instance!r} did not accept the goal in time")
        self._handle = handle
        result_future = handle.get_result_async()
        self._result_future = result_future
        result_future.add_done_callback(self._on_result)
        return handle

    async def await_result(self) -> object:
        if self._result_future is None:
            raise RuntimeError("No goal in flight; call send_goal first.")
        return await self.node.await_ros(self._result_future)

    def is_done(self) -> bool | None:
        return self._done

    def cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel_goal_async()

    @property
    def status(self) -> int | None:
        return self._status

    @property
    def reason(self) -> str | None:
        return self._reason

    def _on_result(self, future: Future) -> None:
        if future is not self._result_future:
            return
        self._done = True
        try:
            response = future.result()
        except Exception as exc:
            self._status, self._reason = STATUS_ABORTED, f"gripper result failed: {exc!r}"
            return
        if response is None:
            self._status, self._reason = STATUS_ABORTED, "gripper result missing"
            return
        result = response.result
        if response.status == GoalStatus.STATUS_SUCCEEDED or result.reached_goal or result.stalled:
            self._status, self._reason = STATUS_SUCCEEDED, None
        else:
            self._status, self._reason = STATUS_ABORTED, f"gripper goal ended with status {response.status}"
