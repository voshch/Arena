"""One RunTask goal and the feedback it owes, read off the phase states of its robot's runner."""

from __future__ import annotations

import asyncio
import typing

import attrs

if typing.TYPE_CHECKING:
    from rclpy.action.server import ServerGoalHandle

    from arena_runtime.env.robot_manager.task_runner import Outcome, TaskRunner


@attrs.define
class RunningTask:
    """A RunTask goal, its episode, the runner request it submitted and the future its action result waits on."""

    handle: ServerGoalHandle
    episode_id: int
    request: int
    done: asyncio.Future[Outcome]
    phases: list[int] = attrs.Factory(list)
    reported: set[int] = attrs.Factory(set)
    cancelled: bool = False

    @property
    def goal_id(self) -> bytes:
        return bytes(self.handle.goal_id.uuid)

    def ended(self, runner: TaskRunner | None) -> list[tuple[int, Outcome]]:
        """The phases that left pending since the last call, `runner` being None when it no longer holds the goal's phases."""
        if runner is None:
            ended: list[tuple[int, Outcome]] = [(index, "dropped") for index in self.phases if index not in self.reported]
        else:
            ours = [state for state in runner.phases if state.request == self.request]
            self.phases = [state.index for state in ours]
            ended = [(state.index, state.outcome) for state in ours if state.index not in self.reported and state.outcome != "pending"]
        self.reported.update(index for index, _ in ended)
        return ended
