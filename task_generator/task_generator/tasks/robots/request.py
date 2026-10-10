"""Typed task requests submitted to robots, the shared types plus their action kinds."""

from __future__ import annotations

from arena_robots.task_kinds import TaskKind
from arena_simulation_setup.shared.task import GoToPhase, PlayGesturePhase, ReachPhase, TaskPhase, TaskRequest


def kind_of(phase: TaskPhase) -> TaskKind:
    """Action kind dispatched for a phase."""
    return TaskKind(phase.kind)


__all__ = [
    "TaskKind",
    "TaskPhase",
    "GoToPhase",
    "ReachPhase",
    "PlayGesturePhase",
    "TaskRequest",
    "kind_of",
]
