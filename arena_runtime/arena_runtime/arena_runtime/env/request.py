"""Typed task requests submitted to robots, the shared types plus their action kinds."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import attrs
from arena_robots.task_kinds import TaskKind
from arena_simulation_setup.shared.task import GoToPhase, LoopPhase, PlayGesturePhase, ReachPhase, TaskPhase, TaskRequest


def kind_of(phase: TaskPhase) -> TaskKind:
    """Action kind dispatched for a phase, loops have none."""
    return TaskKind(phase.kind)


def leaf_phases(phases: Iterable[TaskPhase]) -> Iterator[TaskPhase]:
    """The dispatchable phases in order, loop bodies descended into."""
    for phase in phases:
        if isinstance(phase, LoopPhase):
            yield from leaf_phases(phase.phases)
        else:
            yield phase


@attrs.define
class ElevatorRidePhase(GoToPhase):
    """Board the `target` cabin and stay until the elevator delivers the robot, `until` names the destination cabin."""

    destination: str = attrs.field(default="", kw_only=True)
    exit_cabin: str = attrs.field(default="", kw_only=True)


@attrs.define
class ElevatorLegPhase(GoToPhase):
    """Door-aligned approach or exit waypoint around an elevator ride."""


@attrs.define
class StairsRidePhase(GoToPhase):
    """Drive to a pose past the boarding half's trigger line, judged on the robot standing on the `target` partner half after the seam teleport."""


@attrs.define
class StairsLegPhase(GoToPhase):
    """Axis-aligned approach or exit waypoint around a stair crossing."""


__all__ = [
    "TaskKind",
    "TaskPhase",
    "GoToPhase",
    "LoopPhase",
    "ElevatorRidePhase",
    "ElevatorLegPhase",
    "StairsRidePhase",
    "StairsLegPhase",
    "ReachPhase",
    "PlayGesturePhase",
    "TaskRequest",
    "kind_of",
    "leaf_phases",
]
