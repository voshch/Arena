"""FleetManager: matches TaskModeSpec entries to RobotManager instances at reset."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import yaml

from task_generator.tasks.robots.request import TaskKind

if TYPE_CHECKING:
    from task_generator.manager.robot_manager.robot_manager import RobotManager


def _task_kind(value: TaskKind | str) -> TaskKind:
    if isinstance(value, TaskKind):
        return value
    return TaskKind[str(value).upper()]


def _produces(value: object) -> frozenset[TaskKind]:
    if isinstance(value, (TaskKind, str)):
        return frozenset({_task_kind(value)})
    return frozenset(_task_kind(v) for v in value)  # type: ignore[union-attr]


@attrs.define(eq=False)
class TaskModeSpec:
    """A single entry in the episode-level task_modes list."""

    kind: str
    produces: frozenset[TaskKind] = attrs.field(default=frozenset({TaskKind.GOTO_POSE}), converter=_produces)
    assignments: list[str] = attrs.field(factory=list)
    config: dict = attrs.field(factory=dict)

    @classmethod
    def parse(cls, obj: dict) -> TaskModeSpec:
        if not isinstance(obj, dict) or "kind" not in obj:
            raise ValueError(f"task_modes entry must be a mapping with 'kind', got {obj!r}")
        kind = obj["kind"]
        return cls(
            kind="null" if kind is None else str(kind),
            produces=obj.get("produces") or TaskKind.GOTO_POSE,
            assignments=[str(a) for a in obj.get("assignments") or []],
            config=dict(obj.get("config") or {}),
        )


def load_task_config(path: str | Path) -> list[TaskModeSpec]:
    """Read a task config file into its `task_modes` specs."""
    with open(path) as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict) or not isinstance(obj.get("task_modes"), list):
        raise ValueError(f"task config {path} must contain a top-level 'task_modes' list")
    return [TaskModeSpec.parse(entry) for entry in obj["task_modes"]]


class FleetManager:
    """Greedy allocator pairing TM specs to robot managers."""

    @staticmethod
    def match(
        task_modes: list[TaskModeSpec],
        robots: Iterable[RobotManager],
    ) -> dict[TaskModeSpec, list[RobotManager]]:
        """Allocate each RobotManager to exactly one TM spec."""
        robots_list: list[RobotManager] = list(robots)
        by_name: dict[str, RobotManager] = {r.name: r for r in robots_list}

        allocation: dict[TaskModeSpec, list[RobotManager]] = {spec: [] for spec in task_modes}
        used: set[str] = set()

        for spec in task_modes:
            for name in spec.assignments:
                if name not in by_name:
                    raise KeyError(f"task_mode {spec.kind!r} pins robot {name!r} but no such robot exists; known: {sorted(by_name.keys())}")
                if name in used:
                    raise AssertionError(f"robot {name!r} is pinned to multiple task_modes")
                robot = by_name[name]
                if not spec.produces.issubset(robot.accepts):
                    raise AssertionError(f"task_mode {spec.kind!r} requires task kinds {spec.produces!r} but robot {name!r} accepts only {sorted(k.name for k in robot.accepts)}")
                allocation[spec].append(robot)
                used.add(name)

        unpinned_specs = [spec for spec in task_modes if not spec.assignments]
        for robot in robots_list:
            if robot.name in used:
                continue
            for spec in unpinned_specs:
                if spec.produces.issubset(robot.accepts):
                    allocation[spec].append(robot)
                    used.add(robot.name)
                    break

        null_spec: TaskModeSpec | None = next(
            (spec for spec in task_modes if spec.kind == "null"),
            None,
        )
        if null_spec is not None:
            for robot in robots_list:
                if robot.name in used:
                    continue
                allocation[null_spec].append(robot)
                used.add(robot.name)

        if unallocated := sorted(set(by_name) - used):
            raise AssertionError(f"robots {unallocated} match no task_mode; add a 'null' sink or a spec they can accept")

        return allocation


__all__ = ["TaskModeSpec", "FleetManager", "load_task_config"]
