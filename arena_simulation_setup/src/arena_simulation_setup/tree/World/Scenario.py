import abc
import functools
import itertools
import os
import traceback
import typing
import warnings
from collections.abc import Iterable

import attrs
import yaml

from arena_simulation_setup.shared import DynamicObstacle, EpisodeCondition, Obstacle, Pose, Position, Sound
from arena_simulation_setup.tree import Identifier, PathView
from arena_simulation_setup.utils.cattrs import ArenaConverter, Parseable, converter


@attrs.define
class ScenarioGotoPhase:
    goto: Pose = attrs.field(converter=Pose.converter)


@attrs.define
class ScenarioGesturePhase:
    gesture: str
    instance: str = ""  # arm cap instance (mount name), empty = sole arm


class ScenarioPhase(abc.ABC):
    """Discriminated union of scenario phases. Dispatch on key presence."""

    @classmethod
    @abc.abstractmethod
    def parse(cls, value: dict) -> "ScenarioPhase":
        if "goto" in value:
            return ScenarioGotoPhase(goto=Pose.parse(value["goto"]))
        if "gesture" in value:
            return ScenarioGesturePhase(gesture=str(value["gesture"]), instance=str(value.get("instance", "")))
        raise ValueError(f"ScenarioPhase requires 'goto' or 'gesture' key; got {list(value.keys())}")


converter.register_structure_hook(ScenarioPhase, lambda v, _t: ScenarioPhase.parse(v))


@attrs.define
class RobotGoal(Parseable):
    start: Pose = attrs.field(converter=Pose.converter)
    start_floor: str = attrs.field(default="")
    goal_floor: str = attrs.field(default="")
    goal: Pose | None = attrs.field(default=None)
    phases: list[ScenarioPhase] | None = attrs.field(default=None)

    @classmethod
    def parse(cls, obj: dict) -> "RobotGoal":
        if "start" not in obj:
            raise ValueError("RobotGoal requires 'start' field")

        raw_phases = obj.get("phases")
        phases: list[ScenarioPhase] | None = None
        if raw_phases is not None:
            phases = [ScenarioPhase.parse(p) for p in raw_phases]

        raw_goal = obj.get("goal")
        goal: Pose | None = Pose.parse(raw_goal) if raw_goal is not None else None

        return cls(
            start_floor=obj.get("start_floor", ""),
            goal_floor=obj.get("goal_floor", ""),
            start=Pose.parse(obj["start"]),
            goal=goal,
            phases=phases,
        )

    def phase_list(self) -> list[ScenarioPhase]:
        if self.phases is not None and len(self.phases) > 0:
            return self.phases
        if self.goal is not None:
            warnings.warn(
                "scenario.yaml: 'goal:' on a robot is deprecated; use 'phases: [{goto: ...}]'",
                DeprecationWarning,
                stacklevel=2,
            )
            return [ScenarioGotoPhase(goto=self.goal)]
        return []


@attrs.define
class RegionAssignment:
    type: str = ""  # "source" | "sink"
    polygon: list[Position] = attrs.field(factory=list)  # zone corners (resolved from ref by zone_converter)
    config: dict = attrs.Factory(dict)  # type-specific params


@attrs.define
class TimelineEntry(Parseable):
    """One scenario timeline entry: exactly one trigger (at/every/when) plus a set action list."""

    set: list[dict] = attrs.field(factory=list)
    at: float | None = None
    every: float | None = None
    offset: float = 0.0
    until: float | None = None
    when: dict | None = None

    def __attrs_post_init__(self) -> None:
        triggers = [name for name, value in (("at", self.at), ("every", self.every), ("when", self.when)) if value is not None]
        if len(triggers) != 1:
            raise ValueError(f"timeline entry must set exactly one of 'at', 'every', 'when'; got {triggers}")


_MODERN_ONLY_KEYS: frozenset[str] = frozenset({"conditions", "timeline", "regions"})


@attrs.define
class Scenario:
    static: list[Obstacle] = attrs.field(factory=list)
    dynamic: list[DynamicObstacle] = attrs.field(factory=list)
    sounds: list[Sound] = attrs.field(factory=list)
    robots: list[RobotGoal] = attrs.field(factory=list)
    regions: dict[str, RegionAssignment] = attrs.field(factory=dict)
    timeline: list[TimelineEntry] = attrs.field(factory=list)
    conditions: list[EpisodeCondition] = attrs.field(factory=list)


class ScenarioView(PathView):
    _names: typing.ClassVar[Iterable[str]] = [
        "scenario.yaml",
        "scenario.json",
    ]

    @functools.cached_property
    def scenario_path(self) -> str:
        """
        Get the path to the scenario file.
        """
        prefix = functools.partial(os.path.join, self.path)
        scenario = next((p for p in map(prefix, self._names) if os.path.isfile(p)), prefix(next(iter(self._names))))
        return scenario

    def load_legacy(self) -> Scenario:
        with open(self.scenario_path) as f:
            scenario = yaml.safe_load(f)

        if not isinstance(scenario, dict):
            raise ValueError(f"Scenario file {self.scenario_path} must contain a dictionary at the top level.")

        return Scenario(
            static=[converter.structure({**obs, **dict(included_from=self.path)}, Obstacle) for obs in itertools.chain(scenario.get("obstacles", {}).get("static", []), scenario.get("obstacles", {}).get("interactive", []))],
            dynamic=[converter.structure({**obs, **dict(included_from=self.path)}, DynamicObstacle) for obs in scenario.get("obstacles", {}).get("dynamic", [])],
            robots=[RobotGoal.parse(robot) for robot in scenario.get("robots", [])],
        )

    def identifiers(self, converter: ArenaConverter = converter) -> Iterable[Identifier]:
        """Every asset this scenario references. Pass the world's zone converter to read a
        scenario that addresses zones by name."""
        scenario = self.load(converter=converter)
        for obstacle in itertools.chain(scenario.static, scenario.dynamic):
            yield obstacle.model

    def load(self, converter: ArenaConverter = converter) -> Scenario:
        raw: object = None
        load_exc: Exception
        try:
            with open(self.scenario_path) as f:
                raw = yaml.safe_load(f)
            scenario = converter.structure(raw, Scenario)
            for obj in itertools.chain(scenario.static, scenario.dynamic):
                obj.included_from = self.path
            return scenario
        except Exception as e:
            load_exc = e

        if isinstance(raw, dict) and _MODERN_ONLY_KEYS.intersection(raw):
            raise RuntimeError(f"Scenario {self.scenario_path}: modern-format scenario failed to parse, refusing to degrade to legacy. Underlying error:\n{''.join(traceback.format_exception(type(load_exc), load_exc, load_exc.__traceback__))}") from load_exc

        legacy_exc: Exception
        try:
            scenario = self.load_legacy()
            warnings.warn(
                f"Scenario {self.scenario_path}: new-format load failed, falling back to legacy. Underlying error:\n{''.join(traceback.format_exception(type(load_exc), load_exc, load_exc.__traceback__))}",
                UserWarning,
                stacklevel=2,
            )
            warnings.warn("Loading Scenario in legacy format.", DeprecationWarning, stacklevel=2)
            return scenario
        except Exception as e:
            legacy_exc = e

        raise RuntimeError(
            f"Failed to load scenario from {self.scenario_path}:\n - New format error: {load_exc}\n{''.join(traceback.format_exception(type(load_exc), load_exc, load_exc.__traceback__))}\n - Legacy format error: {legacy_exc}\n{''.join(traceback.format_exception(type(legacy_exc), legacy_exc, legacy_exc.__traceback__))}"
        )
