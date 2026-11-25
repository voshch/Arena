import functools
import itertools
import os
import typing
import warnings
from collections.abc import Iterable, Iterator

import attrs
import yaml

from arena_simulation_setup.shared import DynamicObstacle, EpisodeCondition, Light, Obstacle, Pose, Position, Sound, TaskPhase
from arena_simulation_setup.shared.conditions import Node, check_fields, parse_node
from arena_simulation_setup.shared.task import LoopPhase
from arena_simulation_setup.tree import Identifier, PathView
from arena_simulation_setup.tree.assets.Animation import AnimationIdentifier, referenced_clips
from arena_simulation_setup.utils.articulation import Articulation
from arena_simulation_setup.utils.cattrs import ArenaConverter, Parseable, Serializable, converter

ROBOT_KEYS = frozenset({"name", "model", "start", "level", "goal_level", "phases", "conditions"})


@attrs.define
class RobotGoal(Parseable):
    start: Pose = attrs.field(converter=Pose.converter)
    name: str = attrs.field(default="", kw_only=True)
    model: str = attrs.field(default="", kw_only=True)
    level: str = attrs.field(default="")
    start_zone: str = attrs.field(default="")
    goal_level: str = attrs.field(default="")
    phases: list[TaskPhase] = attrs.field(factory=list)
    conditions: list[EpisodeCondition] = attrs.field(factory=list)

    @classmethod
    def parse(cls, obj: dict) -> "RobotGoal":
        if "start" not in obj:
            raise ValueError("robot requires 'start'")
        unknown = sorted(set(obj) - ROBOT_KEYS)
        if unknown:
            raise ValueError(f"robot has unknown keys {unknown}, expected some of {sorted(ROBOT_KEYS)}")
        return cls(
            name=str(obj.get("name", "")),
            model=str(obj.get("model", "")),
            level=str(obj.get("level", "")),
            goal_level=str(obj.get("goal_level", "")),
            start=Pose.parse(obj["start"]),
            start_zone=obj["start"] if isinstance(obj["start"], str) else "",
            phases=[TaskPhase.parse(p) for p in obj.get("phases") or []],
            conditions=[EpisodeCondition.parse(c) for c in obj.get("conditions", [])],
        )

    def phase_list(self) -> list[TaskPhase]:
        return list(self.phases)


@attrs.define
class RegionAssignment:
    type: str = ""  # "source" | "sink"
    polygon: list[Position] = attrs.field(factory=list)  # zone corners (resolved from ref by zone_converter)
    config: dict = attrs.Factory(dict)  # type-specific params


STIMULUS_KEYS = frozenset({"agents", "need", "intensity"})


@attrs.frozen
class StimulusAction(Serializable):
    """Raise `need` by `intensity` on each named pedestrian agent."""

    agents: tuple[str, ...]
    need: str
    intensity: float = 1.0

    def __attrs_post_init__(self) -> None:
        if not self.agents or not all(isinstance(agent, str) and agent for agent in self.agents):
            raise ValueError(f"stimulus 'agents' must be a non-empty list of agent names, got {self.agents!r}")
        if not isinstance(self.need, str) or not self.need:
            raise ValueError(f"stimulus 'need' must be a non-empty string, got {self.need!r}")

    @classmethod
    def parse(cls, value: object) -> "StimulusAction":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError(f"a stimulus is a mapping, got {value!r}")
        unknown = sorted(set(value) - STIMULUS_KEYS)
        if unknown:
            raise ValueError(f"stimulus has unknown keys {unknown}, expected some of {sorted(STIMULUS_KEYS)}")
        agents = value.get("agents")
        if not isinstance(agents, list | tuple):
            raise ValueError(f"stimulus 'agents' must be a list of agent names, got {agents!r}")
        need = value.get("need")
        if not isinstance(need, str):
            raise ValueError(f"stimulus 'need' must be a non-empty string, got {need!r}")
        return cls(agents=tuple(agents), need=need, intensity=float(value.get("intensity", 1.0)))

    def serialize(self) -> dict:
        result: dict = {"agents": list(self.agents), "need": self.need}
        if self.intensity != 1.0:
            result["intensity"] = self.intensity
        return result


converter.register_structure_hook(StimulusAction, lambda value, _: StimulusAction.parse(value))


@attrs.define
class TimelineEntry(Parseable):
    """One scenario timeline entry: exactly one trigger (at/every/when) plus `set` and `stimulus` actions, at least one."""

    set: list[dict] = attrs.field(factory=list)
    stimulus: list[StimulusAction] = attrs.field(factory=list)
    at: float | None = None
    every: float | None = None
    offset: float = 0.0
    until: float | None = None
    when: dict | str | None = None

    def __attrs_post_init__(self) -> None:
        triggers = [name for name, value in (("at", self.at), ("every", self.every), ("when", self.when)) if value is not None]
        if len(triggers) != 1:
            raise ValueError(f"timeline entry must set exactly one of 'at', 'every', 'when'; got {triggers}")
        if not self.set and not self.stimulus:
            raise ValueError("timeline entry needs at least one action in 'set' or 'stimulus'")
        self.when_node()

    def when_node(self) -> Node | None:
        """The `when` trigger as a condition node, an `{entity, field, is}` mapping read as an `eq` field test."""
        if self.when is None:
            return None
        if isinstance(self.when, dict) and "entity" in self.when:
            stray = sorted(set(self.when) - {"entity", "field", "is"})
            if stray:
                raise ValueError(f"timeline 'when' of the {{entity, field, is}} form takes no keys {stray}")
            return parse_node({"field": f"{self.when.get('entity')}.{self.when.get('field')}", "eq": self.when.get("is")})
        return parse_node(self.when)

    @classmethod
    def parse(cls, value: object) -> "TimelineEntry":
        if isinstance(value, dict) and "stimulus" in value:
            stimulus = value["stimulus"]
            if not isinstance(stimulus, list):
                raise ValueError(f"timeline entry 'stimulus' must be a list, got {stimulus!r}")
            value = {**value, "stimulus": [StimulusAction.parse(item) for item in stimulus]}
        return super().parse(value)


SCENARIO_FORMAT = "arena-scenario/2"
LEGACY_SCENARIO_FORMAT = "arena-scenario/1"
SCENARIO_KEYS = frozenset({"format", "objects", "humans", "sounds", "lights", "robots", "regions", "timeline", "conditions"})


@attrs.define
class Scenario:
    objects: list[Obstacle] = attrs.field(factory=list)
    humans: list[DynamicObstacle] = attrs.field(factory=list)
    sounds: list[Sound] = attrs.field(factory=list)
    lights: list[Light] = attrs.field(factory=list)
    robots: list[RobotGoal] = attrs.field(factory=list)
    regions: dict[str, RegionAssignment] = attrs.field(factory=dict)
    timeline: list[TimelineEntry] = attrs.field(factory=list)
    conditions: list[EpisodeCondition] = attrs.field(factory=list)

    def __attrs_post_init__(self) -> None:
        names: set[str] = set()
        for light in self.lights:
            if light.spec.ambient:
                raise ValueError(f'scenario light {light.name!r} uses fixture {light.fixture!r}, which belongs in the level lights of the world')
            if light.name in names:
                raise ValueError(f'duplicate scenario light {light.name!r}')
            names.add(light.name)
        robot_names: set[str] = set()
        for robot in self.robots:
            if robot.name and robot.name in robot_names:
                raise ValueError(f'duplicate scenario robot {robot.name!r}')
            robot_names.add(robot.name)
        humans = sorted(human.name for human in self.humans)
        for index, entry in enumerate(self.timeline):
            for stimulus in entry.stimulus:
                for agent in stimulus.agents:
                    if agent not in humans:
                        raise ValueError(f'timeline[{index}]: stimulus {stimulus.need!r} names agent {agent!r}, which is not a scenario human, expected one of {humans}')


def _articulations(objects: Iterable[Obstacle]) -> dict[str, Articulation]:
    """Articulation of every object whose model resolves to an articulated object, by object name, a later object of a name replacing an earlier one."""
    found: dict[str, Articulation] = {}
    for obj in objects:
        try:
            articulation = obj.model.resolve_sync().articulation
        except (FileNotFoundError, ValueError):
            articulation = None
        if articulation is not None:
            found[obj.name] = articulation
        else:
            found.pop(obj.name, None)
    return found


def condition_sites(scenario: Scenario) -> Iterator[tuple[str, Node]]:
    """Every condition node of a scenario with the path it was authored at."""

    def phases(prefix: str, items: Iterable[TaskPhase]) -> Iterator[tuple[str, Node]]:
        for index, phase in enumerate(items):
            until = phase.until_node
            if until is not None:
                yield f"{prefix}[{index}].until", until
            for c_index, condition in enumerate(phase.conditions):
                yield from clause(f"{prefix}[{index}].conditions[{c_index}]", condition)
            if isinstance(phase, LoopPhase):
                yield from phases(f"{prefix}[{index}].loop", phase.phases)

    def clause(path: str, condition: EpisodeCondition) -> Iterator[tuple[str, Node]]:
        yield f"{path}.p", condition.p_node
        if condition.q_node is not None:
            yield f"{path}.q", condition.q_node

    for index, condition in enumerate(scenario.conditions):
        yield from clause(f"conditions[{index}]", condition)
    for index, entry in enumerate(scenario.timeline):
        when = entry.when_node()
        if when is not None:
            yield f"timeline[{index}].when", when
    for index, robot in enumerate(scenario.robots):
        yield from phases(f"robots[{index}].phases", robot.phases)
        for c_index, condition in enumerate(robot.conditions):
            yield from clause(f"robots[{index}].conditions[{c_index}]", condition)


def _timeline_value(raw: object) -> str | None:
    """A timeline set value as the runtime applies it, None for a value it draws at fire time."""
    if isinstance(raw, bool):
        return "true" if raw else "false"
    value = str(raw)
    return None if value == "random" or ".." in value else value


def articulation_problems(scenario: Scenario, world_objects: Iterable[Obstacle] = ()) -> list[str]:
    """The `joints`, condition fields and timeline sets that the articulated objects of a scenario and its world, by authored name, do not have or refuse."""
    world_objects = list(world_objects)
    articulations = _articulations([*world_objects, *scenario.objects])
    if not articulations:
        return []
    problems: list[str] = []
    for label, objects in (("objects", scenario.objects), ("world objects", world_objects)):
        for index, obj in enumerate(objects):
            articulation = articulations.get(obj.name)
            if articulation is None or obj.joints is None:
                continue
            try:
                articulation.resolve(obj.joints)
            except ValueError as exc:
                problems.append(f"{label}[{index}] {obj.name!r} joints: {exc}")
    names = {name: frozenset(articulation.field_names()) for name, articulation in articulations.items()}
    problems.extend(f"{path}: {problem}" for path, node in condition_sites(scenario) for problem in check_fields(node, names.get))
    for index, entry in enumerate(scenario.timeline):
        for s_index, action in enumerate(entry.set):
            articulation = articulations.get(str(action.get("entity", "")))
            if articulation is None:
                continue
            value = _timeline_value(action.get("value"))
            if value is None:
                continue
            try:
                articulation.target_for(str(action.get("field", "")), value, articulation.initial())
            except ValueError as exc:
                problems.append(f"timeline[{index}].set[{s_index}]: {exc}")
    return problems


def parse_scenario(raw: object, c: ArenaConverter = converter) -> Scenario:
    """A Scenario from the mapping of a format 2 scenario.yaml."""
    if not isinstance(raw, dict):
        raise ValueError(f"a scenario is a mapping, got {raw!r}")
    if raw.get("format") != SCENARIO_FORMAT:
        raise ValueError(f"format: expected {SCENARIO_FORMAT!r}, got {raw.get('format')!r}")
    unknown = sorted(set(raw) - SCENARIO_KEYS)
    if unknown:
        raise ValueError(f"unknown keys {unknown}, expected some of {sorted(SCENARIO_KEYS)}")
    fields: dict = {}
    for key, kind in (("objects", Obstacle), ("humans", DynamicObstacle), ("sounds", Sound), ("lights", Light), ("timeline", TimelineEntry), ("conditions", EpisodeCondition)):
        items = raw.get(key) or []
        parsed = []
        for index, item in enumerate(items):
            try:
                parsed.append(c.structure(item, kind))
            except Exception as exc:
                raise ValueError(f"{key}[{index}]: {exc}") from exc
        fields[key] = parsed
    robots = []
    for index, item in enumerate(raw.get("robots") or []):
        try:
            robots.append(c.structure(item, RobotGoal))
        except Exception as exc:
            raise ValueError(f"robots[{index}]: {exc}") from exc
    fields["robots"] = robots
    fields["regions"] = c.structure(raw.get("regions") or {}, dict[str, RegionAssignment])
    return Scenario(**fields)


def upgrade_scenario(raw: object) -> tuple[dict, list[str]]:
    """A format 2 scenario mapping and the upgrade warnings for a format 1 scenario mapping."""
    from .legacy import _Upgrade

    if not isinstance(raw, dict):
        raise ValueError(f"a scenario is a mapping, got {raw!r}")
    up = _Upgrade()
    raw = dict(raw)
    raw.pop("format", None)
    wrapper = raw.pop("obstacles", None) or {}
    statics = [*(raw.pop("static", None) or []), *(wrapper.get("static") or []), *(wrapper.get("interactive") or [])]
    dynamics = [*(raw.pop("dynamic", None) or []), *(wrapper.get("dynamic") or [])]
    out: dict = {"format": SCENARIO_FORMAT}
    objects = []
    for index, item in enumerate(statics):
        level = item.get("level_id")
        obj = up.obj(f"static[{index}]", item)
        if level:
            obj["level"] = str(level)
        objects.append(obj)
    if objects:
        out["objects"] = objects
    humans = [up.human(item, keep_level=True) for item in dynamics]
    if humans:
        out["humans"] = humans
    if raw.get("sounds"):
        out["sounds"] = [up.sound(f"sounds[{index}]", item) for index, item in enumerate(raw.pop("sounds"))]
    if raw.get("lights"):
        out["lights"] = [up.light(item) for item in raw.pop("lights")]
    robots = []
    for item in raw.pop("robots", None) or []:
        robot = dict(item)
        if "start_floor" in robot:
            robot["level"] = robot.pop("start_floor")
        if "goal_floor" in robot:
            robot["goal_level"] = robot.pop("goal_floor")
        goal = robot.pop("goal", None)
        via = robot.pop("via", None)
        if goal is not None and not robot.get("phases"):
            robot["phases"] = [{"goto": goal, **({"via": via} if via else {})}]
        elif via:
            up.warn(f"robot via {via!r} applies only to a robot goal without phases, dropped")
        for key in ("level", "goal_level"):
            if key in robot and not robot[key]:
                robot.pop(key)
        robots.append(robot)
    if robots:
        out["robots"] = robots
    for key in ("regions", "timeline", "conditions"):
        if raw.get(key):
            out[key] = raw.pop(key)
    for key in ("sounds", "lights", "regions", "timeline", "conditions"):
        raw.pop(key, None)
    if raw:
        up.warn(f"dropped scenario keys {sorted(raw)} that format 2 does not know")
    return out, up.warnings


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

    def read(self) -> dict:
        """The format 2 mapping of the scenario on disk, format 1 upgraded."""
        with open(self.scenario_path) as f:
            raw = yaml.safe_load(f)
        if isinstance(raw, dict) and "format" in raw:
            return raw
        upgraded, _ = upgrade_scenario(raw)
        warnings.warn(f"scenario {self.scenario_path} is in format 1, migrate it with touch_world --migrate", DeprecationWarning, stacklevel=2)
        return upgraded

    def migrate(self) -> list[str]:
        """Rewrite a format 1 scenario in place as format 2 `scenario.yaml`. Returns the upgrade warnings, empty for a format 2 one."""
        from .write import to_yaml

        with open(self.scenario_path) as f:
            raw = yaml.safe_load(f)
        if isinstance(raw, dict) and "format" in raw:
            return []
        upgraded, notes = upgrade_scenario(raw)
        target = os.path.join(self.path, "scenario.yaml")
        with open(target, "w") as f:
            f.write(to_yaml(upgraded))
        if self.scenario_path != target:
            os.remove(self.scenario_path)
        return notes

    def identifiers(self, converter: ArenaConverter = converter) -> Iterable[Identifier]:
        """Every asset this scenario references, sounds and animation clips included, under the given zone converter."""
        from arena_simulation_setup.tree.assets.Sound import SoundIdentifier

        scenario = self.load(converter=converter)
        for obstacle in itertools.chain(scenario.objects, scenario.humans):
            yield obstacle.model
        for sound in scenario.sounds:
            yield SoundIdentifier.parse(sound.asset_id)
        for obstacle in scenario.humans:
            for name in self.agent_clips(obstacle.agent):
                yield AnimationIdentifier.parse(name)

    def agent_clips(self, agent: object) -> Iterator[str]:
        """Clip names an agent entry plays, from its inline config and the scenario-local agent file it names."""
        if not isinstance(agent, dict):
            return
        yield from referenced_clips(agent)
        agent_type = agent.get('agent_type')
        if isinstance(agent_type, str) and ('/' in agent_type or agent_type.endswith('.yaml')):
            with open(self.path / agent_type) as f:
                yield from referenced_clips(yaml.safe_load(f))

    def load(self, converter: ArenaConverter = converter) -> Scenario:
        try:
            scenario = parse_scenario(self.read(), converter)
        except Exception as exc:
            raise RuntimeError(f"Scenario {self.scenario_path}: {exc}") from exc
        for obj in itertools.chain(scenario.objects, scenario.humans):
            obj.included_from = self.path
        return scenario
