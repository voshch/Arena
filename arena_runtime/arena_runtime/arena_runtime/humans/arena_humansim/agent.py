from pathlib import Path

import attrs
import numpy as np
from arena_humansim.core.agents import BUILTIN_AGENTS, AgentType, SampledParams, StepDef, sample_agent_type
from arena_humansim.core.agents.loader import load_agent_type_from_file
from arena_humansim_msgs.msg import AgentTemplate as AgentTemplateMsg
from arena_humansim_msgs.msg import SinkAffinity as SinkAffinityMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg

from arena_runtime.shared import DynamicObstacle


def _is_path_agent_type(name: str) -> bool:
    return "/" in name or name.endswith(".yaml")


def resolve_agent_type_path(agent_type: str, included_from: Path | None) -> str:
    if included_from is not None and _is_path_agent_type(agent_type) and not Path(agent_type).is_absolute():
        return str((Path(included_from) / agent_type).resolve())
    return agent_type


def agent_type_def(agent_type: str) -> AgentType | None:
    """The agent type behind a yaml path or a builtin name, None when neither resolves."""
    if _is_path_agent_type(agent_type):
        path = Path(agent_type)
        return load_agent_type_from_file(path) if path.is_file() else None
    return BUILTIN_AGENTS.get(agent_type)


def agent_type_stimuli(definition: AgentType) -> frozenset[str]:
    """Stimulus names an agent type acts on: its needs and the events its autonomous steps run `until`."""
    events = {step.until for sequence in definition.sequences.values() for step in sequence.steps.values() if isinstance(step, StepDef) and step.until}
    return frozenset(definition.needs) | frozenset(events)


def path_agent_type_name(agent_type: str) -> str | None:
    """The name the engine reports for a path-based agent type, None for a builtin name or a missing file."""
    if not _is_path_agent_type(agent_type):
        return None
    definition = agent_type_def(agent_type)
    return definition.name if definition is not None else None


def flow_agent_template(raw: dict, included_from: Path | None) -> AgentTemplateMsg:
    """AddSource agent template of a source region's `agent:` block, speed and radius left 0 (the type's sample) when unset."""
    tmpl = AgentTemplateMsg()
    vel = raw.get("desired_velocity")
    if isinstance(vel, dict):
        tmpl.desired_velocity_min = float(vel.get("min", 1.0))
        tmpl.desired_velocity_max = float(vel.get("max", 1.5))
    elif vel is not None:
        tmpl.desired_velocity_min = tmpl.desired_velocity_max = float(vel)
    tmpl.agent_radius = float(raw.get("agent_radius", 0.0))
    tmpl.behavior_tree = raw.get("behavior_tree", "default")
    tmpl.agent_type = resolve_agent_type_path(raw.get("agent_type", "adult"), included_from)
    for sa in raw.get("sink_affinity", []):
        tmpl.sink_affinity.append(SinkAffinityMsg(sink_name=sa.get("sink", ""), weight=sa.get("weight", 1.0)))  # pyright: ignore[reportAttributeAccessIssue]
    return tmpl


_WAYPOINT_MODE_MAP = {
    "repeat": WaypointsMsg.MODE_REPEAT,
    "reverse": WaypointsMsg.MODE_REVERSE,
    "once": WaypointsMsg.MODE_ONCE,
    "random": WaypointsMsg.MODE_RANDOM,
}


@attrs.define()
class ArenaHumanDynamicObstacle:
    agent_type: str = "adult"
    desired_velocity_min: float | None = None
    desired_velocity_max: float | None = None
    agent_radius: float | None = None
    waypoint_mode: int = WaypointsMsg.MODE_REPEAT
    _waypoints: list = attrs.field(factory=list, alias="waypoints")

    @classmethod
    def from_dynamic_obstacle(cls, obs: DynamicObstacle) -> "ArenaHumanDynamicObstacle | None":
        raw = obs.agent
        if raw is None:
            return None
        if isinstance(raw, str):
            return cls(
                agent_type=cls._resolve_agent_type_path(raw, obs),
                waypoints=list(obs.waypoints),
                waypoint_mode=cls._parse_waypoint_mode(obs.waypoint_mode),
            )
        if not isinstance(raw, dict):
            return None

        vel = raw.get("desired_velocity")
        if isinstance(vel, dict):
            vel_min = float(vel.get("min", 1.0))
            vel_max = float(vel.get("max", 1.5))
        elif vel is not None:
            vel_min = vel_max = float(vel)
        else:
            vel_min = vel_max = None
        radius = raw.get("radius")

        agent_type = raw.get("agent_type", "adult")
        agent_type = cls._resolve_agent_type_path(agent_type, obs)

        return cls(
            agent_type=agent_type,
            desired_velocity_min=vel_min,
            desired_velocity_max=vel_max,
            agent_radius=None if radius is None else float(radius),
            waypoints=list(obs.waypoints),
            waypoint_mode=cls._parse_waypoint_mode(obs.waypoint_mode),
        )

    @staticmethod
    def _resolve_agent_type_path(agent_type: str, obs: DynamicObstacle) -> str:
        return resolve_agent_type_path(agent_type, obs.included_from)

    @staticmethod
    def _parse_waypoint_mode(raw: str) -> int:
        return _WAYPOINT_MODE_MAP.get(str(raw).lower(), WaypointsMsg.MODE_REPEAT)

    def sample_params(self, rng: np.random.Generator) -> SampledParams | None:
        definition = agent_type_def(self.agent_type)
        if definition is None:
            return None

        params = sample_agent_type(definition, rng)

        # Apply velocity/radius overrides
        if self.desired_velocity_min is not None and self.desired_velocity_max is not None:
            params = attrs.evolve(params, desired_velocity=float(rng.uniform(self.desired_velocity_min, self.desired_velocity_max)))
        if self.agent_radius is not None:
            params = attrs.evolve(params, agent_radius=self.agent_radius)
        return params
