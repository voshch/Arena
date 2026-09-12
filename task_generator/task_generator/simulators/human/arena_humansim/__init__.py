from pathlib import Path

import attrs
import numpy as np
from arena_humansim.core.agents import BUILTIN_AGENTS, SampledParams, sample_agent_type
from arena_humansim.core.agents.loader import load_agent_type_from_file
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg

from task_generator.shared import DynamicObstacle


def _is_path_agent_type(name: str) -> bool:
    return "/" in name or name.endswith(".yaml")


def resolve_agent_type_path(agent_type: str, included_from: Path | None) -> str:
    if included_from is not None and _is_path_agent_type(agent_type) and not Path(agent_type).is_absolute():
        return str((Path(included_from) / agent_type).resolve())
    return agent_type


_WAYPOINT_MODE_MAP = {
    "repeat": WaypointsMsg.MODE_REPEAT,
    "reverse": WaypointsMsg.MODE_REVERSE,
    "once": WaypointsMsg.MODE_ONCE,
    "random": WaypointsMsg.MODE_RANDOM,
}


@attrs.define()
class ArenaHumanDynamicObstacle:
    agent_type: str = "adult"
    desired_velocity_min: float = 1.0
    desired_velocity_max: float = 1.5
    agent_radius: float = 0.35
    waypoint_mode: int = WaypointsMsg.MODE_REPEAT
    _waypoints: list = attrs.Factory(list)

    @classmethod
    def from_dynamic_obstacle(cls, obs: DynamicObstacle) -> "ArenaHumanDynamicObstacle | None":
        extra = getattr(obs, "extra", None) or {}
        raw = extra.get("agent")
        if raw is None:
            return None
        if isinstance(raw, str):
            return cls(
                agent_type=cls._resolve_agent_type_path(raw, obs),
                waypoints=list(getattr(obs, "waypoints", [])),
                waypoint_mode=cls._parse_waypoint_mode(extra),
            )
        if not isinstance(raw, dict):
            return None

        vel = raw.get("desired_velocity", {})
        if isinstance(vel, dict):
            vel_min = vel.get("min", 1.0)
            vel_max = vel.get("max", 1.5)
        elif vel is not None:
            vel_min = vel_max = float(vel)
        else:
            vel_min, vel_max = 1.0, 1.5

        agent_type = raw.get("agent_type", "adult")
        agent_type = cls._resolve_agent_type_path(agent_type, obs)

        return cls(
            agent_type=agent_type,
            desired_velocity_min=float(vel_min),
            desired_velocity_max=float(vel_max),
            agent_radius=float(raw.get("radius", 0.35)),
            waypoints=list(getattr(obs, "waypoints", [])),
            waypoint_mode=cls._parse_waypoint_mode(extra),
        )

    @staticmethod
    def _resolve_agent_type_path(agent_type: str, obs: DynamicObstacle) -> str:
        return resolve_agent_type_path(agent_type, getattr(obs, "included_from", None))

    @staticmethod
    def _parse_waypoint_mode(extra: dict) -> int:
        raw = extra.get("waypoint_mode", "repeat")
        return _WAYPOINT_MODE_MAP.get(str(raw).lower(), WaypointsMsg.MODE_REPEAT)

    def sample_params(self, rng: np.random.Generator) -> SampledParams | None:
        if _is_path_agent_type(self.agent_type):
            p = Path(self.agent_type)
            if p.is_file():
                agent_type_def = load_agent_type_from_file(p)
            else:
                return None
        else:
            agent_type_def = BUILTIN_AGENTS.get(self.agent_type)
        if agent_type_def is None:
            return None

        params = sample_agent_type(agent_type_def, rng)

        # Apply velocity/radius overrides
        desired_velocity = float(rng.uniform(self.desired_velocity_min, self.desired_velocity_max))
        return attrs.evolve(
            params,
            desired_velocity=desired_velocity,
            agent_radius=self.agent_radius,
        )
