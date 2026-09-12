"""Reading agent types, for the one thing the mode still needs from them: a walking speed."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: Fallback pedestrian speed when a type declares none.
DEFAULT_PED_SPEED = 1.1


class AgentTypeError(ValueError):
    """An agent type could not be read."""


def load_raw(agent_type: str) -> dict[str, Any]:
    """An agent type as a self-contained raw dict, with ``extends`` resolved.

    ``agent_type`` is a builtin name (``adult``) or a path to a YAML file. Scenario agents
    commonly use a path (``./types/blind_elder.yaml``); callers resolve it against the
    obstacle's ``included_from`` before calling this.
    """
    from arena_humansim.core.agents.loader import (  # noqa: PLC0415
        _load_default_agent_types_raw,
        load_agent_type_raw_from_file,
        resolve_extends,
    )

    defaults = {name: raw for name, (raw, _src) in _load_default_agent_types_raw().items()}
    candidate = Path(agent_type)
    if candidate.suffix == ".yaml" or candidate.is_file():
        if not candidate.is_file():
            raise AgentTypeError(f"agent type file not found: {agent_type}")
        name, raw, _src = load_agent_type_raw_from_file(candidate)
        if raw.get("extends") is not None:
            raw = resolve_extends({name: raw}, defaults)[name]
        return copy.deepcopy(raw)
    if agent_type not in defaults:
        raise AgentTypeError(f"unknown builtin agent type {agent_type!r}; known: {sorted(defaults)}")
    return copy.deepcopy(defaults[agent_type])


def mean_of(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, Mapping) and "mean" in value:
        try:
            return float(value["mean"])
        except (TypeError, ValueError):
            return None
    return None


def speed_of(raw: Mapping[str, Any]) -> float:
    """Nominal walking speed declared by an agent type."""
    speed = mean_of(raw.get("desired_velocity"))
    return DEFAULT_PED_SPEED if speed is None or speed <= 0 else speed
