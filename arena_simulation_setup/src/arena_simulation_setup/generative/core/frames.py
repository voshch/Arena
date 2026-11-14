"""Coordinates without a `level` key between the working level's frame and the flattened map frame."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _moved(value: object, dx: float, dy: float) -> object:
    if isinstance(value, list) and len(value) >= 2 and all(isinstance(v, (int, float)) for v in value[:2]):
        return [round(float(value[0]) + dx, 2), round(float(value[1]) + dy, 2), *value[2:]]
    if isinstance(value, dict) and "level" not in value and isinstance(value.get("x"), (int, float)) and isinstance(value.get("y"), (int, float)):
        return {**value, "x": round(float(value["x"]) + dx, 2), "y": round(float(value["y"]) + dy, 2)}
    return value


def shift_humans(doc: dict[str, Any], dx: float, dy: float) -> None:
    """Move the pose and waypoints of every human without a `level` by (dx, dy) in place."""
    if not dx and not dy:
        return
    for agent in doc.get("humans") or []:
        if not isinstance(agent, dict) or agent.get("level"):
            continue
        if "pose" in agent:
            agent["pose"] = _moved(agent["pose"], dx, dy)
        if isinstance(agent.get("waypoints"), list):
            agent["waypoints"] = [_moved(w, dx, dy) for w in agent["waypoints"]]


def shift_targets(type_doc: Mapping[str, Any], dx: float, dy: float) -> None:
    """Move every `go_to` target pose of a behavior-tree type by (dx, dy) in place."""
    if not dx and not dy:
        return
    sequences = type_doc.get("sequences")
    if not isinstance(sequences, Mapping):
        return
    for sequence in sequences.values():
        steps = sequence.get("steps") if isinstance(sequence, Mapping) else None
        for step in (steps or {}).values():
            if isinstance(step, dict) and step.get("kind") == "go_to" and isinstance(step.get("target_pose"), dict):
                step["target_pose"] = _moved(step["target_pose"], dx, dy)
