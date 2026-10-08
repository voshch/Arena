"""Driven joints: wire joints a bundle's rig.yaml declares, turned by the ped's motion instead of its gait phase."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import attrs
import yaml

RIG_FILE = "rig.yaml"


@attrs.frozen
class Roller:
    """A wheel `lateral` meters left of the root that rolls without slipping on `radius`."""

    joint: str
    radius: float
    lateral: float


def parse_drivers(rig: object, source: str) -> tuple[Roller, ...]:
    """Drivers of a parsed rig.yaml document, in file order."""
    if not isinstance(rig, dict):
        raise ValueError(f"{source}: a rig file is a mapping, got {type(rig).__name__}")
    section = rig.get("drivers", {})
    if not isinstance(section, dict):
        raise ValueError(f"{source}: drivers takes {{<wire joint>: {{from: distance, radius: <m>, lateral: <m>}}}}, got {section!r}")
    rollers: list[Roller] = []
    for joint, spec in section.items():
        if not isinstance(spec, dict) or spec.get("from") != "distance":
            raise ValueError(f"{source}: driver {joint!r} takes {{from: distance, radius: <m>, lateral: <m>}}, got {spec!r}")
        radius = float(spec.get("radius", 0.0))
        if radius <= 0.0:
            raise ValueError(f"{source}: driver {joint!r} needs a radius above 0 m, got {spec.get('radius')!r}")
        rollers.append(Roller(joint=str(joint), radius=radius, lateral=float(spec.get("lateral", 0.0))))
    return tuple(rollers)


def load_drivers(model_uri: str) -> tuple[Roller, ...]:
    """Drivers of the rig.yaml beside an actor SDF, none without the file."""
    if not model_uri:
        return ()
    rig_path = Path(model_uri).with_name(RIG_FILE)
    if not rig_path.is_file():
        return ()
    return parse_drivers(yaml.safe_load(rig_path.read_text()), str(rig_path))


def with_driven(names: Sequence[str], positions: Sequence[float], angles: Mapping[str, float]) -> tuple[list[str], list[float]]:
    """Joint names and positions with the driven angles set, same-named entries replaced."""
    merged = dict(zip(names, positions, strict=True))
    merged.update(angles)
    return list(merged), list(merged.values())


class DrivenJoints:
    """Per-ped driver angles in radians, integrated from successive poses and never wrapped."""

    def __init__(self) -> None:
        self._pose: dict[int, tuple[float, float, float]] = {}
        self._angles: dict[int, dict[str, float]] = {}

    def advance(self, agent_id: int, drivers: Sequence[Roller], x: float, y: float, yaw: float) -> dict[str, float]:
        """Roll every driver over the step from the ped's last pose, return joint name to angle."""
        previous = self._pose.get(agent_id)
        self._pose[agent_id] = (x, y, yaw)
        forward = turn = 0.0
        if previous is not None:
            turn = math.remainder(yaw - previous[2], math.tau)
            heading = previous[2] + turn / 2.0
            forward = (x - previous[0]) * math.cos(heading) + (y - previous[1]) * math.sin(heading)
        angles = self._angles.setdefault(agent_id, {})
        for driver in drivers:
            angles[driver.joint] = angles.get(driver.joint, 0.0) + (forward - turn * driver.lateral) / driver.radius
        return {driver.joint: angles[driver.joint] for driver in drivers}

    def known(self) -> set[int]:
        """Ids with integrated state."""
        return set(self._pose)

    def forget(self, agent_id: int) -> None:
        """Drop a despawned ped's pose and angles."""
        self._pose.pop(agent_id, None)
        self._angles.pop(agent_id, None)
