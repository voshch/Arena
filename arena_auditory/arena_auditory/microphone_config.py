from __future__ import annotations

import math

import attrs
import yaml
from arena_simulation_setup.tree.World import (
    MICROPHONE_PLACEMENT_TOLERANCE_M,
    WorldDescription,
)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"microphone {field} must be a string")
    result = value.strip()
    if not result or ":" in result:
        raise ValueError(f"microphone {field} must be non-empty and contain no ':'")
    return result


@attrs.frozen
class RobotMicrophoneSpec:
    robot: str
    placement: str
    frame: str
    index: int

    @property
    def listener_id(self) -> str:
        return f"microphone:robot:{self.robot}:{self.placement}:{self.index}"

    def resolve_frame(self, robot_frame_prefix: str) -> str:
        prefix = robot_frame_prefix.strip("/")
        frame = self.frame.strip("/")
        if prefix and (frame == prefix or frame.startswith(f"{prefix}/")):
            return frame
        return "/".join(part for part in (prefix, frame) if part)


@attrs.frozen
class WorldMicrophoneSpec:
    listener_id: str
    zone: str
    placement: str
    frame: str
    position: tuple[float, float, float]
    ceiling_height_m: float | None


def parse_robot_microphones(raw: str) -> tuple[RobotMicrophoneSpec, ...]:
    configured = yaml.safe_load(raw) if raw.strip() else []
    if configured is None:
        configured = []
    if not isinstance(configured, list):
        raise ValueError("robot_microphones must be a YAML list")

    specs: list[RobotMicrophoneSpec] = []
    listener_ids: set[str] = set()
    for item in configured:
        if not isinstance(item, dict):
            raise ValueError("each robot microphone must be a mapping")
        owner = _identifier(item.get("owner", "robot"), "owner")
        if owner != "robot":
            raise ValueError("launch-configured microphones must use owner: robot")
        robot = _identifier(item.get("robot"), "robot")
        placement = _identifier(
            item.get("placement"),
            "placement",
        ).lower()
        frame = item.get("frame")
        if not isinstance(frame, str) or not frame.strip().strip("/"):
            raise ValueError(f"robot microphone {robot!r}/{placement!r} requires a TF frame")
        index = item.get("index", 1)
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            raise ValueError("microphone index must be a positive integer")
        spec = RobotMicrophoneSpec(
            robot=robot,
            placement=placement,
            frame=frame.strip(),
            index=index,
        )
        if spec.listener_id in listener_ids:
            raise ValueError(f"duplicate microphone {spec.listener_id!r}")
        listener_ids.add(spec.listener_id)
        specs.append(spec)
    return tuple(specs)


def world_microphones(
    world: WorldDescription,
    default_ceiling_height_m: float,
    level_origins: dict[str, tuple[float, float]] | None = None,
) -> tuple[WorldMicrophoneSpec, ...]:
    levels = world.levels
    specs = []
    for level_id in sorted(levels):
        zones = {zone.name: zone for zone in levels[level_id].zones}
        for microphone in levels[level_id].microphones:
            zone = zones[microphone.zone]
            frame = microphone.frame.strip().strip("/")
            offset = level_origins.get(str(level_id), (0.0, 0.0)) if level_origins is not None and frame == "map" else (0.0, 0.0)
            ceiling_height = None
            if microphone.placement == "ceiling":
                ceiling_height = float(zone.ceiling_height) if zone.ceiling_height is not None else float(default_ceiling_height_m)
                if microphone.frame.strip().strip("/") == "map" and not math.isclose(
                    float(microphone.position.z),
                    ceiling_height,
                    abs_tol=MICROPHONE_PLACEMENT_TOLERANCE_M,
                ):
                    raise ValueError(f"microphone {microphone.listener_id!r} z={microphone.position.z} does not match ceiling height {ceiling_height}")
            specs.append(
                WorldMicrophoneSpec(
                    listener_id=microphone.listener_id,
                    zone=microphone.zone,
                    placement=microphone.placement,
                    frame=frame,
                    position=(
                        float(microphone.position.x) + offset[0],
                        float(microphone.position.y) + offset[1],
                        float(microphone.position.z),
                    ),
                    ceiling_height_m=ceiling_height,
                )
            )
    return tuple(specs)
