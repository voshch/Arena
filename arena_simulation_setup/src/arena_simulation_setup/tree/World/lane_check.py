"""Lane checks of a format 2 scenario: a source region's lane block resolved on a world's walls."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import cast

import numpy as np
from arena_humansim.params import GLOBAL_PLANNER_INFLATION_RADIUS, GLOBAL_PLANNER_RESOLUTION
from arena_humansim.utils.lanes import build_lane, build_track, resolve_clearance, resolve_lane, sigma_meters, with_goal_corners
from arena_humansim.utils.types import (
    AgentTemplate,
    Lane,
    LaneConfig,
    LaneSide,
    Pose2D,
    Segments,
    Shape,
    ShapeType,
    SinkAffinity,
    SinkConfig,
    SourceConfig,
)

from arena_simulation_setup.shared.lane import LaneBlock
from arena_simulation_setup.shared.navmesh import cabin_walls
from arena_simulation_setup.tree.World.model import World, Zone

PLANNER_CLEARANCE = resolve_clearance(
    cast("float", GLOBAL_PLANNER_INFLATION_RADIUS.default),
    cast("float", GLOBAL_PLANNER_RESOLUTION.default),
)


def lane_config_from_block(flow: str, block: Mapping[str, object]) -> LaneConfig:
    try:
        parsed = LaneBlock.parse(block)
    except ValueError as exc:
        raise ValueError(f"flow {flow!r}: {exc}") from None
    defaults = LaneConfig(centerlines=[], side=LaneSide(parsed.side))
    return LaneConfig(
        centerlines=[[Pose2D(x=x, y=y) for x, y in segment.centerline] for segment in parsed.segments],
        side=defaults.side,
        two_way=parsed.two_way,
        offset=parsed.offset,
        sigma_q=defaults.sigma_q if parsed.sigma_q is None else parsed.sigma_q,
        chord_tolerance=defaults.chord_tolerance if parsed.chord_tolerance is None else parsed.chord_tolerance,
        track_spacing=parsed.track_spacing,
        arrival_radius=parsed.arrival_radius,
        merge_length=parsed.merge_length,
        spawn_length=parsed.spawn_length,
        sweep_length=parsed.sweep_length,
    )


def lane_summary(flow: str, block: Mapping[str, object], walls: Segments, *, agent_radius: float, source_extent: float) -> dict[str, object]:
    """Derived settings, band and track points of a lane block on the walls, or the reason the lane fails."""
    try:
        config = lane_config_from_block(flow, block)
        lane = build_lane(config, walls, flow=flow, agent_radius=agent_radius, clearance=PLANNER_CLEARANCE, source_extent=source_extent)
    except ValueError as exc:
        return {"valid": False, "reason": str(exc)}
    return _summary_of(lane)


def _summary_of(lane: Lane) -> dict[str, object]:
    settings = lane.settings
    half = lane.half_width
    return {
        "valid": True,
        "settings": {
            "track_spacing": settings.track_spacing,
            "arrival_radius": settings.arrival_radius,
            "spawn_length": settings.spawn_length,
            "sweep_length": settings.sweep_length,
            "merge_length": settings.merge_length,
            "separation": settings.separation,
            "clearance": settings.clearance,
            "sigma_q": settings.sigma_q,
            "chord_tolerance": settings.chord_tolerance,
        },
        "band": {"min": float(half.min()), "median": float(np.median(half)), "max": float(half.max())},
        "sigma_m_median": sigma_meters(lane),
        "track_points": len(build_track(lane, 0.0, 0.0).points),
    }


def scenario_lane_results(regions: object, world: World) -> dict[str, dict[str, object]]:
    """Lane summary of each source region of a scenario that has a lane block, keyed by region."""
    lanes, failures = build_scenario_lanes(regions, world)
    return {name: _summary_of(lanes[name]) if name in lanes else {"valid": False, "reason": failures[name]} for name, _, _ in _source_lane_blocks(regions)}


def build_scenario_lanes(regions: object, world: World) -> tuple[dict[str, Lane], dict[str, str]]:
    """Lanes the engine builds for a scenario's source regions with a lane block, and the reason per region that fails."""
    zones = {zone.name: (zone, level) for level in world.all_levels for zone in level.zones}
    sinks = _sink_configs(regions, {name: zone for name, (zone, _) in zones.items()})
    built: dict[str, Lane] = {}
    failures: dict[str, str] = {}
    for name, block, ref in _source_lane_blocks(regions):
        found = zones.get(ref)
        if found is None:
            failures[name] = f"{name}: no zone {ref!r} in the world"
            continue
        zone, level = found
        try:
            source = _source_config(name, regions, block, zone)
            walls = [*level.closed_wall_segments(), *(((wall.start.x, wall.start.y), (wall.end.x, wall.end.y)) for elevator in level.all_elevators for wall in cabin_walls(elevator))]
            built[name] = resolve_lane(source, walls, sinks, PLANNER_CLEARANCE)
        except ValueError as exc:
            failures[name] = str(exc)
    narrowed = {name: with_goal_corners(lane, [other for other_name, other in built.items() if other_name != name]) for name, lane in built.items()}
    return narrowed, failures


def scenario_lane_problems(regions: object, world: World) -> list[str]:
    """Lane problems of a scenario's source regions, each named by its region."""
    return [str(result["reason"]) for result in scenario_lane_results(regions, world).values() if result["valid"] is not True]


def _source_lane_blocks(regions: object) -> Iterator[tuple[str, Mapping[str, object], str]]:
    """Name, lane block and zone reference of each source region that has a lane block."""
    if not isinstance(regions, Mapping):
        return
    for name, region in regions.items():
        if not isinstance(region, Mapping) or region.get("type") != "source":
            continue
        config = region.get("config")
        block = config.get("lane") if isinstance(config, Mapping) else None
        if isinstance(block, Mapping):
            yield str(name), block, str(region.get("ref"))


def _zone_shape(zone: Zone) -> tuple[Pose2D, Shape]:
    """Pose at the zone centroid and the polygon relative to it, as the runtime sends a region."""
    cx = sum(point.x for point in zone.polygon) / len(zone.polygon)
    cy = sum(point.y for point in zone.polygon) / len(zone.polygon)
    vertices = [Pose2D(x=point.x - cx, y=point.y - cy) for point in zone.polygon]
    return Pose2D(x=cx, y=cy), Shape(type=ShapeType.POLYGON, vertices=vertices)


def _region_config(region: object) -> Mapping[str, object]:
    config = region.get("config") if isinstance(region, Mapping) else None
    return config if isinstance(config, Mapping) else {}


def _sink_configs(regions: object, zones: Mapping[str, Zone]) -> dict[str, SinkConfig]:
    sinks: dict[str, SinkConfig] = {}
    if not isinstance(regions, Mapping):
        return sinks
    for name, region in regions.items():
        if not isinstance(region, Mapping) or region.get("type") != "sink":
            continue
        zone = zones.get(str(region.get("ref")))
        if zone is None:
            continue
        pose, shape = _zone_shape(zone)
        config = _region_config(region)
        if "absorption_radius" in config:
            sinks[str(name)] = SinkConfig(name=str(name), pose=pose, shape=shape, absorption_radius=_number(str(name), "absorption_radius", config["absorption_radius"]))
        else:
            sinks[str(name)] = SinkConfig(name=str(name), pose=pose, shape=shape)
    return sinks


def _source_config(name: str, regions: object, block: Mapping[str, object], zone: Zone) -> SourceConfig:
    region = regions.get(name) if isinstance(regions, Mapping) else None
    agent = _region_config(region).get("agent")
    agent = agent if isinstance(agent, Mapping) else {}
    entries = agent.get("sink_affinity")
    affinity = [SinkAffinity(sink_name=str(entry.get("sink", "")), weight=_number(name, "weight", entry.get("weight", 1.0))) for entry in (entries if isinstance(entries, Sequence) and not isinstance(entries, str) else []) if isinstance(entry, Mapping)]
    pose, shape = _zone_shape(zone)
    return SourceConfig(
        name=name,
        pose=pose,
        shape=shape,
        agent=AgentTemplate(agent_radius=_number(name, "agent_radius", agent.get("agent_radius", 0.0)), sink_affinity=affinity),
        lane_config=lane_config_from_block(name, block),
    )


def _number(flow: str, key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"flow {flow!r}: {key} must be a number")
    return float(value)
