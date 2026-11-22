"""Lane checks of a format 2 scenario: a source region's lane block resolved on a world's walls."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import cast

import numpy as np
from arena_humansim.params import GLOBAL_PLANNER_INFLATION_RADIUS, GLOBAL_PLANNER_RESOLUTION
from arena_humansim.utils.lanes import build_lane, build_track, check_endpoints, resolve_clearance, sigma_meters, with_goal_corners
from arena_humansim.utils.types import Lane, LaneConfig, LaneSide, Pose2D, Segments, Shape, ShapeType, SinkConfig, SourceConfig

from arena_simulation_setup.tree.World.model import World, Zone
from arena_simulation_setup.tree.World.pedestrian_walls import wall_segments_of_world

SCENARIO_AGENT_RADIUS = 0.35  # mean of humansim's default agent radius distribution
PLANNER_CLEARANCE = resolve_clearance(
    cast("float", GLOBAL_PLANNER_INFLATION_RADIUS.default),
    cast("float", GLOBAL_PLANNER_RESOLUTION.default),
)


def lane_config_from_block(flow: str, block: Mapping[str, object]) -> LaneConfig:
    side = block.get("side")
    if not isinstance(side, str) or not side:
        raise ValueError(f"flow {flow!r}: a lane needs a side (right, left or center)")
    try:
        lane_side = LaneSide(side)
    except ValueError:
        raise ValueError(f"flow {flow!r}: unknown lane side {side!r}") from None
    segments = block.get("segments")
    if not isinstance(segments, Sequence) or isinstance(segments, str) or not segments:
        raise ValueError(f"flow {flow!r}: a lane needs at least one segment")
    defaults = LaneConfig(centerlines=[], side=lane_side)
    offset = block.get("offset")
    return LaneConfig(
        centerlines=[_centerline(flow, segment) for segment in segments],
        side=lane_side,
        two_way=bool(block.get("two_way", False)),
        offset=None if offset is None else _number(flow, "offset", offset),
        sigma_q=_number(flow, "sigma_q", block.get("sigma_q", defaults.sigma_q)),
        chord_tolerance=_number(flow, "chord_tolerance", block.get("chord_tolerance", defaults.chord_tolerance)),
        track_spacing=_override(flow, block, "track_spacing"),
        arrival_radius=_override(flow, block, "arrival_radius"),
        merge_length=_override(flow, block, "merge_length"),
        spawn_length=_override(flow, block, "spawn_length"),
        sweep_length=_override(flow, block, "sweep_length"),
    )


def lane_summary(flow: str, block: Mapping[str, object], walls: Segments, *, agent_radius: float, source_extent: float) -> dict[str, object]:
    """Derived settings, band and track points of a lane block on the walls, or the reason the lane fails."""
    config = lane_config_from_block(flow, block)
    try:
        lane = _build(flow, config, walls, agent_radius=agent_radius, source_extent=source_extent)
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


def _endpoint_configs(
    name: str,
    regions: object,
    zones: Mapping[str, Zone],
    source_zone: Zone,
) -> tuple[SourceConfig, SinkConfig] | None:
    if not isinstance(regions, Mapping):
        return None
    region = regions.get(name)
    config = region.get("config") if isinstance(region, Mapping) else None
    agent = config.get("agent") if isinstance(config, Mapping) else None
    affinity = agent.get("sink_affinity") if isinstance(agent, Mapping) else None
    if not isinstance(affinity, Sequence) or isinstance(affinity, str) or len(affinity) != 1:
        return None
    entry = affinity[0]
    sink_name = entry.get("sink") if isinstance(entry, Mapping) else None
    sink_region = regions.get(sink_name) if isinstance(sink_name, str) else None
    sink_ref = sink_region.get("ref") if isinstance(sink_region, Mapping) else None
    sink_zone = zones.get(str(sink_ref)) if sink_ref is not None else None
    if sink_zone is None:
        return None
    sink_cfg = sink_region.get("config") if isinstance(sink_region, Mapping) else None
    absorption = sink_cfg.get("absorption_radius", 0.5) if isinstance(sink_cfg, Mapping) else 0.5
    source_shape = Shape(type=ShapeType.POLYGON, vertices=[Pose2D(x=p.x, y=p.y) for p in source_zone.polygon])
    sink_shape = Shape(type=ShapeType.POLYGON, vertices=[Pose2D(x=p.x, y=p.y) for p in sink_zone.polygon])
    return (
        SourceConfig(name=name, shape=source_shape),
        SinkConfig(name=str(sink_name), shape=sink_shape, absorption_radius=_number(name, "absorption_radius", absorption)),
    )


def scenario_lane_results(regions: object, world: World) -> dict[str, dict[str, object]]:
    """Lane summary of each source region of a scenario that has a lane block, keyed by region."""
    walls = wall_segments_of_world(world)
    zones = {zone.name: zone for level in world.all_levels for zone in level.zones}
    results: dict[str, dict[str, object]] = {}
    for name, block, ref in _source_lane_blocks(regions):
        zone = zones.get(ref)
        if zone is None:
            results[name] = {"valid": False, "reason": f"{name}: no zone {ref!r} in the world"}
            continue
        try:
            tangent = _first_tangent(name, block)
            config = lane_config_from_block(name, block)
            lane = _build(name, config, walls, agent_radius=SCENARIO_AGENT_RADIUS, source_extent=_zone_extent(zone, tangent))
            endpoints = _endpoint_configs(name, regions, zones, zone)
            if endpoints is not None:
                check_endpoints(lane, *endpoints)
        except ValueError as exc:
            results[name] = {"valid": False, "reason": str(exc)}
            continue
        results[name] = _summary_of(lane)
    return results


def scenario_lanes(regions: object, world: World) -> dict[str, Lane]:
    """Built lane of each source region of a scenario that has a lane block, keyed by region. An invalid lane raises."""
    walls = wall_segments_of_world(world)
    zones = {zone.name: zone for level in world.all_levels for zone in level.zones}
    built: dict[str, Lane] = {}
    for name, block, ref in _source_lane_blocks(regions):
        zone = zones.get(ref)
        if zone is None:
            raise ValueError(f"{name}: no zone {ref!r} in the world")
        tangent = _first_tangent(name, block)
        config = lane_config_from_block(name, block)
        lane = _build(name, config, walls, agent_radius=SCENARIO_AGENT_RADIUS, source_extent=_zone_extent(zone, tangent))
        endpoints = _endpoint_configs(name, regions, zones, zone)
        if endpoints is not None:
            check_endpoints(lane, *endpoints)
        built[name] = lane
    return {name: with_goal_corners(lane, [other for other_name, other in built.items() if other_name != name]) for name, lane in built.items()}


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


def _build(flow: str, config: LaneConfig, walls: Segments, *, agent_radius: float, source_extent: float) -> Lane:
    return build_lane(config, walls, flow=flow, agent_radius=agent_radius, clearance=PLANNER_CLEARANCE, source_extent=source_extent)


def scenario_lane_problems(regions: object, world: World) -> list[str]:
    """Lane problems of a scenario's source regions, each named by its region."""
    return [str(result["reason"]) for result in scenario_lane_results(regions, world).values() if result["valid"] is not True]


def _first_tangent(flow: str, block: Mapping[str, object]) -> tuple[float, float]:
    first, second = lane_config_from_block(flow, block).centerlines[0][:2]
    dx, dy = second.x - first.x, second.y - first.y
    length = (dx * dx + dy * dy) ** 0.5
    if length <= 1e-9:
        raise ValueError(f"flow {flow!r}: the first segment has no length")
    return dx / length, dy / length


def _zone_extent(zone: Zone, tangent: tuple[float, float]) -> float:
    along = [point.x * tangent[0] + point.y * tangent[1] for point in zone.polygon]
    return max(along) - min(along)


def _centerline(flow: str, segment: object) -> list[Pose2D]:
    points = segment.get("centerline") if isinstance(segment, Mapping) else None
    if not isinstance(points, Sequence) or isinstance(points, str) or len(points) < 2:
        raise ValueError(f"flow {flow!r}: each segment needs a centerline of at least two [x, y] points")
    return [_pose(flow, point) for point in points]


def _pose(flow: str, point: object) -> Pose2D:
    if not isinstance(point, Sequence) or isinstance(point, str) or len(point) != 2:
        raise ValueError(f"flow {flow!r}: a lane point is [x, y]")
    return Pose2D(x=_number(flow, "point x", point[0]), y=_number(flow, "point y", point[1]))


def _override(flow: str, block: Mapping[str, object], key: str) -> float | None:
    value = block.get(key)
    return None if value is None else _number(flow, key, value)


def _number(flow: str, key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"flow {flow!r}: lane {key} must be a number")
    return float(value)
