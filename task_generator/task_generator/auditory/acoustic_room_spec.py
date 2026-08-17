from __future__ import annotations

import math
from typing import Literal

import attrs
from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
from arena_simulation_setup.tree.World import LevelDescription, WorldDescription
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient

from .world_compat import world_zones

Point2D = tuple[float, float]
BoundaryKind = Literal["wall", "door", "opening"]


@attrs.frozen
class AcousticBoundarySpec:
    start: Point2D
    end: Point2D
    material_id: str
    kind: BoundaryKind
    height_m: float | None = None


@attrs.frozen
class AcousticRoomSpec:
    zone_name: str
    boundary: tuple[AcousticBoundarySpec, ...]

    floor_material_id: str
    ceiling_material_id: str
    ceiling_height_m: float

    @property
    def corners_xy(self) -> tuple[Point2D, ...]:
        return tuple(segment.start for segment in self.boundary)

    @property
    def wall_material_ids(self) -> tuple[str, ...]:
        return tuple(segment.material_id for segment in self.boundary)

    @property
    def boundary_material_ids(self) -> tuple[str, ...]:
        return tuple(segment.material_id for segment in self.boundary)

@attrs.frozen
class AcousticRoomSpecConfig:
    ceiling_height_m: float = 3.0
    default_wall_material_id: str = "Acoustic_Default_Wall"
    default_floor_material_id: str = "Acoustic_Default_Floor"
    default_ceiling_material_id: str = "Acoustic_Default_Ceiling"
    opening_material_id: str = "Acoustic_Open"
    door_mode: Literal["closed", "open"] = "closed"
    geometry_tolerance_m: float = 1e-5

def _is_finite_point(point: Point2D) -> bool:
    return math.isfinite(point[0]) and math.isfinite(point[1])

def _normalize_polygon(zone: LevelDescription.Zone) -> Polygon:
    """Validate and orient an Arena zone polygon counter-clockwise."""

    coordinates = [
        (float(corner.x), float(corner.y))
        for corner in zone.corners
    ]

    if len(coordinates) < 3:
        raise ValueError(
            f"zone {zone.name!r} has fewer than three corners"
        )

    if any(
        not _is_finite_point(coordinate)
        for coordinate in coordinates
    ):
        raise ValueError(
            f"zone {zone.name!r} contains non-finite coordinates"
        )

    polygon = Polygon(coordinates)

    if polygon.is_empty:
        raise ValueError(
            f"zone {zone.name!r} has an empty polygon"
        )

    if not polygon.is_valid:
        raise ValueError(
            f"zone {zone.name!r} has invalid polygon geometry"
        )

    if polygon.area <= 1e-8:
        raise ValueError(
            f"zone {zone.name!r} has zero or negligible area"
        )

    return orient(polygon, sign=1.0)

@attrs.frozen
class _SourceSegment:
    start: Point2D
    end: Point2D
    material_id: str
    kind: BoundaryKind
    height_m: float | None = None

def _projection_parameter(
    point: Point2D,
    edge_start: Point2D,
    edge_end: Point2D,
) -> float:
    px, py = point
    ax, ay = edge_start
    bx, by = edge_end

    dx = bx - ax
    dy = by - ay
    length_squared = dx * dx + dy * dy

    if length_squared <= 1e-12:
        raise ValueError("cannot project onto a zero-length edge")

    return (
        (px - ax) * dx + (py - ay) * dy
    ) / length_squared

def _point_on_edge(
    point: Point2D,
    edge_start: Point2D,
    edge_end: Point2D,
    tolerance: float,
) -> bool:
    line = LineString([edge_start, edge_end])

    if line.distance(Point(point)) > tolerance:
        return False

    parameter = _projection_parameter(
        point,
        edge_start,
        edge_end,
    )

    return (
        -tolerance
        <= parameter
        <= 1.0 + tolerance
    )

def _interpolate(
    start: Point2D,
    end: Point2D,
    parameter: float,
) -> Point2D:
    return (
        start[0] + parameter * (end[0] - start[0]),
        start[1] + parameter * (end[1] - start[1]),
    )

def _points_close(
    first: Point2D,
    second: Point2D,
    tolerance: float,
) -> bool:
    return (
        math.hypot(
            first[0] - second[0],
            first[1] - second[1],
        )
        <= tolerance
    )

def _material_name(
    identifier: MaterialIdentifier | None,
    fallback: str,
) -> str:
    if identifier is None:
        return fallback

    name = str(identifier.name).strip()
    return name or fallback

def _segment_covers_point(
    segment: _SourceSegment,
    point: Point2D,
    tolerance: float,
) -> bool:
    return _point_on_edge(
        point,
        segment.start,
        segment.end,
        tolerance,
    )

def _classify_subsegment(
    midpoint: Point2D,
    doors: list[_SourceSegment],
    walls: list[_SourceSegment],
    config: AcousticRoomSpecConfig,
) -> tuple[str, BoundaryKind, float | None]:

    matching_doors = [
        segment
        for segment in doors
        if _segment_covers_point(
            segment,
            midpoint,
            config.geometry_tolerance_m,
        )
    ]

    if matching_doors:
        door = matching_doors[0]

        if config.door_mode == "open":
            return (
                config.opening_material_id,
                "opening",
                door.height_m,
            )

        return door.material_id, "door", door.height_m

    matching_walls = [
        segment
        for segment in walls
        if _segment_covers_point(
            segment,
            midpoint,
            config.geometry_tolerance_m,
        )
    ]

    if matching_walls:
        wall = matching_walls[0]
        return wall.material_id, "wall", wall.height_m

    return config.opening_material_id, "opening", None

def _split_polygon_edge(
    *,
    edge_start: Point2D,
    edge_end: Point2D,
    walls: list[_SourceSegment],
    doors: list[_SourceSegment],
    config: AcousticRoomSpecConfig,
) -> list[AcousticBoundarySpec]:

    breakpoints = {0.0, 1.0}

    for segment in [*walls, *doors]:
        for endpoint in (segment.start, segment.end):
            if not _point_on_edge(
                endpoint,
                edge_start,
                edge_end,
                config.geometry_tolerance_m,
            ):
                continue

            parameter = _projection_parameter(
                endpoint,
                edge_start,
                edge_end,
            )

            breakpoints.add(
                min(max(parameter, 0.0), 1.0)
            )

    ordered = sorted(breakpoints)
    output: list[AcousticBoundarySpec] = []

    for start_parameter, end_parameter in zip(
        ordered,
        ordered[1:],
        strict=False,
    ):
        if end_parameter - start_parameter <= 1e-9:
            continue

        start = _interpolate(
            edge_start,
            edge_end,
            start_parameter,
        )
        end = _interpolate(
            edge_start,
            edge_end,
            end_parameter,
        )
        midpoint = _interpolate(
            edge_start,
            edge_end,
            (start_parameter + end_parameter) / 2.0,
        )

        material_id, kind, height_m = (
            _classify_subsegment(
                midpoint,
                doors,
                walls,
                config,
            )
        )

        output.append(
            AcousticBoundarySpec(
                start=start,
                end=end,
                material_id=material_id,
                kind=kind,
                height_m=height_m,
            )
        )

    return output

def _validate_spec(
    spec: AcousticRoomSpec,
    *,
    tolerance: float,
) -> None:
    if len(spec.boundary) < 3:
        raise ValueError(
            f"room {spec.zone_name!r} has fewer than "
            "three boundary segments"
        )

    if spec.ceiling_height_m <= 0.0:
        raise ValueError(
            f"room {spec.zone_name!r} has a non-positive "
            "ceiling height"
        )

    if not spec.floor_material_id:
        raise ValueError("floor material ID cannot be empty")

    if not spec.ceiling_material_id:
        raise ValueError("ceiling material ID cannot be empty")

    for index, segment in enumerate(spec.boundary):
        following = spec.boundary[
            (index + 1) % len(spec.boundary)
        ]

        if _points_close(
            segment.start,
            segment.end,
            tolerance,
        ):
            raise ValueError(
                f"room {spec.zone_name!r} contains "
                "a zero-length boundary segment"
            )

        if not _points_close(
            segment.end,
            following.start,
            tolerance,
        ):
            raise ValueError(
                f"room {spec.zone_name!r} has a "
                "discontinuous boundary"
            )

        if not segment.material_id:
            raise ValueError(
                f"room {spec.zone_name!r} contains "
                "an empty material ID"
            )

class AcousticRoomSpecBuilder:
    def __init__(
        self,
        config: AcousticRoomSpecConfig | None = None,
    ) -> None:
        self._config = config or AcousticRoomSpecConfig()

    def from_world(
        self,
        world: WorldDescription,
    ) -> tuple[AcousticRoomSpec, ...]:
        return tuple(
            self._from_zone(zone)
            for zone in world_zones(world)
        )

    def _normalize_walls(self, zone: LevelDescription.Zone) -> list[_SourceSegment]:
        return [
            _SourceSegment(
                start=(
                    float(wall.start.x),
                    float(wall.start.y),
                ),
                end=(
                    float(wall.end.x),
                    float(wall.end.y),
                ),
                material_id=_material_name(
                    wall.material,
                    self._config.default_wall_material_id,
                ),
                kind="wall",
            )
            for wall in zone.walls
        ]

    def _normalize_doors(self, zone: LevelDescription.Zone) -> list[_SourceSegment]:
        return [
            _SourceSegment(
                start=(
                    float(door.start.x),
                    float(door.start.y),
                ),
                end=(
                    float(door.end.x),
                    float(door.end.y),
                ),
                material_id=_material_name(
                    door.material,
                    self._config.default_wall_material_id,
                ),
                kind="door",
                height_m=float(door.height),
            )
            for door in zone.doors
        ]

    def _from_zone(self, zone: LevelDescription.Zone) -> AcousticRoomSpec:
        polygon = _normalize_polygon(zone)

        corners = list(polygon.exterior.coords)[:-1]

        walls = self._normalize_walls(zone)
        doors = self._normalize_doors(zone)

        boundary: list[AcousticBoundarySpec] = []

        for index, edge_start in enumerate(corners):
            edge_end = corners[(index + 1) % len(corners)]

            boundary.extend(
                _split_polygon_edge(
                    edge_start=edge_start,
                    edge_end=edge_end,
                    walls=walls,
                    doors=doors,
                    config=self._config,
                )
            )

        floor_material_id = _material_name(
            zone.material,
            self._config.default_floor_material_id,
        )

        spec = AcousticRoomSpec(
            zone_name=str(zone.name),
            boundary=tuple(boundary),
            floor_material_id=floor_material_id,
            ceiling_material_id=(
                self._config.default_ceiling_material_id
            ),
            ceiling_height_m=(
                self._config.ceiling_height_m
            ),
        )

        _validate_spec(
            spec,
            tolerance=self._config.geometry_tolerance_m,
        )

        return spec
