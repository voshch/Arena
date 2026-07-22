from __future__ import annotations

from dataclasses import dataclass
import math
import re

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

from .acoustic_room_spec import AcousticRoomSpec


Point2D = tuple[float, float]
Position3D = tuple[float, float, float]


def _material_name(identifier, fallback: str) -> str:
    if identifier is None:
        return fallback
    name = str(getattr(identifier, "name", "")).strip()
    return name or fallback


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip())


@dataclass(frozen=True)
class UnpairedDoor:
    door_name: str
    owner_zone: str
    start: Point2D
    end: Point2D
    reason: str


@dataclass(frozen=True)
class AcousticPortal:
    portal_id: str
    door_name: str
    zone_a: str
    zone_b: str
    start: Point2D
    end: Point2D
    height_m: float
    material_id: str

    @property
    def center_xy(self) -> Point2D:
        return (
            0.5 * (self.start[0] + self.end[0]),
            0.5 * (self.start[1] + self.end[1]),
        )

    def connects(self, first_zone: str, second_zone: str) -> bool:
        return {first_zone, second_zone} == {self.zone_a, self.zone_b}

    def other_zone(self, zone_name: str) -> str:
        if zone_name == self.zone_a:
            return self.zone_b
        if zone_name == self.zone_b:
            return self.zone_a
        raise KeyError(f"portal {self.portal_id!r} is not connected to {zone_name!r}")


@dataclass(frozen=True)
class AcousticWorldGraph:
    """Acoustic rooms connected by explicitly authored Arena doors."""

    rooms: tuple[AcousticRoomSpec, ...]
    portals: tuple[AcousticPortal, ...]
    zone_polygons: tuple[tuple[str, Polygon], ...]
    unpaired_doors: tuple[UnpairedDoor, ...] = ()

    @classmethod
    def from_world(
        cls,
        world,
        rooms: tuple[AcousticRoomSpec, ...],
        *,
        adjacency_tolerance_m: float = 0.08,
        default_door_material_id: str = "Acoustic_Default_Wall",
    ) -> "AcousticWorldGraph":
        if adjacency_tolerance_m <= 0.0:
            raise ValueError("adjacency_tolerance_m must be positive")

        polygons = tuple(
            (
                str(zone.name),
                Polygon(
                    [(float(corner.x), float(corner.y)) for corner in zone.corners]
                ),
            )
            for zone in world.zones
        )
        room_names = {room.zone_name for room in rooms}
        portals: list[AcousticPortal] = []
        unpaired: list[UnpairedDoor] = []
        seen: set[tuple[str, str, tuple[float, ...]]] = set()

        for zone in world.zones:
            owner = str(zone.name)
            for door in zone.doors:
                name = str(door.name)
                start = (float(door.start.x), float(door.start.y))
                end = (float(door.end.x), float(door.end.y))
                line = LineString([start, end])
                midpoint = line.interpolate(0.5, normalized=True)

                candidates: list[tuple[float, str]] = []
                for candidate_name, polygon in polygons:
                    if candidate_name == owner:
                        continue
                    overlap = line.intersection(
                        polygon.boundary.buffer(adjacency_tolerance_m)
                    ).length
                    distance = polygon.boundary.distance(midpoint)
                    if overlap > 0.5 * line.length or distance <= adjacency_tolerance_m:
                        candidates.append((overlap - distance, candidate_name))

                if not candidates:
                    unpaired.append(
                        UnpairedDoor(
                            door_name=name,
                            owner_zone=owner,
                            start=start,
                            end=end,
                            reason="no_adjacent_acoustic_zone",
                        )
                    )
                    continue

                candidates.sort(reverse=True)
                neighbour = candidates[0][1]
                if owner not in room_names or neighbour not in room_names:
                    unpaired.append(
                        UnpairedDoor(
                            door_name=name,
                            owner_zone=owner,
                            start=start,
                            end=end,
                            reason="adjacent_zone_has_no_room_spec",
                        )
                    )
                    continue

                ordered_zones = tuple(sorted((owner, neighbour)))
                endpoints = sorted((start, end))
                geometry_key = tuple(round(v, 4) for point in endpoints for v in point)
                key = (ordered_zones[0], ordered_zones[1], geometry_key)
                if key in seen:
                    continue
                seen.add(key)

                portals.append(
                    AcousticPortal(
                        portal_id=_safe_id(
                            f"door:{name}:{ordered_zones[0]}:{ordered_zones[1]}"
                        ),
                        door_name=name,
                        zone_a=owner,
                        zone_b=neighbour,
                        start=start,
                        end=end,
                        height_m=max(float(door.height), 0.1),
                        material_id=_material_name(
                            door.material, default_door_material_id
                        ),
                    )
                )

        return cls(
            rooms=rooms,
            portals=tuple(portals),
            zone_polygons=polygons,
            unpaired_doors=tuple(unpaired),
        )

    def room(self, zone_name: str) -> AcousticRoomSpec | None:
        return next((room for room in self.rooms if room.zone_name == zone_name), None)

    def zone_at_xy(self, x: float, y: float) -> str | None:
        point = Point(float(x), float(y))
        return next(
            (name for name, polygon in self.zone_polygons if polygon.covers(point)),
            None,
        )

    def direct_portal(
        self,
        source_zone: str,
        listener_zone: str,
        *,
        source_xy: Point2D | None = None,
        listener_xy: Point2D | None = None,
    ) -> AcousticPortal | None:
        matches = [
            portal
            for portal in self.portals
            if portal.connects(source_zone, listener_zone)
        ]
        if not matches:
            return None
        if len(matches) == 1 or source_xy is None or listener_xy is None:
            return matches[0]
        return min(
            matches,
            key=lambda portal: (
                math.dist(source_xy, portal.center_xy)
                + math.dist(listener_xy, portal.center_xy)
            ),
        )

    def position_inside_portal(
        self,
        portal: AcousticPortal,
        zone_name: str,
        *,
        inset_m: float,
        height_m: float,
    ) -> Position3D:
        polygon = next(
            (polygon for name, polygon in self.zone_polygons if name == zone_name),
            None,
        )
        if polygon is None:
            raise KeyError(f"unknown acoustic zone {zone_name!r}")

        center = Point(portal.center_xy)
        inner = polygon.buffer(-max(inset_m, 1e-4))
        xy = (
            polygon.representative_point()
            if inner.is_empty
            else nearest_points(center, inner)[1]
        )
        room = self.room(zone_name)
        if room is None:
            raise KeyError(f"no acoustic room for zone {zone_name!r}")
        z = min(max(float(height_m), 0.01), room.ceiling_height_m - 0.01)
        return float(xy.x), float(xy.y), z
