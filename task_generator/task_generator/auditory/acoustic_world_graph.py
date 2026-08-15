from __future__ import annotations

import heapq
import itertools
import math
import re
from collections.abc import Hashable
from typing import Literal

import attrs
from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
from arena_simulation_setup.tree.World import WorldDescription
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import nearest_points

from .acoustic_room_spec import AcousticRoomSpec
from .world_compat import world_zone_groups, world_zones

Point2D = tuple[float, float]
Position3D = tuple[float, float, float]


def _material_name(identifier: MaterialIdentifier | None, fallback: str) -> str:
    if identifier is None:
        return fallback
    name = str(identifier.name).strip()
    return name or fallback


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip())


@attrs.frozen
class UnpairedDoor:
    door_name: str
    owner_zone: str
    start: Point2D
    end: Point2D
    reason: str


@attrs.frozen
class AcousticPortal:
    portal_id: str
    door_name: str
    zone_a: str
    zone_b: str
    start: Point2D
    end: Point2D
    height_m: float
    material_id: str
    portal_kind: Literal["door", "opening"] = "door"
    loss_db: float | None = None

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


@attrs.frozen
class AcousticPortalRoute:
    zones: tuple[str, ...]
    portals: tuple[AcousticPortal, ...]
    estimated_loss_db: float
    estimated_distance_m: float

    @property
    def hop_count(self) -> int:
        return len(self.portals)


@attrs.frozen
class AcousticWorldGraph:
    """Acoustic rooms connected by authored doors and shared openings."""

    rooms: tuple[AcousticRoomSpec, ...]
    portals: tuple[AcousticPortal, ...]
    zone_polygons: tuple[tuple[str, Polygon], ...]
    unpaired_doors: tuple[UnpairedDoor, ...] = ()

    @classmethod
    def from_world(
        cls,
        world: WorldDescription,
        rooms: tuple[AcousticRoomSpec, ...],
        *,
        adjacency_tolerance_m: float = 0.08,
        default_door_material_id: str = "Acoustic_Default_Wall",
        derive_opening_portals: bool = True,
        minimum_opening_width_m: float = 0.30,
        door_portal_loss_db: float = 3.0,
        opening_portal_loss_db: float = 0.5,
    ) -> AcousticWorldGraph:
        if adjacency_tolerance_m <= 0.0:
            raise ValueError("adjacency_tolerance_m must be positive")
        if minimum_opening_width_m <= 0.0:
            raise ValueError("minimum_opening_width_m must be positive")

        zones = world_zones(world)
        zone_levels = {
            str(zone.name): level_id
            for level_id, level_zones in world_zone_groups(world)
            for zone in level_zones
        }
        polygons = tuple(
            (
                str(zone.name),
                Polygon(
                    [(float(corner.x), float(corner.y)) for corner in zone.corners]
                ),
            )
            for zone in zones
        )
        room_names = {room.zone_name for room in rooms}
        portals: list[AcousticPortal] = []
        unpaired: list[UnpairedDoor] = []
        seen: set[tuple[str, str, tuple[float, ...]]] = set()

        for zone in zones:
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
                    if zone_levels.get(candidate_name) != zone_levels.get(owner):
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
                        portal_kind="door",
                        loss_db=door_portal_loss_db,
                    )
                )

        if derive_opening_portals:
            room_by_name = {room.zone_name: room for room in rooms}
            for index, (zone_a, polygon_a) in enumerate(polygons):
                room_a = room_by_name.get(zone_a)
                if room_a is None:
                    continue
                openings_a = [
                    LineString((boundary.start, boundary.end))
                    for boundary in room_a.boundary
                    if boundary.kind == "opening"
                ]
                for zone_b, polygon_b in polygons[index + 1:]:
                    if zone_levels.get(zone_b) != zone_levels.get(zone_a):
                        continue
                    if (
                        polygon_a.boundary.distance(polygon_b.boundary)
                        > adjacency_tolerance_m
                    ):
                        continue
                    room_b = room_by_name.get(zone_b)
                    if room_b is None:
                        continue
                    openings_b = [
                        LineString((boundary.start, boundary.end))
                        for boundary in room_b.boundary
                        if boundary.kind == "opening"
                    ]
                    for opening_a in openings_a:
                        for opening_b in openings_b:
                            shared = opening_a.intersection(opening_b)
                            lines = (
                                list(shared.geoms)
                                if isinstance(shared, BaseMultipartGeometry)
                                else [shared]
                            )
                            for line in lines:
                                if (
                                    line.geom_type != "LineString"
                                    or line.length < minimum_opening_width_m
                                ):
                                    continue
                                coordinates = list(line.coords)
                                start = tuple(map(float, coordinates[0]))
                                end = tuple(map(float, coordinates[-1]))
                                ordered_zones = tuple(sorted((zone_a, zone_b)))
                                endpoints = sorted((start, end))
                                geometry_key = tuple(
                                    round(value, 4)
                                    for point in endpoints
                                    for value in point
                                )
                                key = (
                                    ordered_zones[0],
                                    ordered_zones[1],
                                    geometry_key,
                                )
                                if key in seen:
                                    continue
                                # An authored door on the same shared span wins.
                                candidate_line = LineString((start, end))
                                if any(
                                    portal.connects(zone_a, zone_b)
                                    and LineString(
                                        (portal.start, portal.end)
                                    ).distance(candidate_line)
                                    <= adjacency_tolerance_m
                                    and LineString(
                                        (portal.start, portal.end)
                                    ).intersection(candidate_line).length
                                    > 0.5 * candidate_line.length
                                    for portal in portals
                                ):
                                    continue
                                seen.add(key)
                                portals.append(
                                    AcousticPortal(
                                        portal_id=_safe_id(
                                            "opening:"
                                            f"{ordered_zones[0]}:"
                                            f"{ordered_zones[1]}:"
                                            + ":".join(
                                                f"{value:.3f}"
                                                for value in geometry_key
                                            )
                                        ),
                                        door_name="",
                                        zone_a=zone_a,
                                        zone_b=zone_b,
                                        start=start,
                                        end=end,
                                        height_m=min(
                                            room_a.ceiling_height_m,
                                            room_b.ceiling_height_m,
                                        ),
                                        material_id="Acoustic_Open",
                                        portal_kind="opening",
                                        loss_db=opening_portal_loss_db,
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

    def find_portal_route(
        self,
        source_zone: str,
        listener_zone: str,
        *,
        source_xy: Point2D,
        listener_xy: Point2D,
        max_portals: int = 4,
        distance_loss_db_per_m: float = 0.05,
        default_door_loss_db: float = 3.0,
        default_opening_loss_db: float = 0.5,
    ) -> AcousticPortalRoute | None:
        """Return the least-cost simple portal route between two rooms."""
        if source_zone == listener_zone:
            return AcousticPortalRoute(
                zones=(source_zone,),
                portals=(),
                estimated_loss_db=0.0,
                estimated_distance_m=math.dist(source_xy, listener_xy),
            )
        if max_portals <= 0:
            return None

        adjacency: dict[str, list[AcousticPortal]] = {}
        for portal in self.portals:
            adjacency.setdefault(portal.zone_a, []).append(portal)
            adjacency.setdefault(portal.zone_b, []).append(portal)

        # (estimated total cost, sequence, accumulated cost, distance, zone,
        #  anchor, zones, portals)
        queue: list[tuple[Hashable, ...]] = []
        sequence = itertools.count()
        heapq.heappush(
            queue,
            (
                distance_loss_db_per_m * math.dist(source_xy, listener_xy),
                next(sequence),
                0.0,
                0.0,
                source_zone,
                source_xy,
                (source_zone,),
                (),
            ),
        )
        while queue:
            (
                _,
                _,
                cost,
                distance,
                zone,
                anchor,
                zones,
                route,
            ) = heapq.heappop(queue)
            if zone == listener_zone and route:
                final_distance = distance + math.dist(anchor, listener_xy)
                final_cost = cost + (
                    distance_loss_db_per_m * math.dist(anchor, listener_xy)
                )
                return AcousticPortalRoute(
                    zones=zones,
                    portals=route,
                    estimated_loss_db=final_cost,
                    estimated_distance_m=final_distance,
                )
            if len(route) >= max_portals:
                continue
            for portal in adjacency.get(zone, ()):
                neighbour = portal.other_zone(zone)
                if neighbour in zones:
                    continue
                center = portal.center_xy
                segment_distance = math.dist(anchor, center)
                portal_loss = (
                    portal.loss_db
                    if portal.loss_db is not None
                    else (
                        default_opening_loss_db
                        if portal.portal_kind == "opening"
                        else default_door_loss_db
                    )
                )
                next_cost = (
                    cost
                    + portal_loss
                    + distance_loss_db_per_m * segment_distance
                )
                heuristic = (
                    distance_loss_db_per_m
                    * math.dist(center, listener_xy)
                )
                heapq.heappush(
                    queue,
                    (
                        next_cost + heuristic,
                        next(sequence),
                        next_cost,
                        distance + segment_distance,
                        neighbour,
                        center,
                        (*zones, neighbour),
                        (*route, portal),
                    ),
                )
        return None

    def connected_components(self) -> tuple[tuple[str, ...], ...]:
        remaining = {room.zone_name for room in self.rooms}
        adjacency = {name: set() for name in remaining}
        for portal in self.portals:
            adjacency.setdefault(portal.zone_a, set()).add(portal.zone_b)
            adjacency.setdefault(portal.zone_b, set()).add(portal.zone_a)
        components: list[tuple[str, ...]] = []
        while remaining:
            seed = min(remaining)
            stack = [seed]
            component: set[str] = set()
            while stack:
                zone = stack.pop()
                if zone in component:
                    continue
                component.add(zone)
                stack.extend(adjacency.get(zone, ()) - component)
            remaining -= component
            components.append(tuple(sorted(component)))
        return tuple(components)

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
