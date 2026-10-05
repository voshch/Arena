from __future__ import annotations

import attrs
import shapely
from arena_simulation_setup.tree.World import WorldDescription
from geometry_msgs.msg import Point

from .world_compat import world_zones


@attrs.frozen
class AcousticWall:
    start: tuple[float, float]
    end: tuple[float, float]
    material_id: str

    @property
    def geometry(self) -> shapely.LineString:
        return shapely.LineString([self.start, self.end])


@attrs.frozen
class AcousticZone:
    name: str
    polygon: shapely.Polygon
    floor_material_id: str


@attrs.frozen
class AcousticScene:
    zones: tuple[AcousticZone, ...]
    walls: tuple[AcousticWall, ...]
    ceiling_height_m: float = 3.0
    # Covers a robot-mounted microphone that is just beyond an authored room
    # edge while the robot base is on that edge. The corresponding RIR point
    # is moved just inside the room by PyroomacousticsAdapter.
    zone_lookup_tolerance_m: float = 0.35

    @classmethod
    def from_world(cls, world: WorldDescription) -> AcousticScene:
        zones = []
        walls = []

        for zone in world_zones(world):
            polygon = shapely.Polygon([(corner.x, corner.y) for corner in zone.corners])
            zones.append(AcousticZone(name=zone.name, polygon=polygon, floor_material_id=zone.material.name))

            for wall in zone.walls:
                material_id = wall.material.name if wall.material is not None else "default"
                walls.append(AcousticWall(start=(wall.start.x, wall.start.y), end=(wall.end.x, wall.end.y), material_id=material_id))

        return cls(zones=tuple(zones), walls=tuple(walls))

    def zone_at(self, point: Point) -> AcousticZone | None:
        return self.zone_at_xy(point.x, point.y)

    def zone_at_xy(self, x: float, y: float) -> AcousticZone | None:
        candidate = shapely.Point(float(x), float(y))
        # An exact match must win over a buffered match in an earlier zone.
        # Otherwise microphones near a portal can be assigned to the adjacent
        # room even though their physical coordinate is inside this one.
        exact = next(
            (zone for zone in self.zones if zone.polygon.covers(candidate)),
            None,
        )
        if exact is not None:
            return exact

        tolerance = max(float(self.zone_lookup_tolerance_m), 0.0)
        if tolerance == 0.0:
            return None
        nearby = [(zone.polygon.distance(candidate), index, zone) for index, zone in enumerate(self.zones) if zone.polygon.distance(candidate) <= tolerance]
        return min(nearby, default=(0.0, 0, None))[2]

    def intersecting_walls(self, source: Point, listener: Point) -> list[AcousticWall]:
        path = shapely.LineString([(source.x, source.y), (listener.x, listener.y)])
        return [wall for wall in self.walls if path.crosses(wall.geometry)]
