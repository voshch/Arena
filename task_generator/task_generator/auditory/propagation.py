from __future__ import annotations

import math

import attrs
import shapely
from geometry_msgs.msg import Point

from .acoustic_scene import AcousticScene, AcousticWall, AcousticZone
from .material_catalog import AcousticMaterialCatalog

SPEED_OF_SOUND_MPS = 343.0


@attrs.frozen
class PropagationPath:
    delay_sec: float
    gain_db: float
    bearing_rad: float
    reflection_point: tuple[float, float] | None
    interaction_type: str
    material_id: str = ""


@attrs.frozen
class PropagationResult:
    received_volume_db: float
    direct_delay_sec: float
    occluded: bool
    paths: tuple[PropagationPath, ...]
    rt60_sec: float
    reverb_gain_db: float
    source_zone: str
    listener_zone: str


class Level3Propagation:
    def __init__(
        self,
        materials: AcousticMaterialCatalog,
        *,
        max_reflections: int = 8,
        reflection_floor_db: float = -60.0,
    ) -> None:
        self._materials = materials
        self._max_reflections = max_reflections
        self._reflection_floor_db = reflection_floor_db

    def calculate(
        self,
        scene: AcousticScene,
        source: Point,
        listener: Point,
        source_level_db: float,
    ) -> PropagationResult:
        distance = max(self._distance(source, listener), 1.0)
        direct_loss = 20.0 * math.log10(distance)

        source_zone = scene.zone_at(source)
        listener_zone = scene.zone_at(listener)
        floor_loss_db = 0.0
        if source_zone is not None:
            floor_loss_db += self._materials.surface_damping_db(
                source_zone.floor_material_id,
                "floor",
            )
        if listener_zone is not None and (source_zone is None or source_zone.name != listener_zone.name):
            floor_loss_db += self._materials.surface_damping_db(
                listener_zone.floor_material_id,
                "floor",
            )

        crossed = scene.intersecting_walls(source, listener)
        transmission_loss = sum(self._materials.surface_damping_db(wall.material_id, "wall") for wall in crossed)

        direct_level = source_level_db - direct_loss - floor_loss_db - transmission_loss

        paths = [
            PropagationPath(
                delay_sec=distance / SPEED_OF_SOUND_MPS,
                gain_db=direct_level - source_level_db,
                bearing_rad=self._bearing(source, listener),
                reflection_point=None,
                interaction_type="direct",
            )
        ]

        reflections = self._first_order_reflections(scene, source, listener, source_level_db)
        paths.extend(reflections[: self._max_reflections])

        combined_level = self._sum_decibels([source_level_db + path.gain_db for path in paths])

        reverb_zone = listener_zone or source_zone

        rt60 = self._estimate_rt60(scene, reverb_zone) if reverb_zone is not None else 0.0

        return PropagationResult(
            received_volume_db=combined_level,
            direct_delay_sec=distance / SPEED_OF_SOUND_MPS,
            occluded=bool(crossed),
            paths=tuple(paths),
            rt60_sec=rt60,
            reverb_gain_db=-12.0 if rt60 > 0.0 else -math.inf,
            source_zone=source_zone.name if source_zone else "",
            listener_zone=listener_zone.name if listener_zone else "",
        )

    def _first_order_reflections(self, scene: AcousticScene, source: Point, listener: Point, source_level_db: float) -> list[PropagationPath]:
        paths = []

        for wall in scene.walls:
            image = self._reflect_point(source, wall)
            ray = shapely.LineString([(image.x, image.y), (listener.x, listener.y)])
            intersection = ray.intersection(wall.geometry)

            if intersection.is_empty or intersection.geom_type != "Point":
                continue

            reflection = Point(
                x=float(intersection.x),
                y=float(intersection.y),
                z=source.z,
            )

            d1 = self._distance(source, reflection)
            d2 = self._distance(reflection, listener)
            total_distance = max(d1 + d2, 1.0)

            material = self._materials.get(wall.material_id)
            absorption = sum(material.absorption) / len(material.absorption)
            reflection_coefficient = math.sqrt(max(1.0 - absorption, 1e-6))

            reflection_loss = -20.0 * math.log10(max(reflection_coefficient, 1e-6))
            gain_db = -20.0 * math.log10(total_distance) - reflection_loss

            if source_level_db + gain_db < self._reflection_floor_db:
                continue

            paths.append(
                PropagationPath(
                    delay_sec=total_distance / SPEED_OF_SOUND_MPS,
                    gain_db=gain_db,
                    bearing_rad=self._bearing(reflection, listener),
                    reflection_point=(reflection.x, reflection.y),
                    interaction_type="reflection",
                    material_id=wall.material_id,
                )
            )

        return sorted(paths, key=lambda path: path.delay_sec)

    @staticmethod
    def _reflect_point(point: Point, wall: AcousticWall) -> Point:
        x1, y1 = wall.start
        x2, y2 = wall.end

        dx = x2 - x1
        dy = y2 - y1
        length_squared = dx * dx + dy * dy
        if length_squared <= 1e-12:
            return Point(x=point.x, y=point.y, z=point.z)

        projection = ((point.x - x1) * dx + (point.y - y1) * dy) / length_squared

        projected_x = x1 + projection * dx
        projected_y = y1 + projection * dy

        return Point(x=2.0 * projected_x - point.x, y=2.0 * projected_y - point.y, z=point.z)

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        dx = float(a.x - b.x)
        dy = float(a.y - b.y)
        dz = float(a.z - b.z)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _bearing(source: Point, listener: Point) -> float:
        dx = float(source.x - listener.x)
        dy = float(source.y - listener.y)
        return math.atan2(dy, dx)

    def _broadband_transmission_loss(self, wall: AcousticWall) -> float:
        material = self._materials.get(wall.material_id)
        if not material.transmission_loss_db:
            return 0.0

        return float(sum(material.transmission_loss_db) / len(material.transmission_loss_db))

    @staticmethod
    def _sum_decibels(levels_db: list[float]) -> float:
        powers = [10.0 ** (level / 10.0) for level in levels_db if math.isfinite(level)]

        if not powers:
            return -math.inf

        return 10.0 * math.log10(sum(powers))

    def _estimate_rt60(self, scene: AcousticScene, zone: AcousticZone) -> float:
        floor_area = max(zone.polygon.area, 1e-6)
        perimeter = zone.polygon.length
        height = scene.ceiling_height_m

        volume = floor_area * height
        floor_material = self._materials.get(zone.floor_material_id)

        floor_absorption = floor_area * sum(floor_material.absorption) / len(floor_material.absorption)

        wall_absorption = perimeter * height * 0.10
        ceiling_absorption = floor_area * 0.10
        total_absorption = max(
            floor_absorption + wall_absorption + ceiling_absorption,
            1e-6,
        )

        return min(0.161 * volume / total_absorption, 10.0)
