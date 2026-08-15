import io
import logging
import math
import os
import tarfile
import time
import typing
from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path

import attrs
import numpy as np
import yaml
from typing_extensions import Self

from arena_simulation_setup import ASS_DIR
from arena_simulation_setup.shared import (
    Ceiling,
    Door,
    DynamicObstacle,
    Elevator,
    Floor,
    Obstacle,
    Schedule,
    SemanticCfg,
    Signal,
    Wall,
)
from arena_simulation_setup.shared.semantics import parse_semantics
from arena_simulation_setup.tree import FallbackResolver, Identifier, PathView, SimplePathResolver
from arena_simulation_setup.tree.assets.Material import (
    Material,
    MaterialIdentifier,
)
from arena_simulation_setup.utils.cattrs import ArenaConverter, converter
from arena_simulation_setup.utils.geometry import PointResolver, Position

from .Map import Map
from .Scenario import RegionAssignment, ScenarioView


@attrs.define
class LevelDescription:
    """
    Description of one level of the world.
    """

    @attrs.define
    class Zone:
        """
        Description of a zone (e.g. room) within the 3D world
        """

        @attrs.define
        class WorldEntities:
            """
            Description of the entities within the 3D world
            """

            static: list[Obstacle] = attrs.field(factory=list)
            dynamic: list[DynamicObstacle] = attrs.field(factory=list)

        name: str
        description: str = ''
        material: MaterialIdentifier = attrs.field(
            converter=MaterialIdentifier.converter,
            default=Material.default('floor'),
        )
        corners: list[Position] = attrs.field(factory=list)
        walls: list[Wall] = attrs.field(factory=list)
        doors: list[Door] = attrs.field(factory=list)
        elevators: list[Elevator] = attrs.field(factory=list)
        schedules: list[Schedule] = attrs.field(factory=list)
        signals: list[Signal] = attrs.field(factory=list)
        entities: WorldEntities = attrs.field(factory=WorldEntities)
        ceiling: bool = attrs.field(default=True)
        ceiling_height: float | None = attrs.field(default=None)
        ceiling_cast_shadows: bool = attrs.field(default=False)
        ceiling_material: MaterialIdentifier = attrs.field(
            converter=MaterialIdentifier.converter,
            default=Material.default('ceiling'),
        )
        wall_material: MaterialIdentifier = attrs.field(
            converter=MaterialIdentifier.converter,
            default=Material.default('wall'),
        )
        semantics: list[SemanticCfg] = attrs.field(factory=list, converter=parse_semantics)

        @property
        def floor(self) -> Floor:
            x_min = min(corner.x for corner in self.corners)
            y_min = min(corner.y for corner in self.corners)
            x_max = max(corner.x for corner in self.corners)
            y_max = max(corner.y for corner in self.corners)
            pos = Position(x=(x_min + x_max) / 2, y=(y_min + y_max) / 2)
            x_length = x_max - x_min
            y_length = y_max - y_min
            return Floor(name=self.name, pos=pos, x_length=x_length, y_length=y_length, material=self.material)

    zones: list[Zone] = attrs.field(factory=list)

    @property
    def all_walls(self) -> typing.Iterable[Wall]:
        return (wall for zone in self.zones for wall in zone.walls if wall.material is None or wall.material.name)

    @property
    def all_doors(self) -> typing.Iterable[Door]:
        return (door for zone in self.zones for door in zone.doors)

    @property
    def all_elevators(self) -> typing.Iterable[Elevator]:
        return (elevator for zone in self.zones for elevator in zone.elevators)

    @property
    def all_schedules(self) -> typing.Iterable[Schedule]:
        return (schedule for zone in self.zones for schedule in zone.schedules)

    @property
    def all_signals(self) -> typing.Iterable[Signal]:
        return (signal for zone in self.zones for signal in zone.signals)

    @property
    def all_floors(self) -> typing.Iterable[Floor]:
        return (zone.floor for zone in self.zones if zone.material.name)

    async def all_ceilings(self) -> list[Ceiling]:
        result: list[Ceiling] = []
        for zone in self.zones:
            if not zone.ceiling:
                continue
            if not zone.ceiling_material.name:
                continue
            if not zone.corners:
                continue
            x_min = min(corner.x for corner in zone.corners)
            y_min = min(corner.y for corner in zone.corners)
            x_max = max(corner.x for corner in zone.corners)
            y_max = max(corner.y for corner in zone.corners)
            pos = Position(x=(x_min + x_max) / 2, y=(y_min + y_max) / 2)
            x_length = x_max - x_min
            y_length = y_max - y_min
            if zone.ceiling_height is not None:
                z = zone.ceiling_height
            else:
                z = 2.0
                for wall in zone.walls:
                    segments, _ = await wall.assets()
                    for segment in segments:
                        z = max(z, segment.start.z + segment.height)
            result.append(
                Ceiling(
                    name=zone.name,
                    pos=pos,
                    x_length=x_length,
                    y_length=y_length,
                    z=z,
                    cast_shadows=zone.ceiling_cast_shadows,
                    material=zone.ceiling_material,
                )
            )
        return result

    @property
    def all_static_entities(self) -> typing.Iterable[Obstacle]:
        return (entity for zone in self.zones for entity in zone.entities.static)

    @property
    def all_dynamic_entities(self) -> typing.Iterable[DynamicObstacle]:
        return (entity for zone in self.zones for entity in zone.entities.dynamic)

    def shift_all_positions(self, dx: float, dy: float):
        diff: Position = Position(dx, dy)
        for zone in self.zones:
            for idx, corner in enumerate(zone.corners):
                zone.corners[idx] = corner + diff
        for wall in self.all_walls:
            wall.start = wall.start + diff
            wall.end = wall.end + diff
        for door in self.all_doors:
            door.start = door.start + diff
            door.end = door.end + diff
        for elevator in self.all_elevators:
            elevator.position = elevator.position + diff
        for static_entity in self.all_static_entities:
            static_entity.pose.position = static_entity.pose.position + diff
        for dynamic_entity in self.all_dynamic_entities:
            dynamic_entity.pose.position = dynamic_entity.pose.position + diff
            for idx, wp in enumerate(dynamic_entity.waypoints):
                dynamic_entity.waypoints[idx] = wp + diff

    def lookup_zone_polygon(self, name: str) -> list[Position] | None:
        """Look up a zone, door, or elevator by name and return its polygon vertices."""
        for zone in self.zones:
            if zone.name == name:
                return zone.corners
            for door in zone.doors:
                if door.name == name:
                    return _door_polygon(door.start, door.end)
            for elevator in zone.elevators:
                if elevator.name == name:
                    return elevator.cabin_corners()
        return None

    def zone_ref_names(self) -> list[str]:
        """Every name resolvable by lookup_zone_polygon (zones, doors, elevators)."""
        names: list[str] = []
        for zone in self.zones:
            names.append(zone.name)
            names.extend(door.name for door in zone.doors)
            names.extend(elevator.name for elevator in zone.elevators)
        return names

    def point_resolver(
        self,
        rng: np.random.Generator,
        *,
        is_valid: typing.Callable[[Position], bool] | None = None,
    ) -> PointResolver:
        """Resolver that samples a point inside a named zone/door/elevator polygon."""
        return PointResolver(
            lookup=self.lookup_zone_polygon,
            rng=rng,
            is_valid=is_valid,
            candidates=self.zone_ref_names,
        )

    def zone_converter(
        self,
        rng: np.random.Generator,
        *,
        is_valid: typing.Callable[[Position], bool] | None = None,
    ) -> ArenaConverter:
        """Return a converter that resolves zone/door/elevator names to geometry.

        String values for Pose/Position/Waypoint fields are resolved by the active
        PointResolver (a random point sampled within the named zone polygon).
        RegionAssignment dicts with a ``ref`` key get their polygon resolved here.
        """
        lookup = self.lookup_zone_polygon
        base_region_hook = converter.get_structure_hook(RegionAssignment)

        def region_hook(v: object, t: type) -> RegionAssignment:
            if isinstance(v, dict) and 'ref' in v:
                ref = v.pop('ref')
                polygon = lookup(ref)
                if polygon is None:
                    raise ValueError(f"region ref '{ref}' not found in world")
                v['polygon'] = polygon
            return base_region_hook(v, t)

        c = converter.copy()
        c.set_resolver(self.point_resolver(rng, is_valid=is_valid))
        c.register_structure_hook(RegionAssignment, region_hook)
        return c

    def _rasterize_kwargs(
        self,
        *,
        default_asset_bbox: tuple[tuple[float, float], tuple[float, float]] | None = None,
        asset_color: str | None = None,
        asset_name_color: str | None = None,
    ) -> dict[str, typing.Any]:
        import shapely
        import shapely.affinity

        map_kwargs: dict[str, typing.Any] = {
            "rooms": shapely.MultiPolygon([shapely.Polygon(zone.corners) for zone in self.zones]),
            "doors": shapely.MultiPolygon([poly for door in self.all_doors for poly in _render_door_polygons(door)] + [poly for elevator in self.all_elevators for poly in _render_elevator_polygons(elevator)]),
            "walls": shapely.MultiLineString(list(self.all_walls)),
            "padding": 5,
        }

        if asset_color is not None:
            static_objects: list[tuple[str, shapely.Polygon]] = []
            for entity in self.all_static_entities:
                bbox = entity.asdict(expand_extra=True).get('bbox') or default_asset_bbox
                if bbox is None:
                    continue
                try:
                    (x_min, x_max), (y_min, y_max), *z_pair = bbox
                except (ValueError, TypeError):
                    continue
                if z_pair:
                    (z_min, _z_max) = z_pair[0]
                    # skip fixtures mounted above passage height (lamps, signs) whose 2D footprint would falsely block the floor
                    if entity.pose.position.z + z_min > _PASSAGE_CLEARANCE:
                        continue
                poly = shapely.box(x_min, y_min, x_max, y_max)
                poly = shapely.affinity.rotate(poly, entity.pose.orientation.to_yaw(), use_radians=True)
                poly = shapely.affinity.translate(poly, entity.pose.position.x, entity.pose.position.y)
                static_objects.append((entity.name, poly))

            map_kwargs["static_objects"] = static_objects
            map_kwargs["asset_color"] = asset_color
            map_kwargs["asset_name_color"] = asset_name_color

        return map_kwargs

    def render(
        self,
        resolution: float = 0.05,
        *,
        default_asset_bbox: tuple[tuple[float, float], tuple[float, float]] | None = None,
        asset_color: str | None = None,
        asset_name_color: str | None = None,
    ) -> tuple[bytes, tuple[float, float]]:
        """
        Render the world description to a PNG image.

        Args:
            resolution (float): The resolution of the rendered image in meters per pixel.
            default_asset_bbox (Optional[tuple[tuple[float, float], tuple[float, float]]]): Default bounding box ((xmin, xmax), (ymin, ymax)) to use for static entities if not specified individually.
            asset_color (Optional[str]): Color used to fill static objects in the map.
            asset_name_color (Optional[str]): Color used for static object names in the map.

        Returns:
            Tuple[bytes, tuple[float, float]]: PNG image bytes and the origin (x, y) of the map.

        Notes:
            - Static objects are drawn only if their dimensions can be determined from bbox, width/height, or default_asset_bbox.
            - If asset_color is None, static objects are not drawn.
        """
        return Map.generate_png(
            resolution=resolution,
            **self._rasterize_kwargs(
                default_asset_bbox=default_asset_bbox,
                asset_color=asset_color,
                asset_name_color=asset_name_color,
            ),
        )

    def render_grid(
        self,
        resolution: float = 0.05,
        *,
        default_asset_bbox: tuple[tuple[float, float], tuple[float, float]] | None = None,
        asset_color: str | None = None,
        asset_name_color: str | None = None,
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Like `render` but returns a uint8 numpy array (255=free, 0=occupied) + origin."""
        return Map.rasterize(
            resolution=resolution,
            **self._rasterize_kwargs(
                default_asset_bbox=default_asset_bbox,
                asset_color=asset_color,
                asset_name_color=asset_name_color,
            ),
        )

    def render_map_files(
        self,
        level_origins: dict[str, tuple[float, float]] | None = None,
        resolution: float = 0.05,
        *,
        default_asset_bbox: tuple[tuple[float, float], tuple[float, float]] | None = None,
        asset_color: str | None = None,
        asset_name_color: str | None = None,
    ) -> tuple[bytes, str]:
        """Render this compacted description to (map.png bytes, map.yaml text); embeds per-level origins under the `origins` key."""
        png_bytes, origin = self.render(
            resolution=resolution,
            default_asset_bbox=default_asset_bbox,
            asset_color=asset_color,
            asset_name_color=asset_name_color,
        )
        map_yaml_text = Map.generate_map_yaml(resolution=resolution, filename='map.png', origin=origin)
        if level_origins:
            map_yaml_data = yaml.safe_load(map_yaml_text)
            map_yaml_data['origins'] = {fid: list(off) for fid, off in level_origins.items()}
            map_yaml_text = yaml.safe_dump(map_yaml_data)
        return png_bytes, map_yaml_text

    def export(self, resolution: float = 0.05, extra_files: dict[str, bytes] | None = None, **kwargs: object) -> tarfile.TarFile:
        """
        Export the world description to world.yaml, map.png, map.yaml
        """

        if extra_files is None:
            extra_files = {}
        files: dict[str, bytes] = {**extra_files}

        files['world.yaml'] = typing.cast(bytes, yaml.safe_dump(converter.unstructure(self), encoding='utf-8', sort_keys=False))

        render_args: dict[str, typing.Any] = {"resolution": resolution}
        if "default_asset_bbox" in kwargs:
            render_args["default_asset_bbox"] = kwargs["default_asset_bbox"]
        if "asset_color" in kwargs:
            render_args["asset_color"] = kwargs["asset_color"]
        if "asset_name_color" in kwargs:
            render_args["asset_name_color"] = kwargs["asset_name_color"]

        files['map/map.png'], origin = self.render(**render_args)

        files['map/map.yaml'] = Map.generate_map_yaml(resolution=resolution, filename='map.png', origin=origin).encode('utf-8')

        with io.BytesIO() as tar_stream:
            with tarfile.open(mode='w', fileobj=tar_stream) as tarball:
                for filename, content in files.items():
                    info = tarfile.TarInfo(name=os.path.normpath(filename))
                    info.size = len(content)
                    tarball.addfile(tarinfo=info, fileobj=io.BytesIO(content))
            tar_stream.seek(0)
            return tarfile.open(fileobj=io.BytesIO(tar_stream.getvalue()))


_ZONE_KEY_ALIASES = {'mat': 'material', 'ceiling_mat': 'ceiling_material', 'wall_mat': 'wall_material'}


def _structure_zone(value: object, cls: type) -> object:
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        remapped = dict(value)
        for alias, field in _ZONE_KEY_ALIASES.items():
            if alias in remapped and field not in remapped:
                remapped[field] = remapped.pop(alias)
        value = remapped
    return converter.structure_attrs_fromdict(value, cls)


converter.register_structure_hook(LevelDescription.Zone, _structure_zone)


@attrs.define
class ElevatorDescriptor:
    """
    Elevator descriptor in multi-level context.
    Stores destination mapping by level id.
    """

    name: str = attrs.field(default='')
    destinations_dict: dict[str, str] = attrs.field(factory=dict)

    @classmethod
    def from_elevator(cls, elevator: Elevator) -> 'ElevatorDescriptor':
        return cls(name=elevator.name)

    @property
    def all_destinations(self) -> typing.Iterable[typing.Tuple[str, str]]:
        return ((level_id, elevator_name) for level_id, elevator_name in self.destinations_dict.items())

    def add_destination(self, destination: str, level_id: str):
        if level_id in self.destinations_dict:
            raise RuntimeError(f"Error occured while adding a new destination to elevator {self.name}: level {level_id} of destination is already occupied")
        self.destinations_dict[level_id] = destination


@attrs.define
class Level(LevelDescription):
    levelElevators: list[ElevatorDescriptor] = attrs.field(factory=list)

    @classmethod
    def from_level_description(cls, level_description: LevelDescription) -> 'Level':
        return cls(zones=level_description.zones)


@attrs.define
class WorldDescription:
    levels: dict[str, Level] = attrs.field(factory=dict)  # level by its level id

    @property
    def level_ids(self) -> typing.Iterable[str]:
        return self.levels.keys()

    @property
    def all_levels(self) -> typing.Iterable[Level]:
        return (level for level in self.levels.values())

    @property
    def all_static_entities(self) -> typing.Iterable[Obstacle]:
        return (obstacle for level in self.all_levels for obstacle in level.all_static_entities)

    @property
    def all_dynamic_entities(self) -> typing.Iterable[DynamicObstacle]:
        return (d_obstacle for level in self.all_levels for d_obstacle in level.all_dynamic_entities)

    def get_level(self, level_id: str) -> Level | None:
        return self.levels.get(level_id, None)

    def compact_world(self, origins: dict[str, tuple[float, float]]) -> LevelDescription:
        """Return a single LevelDescription that has all the levels but with shifted origins so that they don't stack with each other."""
        out = LevelDescription()
        for level_id, level in self.levels.items():
            try:
                origin = origins[level_id]
                _level = deepcopy(level)
                _level.shift_all_positions(*origin)
                out.zones.extend(_level.zones)

            except KeyError as e:
                raise KeyError(f"when creating compacted single world from WorldDescription, the origin for level {level_id} was not given") from e

        return out

    def validate(self, loaded_level_ids: set[str] | None = None) -> list[str]:
        """Returns warnings for destinations into unloaded levels, raises on everything else."""
        warnings: list[str] = []

        elevator_levels: dict[str, str] = {}
        for level_id, level in self.levels.items():
            for elevator in level.all_elevators:
                if elevator.name in elevator_levels:
                    raise RuntimeError(f"elevator '{elevator.name}' appears in multiple levels: {elevator_levels[elevator.name]} and {level_id}")
                elevator_levels[elevator.name] = level_id

        for level_id, level in self.levels.items():
            for level_elevator in level.levelElevators:
                if level_elevator.name not in elevator_levels:
                    raise RuntimeError(f"level elevator '{level_elevator.name}' in level '{level_id}' has no matching elevator entity in levels")
                for destination_level_id, destination_name in level_elevator.all_destinations:
                    if destination_level_id not in self.levels:
                        if loaded_level_ids is not None and destination_level_id not in loaded_level_ids:
                            msg = f"elevator '{level_elevator.name}' in level '{level_id}' references destination level '{destination_level_id}' which was not loaded (selective load); elevator will have no destination"
                            warnings.append(msg)
                            logging.getLogger(__name__).warning(msg)
                            continue
                        raise RuntimeError(f"destination level '{destination_level_id}' referenced by level elevator '{level_elevator.name}' does not exist")
                    if destination_name not in elevator_levels:
                        raise RuntimeError(f"destination elevator '{destination_name}' referenced by level elevator '{level_elevator.name}' does not exist")
                    if elevator_levels[destination_name] != destination_level_id:
                        raise RuntimeError(f"destination mapping mismatch for '{destination_name}': declared level '{destination_level_id}', actual level '{elevator_levels[destination_name]}'")

        return warnings

    def apply_elevator_door_sides(self, *, max_distance: float = 1.0) -> None:
        """Infer and set `door_side` for elevators.

        For each elevator on every level, find the nearest wall segment in the same
        level. Once the wall orientation is known, use the projected elevator
        coordinate on the perpendicular axis to compare only walls that live on the
        same slice of the floor. The outermost wall on that slice determines the
        outward direction: left-most => ``+x``, right-most => ``-x``, bottom-most
        => ``+y``, top-most => ``-y``. Elevators farther than ``max_distance`` from
        any wall are left unchanged.

        Elevators with an explicitly non-default ``door_side`` (i.e., anything other
        than the default ``'+x'``) are skipped: they keep their explicit value.
        """
        for _, level in self.levels.items():
            walls = list(level.all_walls)
            if not walls:
                continue

            for elevator in level.all_elevators:
                if elevator.door_side != '+x':
                    continue
                pos = np.array([elevator.position.x, elevator.position.y])

                best_wall = None
                best_proj = None
                best_dist = float('inf')

                for wall in walls:
                    a = np.array([wall.start.x, wall.start.y])
                    b = np.array([wall.end.x, wall.end.y])
                    ab = b - a
                    ab_len2 = float(np.dot(ab, ab))
                    if ab_len2 == 0.0:
                        proj = a
                    else:
                        t = float(np.dot(pos - a, ab) / ab_len2)
                        t = max(0.0, min(1.0, t))
                        proj = a + t * ab
                    dist = float(np.linalg.norm(pos - proj))
                    if dist < best_dist:
                        best_dist = dist
                        best_wall = wall
                        best_proj = proj

                if best_wall is None or best_dist > max_distance:
                    # no nearby wall to infer from
                    continue
                if best_proj is None:
                    continue

                best_dx = best_wall.end.x - best_wall.start.x
                best_dy = best_wall.end.y - best_wall.start.y
                wall_axis = 'y' if abs(best_dy) >= abs(best_dx) else 'x'

                projected_coord = float(best_proj[0] if wall_axis == 'y' else best_proj[1])
                comparable_walls = [wall for wall in walls if (abs((wall.start.x + wall.end.x) / 2.0 - projected_coord) if wall_axis == 'y' else abs((wall.start.y + wall.end.y) / 2.0 - projected_coord)) <= max_distance]
                if not comparable_walls:
                    comparable_walls = [best_wall]

                if wall_axis == 'y':
                    axis_coords = [(wall.start.x + wall.end.x) / 2.0 for wall in comparable_walls]
                    best_coord = (best_wall.start.x + best_wall.end.x) / 2.0
                else:
                    axis_coords = [(wall.start.y + wall.end.y) / 2.0 for wall in comparable_walls]
                    best_coord = (best_wall.start.y + best_wall.end.y) / 2.0
                coord_min = min(axis_coords)
                coord_max = max(axis_coords)

                if wall_axis == 'y':
                    if abs(best_coord - coord_min) <= abs(best_coord - coord_max):
                        elevator.door_side = '+x'
                    else:
                        elevator.door_side = '-x'
                else:
                    if abs(best_coord - coord_min) <= abs(best_coord - coord_max):
                        elevator.door_side = '+y'
                    else:
                        elevator.door_side = '-y'

    @staticmethod
    def _parse_destinations(destination: str) -> list[str]:
        raw = str(destination or '')
        return [part.strip() for part in raw.split(',') if part.strip()]

    @classmethod
    def _all_destinations_for_elevator(cls, elevator: Elevator) -> typing.Iterable[str]:
        return (destination for destination in cls._parse_destinations(elevator.destination))

    @classmethod
    def from_levels(cls, *levels: LevelDescription) -> 'WorldDescription':
        """Construct a WorldDescription from level descriptions, indexed in order as '0', '1', ..."""
        if not cls.unique_elevator_ids(levels):
            raise RuntimeError('from_levels constructor of WorldDescription expects elevator names to be unique across all provided LevelDescription instances')

        levels_by_id: dict[str, Level] = {}
        for idx, level_desc in enumerate(levels):
            level = Level.from_level_description(level_desc)
            level_elevators: list[ElevatorDescriptor] = []
            for elevator in level_desc.all_elevators:
                level_elevator = ElevatorDescriptor.from_elevator(elevator)
                for destination in cls._all_destinations_for_elevator(elevator):
                    level_id = cls._elevator_level_index(destination, levels)
                    if level_id == '':
                        raise RuntimeError('_elevator_level_index returned an empty string. Check elevator destination mapping logic.')
                    level_elevator.add_destination(destination, level_id)
                level_elevators.append(level_elevator)

            level.levelElevators = level_elevators
            levels_by_id[f'{idx}'] = level

        return cls(levels=levels_by_id)

    @classmethod
    def unique_elevator_ids(cls, _level_descs: typing.Iterable[LevelDescription]) -> bool:
        del cls
        seen = set()
        for level_desc in _level_descs:
            for elevator in level_desc.all_elevators:
                if elevator.name in seen:
                    return False
                seen.add(elevator.name)
        return True

    @classmethod
    def _elevator_level_index(
        cls,
        elevator_name: str,
        _level_descs: typing.Iterable[LevelDescription],
    ) -> str:
        del cls
        for idx, level_desc in enumerate(_level_descs):
            if any(elevator.name == elevator_name for elevator in level_desc.all_elevators):
                return f'{idx}'
        return ''

    def level_bbox(self) -> list[tuple[float, float]]:
        bboxes: list[tuple[float, float]] = []
        for level in self.levels.values():
            corners = [corner for zone in level.zones for corner in zone.corners]
            if not corners:
                bboxes.append((0.0, 0.0))
                continue

            x_min = min(corner.x for corner in corners)
            y_min = min(corner.y for corner in corners)
            x_max = max(corner.x for corner in corners)
            y_max = max(corner.y for corner in corners)
            bboxes.append((x_max - x_min, y_max - y_min))
        return bboxes

    def max_level_bbox_dim(self) -> tuple[float, float]:
        bboxes = self.level_bbox()
        if not bboxes:
            return (0.0, 0.0)
        return (
            max(width for width, _ in bboxes),
            max(height for _, height in bboxes),
        )

    def _render_whole_with_origins(
        self,
        resolution: float = 0.05,
        preferred_pixel_width: int = 500,
        margin_width_in_meter: float = 5,
        margin_height_in_meter: float = 5,
        *,
        default_asset_bbox: tuple[tuple[float, float], tuple[float, float]] | None = None,
        asset_color: str | None = None,
        asset_name_color: str | None = None,
    ) -> tuple[bytes, dict[str, tuple[float, float]]]:
        """Render all levels into a single PNG and return per-level offsets.

        The returned offsets map each level_id to the (x, y) shift applied
        when flattening into the shared map frame.
        """

        if not self.levels:
            raise RuntimeError('Cannot render an empty WorldDescription')

        def _level_bbox(level: Level) -> tuple[float, float, float, float]:
            corners = [corner for zone in level.zones for corner in zone.corners]
            if not corners:
                return (0.0, 0.0, 0.0, 0.0)

            x_min = min(corner.x for corner in corners)
            y_min = min(corner.y for corner in corners)
            x_max = max(corner.x for corner in corners)
            y_max = max(corner.y for corner in corners)
            return (x_min, y_min, x_max, y_max)

        max_bbox_width, max_bbox_height = self.max_level_bbox_dim()

        # determine how many levels should be placed in one row
        max_pixel_width_per_level = max((max_bbox_width + margin_width_in_meter) / resolution, 1)
        max_level_counts_per_row = max(1, int(preferred_pixel_width // max_pixel_width_per_level))

        level_counts_per_row = 0
        row_count = 0
        shifted_world = deepcopy(self)
        flattened_world = LevelDescription()
        level_origins: dict[str, tuple[float, float]] = {}
        for level_id, level in shifted_world.levels.items():
            x_min, y_min, _, _ = _level_bbox(level)
            target_x = level_counts_per_row * (max_bbox_width + margin_width_in_meter)
            target_y = -1 * row_count * (max_bbox_height + margin_height_in_meter)
            offset_x = target_x - x_min
            offset_y = target_y - y_min

            level.shift_all_positions(offset_x, offset_y)
            flattened_world.zones.extend(level.zones)
            level_origins[level_id] = (offset_x, offset_y)

            level_counts_per_row += 1
            if level_counts_per_row >= max_level_counts_per_row:
                row_count += 1
                level_counts_per_row = 0

        png = flattened_world.render(
            resolution,
            default_asset_bbox=default_asset_bbox,
            asset_color=asset_color,
            asset_name_color=asset_name_color,
        )[0]
        return png, level_origins

    def export(self, resolution: float = 0.05, extra_files: dict[str, bytes] | None = None, **kwargs: object) -> tarfile.TarFile:
        if extra_files is None:
            extra_files = {}
        files: dict[str, bytes] = {**extra_files}

        render_args: dict[str, typing.Any] = {"resolution": resolution}
        if "default_asset_bbox" in kwargs:
            render_args["default_asset_bbox"] = kwargs["default_asset_bbox"]
        if "asset_color" in kwargs:
            render_args["asset_color"] = kwargs["asset_color"]
        if "asset_name_color" in kwargs:
            render_args["asset_name_color"] = kwargs["asset_name_color"]

        for level_id, level in self.levels.items():
            level_desc = LevelDescription(zones=list(level.zones))
            level_yaml = yaml.safe_dump(converter.unstructure(level_desc), sort_keys=False)
            files[f'{level_id}/world.yaml'] = level_yaml.encode('utf-8')
            png, origin = level_desc.render(**render_args)
            files[f'{level_id}/map.png'] = png
            files[f'{level_id}/map.yaml'] = Map.generate_map_yaml(
                resolution=resolution,
                filename='map.png',
                origin=origin,
            ).encode('utf-8')

        with io.BytesIO() as tar_stream:
            with tarfile.open(mode='w', fileobj=tar_stream) as tarball:
                for filename, content in files.items():
                    info = tarfile.TarInfo(name=os.path.normpath(filename))
                    info.size = len(content)
                    tarball.addfile(tarinfo=info, fileobj=io.BytesIO(content))
            tar_stream.seek(0)
            return tarfile.open(fileobj=io.BytesIO(tar_stream.getvalue()))


# -- Zone geometry helpers ---------------------------------------------------


def _door_polygon(start: Position, end: Position) -> list[Position]:
    dx = end.x - start.x
    dy = end.y - start.y
    length = math.sqrt(dx * dx + dy * dy)
    if length > 0:
        nx, ny = -dy / length, dx / length
    else:
        nx, ny = 0.0, 1.0
    t = 0.3  # half-thickness
    return [
        Position(start.x - nx * t, start.y - ny * t),
        Position(end.x - nx * t, end.y - ny * t),
        Position(end.x + nx * t, end.y + ny * t),
        Position(start.x + nx * t, start.y + ny * t),
    ]


_ELEVATOR_DOORWAY_DEPTH = 0.3

# robot-agnostic clearance above which a static entity is treated as overhead and not rasterized
_PASSAGE_CLEARANCE = 2.0


def _door_axis(door_side: str) -> tuple[tuple[float, float], tuple[float, float]]:
    return {
        '+x': ((1.0, 0.0), (0.0, 1.0)),
        '-x': ((-1.0, 0.0), (0.0, 1.0)),
        '+y': ((0.0, 1.0), (1.0, 0.0)),
        '-y': ((0.0, -1.0), (1.0, 0.0)),
    }[door_side]


def _elevator_doorway_corners(elevator: Elevator) -> list[Position]:
    outward, tangent = _door_axis(elevator.door_side)
    hx, hy = elevator.size[0] / 2.0, elevator.size[1] / 2.0
    out_extent = hx if outward[0] != 0 else hy
    tan_extent = hy if outward[0] != 0 else hx
    inner_cx = elevator.position.x + outward[0] * out_extent
    inner_cy = elevator.position.y + outward[1] * out_extent
    outer_cx = inner_cx + outward[0] * _ELEVATOR_DOORWAY_DEPTH
    outer_cy = inner_cy + outward[1] * _ELEVATOR_DOORWAY_DEPTH
    return [
        Position(inner_cx - tangent[0] * tan_extent, inner_cy - tangent[1] * tan_extent),
        Position(outer_cx - tangent[0] * tan_extent, outer_cy - tangent[1] * tan_extent),
        Position(outer_cx + tangent[0] * tan_extent, outer_cy + tangent[1] * tan_extent),
        Position(inner_cx + tangent[0] * tan_extent, inner_cy + tangent[1] * tan_extent),
    ]


def _render_door_polygons(door: Door) -> list:
    import shapely

    return [shapely.Polygon(door.corners)]


def _render_elevator_polygons(elevator: Elevator) -> list:
    import shapely

    return [shapely.Polygon(elevator.cabin_corners()), shapely.Polygon(_elevator_doorway_corners(elevator))]


class MultiLevelWorldView(PathView):
    @property
    def scenario(self) -> type[Identifier[ScenarioView]]:
        class ScenarioIdentifier(Identifier[ScenarioView]):
            @classmethod
            def listall(cls, **kwargs: object) -> Iterator[Self]:
                scenarios_dir = self.path / 'scenarios'
                if not scenarios_dir.is_dir():
                    yield from ()
                    return
                yield from (cls(entry.name) for entry in os.scandir(scenarios_dir) if entry.is_dir())

            def load(self, path: Path, /, **kwargs: object) -> ScenarioView:
                del kwargs
                return ScenarioView(path)

        ScenarioIdentifier.use(FallbackResolver(ScenarioIdentifier, self.path / 'scenarios'))
        return ScenarioIdentifier

    @property
    def world_path(self) -> Path:
        return self.path / 'world.yaml'

    def load(self, validate: bool = True, level_filter: set[str] | None = None) -> WorldDescription:
        """Load the WorldDescription from disk.

        Enumerates per-level subdirectories containing ``world.yaml``. Each
        subdirectory name is treated as the level id. Falls back to legacy
        single-file ``world.yaml`` at the root when no per-level subdirs exist
        (backward compat: synthesizes a virtual level ``'0'``).

        Args:
            validate: Run WorldDescription.validate() after loading.
            level_filter: When provided, only load levels whose id is in this
                set. Levels not in the set are skipped entirely (selective load).
        """
        level_subdirs = sorted(entry for entry in self.path.iterdir() if entry.is_dir() and (entry / 'world.yaml').exists() and entry.name not in {'scenarios', 'assets', 'map'})

        if level_subdirs:
            levels: dict[str, Level] = {}
            for subdir in level_subdirs:
                level_id = subdir.name
                if level_filter is not None and level_id not in level_filter:
                    continue
                with open(subdir / 'world.yaml', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                level_desc = converter.structure(data, LevelDescription)
                level = Level.from_level_description(level_desc)
                level_elevators: list[ElevatorDescriptor] = []
                for elevator in level_desc.all_elevators:
                    level_elevator = ElevatorDescriptor.from_elevator(elevator)
                    if elevator.destination:
                        for dest_str in WorldDescription._parse_destinations(elevator.destination):
                            if '.' in dest_str:
                                dest_level_id, dest_name = dest_str.split('.', 1)
                            else:
                                dest_level_id = ''
                                dest_name = dest_str
                            level_elevator.add_destination(dest_name, dest_level_id)
                    level_elevators.append(level_elevator)
                level.levelElevators = level_elevators
                levels[level_id] = level
            world_description = WorldDescription(levels=levels)

        elif self.world_path.exists():
            with open(self.world_path, encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if isinstance(data, dict) and 'zones' in data and 'levels' not in data:
                level_desc = converter.structure(data, LevelDescription)
                level = Level.from_level_description(level_desc)
                level_elevators = []
                for elevator in level_desc.all_elevators:
                    level_elevator = ElevatorDescriptor.from_elevator(elevator)
                    if elevator.destination:
                        for dest_str in WorldDescription._parse_destinations(elevator.destination):
                            if '.' in dest_str:
                                dest_level_id, dest_name = dest_str.split('.', 1)
                            else:
                                dest_level_id = ''
                                dest_name = dest_str
                            level_elevator.add_destination(dest_name, dest_level_id)
                    level_elevators.append(level_elevator)
                level.levelElevators = level_elevators
                world_description = WorldDescription(levels={'0': level})
            else:
                world_description = converter.structure(data, WorldDescription)

        else:
            raise FileNotFoundError(f'could not find per-level subdirs or world.yaml in {self.path}')

        if validate:
            world_description.validate(loaded_level_ids={entry.name for entry in level_subdirs} if level_subdirs else None)
        return world_description

    def save(self, world_description: WorldDescription, map_only: bool = False, validate: bool = True, **kwargs: object) -> Path:
        if validate:
            world_description.validate()

        os.makedirs(self.path, exist_ok=True)
        tarball = world_description.export(**kwargs)

        if not hasattr(tarfile, 'data_filter'):
            tarball.extractall(self.path)
        else:
            _filter = tarfile.data_filter
            if map_only:

                def map_only_filter(member: tarfile.TarInfo, destpath: str) -> tarfile.TarInfo | None:
                    if not tarfile.data_filter(member, destpath):
                        return None
                    parts = member.name.split('/', 1)
                    if len(parts) == 2 and parts[1] in {'map.png', 'map.yaml'}:
                        return member
                    return None

                _filter = map_only_filter

            tarball.extractall(self.path, filter=_filter)

        # tar members carry epoch-0 mtime; stamp extracted files with the write time so a re-saved world is detectable
        now = time.time()
        for member in tarball.getmembers():
            extracted = os.path.join(self.path, member.name)
            if member.isfile() and os.path.exists(extracted):
                os.utime(extracted, (now, now))

        return self.path

    def level_origins(self) -> dict[str, tuple[float, float]] | None:
        """Return per-level offsets used to flatten all levels into a single map frame.

        Computed from the world geometry rather than from any on-disk artifact,
        so no pre-rendered map.yaml is required. Returns None for single-level worlds.
        """
        world = self.load(validate=False)
        if len(world.levels) <= 1:
            return None
        _, origins = world._render_whole_with_origins()
        return origins


class WorldIdentifier(Identifier[MultiLevelWorldView]):
    @classmethod
    def listall(cls, **kwargs: object) -> Iterator[Self]:
        del kwargs
        seen: set[str] = set()
        for root in [*_world_search_roots(), ASS_DIR / 'worlds']:
            if not root.is_dir():
                continue
            for name in os.listdir(root):
                if name.lower() != 'readme.md' and name not in seen:
                    seen.add(name)
                    yield WorldIdentifier(name)

    def load(self, path: Path, /, **kwargs: object) -> MultiLevelWorldView:
        del kwargs
        return MultiLevelWorldView(path)

    @classmethod
    def parse(cls, raw: str) -> 'tuple[str, set[str] | None]':
        """Parse a world identifier with optional level filter.

        ``'myworld'`` returns ``('myworld', None)``.
        ``'myworld[1,3]'`` returns ``('myworld', {'1', '3'})``.
        """
        if '[' not in raw:
            return raw, None
        name, _, rest = raw.partition('[')
        ids = {s.strip() for s in rest.rstrip(']').split(',') if s.strip()}
        return name, ids or None


def _world_search_roots() -> list[Path]:
    """Extra world roots from ARENA_WORLD_PATH (colon-separated), searched before the canonical tree."""
    return [Path(p) for p in os.environ.get('ARENA_WORLD_PATH', '').split(':') if p]


WorldIdentifier.use(*(SimplePathResolver(WorldIdentifier, root) for root in _world_search_roots()))
WorldIdentifier.use(FallbackResolver(WorldIdentifier, ASS_DIR / 'worlds'))
