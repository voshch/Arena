"""World format 2: one world of levels, zones as regions, level-scope structure and things."""

from __future__ import annotations

import itertools
import logging
import math
import typing
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path

import attrs
import numpy as np
from typing_extensions import Self

from arena_simulation_setup.shared import (
    LIGHT_FIXTURES,
    Ceiling,
    CeilingLights,
    Door,
    DynamicObstacle,
    Elevator,
    Floor,
    Light,
    Obstacle,
    Schedule,
    SemanticCfg,
    Signal,
    Sound,
    Stairs,
    Wall,
)
from arena_simulation_setup.shared.entities import Waypoint
from arena_simulation_setup.shared.semantics import parse_semantics
from arena_simulation_setup.shared.world import STOREY, clear_door_pockets
from arena_simulation_setup.tree import Identifier
from arena_simulation_setup.tree.assets.Material import Material, MaterialIdentifier
from arena_simulation_setup.utils.cattrs import ArenaConverter, converter
from arena_simulation_setup.utils.geometry import Orientation, PointResolver, Pose, Position

from .Map import Map
from .Scenario import RegionAssignment

if typing.TYPE_CHECKING:
    import shapely

    from arena_simulation_setup.kg import Scene

FORMAT = 'arena-world/2'
LEGACY_FORMAT = 'arena-world/1'
MICROPHONE_PLACEMENT_TOLERANCE_M = 0.05
WORLD_LIGHTING_MODES = ('authored', 'auto')
AUTO_CEILING_LIGHTS = CeilingLights(fixture='panel', spacing=2.4, lumens=7200.0, cct_K=6500.0)

_ON_LINE_M = 0.01
_MIN_PIECE_M = 1e-4
_ADJACENT_M = 0.3
_PASSAGE_CLEARANCE = 2.0
_ELEVATOR_DOORWAY_DEPTH = 0.3


def _material(value: object) -> MaterialIdentifier:
    return MaterialIdentifier.converter(value)


def _optional_material(value: object) -> MaterialIdentifier | None:
    return None if value is None else _material(value)


@attrs.define
class WorldMicrophone:
    zone: str = attrs.field(converter=lambda value: str(value).strip())
    placement: str = attrs.field(converter=lambda value: str(value).strip().lower())
    position: Position = attrs.field(converter=Position.instance_or(Position.parse))
    frame: str = attrs.field(converter=lambda value: str(value).strip().strip('/'), default='map')
    index: int = attrs.field(converter=int, default=1)

    @property
    def listener_id(self) -> str:
        return f'microphone:zone:{self.zone}:{self.placement}:{self.index}'


def _ceiling_lights(value: object) -> CeilingLights | None:
    if value is None or isinstance(value, CeilingLights):
        return value
    return converter.structure(value, CeilingLights)


@attrs.define
class ZoneCeiling:
    """A zone's ceiling: its top height (derived from the walls when None), material, shadows and light rig."""

    height: float | None = attrs.field(default=None, converter=lambda value: None if value is None else float(value))
    material: MaterialIdentifier = attrs.field(converter=_material, factory=lambda: Material.default('ceiling'))
    cast_shadows: bool = False
    lights: CeilingLights | None = attrs.field(default=None, converter=lambda value: _ceiling_lights(value))


@attrs.define
class ZoneWalls:
    """Walls drawn along a zone's polygon, leaving open the edges it shares with `open_to`."""

    open_to: list[str] = attrs.field(factory=list, converter=lambda value: [str(name) for name in value])
    kind: str = ''
    material: MaterialIdentifier | None = attrs.field(default=None, converter=_optional_material)


@attrs.define
class WallPath:
    """A polyline of walls, closed back to its first point when `closed`."""

    path: list[Position] = attrs.field(converter=lambda value: [Position.converter(point) if isinstance(point, Position) else Position.parse(point) for point in value])
    closed: bool = False
    kind: str = ''
    material: MaterialIdentifier | None = attrs.field(default=None, converter=_optional_material)

    def __attrs_post_init__(self) -> None:
        if len(self.path) < 2:
            raise ValueError('a wall path needs at least two points')

    def segments(self) -> list[Wall]:
        points = [*self.path, self.path[0]] if self.closed and len(self.path) > 2 else self.path
        return [Wall(start=a, end=b, kind=self.kind, material=self.material) for a, b in itertools.pairwise(points)]


@attrs.define
class Zone:
    """A named region of a level: its polygon, floor, ceiling, outline walls and runtime semantics."""

    name: str
    polygon: list[Position] = attrs.field(factory=list, converter=lambda value: _open_ring([point if isinstance(point, Position) else Position.parse(point) for point in value]))
    floor: MaterialIdentifier | None = attrs.field(factory=lambda: Material.default('floor'), converter=_optional_material)
    ceiling: ZoneCeiling | None = attrs.field(factory=ZoneCeiling)
    walls: ZoneWalls | None = None
    semantics: list[SemanticCfg] = attrs.field(factory=list, converter=parse_semantics)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = [corner.x for corner in self.polygon]
        ys = [corner.y for corner in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def floor_patch(self) -> Floor:
        x_min, y_min, x_max, y_max = self.bounds
        return Floor(name=self.name, pos=Position(x=(x_min + x_max) / 2, y=(y_min + y_max) / 2), x_length=x_max - x_min, y_length=y_max - y_min, material=self.floor if self.floor is not None else Material.default('floor'))

    def shape(self) -> shapely.Polygon:
        import shapely

        return shapely.Polygon([(corner.x, corner.y) for corner in self.polygon])


def _cut_rect(rect: tuple[float, float, float, float], holes: typing.Sequence[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Axis-aligned rectangle `rect` minus the `holes`, as rectangles."""
    pieces = [rect]
    for hx0, hy0, hx1, hy1 in holes:
        cut: list[tuple[float, float, float, float]] = []
        for x0, y0, x1, y1 in pieces:
            if hx0 >= x1 or hx1 <= x0 or hy0 >= y1 or hy1 <= y0:
                cut.append((x0, y0, x1, y1))
                continue
            ix0, iy0, ix1, iy1 = max(x0, hx0), max(y0, hy0), min(x1, hx1), min(y1, hy1)
            candidates = [(x0, y0, ix0, y1), (ix1, y0, x1, y1), (ix0, y0, ix1, iy0), (ix0, iy1, ix1, y1)]
            cut.extend(piece for piece in candidates if piece[2] - piece[0] > 1e-6 and piece[3] - piece[1] > 1e-6)
        pieces = cut
    return pieces


def _open_ring(points: list[Position]) -> list[Position]:
    if len(points) > 3 and math.isclose(points[0].x, points[-1].x, abs_tol=1e-9) and math.isclose(points[0].y, points[-1].y, abs_tol=1e-9):
        return points[:-1]
    return points


def _place_on(light: Light, entity: Obstacle) -> Light:
    """Place a light at its offset on a static entity, the offset scaled and carried by the entity pose, on the entity's level or zone."""
    offset = light.offset
    scale = entity.scale
    if scale is not None:
        offset = Pose(position=Position(offset.position.x * scale.x, offset.position.y * scale.y, offset.position.z * scale.z), orientation=offset.orientation)
    placed = entity.pose * offset
    return attrs.evolve(
        light,
        attach='',
        owner=entity.name,
        pose=placed if light.spec.directed else Pose(position=placed.position),
        offset=Pose(),
        level=entity.level,
        extra={**light.extra, 'placement_zone': entity.placement_zone} if entity.placement_zone else dict(light.extra),
    )


def anchor_light(light: Light, statics: Iterable[Obstacle]) -> Light:
    """Resolve a light attached to a static entity onto that entity."""
    if not light.attach:
        return light
    matches = [entity for entity in statics if entity.name == light.attach]
    if len(matches) != 1:
        raise ValueError(f"light {light.name!r} is attached to {'an ambiguous' if matches else 'an unknown'} static entity {light.attach!r}")
    return _place_on(light, matches[0])


async def entity_lights(entity: Obstacle) -> list[Light]:
    """Lights the object annotation of an entity carries, placed on the entity."""
    from arena_simulation_setup.shared import object_light

    try:
        view = await entity.model.resolve()
    except FileNotFoundError:
        return []
    entries = (view.annotation or {}).get('lights') or []
    return [_place_on(object_light(entity.name, entry, entity.light), entity) for entry in entries]


def _param(a: Position, b: Position, p: Position) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    length2 = dx * dx + dy * dy
    return 0.0 if length2 == 0.0 else ((p.x - a.x) * dx + (p.y - a.y) * dy) / length2


def _off_line(a: Position, b: Position, p: Position) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    length = math.hypot(dx, dy)
    return math.hypot(p.x - a.x, p.y - a.y) if length == 0.0 else abs((p.x - a.x) * dy - (p.y - a.y) * dx) / length


def _on_line(a: Position, b: Position, *points: Position) -> bool:
    return all(_off_line(a, b, p) <= _ON_LINE_M for p in points)


def _at(a: Position, b: Position, t: float) -> Position:
    return Position(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z))


def _subtract(keep: list[tuple[float, float]], cut: tuple[float, float], length: float) -> list[tuple[float, float]]:
    c0, c1 = sorted(cut)
    result: list[tuple[float, float]] = []
    for k0, k1 in keep:
        if c1 <= k0 or c0 >= k1 or (min(c1, k1) - max(c0, k0)) * length <= _MIN_PIECE_M:
            result.append((k0, k1))
            continue
        if (c0 - k0) * length > _MIN_PIECE_M:
            result.append((k0, c0))
        if (k1 - c1) * length > _MIN_PIECE_M:
            result.append((c1, k1))
    return result


def _cut(wall: Wall, spans: Iterable[tuple[Position, Position]]) -> list[Wall]:
    """The pieces of a wall left after removing every collinear span lying on it."""
    length = math.hypot(wall.end.x - wall.start.x, wall.end.y - wall.start.y)
    if length == 0.0:
        return [wall]
    keep = [(0.0, 1.0)]
    for c, d in spans:
        if _on_line(wall.start, wall.end, c, d):
            keep = _subtract(keep, (_param(wall.start, wall.end, c), _param(wall.start, wall.end, d)), length)
    if keep == [(0.0, 1.0)]:
        return [wall]
    return [attrs.evolve(wall, start=_at(wall.start, wall.end, t0), end=_at(wall.start, wall.end, t1)) for t0, t1 in keep]


def _style(wall: Wall) -> tuple:
    material = wall.material
    return (wall.kind, None if material is None else (material.shortname, tuple(sorted(material.modifiers.items()))))


def _line_key(wall: Wall) -> tuple[int, int]:
    dx, dy = wall.end.x - wall.start.x, wall.end.y - wall.start.y
    theta = math.atan2(dy, dx) % math.pi
    offset = wall.start.x * math.sin(theta) - wall.start.y * math.cos(theta)
    return round(theta * 100), round(offset * 10)


def _merge(walls: list[Wall]) -> list[Wall]:
    """Drop repeated walls and merge plain walls of one style that overlap along one line."""
    seen: set[tuple] = set()
    unique: list[Wall] = []
    for wall in walls:
        ends = sorted(((round(wall.start.x, 6), round(wall.start.y, 6), round(wall.start.z, 6)), (round(wall.end.x, 6), round(wall.end.y, 6), round(wall.end.z, 6))))
        key = (*ends, _style(wall), wall.top, wall.bottom, wall.height)
        if key in seen:
            continue
        seen.add(key)
        unique.append(wall)
    groups: list[list[Wall]] = []
    buckets: dict[tuple[int, int], list[int]] = {}
    order: list[tuple[int, Wall | None]] = []
    for wall in unique:
        if wall.kind or wall.top is not None or wall.height is not None or (wall.start.x, wall.start.y) == (wall.end.x, wall.end.y):
            order.append((-1, wall))
            continue
        theta_key, offset_key = _line_key(wall)
        found = None
        for dt, do in itertools.product((-1, 0, 1), (-1, 0, 1)):
            for index in buckets.get(((theta_key + dt) % 315, offset_key + do), ()):
                ref = groups[index][0]
                if _style(ref) == _style(wall) and math.isclose(ref.start.z, wall.start.z, abs_tol=1e-6) and _on_line(ref.start, ref.end, wall.start, wall.end):
                    found = index
                    break
            if found is not None:
                break
        if found is None:
            found = len(groups)
            groups.append([])
            buckets.setdefault((theta_key % 315, offset_key), []).append(found)
            order.append((found, None))
        groups[found].append(wall)
    merged: dict[int, list[Wall]] = {}
    for index, group in enumerate(groups):
        ref = group[0]
        length = math.hypot(ref.end.x - ref.start.x, ref.end.y - ref.start.y)
        spans = sorted(((min(t0, t1), max(t0, t1), position, wall) for position, wall in enumerate(group) for t0, t1 in [(_param(ref.start, ref.end, wall.start), _param(ref.start, ref.end, wall.end))]), key=lambda span: (span[0], span[2]))
        out: list[Wall] = []
        cluster: list[tuple[float, float, int, Wall]] = []
        for span in [*spans, None]:
            if span is not None and cluster and (span[0] - max(item[1] for item in cluster)) * length < -_MIN_PIECE_M:
                cluster.append(span)
                continue
            if len(cluster) == 1:
                out.append(cluster[0][3])
            elif cluster:
                t0, t1 = cluster[0][0], max(item[1] for item in cluster)
                widest = next((item for item in cluster if math.isclose(item[0], t0, abs_tol=1e-9) and math.isclose(item[1], t1, abs_tol=1e-9)), None)
                out.append(widest[3] if widest is not None else attrs.evolve(cluster[0][3], start=_at(ref.start, ref.end, t0), end=_at(ref.start, ref.end, t1)))
            cluster = [span] if span is not None else []
        merged[index] = out
    result: list[Wall] = []
    for index, wall in order:
        if index < 0:
            result.append(typing.cast(Wall, wall))
        else:
            result.extend(merged[index])
    return result


def _covers_middle(shape: shapely.Polygon, start: Position, end: Position) -> bool:
    import shapely

    return bool(shape.covers(shapely.Point((start.x + end.x) / 2, (start.y + end.y) / 2)))


def _shared_spans(a: Position, b: Position, polygons: Iterable[list[Position]]) -> list[tuple[Position, Position]]:
    """Edges of the given polygons that lie on the line through a and b."""
    return [(c, d) for polygon in polygons for c, d in zip(polygon, [*polygon[1:], *polygon[:1]], strict=True) if _on_line(a, b, c, d)]


@attrs.define
class Level:
    """One level of a world: zones as regions, plus level-scope structure and things."""

    id: str = ''
    zones: list[Zone] = attrs.field(factory=list)
    walls: list[WallPath] = attrs.field(factory=list)
    doors: list[Door] = attrs.field(factory=list, converter=clear_door_pockets)
    elevators: list[Elevator] = attrs.field(factory=list)
    stairs: list[Stairs] = attrs.field(factory=list)
    elevation: float | None = None
    objects: list[Obstacle] = attrs.field(factory=list)
    humans: list[DynamicObstacle] = attrs.field(factory=list)
    lights: list[Light] = attrs.field(factory=list)
    sounds: list[Sound] = attrs.field(factory=list)
    microphones: list[WorldMicrophone] = attrs.field(factory=list)
    pedestrian_walls: bool = True
    source: str = ''
    prefix: str = ''

    @property
    def all_walls(self) -> list[Wall]:
        """Every wall segment of the level: paths and zone outlines, repeats merged, cut open at the doors."""
        by_name = {zone.name: zone for zone in self.zones}
        raw: list[Wall] = [segment for path in self.walls for segment in path.segments()]
        raw.extend(wall for stair in self.stairs for wall in stair.walls())
        for zone in self.zones:
            if zone.walls is None or len(zone.polygon) < 2:
                continue
            open_polygons = [by_name[name].polygon for name in zone.walls.open_to if name in by_name]
            for a, b in zip(zone.polygon, [*zone.polygon[1:], *zone.polygon[:1]], strict=True):
                edge = Wall(start=a, end=b, kind=zone.walls.kind, material=zone.walls.material)
                raw.extend(_cut(edge, _shared_spans(a, b, open_polygons)))
        door_spans = [(door.start, door.end) for door in self.doors]
        return [piece for wall in _merge(raw) for piece in _cut(wall, door_spans)]

    def adjacent_zones(self, start: Position, end: Position) -> list[Zone]:
        """Zones whose polygon, widened by a margin, holds the middle of a segment."""
        return [zone for zone, shape in self._widened() if _covers_middle(shape, start, end)]

    def _widened(self) -> list[tuple[Zone, shapely.Polygon]]:
        import shapely

        result = []
        for zone in self.zones:
            if len(zone.polygon) >= 3:
                shape = zone.shape().buffer(_ADJACENT_M)
                shapely.prepare(shape)
                result.append((zone, shape))
        return result

    def with_lighting(self, mode: str) -> Self:
        """The level under a world.lighting mode: auto gives every zone with a ceiling and no lights of its own the calibrated ceiling rig."""
        if mode not in WORLD_LIGHTING_MODES:
            raise ValueError(f'world.lighting {mode!r} is not one of {", ".join(WORLD_LIGHTING_MODES)}')
        if mode == 'authored':
            return self
        import shapely

        placed = [shapely.Point(light.position.x, light.position.y) for light in (self._anchor_light(light) for light in self.lights) if light.pose is not None and not light.spec.ambient]
        zones = []
        for zone in self.zones:
            ceiling = zone.ceiling
            if ceiling is not None and ceiling.material.name and ceiling.lights is None and len(zone.polygon) >= 3 and not any(zone.shape().covers(point) for point in placed):
                zone = attrs.evolve(zone, ceiling=attrs.evolve(ceiling, lights=AUTO_CEILING_LIGHTS))
            zones.append(zone)
        return attrs.evolve(self, zones=zones)

    def _has_ceiling(self, zone: Zone) -> bool:
        return zone.ceiling is not None and bool(zone.ceiling.material.name)

    async def ceiling_tops(self, walls: list[Wall] | None = None) -> dict[str, float]:
        """Top of every zone ceiling by zone name: its height, else the tallest wall around or inside the zone, at least 2 m."""
        walls = self.all_walls if walls is None else walls
        wall_tops: dict[int, float] = {}
        tops: dict[str, float] = {}
        for zone, shape in [(zone, None) for zone in self.zones if self._has_ceiling(zone) and len(zone.polygon) < 3] + [(zone, shape) for zone, shape in self._widened() if self._has_ceiling(zone)]:
            ceiling = typing.cast(ZoneCeiling, zone.ceiling)
            if ceiling.height is not None:
                tops[zone.name] = ceiling.height
                continue
            z = 2.0
            if shape is not None:
                for index, wall in enumerate(walls):
                    if not _covers_middle(shape, wall.start, wall.end):
                        continue
                    if index not in wall_tops:
                        segments, _ = await wall.assets()
                        wall_tops[index] = max((segment.start.z + segment.height for segment in segments), default=0.0)
                    z = max(z, wall_tops[index])
            tops[zone.name] = z
        return tops

    async def ceiling_z(self, zone: Zone) -> float:
        """Top of a zone's ceiling."""
        return (await self.ceiling_tops()).get(zone.name, 2.0)

    async def closed_walls(self) -> list[Wall]:
        """Walls the simulator builds: those of the level, each without a set top or height raised to the highest ceiling of a zone it borders or stands in, and the stairs' open walls."""
        walls = self.all_walls
        tops = await self.ceiling_tops(walls)
        widened = [(zone, shape) for zone, shape in self._widened() if zone.name in tops]
        result: list[Wall] = []
        for wall in walls:
            covering = [tops[zone.name] for zone, shape in widened if _covers_middle(shape, wall.start, wall.end)] if wall.top is None and wall.height is None else []
            result.append(attrs.evolve(wall, top=max(covering)) if covering else wall)
        return result + [wall for stair in self.stairs for wall in stair.open_walls()]

    async def door_lintels(self) -> list[Wall]:
        """Visual-only wall pieces above the doors that are lower than a ceiling they open under."""
        walls = self.all_walls
        tops = await self.ceiling_tops(walls)
        widened = [(zone, shape) for zone, shape in self._widened() if zone.name in tops]
        result: list[Wall] = []
        for door in self.doors:
            covering = [tops[zone.name] for zone, shape in widened if _covers_middle(shape, door.start, door.end)]
            if not covering:
                continue
            top = max(covering)
            bottom = door.start.z + door.height
            if bottom >= top - 1e-6:
                continue
            around = next((wall for wall in walls if _on_line(wall.start, wall.end, door.start, door.end)), None)
            result.append(Wall(start=door.start, end=door.end, kind='', material=around.material if around is not None and not around.kind else None, top=top - door.start.z, bottom=door.height))
        return result

    @property
    def all_doors(self) -> list[Door]:
        return list(self.doors)

    @property
    def all_elevators(self) -> list[Elevator]:
        return list(self.elevators)

    @property
    def all_sounds(self) -> list[Sound]:
        return list(self.sounds)

    @property
    def all_floors(self) -> list[Floor]:
        """Zone floor rectangles, cut around the wells and escalator pits of the level's stairs."""
        holes = [hole for stair in self.stairs for hole in stair.floor_openings()]
        floors: list[Floor] = []
        for zone in self.zones:
            if zone.floor is None or not zone.floor.name or not zone.polygon:
                continue
            patch = zone.floor_patch
            pieces = _cut_rect(zone.bounds, holes)
            floors.extend(attrs.evolve(patch, name=patch.name if len(pieces) == 1 else f'{patch.name}_{i}', pos=Position(x=(x0 + x1) / 2, y=(y0 + y1) / 2), x_length=x1 - x0, y_length=y1 - y0) for i, (x0, y0, x1, y1) in enumerate(pieces))
        return floors

    async def all_ceilings(self) -> list[Ceiling]:
        tops = await self.ceiling_tops()
        holes = [hole for stair in self.stairs for hole in stair.ceiling_openings()]
        result: list[Ceiling] = []
        for zone in self.zones:
            if not self._has_ceiling(zone) or not zone.polygon:
                continue
            ceiling = typing.cast(ZoneCeiling, zone.ceiling)
            pieces = _cut_rect(zone.bounds, holes)
            result.extend(
                Ceiling(
                    name=zone.name if len(pieces) == 1 else f'{zone.name}_{i}',
                    pos=Position(x=(x_min + x_max) / 2, y=(y_min + y_max) / 2),
                    x_length=x_max - x_min,
                    y_length=y_max - y_min,
                    z=tops[zone.name],
                    cast_shadows=ceiling.cast_shadows,
                    material=ceiling.material,
                )
                for i, (x_min, y_min, x_max, y_max) in enumerate(pieces)
            )
        return result

    def _ceiling_rig(self, zone: Zone, cfg: CeilingLights, ceiling_z: float) -> Light:
        import shapely

        spec = LIGHT_FIXTURES[cfg.fixture]
        polygon = zone.shape()
        x_min, y_min, x_max, y_max = zone.bounds
        nx = max(1, int((x_max - x_min) // cfg.spacing))
        ny = max(1, int((y_max - y_min) // cfg.spacing))
        z = ceiling_z - 0.02
        fixtures: list[Position] = []
        for iy in range(ny):
            y = (y_min + y_max) / 2 + (iy - (ny - 1) / 2) * cfg.spacing
            for ix in range(nx):
                x = (x_min + x_max) / 2 + (ix - (nx - 1) / 2) * cfg.spacing
                if spec.shape == 'rect':
                    footprint = shapely.box(x - spec.x_length / 2, y - spec.y_length / 2, x + spec.x_length / 2, y + spec.y_length / 2)
                else:
                    footprint = shapely.Point(x, y).buffer(spec.radius)
                if polygon.covers(footprint):
                    fixtures.append(Position(x, y, z))
        if not fixtures:
            point = polygon.representative_point()
            fixtures.append(Position(point.x, point.y, z))
        return Light(
            name=zone.name,
            fixture=cfg.fixture,
            lumens=cfg.lumens,
            cct_K=cfg.cct_K,
            cast_shadows=cfg.cast_shadows,
            light_on=cfg.light_on,
            lit=cfg.lit,
            dim=cfg.dim,
            dead_fraction=cfg.dead_fraction,
            rig=True,
            fixtures=fixtures,
            fixture_ranks=[zlib.crc32(f'{zone.name}:{i}'.encode()) / 2**32 for i in range(len(fixtures))],
        )

    def _anchor_light(self, light: Light) -> Light:
        return anchor_light(light, self.objects)

    @property
    def authored_lights(self) -> list[Light]:
        return list(self.lights)

    async def all_lights(self) -> list[Light]:
        result: list[Light] = [self._anchor_light(light) for light in self.lights]
        rigged = [zone for zone in self.zones if zone.ceiling is not None and zone.ceiling.lights is not None]
        tops = await self.ceiling_tops() if rigged else {}
        for zone in rigged:
            result.append(self._ceiling_rig(zone, typing.cast(CeilingLights, typing.cast(ZoneCeiling, zone.ceiling).lights), tops.get(zone.name, 2.0)))
        names = {light.name for light in result}
        for entity in self.objects:
            for light in await entity_lights(entity):
                if light.name in names:
                    raise ValueError(f'object light {light.name!r} collides with another light of that name')
                names.add(light.name)
                result.append(light)
        return result

    @property
    def all_static_entities(self) -> list[Obstacle]:
        return list(self.objects)

    @property
    def all_dynamic_entities(self) -> list[DynamicObstacle]:
        return list(self.humans)

    def refs(self) -> dict[str, list[str]]:
        """The semantic id kind of each named list, with its names."""
        return {
            'room': [zone.name for zone in self.zones],
            'door': [door.name for door in self.doors],
            'elevator': [elevator.name for elevator in self.elevators],
            'stair': [stair.name for stair in self.stairs],
            'obj': [entity.name for entity in self.objects],
            'light': [light.name for light in self.lights],
            'sound': [sound.name for sound in self.sounds],
        }

    def prefixed(self, prefix: str) -> Level:
        """A copy with `prefix` before every name of the level and every reference to one."""
        zones = {zone.name for zone in self.zones}
        objects = {entity.name for entity in self.objects}
        elevators = {elevator.name for elevator in self.elevators}

        def among(names: set[str], name: str) -> str:
            return prefix + name if name in names else name

        return attrs.evolve(
            self,
            zones=[attrs.evolve(zone, name=prefix + zone.name, walls=None if zone.walls is None else attrs.evolve(zone.walls, open_to=[among(zones, name) for name in zone.walls.open_to])) for zone in self.zones],
            doors=[attrs.evolve(door, name=prefix + door.name) for door in self.doors],
            elevators=[attrs.evolve(elevator, name=prefix + elevator.name, destinations=[among(elevators, name) for name in elevator.destinations]) for elevator in self.elevators],
            stairs=[attrs.evolve(stair, name=prefix + stair.name, to=prefix + stair.to if stair.to else '', group=prefix + stair.group if stair.group else '') for stair in self.stairs],
            objects=[attrs.evolve(entity, name=prefix + entity.name) for entity in self.objects],
            humans=[attrs.evolve(human, name=prefix + human.name) for human in self.humans],
            lights=[attrs.evolve(light, name=prefix + light.name, attach=among(objects, light.attach)) for light in self.lights],
            sounds=[attrs.evolve(sound, name=prefix + sound.name, attach=among(objects, sound.attach)) for sound in self.sounds],
            microphones=[attrs.evolve(microphone, zone=among(zones, microphone.zone)) for microphone in self.microphones],
        )

    def shifted(self, dx: float, dy: float) -> Level:
        """A copy of the level moved by (dx, dy) in the level frame."""
        return typing.cast(Level, _shift(self, Position(dx, dy)))

    def validate_microphones(self) -> None:
        zones = {zone.name: zone for zone in self.zones}
        listener_ids: set[str] = set()
        for microphone in self.microphones:
            zone_name = microphone.zone.strip()
            placement = microphone.placement.strip()
            frame = microphone.frame.strip().strip('/')
            if not zone_name or ':' in zone_name:
                raise ValueError(f'invalid microphone zone {microphone.zone!r}')
            if not placement or ':' in placement:
                raise ValueError(f'invalid microphone placement {microphone.placement!r}')
            if not frame:
                raise ValueError(f'microphone in zone {zone_name!r} requires a TF frame')
            if microphone.index < 1:
                raise ValueError('microphone index must be at least 1')
            zone = zones.get(zone_name)
            if zone is None:
                raise ValueError(f'microphone references unknown zone {zone_name!r}')
            listener_id = microphone.listener_id
            if listener_id in listener_ids:
                raise ValueError(f'duplicate microphone {listener_id!r}')
            listener_ids.add(listener_id)
            if placement == 'ceiling' and not self._has_ceiling(zone):
                raise ValueError(f'ceiling microphone {listener_id!r} is in a zone without a ceiling')
            if frame != 'map':
                continue
            import shapely

            if not zone.shape().covers(shapely.Point(microphone.position.x, microphone.position.y)):
                raise ValueError(f'microphone {listener_id!r} is outside zone {zone_name!r}')
            height = zone.ceiling.height if zone.ceiling is not None else None
            if placement == 'ceiling' and height is not None and not math.isclose(microphone.position.z, height, abs_tol=MICROPHONE_PLACEMENT_TOLERANCE_M):
                raise ValueError(f'microphone {listener_id!r} z={microphone.position.z} does not match ceiling height {height}')

    def validate_levels(self) -> None:
        for entity in (*self.objects, *self.humans, *self.lights, *self.sounds):
            if entity.level:
                raise ValueError(f'{entity.name!r} sits on the level that lists it and takes no level')

    def validate_lights(self) -> None:
        names: set[str] = set()
        for light in self.lights:
            self._anchor_light(light)
            if light.name in names:
                raise ValueError(f'duplicate light {light.name!r}')
            names.add(light.name)
        for zone in self.zones:
            if zone.ceiling is None or zone.ceiling.lights is None:
                continue
            if not zone.ceiling.material.name:
                raise ValueError(f'zone {zone.name!r} has ceiling lights but no ceiling')
            if len(zone.polygon) < 3:
                raise ValueError(f'zone {zone.name!r} has ceiling lights but no polygon')
            if zone.name in names:
                raise ValueError(f'duplicate light {zone.name!r}')
            names.add(zone.name)

    def lookup_zone_polygon(self, name: str) -> list[Position] | None:
        """Look up a zone, door, elevator or stair by name and return its polygon vertices."""
        for zone in self.zones:
            if zone.name == name:
                return zone.polygon
        for door in self.doors:
            if door.name == name:
                return door_polygon(door.start, door.end)
        for elevator in self.elevators:
            if elevator.name == name:
                return elevator.cabin_corners()
        for stair in self.stairs:
            if name in (stair.name, stair.group):
                return stair.corners()
        return None

    def zone_ref_names(self) -> list[str]:
        """Every name resolvable by lookup_zone_polygon (zones, doors, elevators, stairs)."""
        return [*(zone.name for zone in self.zones), *(door.name for door in self.doors), *(elevator.name for elevator in self.elevators), *(stair.name for stair in self.stairs), *dict.fromkeys(stair.group for stair in self.stairs if stair.group)]

    def point_resolver(self, rng: np.random.Generator, *, is_valid: typing.Callable[[Position], bool] | None = None) -> PointResolver:
        """Resolver that samples a point inside a named zone/door/elevator polygon."""
        return PointResolver(lookup=self.lookup_zone_polygon, rng=rng, is_valid=is_valid, candidates=self.zone_ref_names)

    def zone_converter(self, rng: np.random.Generator, *, is_valid: typing.Callable[[Position], bool] | None = None) -> ArenaConverter:
        """A converter that resolves zone/door/elevator names to geometry: a sampled point for a pose or position, the polygon for a region `ref`."""
        lookup = self.lookup_zone_polygon
        base_region_hook = converter.get_structure_hook(RegionAssignment)

        def region_hook(v: object, t: type) -> RegionAssignment:
            if isinstance(v, dict) and 'ref' in v:
                v = dict(v)
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
        static_objects: Iterable[tuple[str, shapely.Polygon]] = (),
    ) -> dict[str, typing.Any]:
        import shapely
        import shapely.affinity

        map_kwargs: dict[str, typing.Any] = {
            'rooms': shapely.MultiPolygon([zone.shape() for zone in self.zones if len(zone.polygon) >= 3]),
            'doors': shapely.MultiPolygon([poly for door in self.doors for poly in render_door_polygons(door)]),
            'cabins': shapely.MultiPolygon([poly for elevator in self.elevators for poly in render_elevator_polygons(elevator)]),
            'walls': shapely.MultiLineString([[(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)] for wall in self.all_walls]),
            'padding': 5,
        }
        if asset_color is not None:
            footprints: list[tuple[str, shapely.Polygon]] = list(static_objects)
            for entity in self.objects:
                bbox = entity.extra.get('bbox') or default_asset_bbox
                if bbox is None:
                    continue
                (x_min, x_max), (y_min, y_max), *z_pair = bbox
                if z_pair:
                    (z_min, _z_max) = z_pair[0]
                    if entity.pose.position.z + z_min > _PASSAGE_CLEARANCE:
                        continue
                poly = shapely.box(x_min, y_min, x_max, y_max)
                poly = shapely.affinity.rotate(poly, entity.pose.orientation.to_yaw(), use_radians=True)
                poly = shapely.affinity.translate(poly, entity.pose.position.x, entity.pose.position.y)
                footprints.append((entity.name, poly))
            map_kwargs['static_objects'] = footprints
            map_kwargs['asset_color'] = asset_color
            map_kwargs['asset_name_color'] = asset_name_color
        return map_kwargs

    def render(
        self,
        resolution: float = 0.05,
        *,
        default_asset_bbox: tuple[tuple[float, float], tuple[float, float]] | None = None,
        asset_color: str | None = None,
        asset_name_color: str | None = None,
    ) -> tuple[bytes, tuple[float, float]]:
        """The level as a PNG map and its origin, static objects drawn only with asset_color and a known box."""
        return Map.generate_png(resolution=resolution, **self._rasterize_kwargs(default_asset_bbox=default_asset_bbox, asset_color=asset_color, asset_name_color=asset_name_color))

    def render_grid(
        self,
        resolution: float = 0.05,
        *,
        default_asset_bbox: tuple[tuple[float, float], tuple[float, float]] | None = None,
        asset_color: str | None = None,
        asset_name_color: str | None = None,
        static_objects: Iterable[tuple[str, shapely.Polygon]] = (),
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Like `render` but returns a uint8 numpy array (255=free, 0=occupied) + origin, drawing `static_objects` (name, map-frame polygon) alongside the entity boxes."""
        return Map.rasterize(resolution=resolution, **self._rasterize_kwargs(default_asset_bbox=default_asset_bbox, asset_color=asset_color, asset_name_color=asset_name_color, static_objects=static_objects))

    def render_map_files(
        self,
        level_origins: dict[str, tuple[float, float]] | None = None,
        resolution: float = 0.05,
        *,
        default_asset_bbox: tuple[tuple[float, float], tuple[float, float]] | None = None,
        asset_color: str | None = None,
        asset_name_color: str | None = None,
    ) -> tuple[bytes, str]:
        """Render this level to (map.png bytes, map.yaml text), embedding per-level origins under the `origins` key."""
        import yaml

        png_bytes, origin = self.render(resolution=resolution, default_asset_bbox=default_asset_bbox, asset_color=asset_color, asset_name_color=asset_name_color)
        map_yaml_text = Map.generate_map_yaml(resolution=resolution, filename='map.png', origin=origin)
        if level_origins:
            map_yaml_data = yaml.safe_load(map_yaml_text)
            map_yaml_data['origins'] = {fid: list(off) for fid, off in level_origins.items()}
            map_yaml_text = yaml.safe_dump(map_yaml_data)
        return png_bytes, map_yaml_text


def _shift(value: object, d: Position) -> object:
    """A copy of value with every level-frame position moved by d."""
    if isinstance(value, Waypoint):
        return value + d
    if isinstance(value, Position):
        return value + d
    if isinstance(value, Pose):
        return Pose(position=value.position + d, orientation=value.orientation)
    if isinstance(value, list):
        return [_shift(item, d) for item in value]
    if isinstance(value, (Wall, Light, Sound, WorldMicrophone, Door, Elevator, Stairs, Obstacle, DynamicObstacle, Zone, WallPath, Level)):
        skip = {'offset'}
        if isinstance(value, Light) and value.spec.ambient:
            skip.add('pose')
        if isinstance(value, WorldMicrophone) and value.frame != 'map':
            skip.add('position')
        changes = {field.name: _shift(getattr(value, field.name), d) for field in attrs.fields(type(value)) if field.init and field.name not in skip and _moves(getattr(value, field.name))}
        return attrs.evolve(value, **changes) if changes else value
    return value


def _moves(value: object) -> bool:
    if isinstance(value, (Position, Pose)):
        return True
    if isinstance(value, list):
        return any(_moves(item) for item in value)
    return isinstance(value, (Wall, Light, Sound, WorldMicrophone, Door, Elevator, Stairs, Obstacle, DynamicObstacle, Zone, WallPath))


@attrs.define
class World:
    """A world: its levels by id, world-scope schedules and signals, and the optional KG."""

    levels: dict[str, Level] = attrs.field(factory=dict)
    layout: typing.Literal['stacked', 'spread'] = 'spread'
    defaults: dict = attrs.field(factory=dict)
    schedules: list[Schedule] = attrs.field(factory=list)
    signals: list[Signal] = attrs.field(factory=list)
    kg: Scene | None = attrs.field(default=None, eq=False)
    sources: list[Path] = attrs.field(factory=list, eq=False)
    linked_kg: Scene | None = attrs.field(default=None, eq=False)
    linked_scope: list[Schedule | Signal] = attrs.field(factory=list, eq=False)

    @property
    def level_ids(self) -> list[str]:
        return list(self.levels)

    @property
    def all_levels(self) -> list[Level]:
        return list(self.levels.values())

    @property
    def all_static_entities(self) -> list[Obstacle]:
        return [obstacle for level in self.all_levels for obstacle in level.objects]

    @property
    def all_dynamic_entities(self) -> list[DynamicObstacle]:
        return [human for level in self.all_levels for human in level.humans]

    def get_level(self, level_id: str) -> Level | None:
        return self.levels.get(level_id)

    def elevator_levels(self) -> dict[str, str]:
        """The level of every elevator cabin, by cabin name."""
        return {elevator.name: level_id for level_id, level in self.levels.items() for elevator in level.elevators}

    def stair_levels(self) -> dict[str, str]:
        """The level of every stair half, by name."""
        return {stair.name: level_id for level_id, level in self.levels.items() for stair in level.stairs}

    def level_bounds(self) -> dict[str, tuple[float, float, float, float]]:
        """The level-local bounds (x_min, y_min, x_max, y_max) of every level's zones."""
        bounds: dict[str, tuple[float, float, float, float]] = {}
        for level_id, level in self.levels.items():
            corners = [corner for zone in level.zones for corner in zone.polygon]
            if corners:
                bounds[level_id] = (min(c.x for c in corners), min(c.y for c in corners), max(c.x for c in corners), max(c.y for c in corners))
        return bounds

    def elevation(self, level_id: str) -> float:
        """Floor height of a level: its own `elevation`, else a storey per level before it."""
        level = self.levels[level_id]
        return level.elevation if level.elevation is not None else list(self.levels).index(level_id) * STOREY

    def stack_offsets(self) -> dict[str, tuple[float, float, float]]:
        """Per level, the offset that stacks it at its elevation over the shared floor plan of a stacked world."""
        if self.layout != 'stacked':
            raise ValueError('a spread world gives every level a floor plan of its own, it stacks only with layout: stacked')
        return {level_id: (0.0, 0.0, self.elevation(level_id)) for level_id in self.levels}

    def names(self) -> dict[str, list[str]]:
        """Every authored name of the world by kind."""
        return {
            'zone': [zone.name for level in self.all_levels for zone in level.zones],
            'door': [door.name for level in self.all_levels for door in level.doors],
            'elevator': [elevator.name for level in self.all_levels for elevator in level.elevators],
            'stair': [*(stair.name for level in self.all_levels for stair in level.stairs), *dict.fromkeys(stair.group for level in self.all_levels for stair in level.stairs if stair.group)],
            'obj': [entity.name for level in self.all_levels for entity in level.objects],
            'human': [human.name for level in self.all_levels for human in level.humans],
            'light': [light.name for level in self.all_levels for light in level.lights],
            'sound': [sound.name for level in self.all_levels for sound in level.sounds],
            'schedule': [schedule.name for schedule in self.schedules],
            'signal': [signal.name for signal in self.signals],
        }

    def identifiers(self) -> Iterable[Identifier]:
        """Every asset this world references, for preloading and publish preflight."""
        from arena_simulation_setup.tree.assets.Sound import SoundIdentifier
        from arena_simulation_setup.tree.Wall import WallIdentifier

        for level in self.all_levels:
            for zone in level.zones:
                if zone.floor is not None:
                    yield zone.floor
                if zone.ceiling is not None:
                    yield zone.ceiling.material
                if zone.walls is not None:
                    if zone.walls.material is not None:
                        yield zone.walls.material
                    if zone.walls.kind:
                        yield WallIdentifier(zone.walls.kind)
            for path in level.walls:
                if path.material is not None:
                    yield path.material
                if path.kind:
                    yield WallIdentifier(path.kind)
            if any(path.material is None and not path.kind for path in level.walls) or any(zone.walls is not None and zone.walls.material is None and not zone.walls.kind for zone in level.zones):
                yield Material.default('wall')
            for door in level.doors:
                yield door.material
            for elevator in level.elevators:
                yield elevator.material
            for stair in level.stairs:
                yield stair.material
            for entity in level.objects:
                yield entity.model
            for human in level.humans:
                yield human.model
            for sound in level.sounds:
                yield SoundIdentifier.parse(sound.asset_id)

    def compact_world(self, origins: Mapping[str, tuple[float, float]]) -> Level:
        """All levels in one, each moved by its origin so that they lie side by side."""
        out = Level()
        for level_id, level in self.levels.items():
            if level_id not in origins:
                raise KeyError(f'when creating compacted single world from World, the origin for level {level_id} was not given')
            moved = level.shifted(*origins[level_id])
            for field in attrs.fields(Level):
                if field.name in ('id', 'elevation', 'pedestrian_walls', 'source', 'prefix'):
                    continue
                getattr(out, field.name).extend(getattr(moved, field.name))
        out.pedestrian_walls = not self.levels or any(level.pedestrian_walls for level in self.levels.values())
        return out

    def validate(self, loaded_level_ids: set[str] | None = None) -> list[str]:
        """Returns warnings for destinations into unloaded levels, raises on everything else."""
        warnings: list[str] = []
        seen: dict[str, str] = {}
        for kind, names in self.names().items():
            for name in names:
                if not name:
                    raise ValueError(f'a {kind} has an empty name')
                if name in seen:
                    raise ValueError(f'name {name!r} is used by a {seen[name]} and a {kind}, names are unique across the world')
                seen[name] = kind
        microphone_ids: set[str] = set()
        for level in self.all_levels:
            level.validate_microphones()
            level.validate_levels()
            level.validate_lights()
            zone_names = {zone.name for zone in level.zones}
            for zone in level.zones:
                if zone.walls is not None:
                    unknown = [name for name in zone.walls.open_to if name not in zone_names]
                    if unknown:
                        raise ValueError(f'zone {zone.name!r} is open to unknown zones {unknown}')
            for microphone in level.microphones:
                if microphone.listener_id in microphone_ids:
                    raise ValueError(f'microphone {microphone.listener_id!r} appears in multiple levels')
                microphone_ids.add(microphone.listener_id)
        cabins = self.elevator_levels()
        for level in self.all_levels:
            for elevator in level.elevators:
                for destination in elevator.destinations:
                    if destination not in cabins:
                        if loaded_level_ids is not None:
                            msg = f'elevator {elevator.name!r} has destination {destination!r} on a level that was not loaded, it will have no destination'
                            warnings.append(msg)
                            logging.getLogger(__name__).warning(msg)
                            continue
                        raise ValueError(f'elevator {elevator.name!r} has unknown destination {destination!r}')
        stairs = self.stair_levels()
        for level in self.all_levels:
            for stair in level.stairs:
                if stair.to and stair.to not in stairs:
                    if loaded_level_ids is None:
                        raise ValueError(f'stair {stair.name!r} leads to unknown stair {stair.to!r}')
                    msg = f'stair {stair.name!r} leads to {stair.to!r} on a level that was not loaded, it will not board'
                    warnings.append(msg)
                    logging.getLogger(__name__).warning(msg)
        return warnings

    def apply_elevator_door_sides(self, *, max_distance: float = 1.0) -> None:
        """Set the door side of every elevator that has none from the nearest wall on its level, `+x` when no wall is within max_distance."""
        for level in self.levels.values():
            walls = level.all_walls
            for elevator in level.elevators:
                if elevator.door_side is None:
                    elevator.door_side = _infer_door_side(elevator, walls, max_distance)

    @classmethod
    def from_levels(cls, *levels: Level) -> World:
        """A world of the given levels, with ids '0', '1', ... where a level has none."""
        result: dict[str, Level] = {}
        for index, level in enumerate(levels):
            level_id = level.id or f'{index}'
            result[level_id] = attrs.evolve(level, id=level_id)
        return cls(levels=result)

    def level_bbox(self) -> list[tuple[float, float]]:
        bboxes: list[tuple[float, float]] = []
        for level in self.levels.values():
            corners = [corner for zone in level.zones for corner in zone.polygon]
            if not corners:
                bboxes.append((0.0, 0.0))
                continue
            bboxes.append((max(c.x for c in corners) - min(c.x for c in corners), max(c.y for c in corners) - min(c.y for c in corners)))
        return bboxes

    def max_level_bbox_dim(self) -> tuple[float, float]:
        bboxes = self.level_bbox()
        if not bboxes:
            return (0.0, 0.0)
        return (max(width for width, _ in bboxes), max(height for _, height in bboxes))

    def level_origins(self, margin_m: float = 5.0, resolution: float = 0.05, preferred_pixel_width: int = 500) -> dict[str, tuple[float, float]]:
        """Per-level shifts that lay the levels side by side in one map frame, rows of at most preferred_pixel_width pixels."""
        max_width, max_height = self.max_level_bbox_dim()
        per_row = max(1, int(preferred_pixel_width // max((max_width + margin_m) / resolution, 1)))
        origins: dict[str, tuple[float, float]] = {}
        for index, (level_id, level) in enumerate(self.levels.items()):
            corners = [corner for zone in level.zones for corner in zone.polygon]
            x_min = min((c.x for c in corners), default=0.0)
            y_min = min((c.y for c in corners), default=0.0)
            column, row = index % per_row, index // per_row
            origins[level_id] = (column * (max_width + margin_m) - x_min, -row * (max_height + margin_m) - y_min)
        return origins


def _infer_door_side(elevator: Elevator, walls: list[Wall], max_distance: float) -> typing.Literal['+x', '-x', '+y', '-y']:
    if not walls:
        return '+x'
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
            t = max(0.0, min(1.0, float(np.dot(pos - a, ab) / ab_len2)))
            proj = a + t * ab
        dist = float(np.linalg.norm(pos - proj))
        if dist < best_dist:
            best_dist, best_wall, best_proj = dist, wall, proj
    if best_wall is None or best_proj is None or best_dist > max_distance:
        return '+x'
    wall_axis = 'y' if abs(best_wall.end.y - best_wall.start.y) >= abs(best_wall.end.x - best_wall.start.x) else 'x'
    projected = float(best_proj[0] if wall_axis == 'y' else best_proj[1])

    def middle(wall: Wall) -> float:
        return (wall.start.x + wall.end.x) / 2.0 if wall_axis == 'y' else (wall.start.y + wall.end.y) / 2.0

    comparable = [wall for wall in walls if abs(middle(wall) - projected) <= max_distance] or [best_wall]
    coords = [middle(wall) for wall in comparable]
    best = middle(best_wall)
    nearer_low = abs(best - min(coords)) <= abs(best - max(coords))
    if wall_axis == 'y':
        return '+x' if nearer_low else '-x'
    return '+y' if nearer_low else '-y'


def door_polygon(start: Position, end: Position) -> list[Position]:
    dx = end.x - start.x
    dy = end.y - start.y
    length = math.sqrt(dx * dx + dy * dy)
    nx, ny = (-dy / length, dx / length) if length > 0 else (0.0, 1.0)
    t = 0.3
    return [
        Position(start.x - nx * t, start.y - ny * t),
        Position(end.x - nx * t, end.y - ny * t),
        Position(end.x + nx * t, end.y + ny * t),
        Position(start.x + nx * t, start.y + ny * t),
    ]


def _door_axis(door_side: str) -> tuple[tuple[float, float], tuple[float, float]]:
    return {
        '+x': ((1.0, 0.0), (0.0, 1.0)),
        '-x': ((-1.0, 0.0), (0.0, 1.0)),
        '+y': ((0.0, 1.0), (1.0, 0.0)),
        '-y': ((0.0, -1.0), (1.0, 0.0)),
    }[door_side]


def elevator_doorway_corners(elevator: Elevator) -> list[Position]:
    outward, tangent = _door_axis(elevator.door_side or '+x')
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


def elevator_doorway_pose(elevator: Elevator, standoff: float = 0.0) -> Pose:
    """Pose facing the cabin on the outer edge of its doorway, moved `standoff` further out along the door normal."""
    outward, _tangent = _door_axis(elevator.door_side or '+x')
    out_extent = elevator.size[0] / 2.0 if outward[0] != 0 else elevator.size[1] / 2.0
    reach = out_extent + _ELEVATOR_DOORWAY_DEPTH + standoff
    return Pose(
        position=Position(elevator.position.x + outward[0] * reach, elevator.position.y + outward[1] * reach, elevator.position.z),
        orientation=Orientation.from_yaw(math.atan2(-outward[1], -outward[0])),
    )


def render_door_polygons(door: Door) -> list:
    import shapely

    return [shapely.Polygon(door.corners)]


def render_elevator_polygons(elevator: Elevator) -> list:
    import shapely

    return [shapely.Polygon(elevator.cabin_corners()), shapely.Polygon(elevator_doorway_corners(elevator))]
