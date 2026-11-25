"""NAV@r: share of a disc robot's traversable configuration space that a spawn can reach on the static occupancy."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import attrs
import numpy as np
import shapely
import shapely.affinity
from PIL import Image
from scipy import ndimage

from arena_simulation_setup.shared import Obstacle
from arena_simulation_setup.tree import Identifier
from arena_simulation_setup.tree.World import Level, WorldIdentifier, WorldView, world_assets
from arena_simulation_setup.tree.World.model import _PASSAGE_CLEARANCE

DEFAULT_RADIUS = 0.267
SPAWN_CLEARANCE = 0.45
RESOLUTION = 0.05
CUT_OFF_AREA = 2.0

_FREE = 255
_CONNECTIVITY = np.ones((3, 3), dtype=bool)
_PAINT_WALL = (20, 20, 20)
_PAINT_ENTITY = (120, 82, 60)
_PAINT_TIGHT = (215, 215, 215)
_PAINT_DOMINANT = (44, 162, 95)
_PAINT_STRANDED = ((230, 85, 13), (117, 107, 177), (222, 45, 38), (49, 130, 189), (253, 174, 107), (158, 154, 200))


@attrs.frozen
class ZoneReach:
    """Floor area of one zone and the part of it inside the dominant component, in square meters."""

    name: str
    area: float
    reachable_area: float
    cut_off: bool


@attrs.frozen
class NavAtR:
    """NAV@r of one level, the expectation sum_i w_i m_i over traversable components, NaN when no cell admits a spawn."""

    level: str
    value: float
    radius: float
    spawn_clearance: float
    walls_only: bool
    traversable_area: float
    masses: tuple[float, ...]
    spawn_shares: tuple[float, ...]
    zones: tuple[ZoneReach, ...]
    entities: int
    unresolved: int
    start_share: float | None = None
    start_snap: float | None = None

    @property
    def components(self) -> int:
        return len(self.masses)

    @property
    def largest_share(self) -> float:
        return self.masses[0] if self.masses else 0.0

    @property
    def cut_off_zones(self) -> tuple[str, ...]:
        return tuple(zone.name for zone in self.zones if zone.cut_off)

    def asdict(self) -> dict:
        """JSON-ready dict with the derived fields, NaN as None."""
        data = attrs.asdict(self)
        data.update(components=self.components, largest_share=self.largest_share, cut_off_zones=list(self.cut_off_zones))
        if math.isnan(self.value):
            data['value'] = None
        return data


def resolve_world(raw: str) -> tuple[WorldView, set[str] | None]:
    """World directory, or name with an optional `[level,...]` filter resolved like the runtime (ARENA_WORLD_PATH, the writable local root, installed worlds, world buckets)."""
    name, levels = WorldIdentifier.split_filter(raw)
    path = Path(name).expanduser()
    if path.is_dir():
        return WorldView(path.resolve()), levels
    return WorldIdentifier(name).resolve_sync(), levels


def nav_at_r(
    view: WorldView,
    *,
    radius: float = DEFAULT_RADIUS,
    walls_only: bool = False,
    start: tuple[float, float] | None = None,
    levels: set[str] | None = None,
    resolution: float = RESOLUTION,
    spawn_clearance: float = SPAWN_CLEARANCE,
    image: Path | None = None,
) -> dict[str, NavAtR]:
    """NAV@r per level id, with static entities as occupancy unless `walls_only`, the single-start share when `start` is given, and the components painted to `image`."""
    world = view.load(validate=False, level_filter=levels)
    single = len(world.levels) == 1
    with world_assets(view, world):
        return {
            level_id: _level_nav_at_r(
                level_id,
                level,
                radius=radius,
                walls_only=walls_only,
                start=start,
                resolution=resolution,
                spawn_clearance=spawn_clearance,
                image=None if image is None else image if single else image.with_name(f'{image.stem}_{level_id}{image.suffix}'),
            )
            for level_id, level in world.levels.items()
        }


def _level_nav_at_r(
    level_id: str,
    level: Level,
    *,
    radius: float,
    walls_only: bool,
    start: tuple[float, float] | None,
    resolution: float,
    spawn_clearance: float,
    image: Path | None,
) -> NavAtR:
    entities = list(level.objects)
    walls_grid, origin = level.render_grid(resolution=resolution)
    distance = ndimage.distance_transform_edt(walls_grid == _FREE)
    assert isinstance(distance, np.ndarray)
    clearance = distance * resolution
    walls_traversable = clearance >= radius
    unresolved = 0
    grid = walls_grid
    if not walls_only:
        footprints, unresolved = _footprints(entities)
        grid, _ = level.render_grid(resolution=resolution, asset_color='black', asset_name_color=None, static_objects=footprints)
        distance = ndimage.distance_transform_edt(grid == _FREE)
        assert isinstance(distance, np.ndarray)
        clearance = distance * resolution
    traversable = clearance >= radius
    spawnable = traversable & (clearance >= spawn_clearance)

    labelled = ndimage.label(traversable, structure=_CONNECTIVITY)
    assert isinstance(labelled, tuple)
    labels, count = labelled
    cells = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    spawns = np.bincount(labels[spawnable], minlength=count + 1)[1:]
    order = np.argsort(-cells, kind='stable')
    masses = cells[order] / cells.sum() if count else np.zeros(0)
    shares = spawns[order] / spawns.sum() if spawns.sum() else np.zeros(count)
    value = float(np.dot(masses, shares)) if spawns.sum() else math.nan
    rank = np.zeros(count + 1, dtype=np.int64)
    rank[order + 1] = np.arange(1, count + 1)
    ranked = rank[labels]  # pyright: ignore[reportCallIssue, reportArgumentType]

    if image is not None:
        _paint(walls_grid, grid, ranked).save(image)

    start_share = start_snap = None
    if start is not None:
        start_share, start_snap = _start_share(ranked, masses, origin, resolution, start)

    return NavAtR(
        level=level_id,
        value=value,
        radius=radius,
        spawn_clearance=max(spawn_clearance, radius),
        walls_only=walls_only,
        traversable_area=float(cells.sum()) * resolution**2,
        masses=tuple(float(m) for m in masses),
        spawn_shares=tuple(float(s) for s in shares),
        zones=tuple(_zone_reach(zone.name, zone.polygon, ranked, walls_traversable, origin, resolution) for zone in level.zones if len(zone.polygon) >= 3),
        entities=len(entities),
        unresolved=unresolved,
        start_share=start_share,
        start_snap=start_snap,
    )


def _paint(walls_grid: np.ndarray, grid: np.ndarray, ranked: np.ndarray) -> Image.Image:
    """Occupancy map with the dominant traversable component in green and every stranded component in another color."""
    canvas = np.empty((*grid.shape, 3), dtype=np.uint8)
    canvas[:] = _PAINT_WALL
    canvas[(walls_grid == _FREE) & (grid != _FREE)] = _PAINT_ENTITY
    canvas[grid == _FREE] = _PAINT_TIGHT
    canvas[ranked == 1] = _PAINT_DOMINANT
    stranded = ranked > 1
    canvas[stranded] = np.array(_PAINT_STRANDED, dtype=np.uint8)[(ranked[stranded] - 2) % len(_PAINT_STRANDED)]
    return Image.fromarray(canvas)


def _footprints(entities: Iterable[Obstacle]) -> tuple[list[tuple[str, shapely.Polygon]], int]:
    pending = [entity for entity in entities if not entity.extra.get('bbox')]

    async def _gather() -> list[object]:
        return await asyncio.gather(*(_annotation_bbox(entity) for entity in pending))

    footprints: list[tuple[str, shapely.Polygon]] = []
    unresolved = 0
    for entity, bbox in zip(pending, Identifier._run_sync(_gather()), strict=True):
        if not isinstance(bbox, Sequence):
            unresolved += 1
            continue
        try:
            (x_min, x_max), (y_min, y_max), *z_pair = bbox
        except (TypeError, ValueError):
            unresolved += 1
            continue
        if z_pair and entity.pose.position.z + z_pair[0][0] > _PASSAGE_CLEARANCE:
            continue
        poly = shapely.box(x_min, y_min, x_max, y_max)
        poly = shapely.affinity.rotate(poly, entity.pose.orientation.to_yaw(), origin=(0, 0), use_radians=True)
        footprints.append((entity.name, shapely.affinity.translate(poly, xoff=entity.pose.position.x, yoff=entity.pose.position.y)))
    return footprints, unresolved


async def _annotation_bbox(entity: Obstacle) -> object:
    try:
        view = await entity.model.resolve()
    except FileNotFoundError:
        return None
    return (view.annotation or {}).get('bounding_box')


def _cell_centers(shape: tuple[int, int], origin: tuple[float, float], resolution: float, rows: slice, cols: slice) -> tuple[np.ndarray, np.ndarray]:
    height = shape[0]
    xs = origin[0] + (np.arange(cols.start, cols.stop) + 0.5) * resolution
    ys = origin[1] + (height - 1 - np.arange(rows.start, rows.stop) + 0.5) * resolution
    return np.meshgrid(xs, ys)


def _zone_reach(name: str, corners: Sequence, ranked: np.ndarray, walls_traversable: np.ndarray, origin: tuple[float, float], resolution: float) -> ZoneReach:
    polygon = shapely.Polygon([(corner.x, corner.y) for corner in corners])
    height, width = ranked.shape
    x_min, y_min, x_max, y_max = polygon.bounds
    cols = slice(max(0, math.floor((x_min - origin[0]) / resolution)), min(width, math.ceil((x_max - origin[0]) / resolution)))
    rows = slice(max(0, height - math.ceil((y_max - origin[1]) / resolution)), min(height, height - math.floor((y_min - origin[1]) / resolution)))
    if cols.stop <= cols.start or rows.stop <= rows.start:
        return ZoneReach(name=name, area=polygon.area, reachable_area=0.0, cut_off=False)
    xx, yy = _cell_centers(ranked.shape, origin, resolution, rows, cols)
    inside = shapely.contains_xy(polygon, xx, yy)
    reachable = int(np.count_nonzero(inside & (ranked[rows, cols] == 1)))
    cut_off = polygon.area >= CUT_OFF_AREA and reachable == 0 and bool(np.any(inside & walls_traversable[rows, cols]))
    return ZoneReach(name=name, area=polygon.area, reachable_area=reachable * resolution**2, cut_off=cut_off)


def _start_share(ranked: np.ndarray, masses: np.ndarray, origin: tuple[float, float], resolution: float, start: tuple[float, float]) -> tuple[float, float | None]:
    height, width = ranked.shape
    row = height - 1 - math.floor((start[1] - origin[1]) / resolution)
    col = math.floor((start[0] - origin[0]) / resolution)
    if 0 <= row < height and 0 <= col < width and ranked[row, col] > 0:
        return float(masses[ranked[row, col] - 1]), 0.0
    cells = np.argwhere(ranked > 0)
    if not len(cells):
        return 0.0, None
    distances = np.hypot(origin[0] + (cells[:, 1] + 0.5) * resolution - start[0], origin[1] + (height - 1 - cells[:, 0] + 0.5) * resolution - start[1])
    nearest = int(distances.argmin())
    row, col = cells[nearest]
    return float(masses[ranked[row, col] - 1]), float(distances[nearest])


def _parse_start(raw: str) -> tuple[float, float]:
    try:
        x, y = (float(part) for part in raw.split(','))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f'expected X,Y in meters in the level frame, e.g. --start 1.5,2.0, got {raw!r}') from e
    return x, y


def _parse_meters(raw: str) -> float:
    try:
        meters = float(raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f'expected a length in meters, e.g. 0.267, got {raw!r}') from e
    if not meters > 0:
        raise argparse.ArgumentTypeError(f'expected a positive length in meters, got {raw!r}')
    return meters


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='nav_at_r',
        description=(
            'NAV@r of an Arena world: the share of a disc robot\'s traversable configuration space (cells with clearance >= r on the static occupancy map) '
            f'that a uniform spawn (cells with clearance >= {SPAWN_CLEARANCE} m, or r when larger) reaches, sum_i w_i m_i over the 8-connected components. '
            '1.0 means every traversable cell is reachable from every spawn. Computed per level, static entities are occupancy through their annotation bounding boxes.'
        ),
        epilog=(
            'output: per level the NAV@r value, the number of traversable components, the largest component\'s share, '
            'the start component\'s share with --start, entities without a resolvable bbox, '
            f'and the cut-off zones (>= {CUT_OFF_AREA:g} m2, traversable on walls only, but outside the dominant component). '
            'examples: nav_at_r hospital_1 | nav_at_r ./worlds/office_a --radius 0.267 --walls-only | nav_at_r "three_storied_residential[1]" --radius 0.3 --start 2,3 --json. '
            'exit 0 on success, 2 on a bad argument or an unresolvable world.'
        ),
    )
    parser.add_argument('world', help='world directory, or world name resolved like the runtime (ARENA_WORLD_PATH roots, then the installed worlds, then the world buckets), optionally with a level filter as name[0,1]')
    parser.add_argument('--radius', type=_parse_meters, metavar='M', help=f'footprint radius r in meters (default {DEFAULT_RADIUS}, the Jackal collision radius)')
    parser.add_argument('--walls-only', action='store_true', help='ignore static entities, to tell layout faults (doors, walls) from furniture blockages')
    parser.add_argument('--start', type=_parse_start, metavar='X,Y', help='start pose in meters in each level\'s frame, adds the single-start share (snapped to the nearest traversable cell)')
    parser.add_argument('--resolution', type=_parse_meters, default=RESOLUTION, metavar='M', help=f'raster resolution in meters per cell (default {RESOLUTION})')
    parser.add_argument('--image', type=Path, metavar='PNG', help='paint the map to this file: dominant component green, stranded components in other colors, free cells with clearance below r light grey, entities brown, walls black. With several levels the level id is appended to the file name')
    parser.add_argument('--json', action='store_true', help='print one JSON object with every per-level and per-zone field instead of the table')
    return parser


def _table(world: str, results: dict[str, NavAtR], *, radius: float, walls_only: bool, start: bool) -> str:
    header = ['level', 'NAV@r', 'components', 'largest', *(['start'] if start else []), 'no bbox', 'cut-off zones']
    rows = []
    for result in results.values():
        rows.append(
            [
                result.level,
                'n/a' if math.isnan(result.value) else f'{result.value:.3f}',
                str(result.components),
                f'{result.largest_share:.3f}',
                *([f'{result.start_share:.3f}'] if start and result.start_share is not None else []),
                '-' if walls_only else f'{result.unresolved}/{result.entities}',
                ', '.join(result.cut_off_zones) or '-',
            ]
        )
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header) - 1)]
    lines = [f'{world}: r = {radius:g} m, {"walls only" if walls_only else "furnished"}']
    for row in [header, *rows]:
        lines.append('  '.join([*(cell.ljust(width) for cell, width in zip(row, widths, strict=False)), row[-1]]))
    return '\n'.join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point, prints the per-level NAV@r table or JSON."""
    parser = _parser()
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        parser.print_help(sys.stderr)
        return 2
    args = parser.parse_args(args_list)
    radius = args.radius if args.radius is not None else DEFAULT_RADIUS
    try:
        view, levels = resolve_world(args.world)
        results = nav_at_r(view, radius=radius, walls_only=args.walls_only, start=args.start, levels=levels, resolution=args.resolution, image=args.image)
    except FileNotFoundError as e:
        parser.error(f'world {args.world!r} not found: pass a world directory, or a name under ARENA_WORLD_PATH, the installed worlds or the world buckets ({e})')
    if not results:
        parser.error(f'world {args.world!r} has no level matching the filter {sorted(levels or ())}')
    if args.json:
        print(json.dumps({'world': view.name, 'radius': radius, 'walls_only': args.walls_only, 'levels': [result.asdict() for result in results.values()]}, indent=2))
    else:
        print(_table(view.name, results, radius=radius, walls_only=args.walls_only, start=args.start is not None))
    return 0


if __name__ == '__main__':
    sys.exit(main())
