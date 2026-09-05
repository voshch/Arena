"""
This file exists to make world_manager more readable
"""

from collections.abc import Callable, Collection
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, TypeVar

import attrs
import numpy as np
import scipy.interpolate
import shapely
import yaml
from arena_rclpy_mixins.Time import Time
from PIL import Image, ImageDraw

from task_generator.shared import Position, PositionRadius, Wall

if TYPE_CHECKING:
    from arena_simulation_setup.tree.World.World import LevelDescription as WorldDescription

# CONVERTERS

T = TypeVar('T')


def list_from_any[T](v: T | list[T]) -> list[T]:
    if not isinstance(v, list):
        return list((v,))
    return v


def check_list[T](t: type[T], l: list[T]) -> list[T]:
    """
    runtime check list for type
    """
    if any(not isinstance(v, t) for v in l):
        raise RuntimeError(f'list {l} contains value not of type {t}')
    return l


# TYPES


WorldWalls = Collection[Wall]


class WorldOccupancy:
    FULL: np.uint8 = np.uint8(np.iinfo(np.uint8).min)
    EMPTY: np.uint8 = np.uint8(np.iinfo(np.uint8).max)

    _grid: np.ndarray

    def __init__(self, grid: np.ndarray):
        self._grid = grid

    @staticmethod
    def from_map(input_map: np.ndarray) -> "WorldOccupancy":
        remap = scipy.interpolate.interp1d([input_map.max(), input_map.min()], [WorldOccupancy.EMPTY, WorldOccupancy.FULL])
        return WorldOccupancy(remap(input_map))

    @staticmethod
    def empty(grid: np.ndarray) -> np.ndarray:
        return np.isclose(grid, WorldOccupancy.EMPTY)

    @staticmethod
    def not_empty(grid: np.ndarray) -> np.ndarray:
        return np.invert(WorldOccupancy.empty(grid))

    @staticmethod
    def emptyish(grid: np.ndarray, thresh: float | None = None) -> np.ndarray:
        if thresh is None:
            thresh = float((int(WorldOccupancy.FULL) + int(WorldOccupancy.EMPTY)) / 2)
        return grid >= thresh

    @staticmethod
    def full(grid: np.ndarray) -> np.ndarray:
        return np.isclose(grid, WorldOccupancy.FULL)

    @staticmethod
    def not_full(grid: np.ndarray) -> np.ndarray:
        return np.invert(WorldOccupancy.full(grid))

    @staticmethod
    def fullish(grid: np.ndarray, thresh: float | None = None) -> np.ndarray:
        return np.invert(WorldOccupancy.emptyish(grid, thresh))

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def clear(self):
        self.grid.fill(WorldOccupancy.EMPTY)

    def occupy(self, lo: tuple[int, int], hi: tuple[int, int]):
        rows = np.clip(np.array([lo[0], hi[0]]), 0, self._grid.shape[0] - 1)
        cols = np.clip(np.array([lo[1], hi[1]]), 0, self._grid.shape[1] - 1)
        self._grid[int(rows.min()) : int(rows.max()), int(cols.min()) : int(cols.max())] = WorldOccupancy.FULL

    def occupy_mask(self, mask: np.ndarray):
        self._grid[mask] = WorldOccupancy.FULL


class WorldLayers:
    _walls: WorldOccupancy  # walls
    _obstacle: WorldOccupancy  # intrinsic obstcales
    _forbidden: WorldOccupancy  # task obstacles

    def __init__(self, walls: WorldOccupancy):
        self._walls = walls
        self._obstacle = WorldOccupancy(np.full(walls.grid.shape, WorldOccupancy.EMPTY))
        self._forbidden = WorldOccupancy(np.full(walls.grid.shape, WorldOccupancy.EMPTY))

        self._combined_cache = None

    _combined_cache: WorldOccupancy | None

    def _invalidate_combined_cache(self):
        self._combined_cache = None

    @property
    def _combined(self) -> WorldOccupancy:
        if self._combined_cache is None:
            self._combined_cache = WorldOccupancy(np.minimum.reduce([self._walls.grid, self._obstacle.grid, self._forbidden.grid]))

        return self._combined_cache

    @property
    def grid(self) -> np.ndarray:
        return self._combined.grid

    @property
    def shape(self) -> tuple[int, ...]:
        return self._walls.grid.shape

    @property
    def walls(self) -> np.ndarray:
        return self._walls.grid

    def detect_walls(
        self,
        transform: Callable[[tuple[float, float]], Position] | None = None,
    ) -> WorldWalls:
        return occupancy_to_walls(self._walls.grid, transform)

    # obstacle interface
    def obstacle_occupy(self, mask: np.ndarray):
        self._obstacle.occupy_mask(mask)
        self._combined.occupy_mask(mask)

    def obstacle_clear(self):
        self._obstacle.clear()
        self._invalidate_combined_cache()

    # forbidden interface
    def forbidden_occupy(self, lo: tuple[int, int], hi: tuple[int, int]):
        self._forbidden.occupy(lo, hi)
        self._combined.occupy(lo, hi)

    def forbidden_clear(self):
        self._forbidden.clear()
        self._invalidate_combined_cache()

    class WorldLayersFork:
        _base: "WorldLayers"
        _grid: WorldOccupancy

        def __init__(self, base: "WorldLayers"):
            self._base = base
            self._grid = WorldOccupancy(self._base.grid.copy())

        def commit(self):
            self._base._forbidden = self._grid
            self._base._invalidate_combined_cache()

        def occupy(self, lo: tuple[int, int], hi: tuple[int, int]):
            self._grid.occupy(lo, hi)

        @property
        def grid(self) -> np.ndarray:
            return self._grid.grid

    def fork(self) -> "WorldLayers.WorldLayersFork":
        return WorldLayers.WorldLayersFork(self)


@attrs.define(frozen=True)
class GridFrame:
    """Row/col <-> map-frame transform of a north-up grid (row 0 is the top edge)."""

    shape: tuple[int, ...]
    origin: Position
    resolution: float

    def pos2grid(self, position: Position) -> tuple[int, int]:
        return np.round(self.shape[0] - (position.y - self.origin.y) / self.resolution), np.round((position.x - self.origin.x) / self.resolution)

    def grid2pos(self, grid_pos: tuple[float, float]) -> Position:
        return Position(x=grid_pos[1] * self.resolution + self.origin.x, y=(self.shape[0] - grid_pos[0]) * self.resolution + self.origin.y)

    def grid2xy(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized grid2pos."""
        return cols * self.resolution + self.origin.x, (self.shape[0] - rows) * self.resolution + self.origin.y

    def poly_window(self, poly: shapely.Polygon, offset: float = 0.5) -> tuple[slice, slice, np.ndarray] | None:
        """Bbox-clipped (rows, cols, inside) of cells whose sample point (row + offset, col + offset) lies inside poly, None when off-grid."""
        min_x, min_y, max_x, max_y = poly.bounds
        r_lo, c_lo = self.pos2grid(Position(x=min_x, y=max_y))
        r_hi, c_hi = self.pos2grid(Position(x=max_x, y=min_y))
        r0, r1 = max(int(r_lo) - 1, 0), min(int(r_hi) + 2, self.shape[0])
        c0, c1 = max(int(c_lo) - 1, 0), min(int(c_hi) + 2, self.shape[1])
        if r0 >= r1 or c0 >= c1:
            return None
        rows, cols = np.mgrid[r0:r1, c0:c1]
        inside = shapely.contains_xy(poly, *self.grid2xy(rows + offset, cols + offset))
        return slice(r0, r1), slice(c0, c1), inside

    def lines_mask(self, segments: Iterable[tuple[tuple[float, float], tuple[float, float]]]) -> np.ndarray:
        """Cells touched by the segments rasterized one cell wide."""
        img = Image.new('1', (int(self.shape[1]), int(self.shape[0])), 0)
        draw = ImageDraw.Draw(img)
        for (x0, y0), (x1, y1) in segments:
            draw.line([self._pixel(x0, y0), self._pixel(x1, y1)], fill=1, width=1)
        return np.array(img, dtype=bool)

    def _pixel(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.origin.x) / self.resolution), int(self.shape[0] - (y - self.origin.y) / self.resolution)


@attrs.define()
class WorldMap:
    occupancy: WorldLayers
    origin: Position
    resolution: float
    time: Time
    level_origins: dict[str, tuple[float, float]] = attrs.field(factory=dict)

    @staticmethod
    def from_map_files(map_yaml_path: str | Path) -> "WorldMap":
        map_yaml_path = Path(map_yaml_path)
        with open(map_yaml_path, encoding='utf-8') as f:
            map_yaml = yaml.safe_load(f)
        if not isinstance(map_yaml, dict):
            raise ValueError(f"map.yaml must be a dictionary: {map_yaml_path}")

        image_path = map_yaml.get('image', '')
        if not image_path:
            raise ValueError(f"map.yaml missing image field: {map_yaml_path}")
        image_path = str(image_path)
        if not image_path.startswith('/'):
            image_path = str(map_yaml_path.parent / image_path)

        img = Image.open(image_path).convert('L')
        img_data = np.array(img, dtype=np.float32) / 255.0

        negate = int(map_yaml.get('negate', 0))
        if negate:
            img_data = 1.0 - img_data

        occupied_thresh = float(map_yaml.get('occupied_thresh', 0.9))
        free_thresh = float(map_yaml.get('free_thresh', 0.1))

        grid_data = np.full(img_data.shape, 50.0, dtype=np.float32)
        grid_data[img_data > free_thresh] = 0.0
        grid_data[img_data < occupied_thresh] = 100.0

        normalized_data = np.interp(grid_data, (100, 0), (WorldOccupancy.EMPTY, WorldOccupancy.FULL)).astype(np.uint8)

        origin = map_yaml.get('origin', [0, 0, 0])
        resolution = float(map_yaml.get('resolution', 0.05))
        level_origins = map_yaml.get('origins', {})
        if level_origins:
            level_origins = {fid: tuple(_origin) for fid, _origin in level_origins.items()}

        return WorldMap(
            occupancy=WorldLayers(walls=WorldOccupancy(normalized_data)),
            origin=Position(x=float(origin[0]), y=float(origin[1])),
            resolution=resolution,
            time=Time(-1, 0),
            level_origins=level_origins,
        )

    @classmethod
    def from_world_description(cls, description: "WorldDescription", resolution: float, time: Time, _level_origins: dict[str, tuple[float, float]] | None = None) -> "WorldMap":
        """Rasterize a WorldDescription into a WorldMap. PIL grayscale matches WorldOccupancy (255=EMPTY, 0=FULL)."""
        grid, origin = description.render_grid(resolution=resolution)
        return WorldMap(
            occupancy=WorldLayers(walls=WorldOccupancy(grid.copy())),
            origin=Position(x=origin[0], y=origin[1]),
            resolution=resolution,
            time=time,
            level_origins=_level_origins if _level_origins is not None else {},
        )

    def get_origin(self, level_id: str = "") -> tuple[float, float]:
        if level_id:
            try:
                return self.level_origins[level_id]
            except KeyError as e:
                raise KeyError(f"level id {level_id} was not recognized by WorldMap") from e
        else:
            return (self.origin.x, self.origin.y)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.occupancy.shape

    @property
    def frame(self) -> GridFrame:
        return GridFrame(shape=self.shape, origin=self.origin, resolution=self.resolution)

    def tf_pos2grid(self, position: Position) -> tuple[int, int]:
        return self.frame.pos2grid(position)

    def tf_grid2pos(self, grid_pos: tuple[float, float]) -> Position:
        return self.frame.grid2pos(grid_pos)

    def tf_grid2xy(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized tf_grid2pos."""
        return self.frame.grid2xy(rows, cols)

    def tf_posr2rect(self, posr: PositionRadius) -> tuple[tuple[int, int], tuple[int, int]]:
        lo = self.tf_pos2grid(
            Position(
                x=posr.x - posr.radius,
                y=posr.y - posr.radius,
            )
        )
        hi = self.tf_pos2grid(
            Position(
                x=posr.x + posr.radius,
                y=posr.y + posr.radius,
            )
        )
        return (lo, hi)

    def tf_poly2mask(self, poly: shapely.Polygon, offset: float = 0.5) -> np.ndarray:
        """Cells whose sample point (row + offset, col + offset) lies inside poly, default cell centers."""
        mask = np.zeros(self.shape[:2], dtype=bool)
        window = self.frame.poly_window(poly, offset)
        if window is not None:
            rows, cols, inside = window
            mask[rows, cols] = inside
        return mask

    def detect_walls(self) -> WorldWalls:
        return self.occupancy.detect_walls(self.tf_grid2pos)


@attrs.define
class MultiLevelMap:
    _maps: dict[str, WorldMap] = attrs.field(factory=dict)

    def __init__(self, maps: dict[str, WorldMap] | None = None):
        self._maps = maps if maps is not None else {}

    @property
    def level_ids(self) -> Iterable[str]:
        return self._maps.keys()

    def get_map(self, level_id: str) -> WorldMap | None:
        return self._maps.get(level_id, None)

    def select_map(self, level_id: str) -> WorldMap:
        return self._maps[level_id]

    @classmethod
    def from_single(cls, world_map: WorldMap, level_id: str = "") -> "MultiLevelMap":
        return cls(maps={level_id: world_map})


# END TYPES


def RLE_1D(grid: np.ndarray) -> list[list[int]]:
    """
    run-length encode walls in 1D (occupancy grid -> run_length[segments][rows])
    """
    res: list[list[int]] = list()
    for major in grid:
        run: int = 1
        last: int = major[0]
        subres: list[int] = [0]
        for minor in major[1:]:
            if minor == last:
                run += 1
            else:
                subres.append(run)
                run = 1
                last = minor
        subres.append(run)
        res.append(subres)
    return res


class _WallLines(dict[float, list[tuple[float, float]]]):
    """
    Helper class for efficiently merging collinear line segments
    """

    WallsT = Collection[tuple[tuple[float, float], tuple[float, float]]]

    _inverted: bool

    def __init__(self, inverted: bool = False, *args: object, **kwargs: object):
        """
        inverted=True for y-axis pass
        """
        super().__init__(*args, **kwargs)
        self._inverted = inverted

    def add(self, major: float, minor: float, length: float = 1):
        """
        add a wall segment in row <major> at position <minor> with length <length> and merge with previous line segment if their endpoints touch
        """
        if major not in self:
            self[major] = [(minor, minor + length)]
            return

        last = self[major][-1]

        if minor == last[1]:
            self[major][-1] = (last[0], minor + length)
        else:
            self[major].append((minor, minor + length))

    @property
    def lines(self) -> WallsT:
        """
        get WorldWalls object
        """
        if not self._inverted:
            return [
                (
                    (start, major),
                    (end, major),
                )
                for major, segment in self.items()
                for start, end in segment
            ]

        else:
            return set([((major, start), (major, end)) for major, segment in self.items() for start, end in segment])

    @classmethod
    def walls(cls, walls: WallsT) -> WorldWalls:
        return [Wall(start=Position(x=sx, y=sy), end=Position(x=ex, y=ey)) for (sx, sy), (ex, ey) in walls]


def RLE_2D(grid: np.ndarray) -> WorldWalls:
    """
    rudimentary (but fast) 2D extension of 1D-RLE to 2D (occupancy grid -> WorldWalls)
    """

    walls_x = _WallLines()
    walls_y = _WallLines(inverted=True)

    for y, rles in enumerate(RLE_1D(grid)):
        distance: int = 0
        for run in rles:
            distance += run
            walls_x.add(distance, y)

    for x, rles in enumerate(RLE_1D(grid.T)):
        distance: int = 0
        for run in rles:
            distance += run
            walls_y.add(distance, x)

    return _WallLines.walls(set().union(walls_x.lines, walls_y.lines))


def occupancy_to_walls(occupancy_grid: np.ndarray, transform: Callable[[tuple[float, float]], Position] | None = None) -> WorldWalls:

    walls = RLE_2D(grid=WorldOccupancy.not_full(occupancy_grid))

    if transform is None:

        def _transform(p: tuple[float, float]) -> Position:
            return Position(x=p[0], y=p[1])

        transform = _transform

    return [
        Wall(
            start=transform((wall.start.x, wall.start.y)),
            end=transform((wall.end.x, wall.end.y)),
        )
        for wall in walls
    ]
