"""Navigation mesh of every walkable surface of a world, for mesh navigation planners."""

from __future__ import annotations

import itertools
import math
import typing
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import attrs
import numpy as np

from arena_simulation_setup.utils.geometry import Position

from .walls import Wall
from .world import Door, Elevator, Floor, Stairs

if typing.TYPE_CHECKING:
    from arena_simulation_setup.tree.World.model import Level, World

NAV_CLIMB = 0.25
NAV_HEADROOM = 0.4
NAV_SKIN = 0.02
WALL_THICKNESS = 0.05
WALL_HEIGHT = 2.0
_EPS = 1e-6


_CABIN_RING = ((1, 1), (-1, 1), (-1, -1), (1, -1))
_CABIN_DOORS = {'+y': 0, '-x': 1, '-y': 2, '+x': 3}


@attrs.frozen
class NavLevel:
    """The floors, blocking walls, stair halves and doors of one level, all in the simulator frame."""

    level: str
    floors: Sequence[Floor]
    walls: Sequence[Wall]
    stairs: Sequence[Stairs]
    doors: Sequence[Door]


def cabin_floor(elevator: Elevator) -> Floor:
    """The floor of an elevator cabin, at the cabin position."""
    return Floor(name=elevator.name, pos=elevator.position, x_length=elevator.size[0], y_length=elevator.size[1])


def cabin_walls(elevator: Elevator) -> list[Wall]:
    """The sides of an elevator cabin but its door side."""
    center, half_x, half_y = elevator.position, elevator.size[0] / 2.0, elevator.size[1] / 2.0
    corners = [Position(center.x + sx * half_x, center.y + sy * half_y, center.z) for sx, sy in _CABIN_RING]
    door = _CABIN_DOORS[elevator.door_side or '+x']
    return [Wall(start=corners[i], end=corners[(i + 1) % len(corners)]) for i in range(len(corners)) if i != door]


def nav_floors(level: Level) -> list[Floor]:
    """The walkable floors of a level in its own frame: zone floors and elevator cabins."""
    return [*level.all_floors, *(cabin_floor(elevator) for elevator in level.all_elevators)]


def nav_walls(level: Level) -> list[Wall]:
    """The blocking walls of a level in its own frame: walls cut open at the doors and the closed sides of elevator cabins."""
    return [*level.all_walls, *(wall for elevator in level.all_elevators for wall in cabin_walls(elevator))]


def door_wall(door: Door) -> Wall:
    """A closed door as a wall across its span."""
    return Wall(start=door.start, end=door.end, height=door.height)


@attrs.frozen(eq=False)
class NavMesh:
    """Triangle mesh of the walkable surfaces, every face counter-clockwise seen from above."""

    vertices: np.ndarray
    faces: np.ndarray

    def components(self) -> np.ndarray:
        """Connected component of every vertex, numbered from 0."""
        parent = np.arange(len(self.vertices))
        edges = np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]]])
        while True:
            a, b = parent[edges[:, 0]], parent[edges[:, 1]]
            if np.array_equal(a, b):
                break
            np.minimum.at(parent, np.maximum(a, b), np.minimum(a, b))
            while not np.array_equal(parent, jumped := parent[parent]):
                parent = jumped
        return np.unique(parent, return_inverse=True)[1].reshape(-1)

    def nearest(self, x: float, y: float, z: float) -> tuple[int, float]:
        """Index of the vertex nearest to a point and its distance, (-1, inf) for an empty mesh."""
        if not len(self.vertices):
            return -1, math.inf
        distances = np.linalg.norm(self.vertices - (x, y, z), axis=1)
        index = int(np.argmin(distances))
        return index, float(distances[index])

    def write_ply(self, path: str | Path) -> Path:
        """Write the mesh as a binary little-endian PLY file."""
        target = Path(path)
        faces = np.empty(len(self.faces), dtype=[('count', 'u1'), ('index', '<u4', (3,))])
        faces['count'] = 3
        faces['index'] = self.faces
        header = '\n'.join(
            [
                'ply',
                'format binary_little_endian 1.0',
                f'element vertex {len(self.vertices)}',
                'property float x',
                'property float y',
                'property float z',
                f'element face {len(self.faces)}',
                'property list uchar uint vertex_indices',
                'end_header',
                '',
            ]
        )
        with target.open('wb') as file:
            file.write(header.encode('ascii'))
            file.write(np.ascontiguousarray(self.vertices, dtype='<f4').tobytes())
            file.write(faces.tobytes())
        return target


@attrs.define
class _Field:
    """Walking height of one level on the grid points from index (i0, j0), NaN where nothing is walkable."""

    i0: int
    j0: int
    z: np.ndarray

    def window(self, x0: float, y0: float, x1: float, y1: float, resolution: float) -> tuple[slice, slice, np.ndarray, np.ndarray] | None:
        """The slices of the field inside a rectangle and the x, y of their grid points."""
        i0, i1 = _span(x0, x1, resolution)
        j0, j1 = _span(y0, y1, resolution)
        i0, i1 = max(i0, self.i0), min(i1, self.i0 + self.z.shape[0] - 1)
        j0, j1 = max(j0, self.j0), min(j1, self.j0 + self.z.shape[1] - 1)
        if i0 > i1 or j0 > j1:
            return None
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing='ij')
        return slice(i0 - self.i0, i1 + 1 - self.i0), slice(j0 - self.j0, j1 + 1 - self.j0), ii * resolution, jj * resolution


def _span(low: float, high: float, resolution: float) -> tuple[int, int]:
    return math.ceil(low / resolution - _EPS), math.floor(high / resolution + _EPS)


def _floor_rect(floor: Floor) -> tuple[float, float, float, float]:
    return floor.pos.x - floor.x_length / 2.0, floor.pos.y - floor.y_length / 2.0, floor.pos.x + floor.x_length / 2.0, floor.pos.y + floor.y_length / 2.0


def _rect(points: Sequence[Position]) -> tuple[float, float, float, float]:
    return min(p.x for p in points), min(p.y for p in points), max(p.x for p in points), max(p.y for p in points)


def _stair_rect(stair: Stairs) -> tuple[float, float, float, float]:
    half = stair.width / 2.0
    alongs = [0.0, stair.length, *(edge for zone in stair.plate_zones for edge in zone)]
    return _rect([stair.local(along, across) for along in alongs for across in (-half, half)])


def _level_field(level: NavLevel, resolution: float) -> _Field | None:
    """The walking heights of a level: floors, overridden by stair halves, overridden by landing plates."""
    floors = [_floor_rect(floor) for floor in level.floors]
    rects = floors + [_stair_rect(stair) for stair in level.stairs]
    if not rects:
        return None
    i0, i1 = _span(min(r[0] for r in rects), max(r[2] for r in rects), resolution)
    j0, j1 = _span(min(r[1] for r in rects), max(r[3] for r in rects), resolution)
    if i0 > i1 or j0 > j1:
        return None
    field = _Field(i0, j0, np.full((i1 - i0 + 1, j1 - j0 + 1), np.nan))
    for floor, rect in zip(level.floors, floors, strict=True):
        window = field.window(*rect, resolution)
        if window is not None:
            field.z[window[0], window[1]] = floor.pos.z
    for stair in level.stairs:
        window = field.window(*_stair_rect(stair), resolution)
        if window is None:
            continue
        si, sj, x, y = window
        along, across = stair.along(x, y), stair.across(x, y)
        beside = np.abs(across) <= stair.width / 2.0 + _EPS
        block = field.z[si, sj]
        covered = beside & (along >= -_EPS) & (along <= stair.length + _EPS)
        for begin, end in stair.plate_zones:
            covered |= beside & (along >= begin - _EPS) & (along <= end + _EPS)
        block[covered] = [stair.position.z + stair.footing(a) for a in along[covered]]
    return field


def _block(fields: Sequence[_Field], wall: Wall, resolution: float) -> None:
    """Clear the grid points a wall covers within the headroom above their walking height."""
    reach = WALL_THICKNESS / 2.0 + resolution / 2.0 + _EPS
    sx, sy, sz = wall.start.x, wall.start.y, wall.start.z
    dx, dy, dz = wall.end.x - sx, wall.end.y - sy, wall.end.z - sz
    span = dx * dx + dy * dy
    low = wall.bottom if wall.bottom is not None else 0.0
    high = wall.top if wall.top is not None else wall.height if wall.height is not None else WALL_HEIGHT
    for field in fields:
        window = field.window(min(sx, sx + dx) - reach, min(sy, sy + dy) - reach, max(sx, sx + dx) + reach, max(sy, sy + dy) + reach, resolution)
        if window is None:
            continue
        si, sj, x, y = window
        t = np.clip(((x - sx) * dx + (y - sy) * dy) / span, 0.0, 1.0) if span > 0.0 else np.zeros_like(x)
        line = sz + t * dz
        z = field.z[si, sj]
        covered = (np.hypot(x - sx - t * dx, y - sy - t * dy) <= reach) & (line + low < z + NAV_HEADROOM) & (line + high > z + NAV_SKIN)
        z[covered] = np.nan


def build_navmesh(levels: Sequence[NavLevel], resolution: float = 0.1, *, climb: float = NAV_CLIMB, shut: Collection[str] = ()) -> NavMesh:
    """Grid-sampled mesh of the floors, stairs and escalators of the levels, bridging height steps up to `climb` within a grid cell, the doors named in `shut` closed."""
    fields = [field for level in levels if (field := _level_field(level, resolution)) is not None]
    closed = frozenset(shut)
    for wall in (wall for level in levels for wall in (*level.walls, *(door_wall(door) for door in level.doors if door.name in closed))):
        _block(fields, wall, resolution)
    ix, iy, z = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    for field in fields:
        ii, jj = np.nonzero(~np.isnan(field.z))
        ix, iy, z = np.concatenate([ix, field.i0 + ii]), np.concatenate([iy, field.j0 + jj]), np.concatenate([z, field.z[ii, jj]])
    empty = NavMesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    if not len(z):
        return empty
    order = np.lexsort((z, iy, ix))
    ix, iy, z = ix[order], iy[order], z[order]
    column = np.ones(len(z), dtype=bool)
    column[1:] = (ix[1:] != ix[:-1]) | (iy[1:] != iy[:-1])
    starts = np.flatnonzero(column | np.concatenate([[True], np.diff(z) > climb]))
    vx, vy, vz = ix[starts], iy[starts], np.maximum.reduceat(z, starts)
    first = np.maximum.accumulate(np.where(column[starts], np.arange(len(starts)), 0))
    slot = np.arange(len(starts)) - first
    gx, gy = vx.min(), vy.min()
    ids = np.full((slot.max() + 1, vx.max() - gx + 1, vy.max() - gy + 1), -1, dtype=np.int64)
    ids[slot, vx - gx, vy - gy] = np.arange(len(starts))
    heights = np.where(ids >= 0, vz[ids], np.nan)
    corners = ((np.s_[:-1], np.s_[:-1]), (np.s_[1:], np.s_[:-1]), (np.s_[1:], np.s_[1:]), (np.s_[:-1], np.s_[1:]))
    triangles: list[np.ndarray] = []
    for slots in itertools.product(range(ids.shape[0]), repeat=4):
        quad = [ids[k][si, sj] for k, (si, sj) in zip(slots, corners, strict=True)]
        tops = np.stack([heights[k][si, sj] for k, (si, sj) in zip(slots, corners, strict=True)])
        cells = np.all(np.stack(quad) >= 0, axis=0) & (tops.max(axis=0) - tops.min(axis=0) <= climb + _EPS)
        v00, v10, v11, v01 = (corner[cells] for corner in quad)
        triangles.extend([np.stack([v00, v10, v11], axis=1), np.stack([v00, v11, v01], axis=1)])
    faces = np.concatenate(triangles)
    if not len(faces):
        return empty
    used, faces = np.unique(faces, return_inverse=True)
    vertices = np.stack([vx[used] * resolution, vy[used] * resolution, vz[used]], axis=1).astype(np.float64)
    return NavMesh(vertices, faces.reshape(-1, 3).astype(np.int64))


def _at(position: Position, shift: tuple[float, float, float]) -> Position:
    return Position(position.x + shift[0], position.y + shift[1], position.z + shift[2])


def navmesh_levels(world: World, shifts: Mapping[str, tuple[float, float, float]]) -> list[NavLevel]:
    """The levels of a world moved by their shifts into the simulator frame, stair pairs joined when the world stacks."""
    stairs = [stair for level in world.levels.values() for stair in level.stairs]
    names = {stair.name for stair in stairs}
    stacked = world.layout == 'stacked' and len(world.levels) > 1
    levels: list[NavLevel] = []
    for level_id, level in world.levels.items():
        shift = shifts.get(level_id, (0.0, 0.0, 0.0))
        halves: list[Stairs] = []
        for stair in level.stairs:
            half = attrs.evolve(stair)
            half.joined = stacked and (stair.to in names or any(other.to == stair.name for other in stairs))
            halves.append(half)
        joined = attrs.evolve(level, stairs=halves)
        placed: list[Stairs] = []
        for half in halves:
            moved = attrs.evolve(half, position=_at(half.position, shift))
            moved.joined = half.joined
            placed.append(moved)
        levels.append(
            NavLevel(
                level=level_id,
                floors=[attrs.evolve(floor, pos=_at(floor.pos, shift)) for floor in nav_floors(joined)],
                walls=[attrs.evolve(wall, start=_at(wall.start, shift), end=_at(wall.end, shift)) for wall in nav_walls(joined)],
                stairs=placed,
                doors=[attrs.evolve(door, start=_at(door.start, shift), end=_at(door.end, shift)) for door in joined.all_doors],
            )
        )
    return levels
