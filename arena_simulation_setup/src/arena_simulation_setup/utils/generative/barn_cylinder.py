import logging
import math
from collections import deque

import numpy as np
import shapely

from . import BaseConfiguration
from .barn import BarnBase
from .utils import to_walls

logger = logging.getLogger(__name__)


class WorldGeneratorBarnCylinder(BarnBase):
    """BARN cylinder field: cellular-automata grid of small cylinders for a Jackal.

    Follows the literal BARN simulation benchmark (cs.utexas.edu/~xiao/BARN/BARN.html):
    random-fill + CA smoothing produces a navigable obstacle field; connectivity is
    verified in C-space before accepting the map. Defaults give a 30x30 grid (BARN scale)
    with a robot_clearance honest for a Jackal (radius plus cylinder).
    """

    SCENARIO = 'barn_cylinder'

    class Configuration(BaseConfiguration):
        width: float = 10.0
        height: float = 10.0
        margin: float = 0.5  # clear border inside the boundary walls
        cell_size: float = 0.3  # CA grid cell and cylinder spacing (m)
        fill_probability: float = 0.45  # initial random wall fraction
        ca_iterations: int = 4  # cellular-automata smoothing passes
        birth_limit: int = 5  # empty cell becomes wall if >= this many of 8 neighbors are walls
        death_limit: int = 4  # wall cell survives only if >= this many of 8 neighbors are walls
        cylinder_radius: float = 0.075  # rendered obstacle radius (m)
        robot_clearance: float = 0.28  # C-space radius for connectivity, Jackal radius plus cylinder (m)
        max_tries: int = 50  # reject-sampling cap before raising

    config: Configuration

    def _build(self):
        if self._built:
            return

        c = self.config
        nx = int((c.width - 2 * c.margin) / c.cell_size)
        ny = int((c.height - 2 * c.margin) / c.cell_size)
        if nx < 3 or ny < 3:
            raise ValueError(f'arena too small for barn_cylinder: grid is {nx}x{ny}, need at least 3x3 (reduce margin/cell_size or grow width/height)')

        def cell_world(i: int, j: int) -> tuple[float, float]:
            return c.margin + (i + 0.5) * c.cell_size, c.margin + (j + 0.5) * c.cell_size

        si, sj = nx // 2, 0
        gi, gj = nx // 2, ny - 1
        start_x, start_y = cell_world(si, sj)
        goal_x, goal_y = cell_world(gi, gj)

        dilate_r = math.ceil(c.robot_clearance / c.cell_size)

        def clear_disc(grid: np.ndarray, ci: int, cj: int) -> None:
            for di in range(-dilate_r, dilate_r + 1):
                for dj in range(-dilate_r, dilate_r + 1):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < nx and 0 <= nj < ny:
                        grid[nj, ni] = False

        def ca_step(grid: np.ndarray) -> np.ndarray:
            # pad with True (wall) so border cells count out-of-bounds as occupied
            padded = np.pad(grid, 1, mode='constant', constant_values=True)
            nbrs = padded[:-2, :-2].astype(np.int8) + padded[:-2, 1:-1].astype(np.int8) + padded[:-2, 2:].astype(np.int8) + padded[1:-1, :-2].astype(np.int8) + padded[1:-1, 2:].astype(np.int8) + padded[2:, :-2].astype(np.int8) + padded[2:, 1:-1].astype(np.int8) + padded[2:, 2:].astype(np.int8)
            return (grid & (nbrs >= c.death_limit)) | (~grid & (nbrs >= c.birth_limit))

        def dilate(grid: np.ndarray, r: int) -> np.ndarray:
            # Chebyshev dilation: blocked if any cell within Chebyshev radius r is occupied
            padded = np.pad(grid, r, mode='constant', constant_values=False)
            blocked = np.zeros((ny, nx), dtype=bool)
            for di in range(2 * r + 1):
                for dj in range(2 * r + 1):
                    blocked |= padded[dj : dj + ny, di : di + nx]
            return blocked

        def bfs_reaches_goal(grid: np.ndarray) -> bool:
            # 4-connected BFS on free cells in dilated C-space
            blocked = dilate(grid, dilate_r)
            if blocked[sj, si] or blocked[gj, gi]:
                return False
            visited = np.zeros((ny, nx), dtype=bool)
            q: deque[tuple[int, int]] = deque()
            q.append((si, sj))
            visited[sj, si] = True
            while q:
                ci, cj = q.popleft()
                if ci == gi and cj == gj:
                    return True
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < nx and 0 <= nj < ny and not visited[nj, ni] and not blocked[nj, ni]:
                        visited[nj, ni] = True
                        q.append((ni, nj))
            return False

        grid: np.ndarray | None = None
        for _ in range(c.max_tries):
            g = np.array(
                [[self.rng.random() < c.fill_probability for _ in range(nx)] for _ in range(ny)],
                dtype=bool,
            )
            clear_disc(g, si, sj)
            clear_disc(g, gi, gj)

            for _ in range(c.ca_iterations):
                g = ca_step(g)
            clear_disc(g, si, sj)
            clear_disc(g, gi, gj)

            if bfs_reaches_goal(g):
                grid = g
                break

        if grid is None:
            raise ValueError(f'could not generate a connected barn_cylinder field in {c.max_tries} tries; lower fill_probability or increase max_tries')

        arena, walls = self._arena()
        n_cylinders = 0
        for j in range(ny):
            for i in range(nx):
                if grid[j, i]:
                    cx, cy = cell_world(i, j)
                    walls += to_walls(shapely.Point(cx, cy).buffer(c.cylinder_radius, quad_segs=2))
                    n_cylinders += 1

        self._set_level('barn_cylinder', arena, walls, f'BARN cylinder field, {n_cylinders} cylinders')
        self._set_scenario((start_x, start_y, math.pi / 2), (goal_x, goal_y, math.pi / 2))

        logger.info(self.config)
        self._built = True
