"""Grid path planning over the world occupancy, for Level C.

Level C designs an encounter *on the robot's route*. Modelling that route as the straight
line from start to goal only works inside one open area: in a furnished multi-room world the
real path bends through doorways, so the straight line cuts through walls and every encounter
designed on it sits somewhere the robot never goes.

This module plans the route the robot will approximately take, so the encounter can be placed
on *that* instead. It does not try to match nav2's plan exactly: the encounter only needs to
be somewhere the robot will pass through at a predictable time, and a shortest free-space
path through the same occupancy grid is close enough. Achieved timing is measured either way.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterable, Sequence

from task_generator.shared import Position

Point = tuple[float, float]
Cell = tuple[int, int]

IsValid = Callable[[Position], bool]

#: 8-connected neighbourhood, with the diagonal cost baked in.
_NEIGHBOURS: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
)

#: Cap on expanded cells. A 109x69 world is ~7.5k cells, so this is generous for a single
#: route while still bounding a pathological search.
_MAX_EXPANSIONS = 200_000

#: Spacing for the line-of-sight check used to simplify the raw cell path.
_LOS_STEP_M = 0.2

#: How far to look for a free cell near an endpoint, in cells. The robot's start and goal are
#: forbidden for the crowd, so A* starts from the nearest genuinely free cell and the true
#: endpoints are attached afterwards.
_ENDPOINT_SEARCH_CELLS = 40


class PathError(RuntimeError):
    """No route could be planned. Recorded and aborted like any other unbuildable case."""


class _Grid:
    """Occupancy lookups in cell space, memoised.

    ``is_valid`` is comparatively expensive - it transforms a point to a rectangle and reduces
    over a slice - and A* revisits cells constantly, so caching is what makes this practical.
    """

    def __init__(self, world_map: object, is_valid: IsValid | None) -> None:
        self._map = world_map
        self._is_valid = is_valid
        self.rows, self.cols = world_map.occupancy.grid.shape  # type: ignore[attr-defined]
        self._cache: dict[Cell, bool] = {}

    def to_cell(self, pt: Point) -> Cell:
        r, c = self._map.tf_pos2grid(Position(x=pt[0], y=pt[1]))  # type: ignore[attr-defined]
        return int(r), int(c)

    def to_point(self, cell: Cell) -> Point:
        pos = self._map.tf_grid2pos((float(cell[0]), float(cell[1])))  # type: ignore[attr-defined]
        return float(pos.x), float(pos.y)

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def free(self, cell: Cell) -> bool:
        if not self.in_bounds(cell):
            return False
        cached = self._cache.get(cell)
        if cached is None:
            cached = True if self._is_valid is None else bool(self._is_valid(Position(*self.to_point(cell))))
            self._cache[cell] = cached
        return cached

    def nearest_free(self, cell: Cell, *, limit: int = _ENDPOINT_SEARCH_CELLS) -> Cell | None:
        """The closest free cell to ``cell``, searched outward in rings.

        Used for both endpoints, which are occupied by construction rather than by geometry.
        Searching adapts to whatever radius ``forbid`` used instead of hardcoding a guess.
        """
        if self.free(cell):
            return cell
        for radius in range(1, limit + 1):
            best: Cell | None = None
            best_d = math.inf
            for dr in range(-radius, radius + 1):
                for dc in (-radius, radius) if abs(dr) != radius else range(-radius, radius + 1):
                    candidate = (cell[0] + dr, cell[1] + dc)
                    if not self.free(candidate):
                        continue
                    d = math.dist(candidate, cell)
                    if d < best_d:
                        best, best_d = candidate, d
            if best is not None:
                return best
        return None


def _reconstruct(came_from: dict[Cell, Cell], cell: Cell) -> list[Cell]:
    path = [cell]
    while cell in came_from:
        cell = came_from[cell]
        path.append(cell)
    path.reverse()
    return path


def _astar(grid: _Grid, start: Cell, goal: Cell) -> list[Cell]:
    """Shortest 8-connected free path between two cells.

    The endpoints are treated as passable whatever the occupancy says. Both are routinely
    occupied by construction: the robots scenario mode ``forbid``s its own start and goal so
    the crowd cannot spawn on them, and the robot itself stands at the start.
    """
    if start == goal:
        return [start]

    open_heap: list[tuple[float, float, Cell]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start))
    came_from: dict[Cell, Cell] = {}
    best: dict[Cell, float] = {start: 0.0}
    seen: set[Cell] = set()
    expansions = 0

    def h(cell: Cell) -> float:
        return math.dist(cell, goal)

    while open_heap:
        _f, g, cell = heapq.heappop(open_heap)
        if cell == goal:
            return _reconstruct(came_from, cell)
        if cell in seen:
            continue
        seen.add(cell)

        expansions += 1
        if expansions > _MAX_EXPANSIONS:
            raise PathError(f"path search exceeded {_MAX_EXPANSIONS} expansions between {start} and {goal}")

        for dr, dc, step in _NEIGHBOURS:
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt in seen or not grid.in_bounds(nxt):
                continue
            if nxt != goal and not grid.free(nxt):
                continue
            # No corner cutting. A diagonal between two free cells can still clip the corner
            # of a wall when both orthogonal neighbours are blocked, which produces a path
            # whose own segments are not navigable - exactly what the route must never be.
            if dr and dc and not (grid.free((cell[0] + dr, cell[1])) and grid.free((cell[0], cell[1] + dc))):
                continue
            tentative = g + step
            if tentative < best.get(nxt, math.inf):
                best[nxt] = tentative
                came_from[nxt] = cell
                heapq.heappush(open_heap, (tentative + h(nxt), tentative, nxt))

    free_seen = sum(1 for c in seen if grid.free(c))
    raise PathError(
        f"no free-space route between cells {start} and {goal} on a {grid.rows}x{grid.cols} grid "
        f"({expansions} expansion(s), {free_seen} free cell(s) reachable from the start). "
        "The two are in disconnected free space."
    )


def line_of_sight(a: Point, b: Point, is_valid: IsValid | None, *, step: float = _LOS_STEP_M) -> bool:
    """Whether the straight segment ``a``->``b`` stays in free space."""
    if is_valid is None:
        return True
    length = math.dist(a, b)
    steps = max(1, int(math.ceil(length / max(step, 1e-6))))
    for i in range(steps + 1):
        f = i / steps
        if not is_valid(Position(x=a[0] + (b[0] - a[0]) * f, y=a[1] + (b[1] - a[1]) * f)):
            return False
    return True


def simplify(points: Sequence[Point], is_valid: IsValid | None) -> list[Point]:
    """Drop intermediate points a straight segment can already see past.

    A raw cell path is a staircase of 0.2 m steps; the encounter geometry wants long straight
    runs so that "the robot's heading at E" is a meaningful direction rather than the angle of
    one grid step.

    Greedy string-pulling: from each kept point, take the *furthest* later point still in line
    of sight, so every segment of the result is verified navigable. When nothing further is
    visible it falls back to the next raw point, an adjacent cell that is always reachable; that
    keeps the occupied endpoints (the robot stands at the start, start and goal are forbidden)
    from breaking it.
    """
    if len(points) <= 2:
        return list(points)

    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not line_of_sight(points[i], points[j], is_valid):
            j -= 1
        out.append(points[j])
        i = j

    deduped = [out[0]]
    for pt in out[1:]:
        if math.dist(pt, deduped[-1]) > 1e-9:
            deduped.append(pt)
    return deduped


def plan_route(
    world_map: object,
    start: Point,
    goal: Point,
    *,
    is_valid: IsValid | None,
) -> list[Point]:
    """The route the robot will approximately take, as a simplified polyline.

    Returns ``[start, goal]`` unchanged when the straight line is already clear, so open-area
    routes keep exactly the behaviour they had before planning existed.

    Raises :class:`PathError` when start or goal is off the map, or no free route joins them.
    """
    if line_of_sight(start, goal, is_valid):
        return [start, goal]

    grid = _Grid(world_map, is_valid)
    endpoints: list[Cell] = []
    for label, pt in (("start", start), ("goal", goal)):
        cell = grid.to_cell(pt)
        if not grid.in_bounds(cell):
            raise PathError(f"robot {label} ({pt[0]:.2f}, {pt[1]:.2f}) is outside the occupancy grid")
        free_cell = grid.nearest_free(cell)
        if free_cell is None:
            raise PathError(
                f"no free cell within {_ENDPOINT_SEARCH_CELLS} cells of the robot's {label} "
                f"({pt[0]:.2f}, {pt[1]:.2f}); it is walled in, not merely forbidden"
            )
        endpoints.append(free_cell)

    cells = _astar(grid, endpoints[0], endpoints[1])

    # Keep the true endpoints rather than their cell centres, so the route still starts where
    # the robot stands and ends where it was actually sent.
    points: list[Point] = [start, *(grid.to_point(c) for c in cells[1:-1]), goal]
    return simplify(points, is_valid)


# Polyline geometry
# -----------------


def route_length(route: Sequence[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(route, route[1:], strict=False))


def point_at_arc(route: Sequence[Point], distance: float) -> tuple[Point, Point]:
    """Position at ``distance`` along the polyline, and the unit heading there.

    Heading is the direction of the segment the point falls on - the robot's direction of
    travel at that moment, which is what the approach angle is measured against.
    """
    if len(route) < 2:
        raise PathError("a route needs at least two points")

    remaining = max(0.0, distance)
    last_dir: Point | None = None
    for a, b in zip(route, route[1:], strict=False):
        seg = math.dist(a, b)
        if seg <= 1e-12:
            continue
        direction = ((b[0] - a[0]) / seg, (b[1] - a[1]) / seg)
        last_dir = direction
        if remaining <= seg:
            f = remaining / seg
            return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), direction
        remaining -= seg

    if last_dir is None:
        raise PathError("route has no non-degenerate segment")
    return route[-1], last_dir


def iter_points(route: Sequence[Point], step: float) -> Iterable[Point]:
    """Sample the polyline at roughly ``step`` metres, endpoints included."""
    total = route_length(route)
    n = max(1, int(math.ceil(total / max(step, 1e-6))))
    for i in range(n + 1):
        yield point_at_arc(route, total * i / n)[0]
