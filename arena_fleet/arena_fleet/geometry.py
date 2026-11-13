import heapq
import math
import typing
from collections.abc import Iterator, Sequence

Point = tuple[float, float]
Wall = tuple[Point, Point]
Box = tuple[float, float, float, float]
Funnel = tuple[float, float, float, float]

RES = 0.1
ROBOT_R = 0.35
C0 = 1.0
CLEAR_K = 6.0


class Occupancy(typing.Protocol):
    def occupied(self, tau: float, x: float, y: float, inflate: bool = True) -> bool: ...


def seg_cross(p: Point, q: Point, a: Point, b: Point) -> bool:
    def orient(u: Point, v: Point, w: Point) -> float:
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])

    return (orient(p, q, a) * orient(p, q, b) < 0) and (orient(a, b, p) * orient(a, b, q) < 0)


def grown(poly: Sequence[Point], m: float) -> Box:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m


def in_box(box: Box, x: float, y: float) -> bool:
    return box[0] <= x <= box[1] and box[2] <= y <= box[3]


def plan_len(p: Sequence[Point]) -> float:
    return sum(math.dist(p[i], p[i + 1]) for i in range(len(p) - 1))


def hungarian(c: Sequence[Sequence[float]]) -> list[int]:
    """Minimum-cost assignment of a square matrix, the column of each row."""
    n = len(c)
    u, v, p, way = [0.0] * (n + 1), [0.0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [math.inf] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = c[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]
    out = [0] * n
    for j in range(1, n + 1):
        out[p[j] - 1] = j - 1
    return out


class Grid:
    """Wall occupancy with per-cell clearance; plans are descents of a cost field from the goal."""

    NB: typing.ClassVar[list[tuple[int, int, float]]] = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

    def __init__(self, walls: Sequence[Wall]) -> None:
        xs = [p[0] for w in walls for p in w]
        ys = [p[1] for w in walls for p in w]
        self.x0, self.y0 = min(xs) - 0.5, min(ys) - 0.5
        self.nx, self.ny = int((max(xs) + 0.5 - self.x0) / RES) + 1, int((max(ys) + 0.5 - self.y0) / RES) + 1
        self.clear = [math.inf] * (self.nx * self.ny)
        pq: list[tuple[float, int]] = []
        for a, b in walls:
            for k in range(int(math.dist(a, b) / (RES / 2)) + 1):
                f = min(k * (RES / 2) / max(math.dist(a, b), 1e-9), 1.0)
                c = self.cell(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
                if self.clear[c] > 0:
                    self.clear[c] = 0.0
                    heapq.heappush(pq, (0.0, c))
        while pq:
            d, c = heapq.heappop(pq)
            if d > self.clear[c] or d > 2.5:
                continue
            for nc, w in self.neigh(c):
                nd = d + w * RES
                if nd < self.clear[nc]:
                    self.clear[nc] = nd
                    heapq.heappush(pq, (nd, nc))

    def cell(self, x: float, y: float) -> int:
        return min(max(int((y - self.y0) / RES), 0), self.ny - 1) * self.nx + min(max(int((x - self.x0) / RES), 0), self.nx - 1)

    def xy(self, c: int) -> Point:
        return self.x0 + (c % self.nx + 0.5) * RES, self.y0 + (c // self.nx + 0.5) * RES

    def clearance(self, p: Point) -> float:
        return self.clear[self.cell(*p)]

    def neigh(self, c: int) -> Iterator[tuple[int, float]]:
        cx, cy = c % self.nx, c // self.nx
        for dx, dy, w in self.NB:
            x, y = cx + dx, cy + dy
            if 0 <= x < self.nx and 0 <= y < self.ny:
                yield y * self.nx + x, w

    def _step(self, c: int, funnel: Funnel | None) -> float:
        x, y = self.xy(c)
        step = 1.0 + CLEAR_K * max(0.0, (C0 - self.clear[c]) / C0) ** 2
        if funnel and abs(x - funnel[0]) <= funnel[2]:
            step *= 1.0 + funnel[3] * (y - funnel[1]) ** 2
        return step

    def field(self, goal: Point, closed: Sequence[Box], funnel: Funnel | None = None) -> list[float]:
        """Dijkstra from the goal around walls and closed boxes; funnel = (door x, door y, length, k) aligns plans with a door."""
        dist = [math.inf] * (self.nx * self.ny)
        g = self.cell(*goal)
        dist[g] = 0.0
        pq = [(0.0, g)]
        while pq:
            d, c = heapq.heappop(pq)
            if d > dist[c]:
                continue
            for nc, w in self.neigh(c):
                if self.clear[nc] < ROBOT_R:
                    continue
                x, y = self.xy(nc)
                if any(in_box(b, x, y) for b in closed):
                    continue
                nd = d + w * RES * self._step(nc, funnel)
                if nd < dist[nc]:
                    dist[nc] = nd
                    heapq.heappush(pq, (nd, nc))
        return dist

    def descend(self, field: Sequence[float], x: float, y: float) -> list[Point]:
        """Path down the field from the nearest passable cell to (x, y)."""
        c = self.cell(x, y)
        if not math.isfinite(field[c]):
            seen, ring, best = {c}, [c], None
            for _ in range(30):
                nxt: list[int] = []
                for k in ring:
                    for nc, _w in self.neigh(k):
                        if nc not in seen:
                            seen.add(nc)
                            nxt.append(nc)
                fin = [k for k in nxt if math.isfinite(field[k])]
                if fin:
                    best = min(fin, key=lambda k: field[k])
                    break
                ring = nxt
            if best is None:
                return []
            c = best
        out = [c]
        for _ in range(4000):
            if field[c] <= 0.0:
                break
            nc = min((k for k, _w in self.neigh(c)), key=lambda k: field[k])
            if field[nc] >= field[c]:
                break
            c = nc
            out.append(c)
        return self.thin(out)

    def thin(self, cells: Sequence[int]) -> list[Point]:
        return [self.xy(k) for k in cells[::2]] + ([self.xy(cells[-1])] if len(cells) % 2 == 0 else [])

    def route(self, field: Sequence[float], x: float, y: float, view: Occupancy, v_nom: float, occ_cost: float, funnel: Funnel | None = None, horizon: float = 30.0) -> list[Point]:
        """A* to the goal with the static field as heuristic; cells the people map marks occupied on arrival cost occ_cost per m."""
        start = self.descend(field, x, y)
        if not start:
            return []
        s0 = self.cell(*start[0])
        best: dict[int, float] = {s0: 0.0}
        parent: dict[int, int] = {}
        pq = [(field[s0], 0.0, 0.0, s0)]
        expanded = 0
        while pq and expanded < 8000:
            _f, g, length, c = heapq.heappop(pq)
            expanded += 1
            if field[c] <= 0.0:
                out = [c]
                while out[-1] in parent:
                    out.append(parent[out[-1]])
                return self.thin(out[::-1])
            if g > best.get(c, math.inf):
                continue
            for nc, w in self.neigh(c):
                if not math.isfinite(field[nc]):
                    continue
                mx, my = self.xy(nc)
                ln = length + w * RES
                busy = ln / v_nom <= horizon and view.occupied(ln / v_nom, mx, my)
                ng = g + w * RES * (self._step(nc, funnel) + (occ_cost if busy else 0.0))
                if ng < best.get(nc, math.inf):
                    best[nc], parent[nc] = ng, c
                    heapq.heappush(pq, (ng + field[nc], ng, ln, nc))
        return start
