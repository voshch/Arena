"""A world as the generator sees it: zones, furniture, the occupancy grid and the robot's planned route."""

from __future__ import annotations

import copy
import math
import random
import typing
from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

import attrs

from arena_simulation_setup.shared.pathing import OccupancyGrid, PathError, Point, plan_route, point_at_arc, route_length

if typing.TYPE_CHECKING:
    import numpy as np
    import shapely

from arena_simulation_setup.generative.core.bodies import RUNTIME_SAFE_DIST

from ..formats import first_level, world_roles
from ..occupancy import level_grid, load_world, world_statics
from ..populate import MIN_ROUTE_CLEARANCE, centroid, clearance, polygon, robot_zones, route_points, work_zones, world_kind
from ..route import MAX_ROUTE_M, MIN_EPISODE_S, ROBOT_SPEED_MPS
from ..zone_screen import rasterize, ref_safe_zones, route_clearance, screen_zones
from ..zones import Role, zones_by_role

#: The default robot radius, the clearance its route keeps from walls and furniture, and the pedestrian body clearance.
ROBOT_CLEAR = 0.35
#: Clearance of the robot's start and goal beyond its radius for nav2's inflation, the radius alone when no such cell is within 3 m.
ENDPOINT_MARGIN = 0.35
#: A planned leg shorter than this is replaced by the longest leg the zones offer: the base scenarios' own floor.
MIN_LEG_M = MIN_EPISODE_S * ROBOT_SPEED_MPS
#: The episode budget minus the case's reserve, where a longer route is truncated.
MAX_LEG_M = MAX_ROUTE_M
#: Name fragments marking a zone a building empties through, best first.
EGRESS_HINTS = ("exit", "entrance", "entry", "lobby", "reception", "hallway", "corridor", "foyer", "living")


def endpoint(grid: Grid, pt: Point, radius: float = ROBOT_CLEAR) -> Point:
    if grid.free(pt, radius + ENDPOINT_MARGIN):
        return pt
    return grid.nearest_free(pt, radius + ENDPOINT_MARGIN, within=3.0) or grid.nearest_free(pt, radius, within=3.0) or pt


def plan(grid: Grid, start: Point, goal: Point, radius: float = RUNTIME_SAFE_DIST) -> list[Point]:
    try:
        return plan_route(grid, start, goal, is_valid=grid.clear_of(radius))
    except PathError as exc:
        raise WorldError(str(exc)) from None


def _truncate(route: Sequence[Point], cap_m: float) -> list[Point]:
    """The prefix of `route` that is `cap_m` long, ending on an interpolated point of it."""
    out: list[Point] = [route[0]]
    left = cap_m
    for a, b in zip(route, route[1:], strict=False):
        leg = math.dist(a, b)
        if leg >= left:
            f = left / leg if leg > 0 else 0.0
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
            return out
        out.append(b)
        left -= leg
    return list(route)


def _longest_leg(grid: Grid, start: Point, candidates: Sequence[Point], *, radius: float, beat: float = 0.0) -> tuple[Point, list[Point]] | None:
    """The snapped goal with the longest drivable route from `start` up to `MAX_LEG_M`, None when none beats `beat` by a meter."""
    best: tuple[float, Point, list[Point]] | None = None
    for pt in candidates:
        goal = endpoint(grid, pt, radius)
        if math.dist(goal, start) < 1.0:
            continue
        try:
            route = plan(grid, start, goal, radius)
        except WorldError:
            continue
        length = route_length(route)
        if length > MAX_LEG_M or length < beat + 1.0:
            continue
        if best is None or length > best[0]:
            best = (length, goal, route)
    return None if best is None else (best[1], best[2])


class WorldError(RuntimeError):
    pass


@attrs.define
class Grid(OccupancyGrid):
    _labels_cache: dict[float, Any] = attrs.Factory(dict)

    @cached_property
    def dist(self) -> np.ndarray:
        import numpy as np
        import scipy.ndimage

        dist = scipy.ndimage.distance_transform_edt(~self.occupied)
        assert isinstance(dist, np.ndarray)
        return dist * self.res

    def labels_at(self, radius: float) -> np.ndarray:
        import scipy.ndimage

        key = round(radius, 2)
        if key not in self._labels_cache:
            labelled = scipy.ndimage.label(self.dist >= key)
            assert isinstance(labelled, tuple)
            self._labels_cache[key] = labelled[0]
        return self._labels_cache[key]

    def clearance(self, pt: Point) -> float:
        cell = self.cell(pt)
        return float(self.dist[cell]) if self.inside(cell) else 0.0

    def free(self, pt: Point, radius: float = ROBOT_CLEAR) -> bool:
        return self.clearance(pt) >= radius

    def component(self, pt: Point, radius: float = ROBOT_CLEAR) -> int:
        cell = self.cell(pt)
        return int(self.labels_at(min(radius, ROBOT_CLEAR))[cell]) if self.inside(cell) else 0

    def nearest_free(self, pt: Point, radius: float, within: float) -> Point | None:
        found = self.nearest(self.cell(pt), within / self.res, lambda rows, cols: self.dist[rows, cols] >= radius)
        return None if found is None else self.xy(found)


@attrs.frozen
class LoadedWorld:
    """The parts of a world every base shares: its working level, KG roles and occupancy grid with furniture stamped."""

    doc: dict[str, Any]
    overrides: dict[str, Role]
    grid: Grid | None
    origin: Point = (0.0, 0.0)
    unresolved: tuple[str, ...] = ()

    @classmethod
    def read(cls, root: str | Path) -> LoadedWorld:
        from arena_simulation_setup.tree.World.view import WorldView

        root = Path(root)
        world = load_world(root)
        doc = first_level(root, world)
        if doc is None:
            raise WorldError(f"{root.name}: no world.yaml")
        statics = world_statics(root, world)
        loaded = level_grid(root, statics, world)
        grid = None if loaded is None else Grid(occupied=loaded[0], res=loaded[1], ox=loaded[2], oy=loaded[3])
        origins = WorldView(root.resolve()).level_origins() or {}
        unresolved = tuple(sorted({str(s["model"]) for s in statics if not s["resolved"]}))
        return cls(doc=doc, overrides=world_roles(root, world), grid=grid, origin=origins.get(str(doc["id"]), (0.0, 0.0)), unresolved=unresolved)


@attrs.define
class WorldContext:
    name: str
    root: Path
    zones: list[dict[str, Any]]
    roles: dict[str, Role]
    robot: dict[str, Any]
    grid: Grid | None
    start: Point
    goal: Point
    route: list[Point]
    homes: frozenset[str]
    kind: str
    level: str = ""
    origin: Point = (0.0, 0.0)
    robot_radius: float = ROBOT_CLEAR

    @classmethod
    def load(cls, root: str | Path, base_doc: Mapping[str, Any], parts: LoadedWorld | None = None, *, robot_radius: float = ROBOT_CLEAR) -> WorldContext:
        """The world at `root` around the base's robot, its route kept `robot_radius` clear, `parts` the already read world."""
        root = Path(root)
        world = root.name
        if parts is None:
            parts = LoadedWorld.read(root)
        doc = copy.deepcopy(parts.doc)
        zones = [z for z in doc.get("zones") or [] if isinstance(z, dict) and z.get("name")]
        robots = base_doc.get("robots") or []
        if not robots or not isinstance(robots[0], dict) or robots[0].get("start") is None:
            raise WorldError(f"{world}: the base scenario declares no robot")
        robot = dict(robots[0])
        pts = route_points(robot, zones)
        if len(pts) < 2:
            raise WorldError(f"{world}: the robot's route has fewer than two resolvable endpoints")
        start, goal = pts[0], pts[1]

        grid = parts.grid
        route = [start, goal]
        if grid is not None:
            start = endpoint(grid, start, robot_radius)
            goal = endpoint(grid, goal, robot_radius)
            route = plan(grid, start, goal, robot_radius)
            length = route_length(route)
            if length > MAX_LEG_M:
                cut = _truncate(route, MAX_LEG_M)
                goal = endpoint(grid, cut[-1], robot_radius)
                replanned = plan(grid, start, goal, robot_radius)
                route = replanned if route_length(replanned) <= MAX_LEG_M * 1.2 else cut
            elif length < MIN_LEG_M:
                longer = _longest_leg(grid, start, [centroid(polygon(z)) for z in zones if len(polygon(z)) >= 3], radius=robot_radius, beat=length)
                if longer is not None:
                    goal, route = longer

        overrides = parts.overrides
        roles = {}
        for role, names in zones_by_role(zones, overrides).items():
            for n in names:
                roles[n] = role
        homes: frozenset[str] = frozenset()
        if grid is not None:
            clearance_map = (grid.dist, grid.res, grid.ox, grid.oy)
            homes = ref_safe_zones(screen_zones(zones, world_root=root, route=route, min_route_clearance=MIN_ROUTE_CLEARANCE, robot_zones=robot_zones(robot), clearance_map=clearance_map))
        return cls(name=world, root=root, zones=zones, roles=roles, robot=robot, grid=grid, start=start, goal=goal, route=route, homes=homes, kind=world_kind(zones, overrides), level=str(doc["id"]), origin=parts.origin, robot_radius=robot_radius)

    @cached_property
    def polygons(self) -> dict[str, list[Point]]:
        return {str(z["name"]): polygon(z) for z in self.zones if len(polygon(z)) >= 3}

    @cached_property
    def shapes(self) -> dict[str, shapely.Polygon]:
        import shapely

        return {name: shapely.Polygon(poly) for name, poly in self.polygons.items()}

    @cached_property
    def centroids(self) -> dict[str, Point]:
        return {name: centroid(poly) for name, poly in self.polygons.items()}

    def zone_names(self) -> list[str]:
        return sorted(self.polygons)

    def zone_of(self, pt: Point) -> str | None:
        import shapely

        here = shapely.Point(pt)
        return next((name for name, shape in self.shapes.items() if shape.covers(here)), None)

    def zone_point(self, name: str, *, need: float = 0.4) -> Point:
        """The reachable point nearest the zone's centroid `need` clear of walls, off the robot's endpoints, else the centroid."""
        if name not in self.centroids:
            raise WorldError(f"{name!r} is not a zone of {self.name}, known: {self.zone_names()}")
        c = self.centroids[name]
        grid = self.grid
        if grid is None:
            return c
        import numpy as np

        mask = rasterize(self.polygons[name], grid.shape, grid.res, grid.ox, grid.oy)
        reachable = grid.labels_at(ROBOT_CLEAR) == grid.component(self.start)
        for clear in (need + 0.3, need, max(0.5, need / 2)):
            ok = mask & (grid.dist >= clear) & reachable
            rows, cols = np.nonzero(ok)
            if rows.size == 0:
                continue
            xs, ys = grid.xy((rows, cols))
            far = (np.hypot(xs - self.start[0], ys - self.start[1]) >= 1.8 + need) & (np.hypot(xs - self.goal[0], ys - self.goal[1]) >= 1.3 + need)
            keep = far & (route_clearance(xs, ys, self.route) >= 1.5 + need)
            if not keep.any():
                keep = far
            if not keep.any():
                continue
            d = np.hypot(xs - c[0], ys - c[1])
            d[~keep] = np.inf
            i = int(np.argmin(d))
            return (float(xs[i]), float(ys[i]))
        raise WorldError(f"zone {name!r} has no room for a formation of radius {need:.1f} m away from the robot's endpoints")

    def zone_sample(self, name: str, rng: random.Random, *, need: float = 0.4) -> Point:
        """A random reachable point in the zone `need` clear of walls and furniture, the centroid without a grid or a clear cell."""
        if name not in self.centroids:
            raise WorldError(f"{name!r} is not a zone of {self.name}, known: {self.zone_names()}")
        grid = self.grid
        if grid is None:
            return self.centroids[name]
        import numpy as np

        clear = rasterize(self.polygons[name], grid.shape, grid.res, grid.ox, grid.oy) & (grid.dist >= need)
        reachable = clear & (grid.labels_at(ROBOT_CLEAR) == grid.component(self.start))
        rows, cols = np.nonzero(reachable if reachable.any() else clear)
        if rows.size == 0:
            return self.centroids[name]
        i = rng.randrange(rows.size)
        return grid.xy((int(rows[i]), int(cols[i])))

    def exit_zone(self) -> str | None:
        transit = sorted(n for n, r in self.roles.items() if r is Role.TRANSIT and n in self.polygons)
        if transit:
            return transit[0]
        for hint in EGRESS_HINTS:
            matched = sorted(n for n in self.polygons if hint in n.lower())
            if matched:
                return matched[0]
        return max(self.shapes, key=lambda n: (self.shapes[n].area, n)) if self.shapes else None

    def work_zone(self) -> str | None:
        wanted = [n for n in work_zones(self.zones, self.roles) if n in self.polygons]
        return max(wanted, key=lambda n: (round(self.shapes[n].area, 1), "work" in n.lower(), n)) if wanted else None

    def other_zones(self, exclude: Sequence[str] = ()) -> list[str]:
        """Zones a walker may head for: not private, not the robot's own."""
        forbidden = set(exclude) | set(robot_zones(self.robot))
        return [n for n in self.zone_names() if self.roles.get(n) is not Role.PRIVATE and n not in forbidden]

    @cached_property
    def route_length(self) -> float:
        return route_length(self.route)

    def point_at(self, fraction: float) -> tuple[Point, Point]:
        """Position and unit heading `fraction` of the way along the planned route, (last point, +x) on a degenerate one."""
        if self.route_length <= 1e-9:
            return self.route[-1], (1.0, 0.0)
        return point_at_arc(self.route, max(0.0, min(1.0, fraction)) * self.route_length)

    def fraction_of(self, pt: Point) -> float:
        """Arc-length fraction of the route point nearest `pt`, 0 on a degenerate route."""
        import shapely

        if self.route_length <= 1e-9:
            return 0.0
        return float(shapely.LineString(self.route).project(shapely.Point(pt))) / self.route_length

    def route_clearance(self, pt: Point) -> float:
        return clearance(pt, self.route)

    def passage_width(self, pt: Point, heading: Point, limit: float = 8.0) -> tuple[float, Point, Point]:
        """Width of free space through `pt` perpendicular to `heading`, and the two wall points."""
        if self.grid is None:
            return limit, (pt[0] - heading[1] * limit / 2, pt[1] + heading[0] * limit / 2), (pt[0] + heading[1] * limit / 2, pt[1] - heading[0] * limit / 2)
        normal = (-heading[1], heading[0])
        ends = []
        for sign in (1.0, -1.0):
            d = 0.0
            while d < limit and self.grid.clearance((pt[0] + sign * normal[0] * d, pt[1] + sign * normal[1] * d)) > 0.05:
                d += 0.05
            ends.append((pt[0] + sign * normal[0] * d, pt[1] + sign * normal[1] * d))
        return math.dist(ends[0], ends[1]), ends[0], ends[1]

    def narrowest_passage(self, fraction: float, span: float = 3.0, step: float = 0.25) -> tuple[Point, Point, float]:
        """(point, heading, width) of the route's narrowest cross-section within `span` meters of `fraction`."""
        total = max(self.route_length, 1e-9)
        samples: list[tuple[float, Point, Point, float]] = []
        n = int(span / step)
        for i in range(-n, n + 1):
            f = fraction + i * step / total
            if not 0.05 < f < 0.95:
                continue
            pt, heading = self.point_at(f)
            if math.dist(pt, self.start) < 2.1 or math.dist(pt, self.goal) < 1.6:
                continue
            width, _l, _r = self.passage_width(pt, heading)
            samples.append((abs(i) * step, pt, heading, width))
        if not samples:
            pt, heading = self.point_at(fraction)
            return pt, heading, self.passage_width(pt, heading)[0]
        narrowest = min(w for _d, _p, _h, w in samples)
        _d, pt, heading, width = min((s for s in samples if s[3] <= narrowest + 0.2), key=lambda s: s[0])
        return pt, heading, width

    def fit_along_route(self, fraction: float, need: float, *, span: float = 0.3, step: float = 0.25) -> tuple[Point, Point, float] | None:
        """(point, heading, fraction) nearest `fraction` where a formation of radius `need` fits on the route, None when none does."""
        total = max(self.route_length, 1e-9)
        n = int(span * total / step)
        for i in range(0, n + 1):
            for sign in (1, -1) if i else (1,):
                f = fraction + sign * i * step / total
                if not 0.05 < f < 0.95:
                    continue
                pt, heading = self.point_at(f)
                if self.grid is not None:
                    if self.grid.clearance(pt) < need + 0.3:
                        continue
                    width, _l, _r = self.passage_width(pt, heading)
                    if width < 2 * need + 0.8:
                        continue
                if math.dist(pt, self.start) < 1.8 + need or math.dist(pt, self.goal) < 1.3 + need:
                    continue
                return pt, heading, f
        return None

    def valid(self, pt: Point, *, radius: float = 0.4, on_route: bool = False) -> str | None:
        """None when a pedestrian may stand at `pt`, otherwise why not."""
        if self.grid is not None:
            if not self.grid.inside(self.grid.cell(pt)):
                return "outside the map"
            if not self.grid.free(pt, radius):
                return f"{self.grid.clearance(pt):.2f} m from a wall or furniture"
            if self.grid.component(pt, radius) != self.grid.component(self.start, radius):
                return "not reachable from the robot's start"
        if math.dist(pt, self.start) < 1.5:
            return "within 1.5 m of the robot's start"
        if math.dist(pt, self.goal) < 1.0:
            return "within 1.0 m of the robot's goal"
        if not on_route and self.route_clearance(pt) < 1.5:
            return "within 1.5 m of the robot's route"
        return None

    def _candidates(self, pt: Point, *, radius: float, on_route: bool, within: float) -> list[Point]:
        """Cell centers within `within` of `pt` clear by `ROBOT_CLEAR` at least that pass the checks of `valid`, nearest first."""
        import numpy as np

        grid = self.grid
        if grid is None:
            return []
        r0, c0 = grid.cell(pt)
        k = int(math.ceil(within / grid.res))
        rows, cols = grid.shape
        rs, re = max(0, r0 - k), min(rows, r0 + k + 1)
        cs, ce = max(0, c0 - k), min(cols, c0 + k + 1)
        if rs >= re or cs >= ce:
            return []
        labels = grid.labels_at(min(radius, ROBOT_CLEAR))[rs:re, cs:ce]
        ok = (grid.dist[rs:re, cs:ce] >= max(radius, ROBOT_CLEAR)) & (labels == grid.component(self.start, radius))
        rr, cc = np.nonzero(ok)
        xs, ys = grid.xy((rr + rs, cc + cs))
        d = np.hypot(xs - pt[0], ys - pt[1])
        keep = (d <= within) & (np.hypot(xs - self.start[0], ys - self.start[1]) >= 1.5) & (np.hypot(xs - self.goal[0], ys - self.goal[1]) >= 1.0)
        if not on_route:
            keep &= route_clearance(xs, ys, self.route) >= 1.5
        order = np.argsort(d[keep], kind="stable")
        return [(float(x), float(y)) for x, y in zip(xs[keep][order], ys[keep][order], strict=True)]

    def resolve(self, pt: Point, *, radius: float = 0.4, on_route: bool = False, within: float = 1.5) -> tuple[Point, str]:
        """`pt` if valid, else the nearest valid point within `within`. Raises when none."""
        why = self.valid(pt, radius=radius, on_route=on_route)
        if why is None:
            return pt, ""
        if self.grid is not None:
            for near in self._candidates(pt, radius=radius, on_route=on_route, within=within):
                if self.valid(near, radius=radius, on_route=on_route) is None:
                    return near, f"moved {math.dist(pt, near):.2f} m ({why})"
        raise WorldError(f"no valid point within {within} m of ({pt[0]:.2f}, {pt[1]:.2f}): {why}")

    def context_for_model(self) -> str:
        """The world as the model sees it: zones with roles and centroids, the route, doors."""
        lines = [f"world: {self.name} ({self.kind})", "zones (name, role, centroid):"]
        for name in self.zone_names():
            c = self.centroids[name]
            lines.append(f"  - {name}: {self.roles.get(name, Role.UNKNOWN).value} at ({c[0]:.1f}, {c[1]:.1f})")
        lines.append(f"robot: start ({self.start[0]:.1f}, {self.start[1]:.1f}) in {self.zone_of(self.start) or '?'} -> goal ({self.goal[0]:.1f}, {self.goal[1]:.1f}) in {self.zone_of(self.goal) or '?'}, route {self.route_length:.1f} m")
        marks = []
        for f in (0.25, 0.5, 0.75):
            pt, _h = self.point_at(f)
            marks.append(f"{f:.2f} -> ({pt[0]:.1f}, {pt[1]:.1f}) in {self.zone_of(pt) or '?'}")
        lines.append("route fractions: " + "; ".join(marks))
        return "\n".join(lines)
