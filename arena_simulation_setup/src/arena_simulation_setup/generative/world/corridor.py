"""Corridor trees on the layout IR."""

import itertools
import math

import pydantic
import shapely

from arena_simulation_setup.tree.World import World

from . import PedestrianConfiguration, WithPedestrians
from .layout import Area, Layout, Segment, compile_layout

Point = tuple[float, float]


class WorldGeneratorCorridor(WithPedestrians):
    """A tree of corridors grown from the center, each branch joined to the nearest node (L1)."""

    class Configuration(PedestrianConfiguration):
        width: float = pydantic.Field(30.0, gt=0)
        height: float = pydantic.Field(30.0, gt=0)
        corridor_width: float = pydantic.Field(2.0, gt=0)
        width_spread: float = pydantic.Field(0.0, ge=0, lt=1)
        branches: int = pydantic.Field(10, ge=1)
        bends: int = pydantic.Field(0, ge=0, le=8)
        wiggle: float = pydantic.Field(0.0, ge=0, le=0.5)
        diagonal: bool = True

    config: Configuration

    def configure(self, configuration: dict):
        self.config = self.Configuration.model_validate(configuration)
        self.warnings = []
        self.diagnostics = None

    def _snap(self, value: float) -> float:
        return round(value / self.config.resolution) * self.config.resolution

    def _kinks(self, a: Point, b: Point, margin: float) -> list[Point]:
        """Up to `bends` points along a-b, each pushed across it by up to `wiggle` of its length."""
        c = self.config
        length = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        n = min(c.bends, int(length // (4 * c.corridor_width)) - 1)
        if n <= 0:
            return []
        nx, ny = (a[1] - b[1]) / length, (b[0] - a[0]) / length
        kinks = []
        for i in range(1, n + 1):
            t, off = i / (n + 1), self.rng.uniform(-c.wiggle, c.wiggle) * length
            x = min(max(a[0] + t * (b[0] - a[0]) + off * nx, margin), c.width - margin)
            y = min(max(a[1] + t * (b[1] - a[1]) + off * ny, margin), c.height - margin)
            kinks.append((self._snap(x), self._snap(y)))
        return kinks

    def _octilinear(self, a: Point, b: Point) -> list[Point]:
        """The turn between a diagonal and an axis leg from a to b, or between two axis legs without `diagonal`."""
        if not self.config.diagonal:
            turn = (b[0], a[1]) if self.rng.random() < 0.5 else (a[0], b[1])
            return [turn] if turn not in (a, b) else []
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = min(abs(dx), abs(dy))
        turn = (a[0] + math.copysign(d, dx), a[1] + math.copysign(d, dy)) if self.rng.random() < 0.5 else (b[0] - math.copysign(d, dx), b[1] - math.copysign(d, dy))
        return [turn] if turn not in (a, b) else []

    def _winding(self, new: Point, near: Point, margin: float, width: float) -> tuple[list[Segment], list[Area]]:
        """Segments and mitered turn pieces of one octilinear branch."""
        path = [new]
        for kink in [*self._kinks(new, near, margin), near]:
            path += [p for p in (*self._octilinear(path[-1], kink), kink) if p != path[-1]]
        half = width / 2
        extent = shapely.box(0, 0, self.config.width, self.config.height)
        (px, py), (qx, qy) = path[-2], path[-1]
        run = math.dist(path[-2], path[-1])
        end = (qx + (qx - px) / run * half, qy + (qy - py) / run * half)
        segments = [Segment(a=a, b=b, width=width) for a, b in itertools.pairwise(path)]
        pieces = [shapely.LineString([path[-1], end]).buffer(half, cap_style='flat')]
        for u, v, w in zip(path, path[1:-1], path[2:], strict=False):
            back = [v[i] + (u[i] - v[i]) * min(1.0, width / math.dist(u, v)) for i in (0, 1)]
            ahead = [v[i] + (w[i] - v[i]) * min(1.0, width / math.dist(v, w)) for i in (0, 1)]
            pieces.append(shapely.LineString([back, v, ahead]).buffer(half, cap_style='flat', join_style='mitre', mitre_limit=2.0))
        turns = []
        for piece in pieces:
            parts = [g for g in shapely.get_parts(piece.intersection(extent)) if isinstance(g, shapely.Polygon) and g.area > 0]
            if parts:
                turns.append(Area(polygon=parts[0] if len(parts) == 1 else shapely.MultiPolygon(parts)))
        return segments, turns

    def _sample(self, nodes: list[Point], margin: float) -> tuple[Point, Point]:
        """A random point and its nearest node (L1)."""
        c = self.config
        new = (self._snap(self.rng.uniform(margin, c.width - margin)), self._snap(self.rng.uniform(margin, c.height - margin)))
        return new, min(nodes, key=lambda node: abs(node[0] - new[0]) + abs(node[1] - new[1]))

    def compute(self) -> World:
        c = self.config
        margin = c.corridor_width * (1 + c.width_spread) / 2
        if min(c.width, c.height) <= 2 * margin:
            raise ValueError(f'a {c.width} x {c.height} m extent leaves no room for corridors up to {2 * margin} m wide')
        nodes: list[Point] = [(self._snap(c.width / 2), self._snap(c.height / 2))]
        layout = Layout()
        for _ in range(c.branches):
            new, near = self._sample(nodes, margin)
            for _ in range(19):
                if not c.bends or math.dist(new, near) >= 3 * c.corridor_width:
                    break
                new, near = self._sample(nodes, margin)
            width = c.corridor_width * self.rng.uniform(1 - c.width_spread, 1 + c.width_spread)
            if c.bends and math.dist(new, near) >= 3 * c.corridor_width:
                segments, turns = self._winding(new, near, margin, width)
                layout.segments.extend(segments)
                layout.areas.extend(turns)
            else:
                elbow = (new[0], near[1]) if self.rng.random() < 0.5 else (near[0], new[1])
                layout.segments.extend(Segment(a=a, b=b, width=width) for a, b in ((new, elbow), (elbow, near)) if a != b)
                half = width / 2
                layout.areas.extend(Area(polygon=shapely.box(x - half, y - half, x + half, y + half)) for x, y in dict.fromkeys((new, elbow, near)))
            nodes.append(new)
        if not layout.segments:
            raise ValueError(f'no branch fits a {c.width} x {c.height} m extent')
        world, self.diagnostics = compile_layout(layout)
        return world
