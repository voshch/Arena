"""Rooms on a grid, joined by doors along a random spanning tree grown from an entrance room."""

from __future__ import annotations

import pydantic
import shapely

from arena_simulation_setup.shared import Door, Position
from arena_simulation_setup.tree.World import World

from . import BaseConfiguration, WorldGeneratorImpl
from .layout import Area, Layout, Region, compile_layout

ENTRANCE = "entrance"
GROW_TRIES = 20

Cell = tuple[int, int]  # (row, col), row 0 holds the entrance


def room_name(cell: Cell, entrance: Cell) -> str:
    return ENTRANCE if cell == entrance else f"room_{cell[0]}_{cell[1]}"


class WorldGeneratorRooms(WorldGeneratorImpl):
    """A building of rows x cols rooms, every room reachable from the entrance through doors."""

    class Configuration(BaseConfiguration):
        rows: int = pydantic.Field(3, ge=1, description='Number of room rows, row 0 holding the entrance.')
        cols: int = pydantic.Field(4, ge=1, description='Number of room columns.')
        room_min: float = pydantic.Field(4.0, gt=0, description='Lower bound in meters of the room side drawn per row and per column.')
        room_max: float = pydantic.Field(7.0, gt=0, description='Upper bound in meters of the room side drawn per row and per column.')
        wall: float = pydantic.Field(0.2, gt=0, description='Wall thickness between neighboring rooms in meters.')
        door_width: float = pydantic.Field(1.0, gt=0, description='Width of each door between two rooms in meters.')
        max_depth: int = pydantic.Field(0, ge=0, description='Maximum number of doors from the entrance to any room (0 = unbounded).')
        deep: float = pydantic.Field(0.5, ge=0.0, le=1.0, description='Probability of growing the tree from the newest room, where high values give long chains.')
        loops: int = pydantic.Field(0, ge=0, description='Number of extra doors beyond the spanning tree, each closing a cycle.')

        @pydantic.model_validator(mode="after")
        def _sized(self) -> WorldGeneratorRooms.Configuration:
            if self.room_max < self.room_min:
                raise ValueError(f"room_max {self.room_max} is below room_min {self.room_min}")
            if self.door_width + 2 * self.wall > self.room_min:
                raise ValueError(f"a {self.door_width} m door with {self.wall} m walls does not fit a {self.room_min} m room side")
            return self

    config: Configuration
    _layout: Layout | None
    depths: dict[Cell, int]
    entrance: Cell

    def configure(self, configuration: dict):
        self.config = self.Configuration.model_validate(configuration)
        self.warnings = []
        self.diagnostics = None
        self._layout = None
        self.depths = {}

    def _neighbours(self, cell: Cell) -> list[Cell]:
        r, c = cell
        return [(rr, cc) for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if 0 <= rr < self.config.rows and 0 <= cc < self.config.cols]

    def _grow(self, deep: float) -> list[tuple[Cell, Cell]] | None:
        """Random tree edges from the entrance within max_depth doors, None when the growth strands a room."""
        limit = self.config.max_depth
        self.depths = {self.entrance: 0}
        order = [self.entrance]
        edges: list[tuple[Cell, Cell]] = []

        def open_from(sources: list[Cell]) -> list[tuple[Cell, Cell]]:
            return [(a, b) for a in sources for b in self._neighbours(a) if b not in self.depths and (limit == 0 or self.depths[a] < limit)]

        while len(self.depths) < self.config.rows * self.config.cols:
            candidates = open_from([order[-1]]) if self.rng.random() < deep else []
            candidates = candidates or open_from(order)
            if not candidates:
                return None
            if deep == 0.0:
                shallowest = min(self.depths[a] for a, _ in candidates)
                candidates = [(a, b) for a, b in candidates if self.depths[a] == shallowest]
            a, b = self.rng.choice(candidates)
            self.depths[b] = self.depths[a] + 1
            order.append(b)
            edges.append((a, b))
        return edges

    def _tree(self) -> list[tuple[Cell, Cell]]:
        """Tree edges from the entrance, keeping every room within max_depth doors; a breadth-first tree when random growth keeps stranding rooms."""
        for _ in range(GROW_TRIES):
            edges = self._grow(self.config.deep)
            if edges is not None:
                return edges
        edges = self._grow(0.0)
        if edges is None:
            raise ValueError(f"{self.config.rows} x {self.config.cols} rooms do not all lie within {self.config.max_depth} doors of an entrance in column {self.entrance[1]}")
        return edges

    def _loops(self, tree: list[tuple[Cell, Cell]]) -> list[tuple[Cell, Cell]]:
        taken = {frozenset(edge) for edge in tree}
        spare = sorted({(a, b) for a in self.depths for b in self._neighbours(a) if a < b and frozenset((a, b)) not in taken})
        return self.rng.sample(spare, min(self.config.loops, len(spare)))

    def compute(self) -> World:
        c = self.config
        widths = [round(self.rng.uniform(c.room_min, c.room_max), 2) for _ in range(c.cols)]
        heights = [round(self.rng.uniform(c.room_min, c.room_max), 2) for _ in range(c.rows)]
        xs = [round(sum(widths[:i]), 2) for i in range(c.cols + 1)]
        ys = [round(sum(heights[:i]), 2) for i in range(c.rows + 1)]
        self.entrance = (0, self.rng.randrange(c.cols))
        tree = self._tree()
        edges = [*tree, *self._loops(tree)]

        def box(cell: Cell) -> shapely.Polygon:
            r, col = cell
            return shapely.box(xs[col], ys[r], xs[col + 1], ys[r + 1])

        areas = [Area(polygon=box(cell).buffer(-c.wall / 2, join_style="mitre")) for cell in self.depths]
        doors: list[Door] = []
        half = c.wall / 2 + c.resolution
        for a, b in edges:
            (r, col), (r2, col2) = sorted((a, b))
            if r == r2:
                x = xs[col + 1]
                mid = round(self.rng.uniform(ys[r] + c.wall + c.door_width / 2, ys[r + 1] - c.wall - c.door_width / 2), 2)
                start, end = (x, mid - c.door_width / 2), (x, mid + c.door_width / 2)
                areas.append(Area(polygon=shapely.box(x - half, start[1], x + half, end[1])))
            else:
                y = ys[r + 1]
                mid = round(self.rng.uniform(xs[col] + c.wall + c.door_width / 2, xs[col + 1] - c.wall - c.door_width / 2), 2)
                start, end = (mid - c.door_width / 2, y), (mid + c.door_width / 2, y)
                areas.append(Area(polygon=shapely.box(start[0], y - half, end[0], y + half)))
            name = f"door_{room_name(a, self.entrance)}__{room_name(b, self.entrance)}"
            doors.append(Door(name=name, start=Position(x=start[0], y=start[1]), end=Position(x=end[0], y=end[1])))

        regions = [
            Region(
                name=room_name(cell, self.entrance),
                polygon=box(cell),
                description=f"{'entrance' if cell == self.entrance else 'room'} of {widths[cell[1]]:.1f} x {heights[cell[0]]:.1f} m, {depth} doors from the entrance",
            )
            for cell, depth in sorted(self.depths.items())
        ]
        self._layout = Layout(areas=areas, regions=regions)
        world, self.diagnostics = compile_layout(self._layout)
        next(iter(world.levels.values())).doors.extend(doors)
        return world

    def layout(self) -> Layout | None:
        return self._layout
