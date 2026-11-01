"""Open fields of square obstacles."""

import pydantic
import shapely

from arena_simulation_setup.tree.World import World

from . import PedestrianConfiguration, WithPedestrians
from .layout import Area, Layout, Region, compile_layout


class WorldGeneratorOpen(WithPedestrians):
    """A walled box with square obstacles at random positions, merged where they overlap, redrawn until the free space is one piece."""

    class Configuration(PedestrianConfiguration):
        width: float = pydantic.Field(20.0, gt=0)
        height: float = pydantic.Field(20.0, gt=0)
        obstacles: int = pydantic.Field(15, ge=0)
        obstacle_size: float = pydantic.Field(1.0, gt=0)
        size_spread: float = pydantic.Field(0.0, ge=0, lt=1)
        max_tries: int = pydantic.Field(50, ge=1)

    config: Configuration

    def configure(self, configuration: dict):
        self.config = self.Configuration.model_validate(configuration)
        self.warnings = []
        self.diagnostics = None

    def _snap(self, value: float) -> float:
        return round(value / self.config.resolution) * self.config.resolution

    def compute(self) -> World:
        c = self.config
        box = shapely.box(0, 0, c.width, c.height)
        for _ in range(c.max_tries):
            blocks = []
            for _ in range(c.obstacles):
                side = c.obstacle_size * self.rng.uniform(1 - c.size_spread, 1 + c.size_spread)
                x, y = self.rng.uniform(0, c.width), self.rng.uniform(0, c.height)
                blocks.append(shapely.box(self._snap(x - side / 2), self._snap(y - side / 2), self._snap(x + side / 2), self._snap(y + side / 2)))
            free = box.difference(shapely.union_all(blocks)) if blocks else box
            if isinstance(free, shapely.Polygon) and not free.is_empty:
                description = f'open field, {c.obstacles} obstacles, {1 - free.area / box.area:.0%} filled'
                world, self.diagnostics = compile_layout(Layout(areas=[Area(polygon=free)], regions=[Region(name='open', polygon=box, description=description)]))
                return world
        raise ValueError(f'{c.obstacles} obstacles of {c.obstacle_size} m left no connected free space in a {c.width} x {c.height} m box in {c.max_tries} tries')
