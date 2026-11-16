import pydantic
import shapely

from arena_simulation_setup.tree.World import Level, World, Zone

from . import (
    BaseConfiguration,
    WorldGeneratorImpl,
)
from .utils import to_corners, to_walls, world_of


class WorldGeneratorEmpty(WorldGeneratorImpl):
    class Configuration(BaseConfiguration):
        width: float = pydantic.Field(15.0, description='Room extent along x in meters.')
        height: float = pydantic.Field(15.0, description='Room extent along y in meters.')

    config: Configuration

    def configure(self, configuration: dict):
        self.config = self.Configuration.model_validate(configuration)

    def compute(self) -> World:

        room = shapely.Polygon(
            [
                (0, 0),
                (self.config.width, 0),
                (self.config.width, self.config.height),
                (0, self.config.height),
            ]
        )

        level = Level(zones=[Zone(name="empty_zone", polygon=to_corners(room))], walls=to_walls(room))
        return world_of(level, {"empty_zone": "An empty zone"})
