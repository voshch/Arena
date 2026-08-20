import shapely

from . import (
    BaseConfiguration,
    LevelDescription,
    WorldGeneratorImpl,
)
from .utils import to_corners, to_walls


class WorldGeneratorEmpty(WorldGeneratorImpl):
    class Configuration(BaseConfiguration):
        width: float = 15.0  # m
        height: float = 15.0  # m

    config: Configuration

    def configure(self, configuration: dict):
        self.config = self.Configuration.model_validate(configuration)

    def compute(self) -> LevelDescription:

        room = shapely.Polygon(
            [
                (0, 0),
                (self.config.width, 0),
                (self.config.width, self.config.height),
                (0, self.config.height),
            ]
        )

        return LevelDescription(
            zones=[
                LevelDescription.Zone(
                    name="empty_zone",
                    corners=to_corners(room),
                    walls=to_walls(room),
                    description="An empty zone",
                )
            ]
        )
