import logging

import pydantic
import shapely

from arena_simulation_setup.shared import Door, Position
from arena_simulation_setup.tree.World import Level, WallPath, World, Zone

from . import BaseConfiguration, WorldGeneratorImpl
from .utils import to_corners, to_walls, world_of

logger = logging.getLogger(__name__)


class WorldGeneratorHallway(WorldGeneratorImpl):
    class Configuration(BaseConfiguration):
        width: float = pydantic.Field(80.0, description='Hallway length along x in meters, which the room widths are scaled to fill.')
        height: float = pydantic.Field(50.0, description='Extent along y in meters, which centers the hallway band.')
        hallway_height: float = pydantic.Field(5.0, description='Width of the horizontal hallway band in meters.')
        rooms_per_side: int = pydantic.Field(7, description='Number of rooms on each side of the hallway.')
        big_width: tuple[float, float] = pydantic.Field((18.0, 28.0), description='Range (min, max) in meters the width of the first and last room of each side is drawn from.')
        big_height: tuple[float, float] = pydantic.Field((12.0, 20.0), description='Range (min, max) in meters the depth of the first and last room of each side is drawn from.')
        small_width: tuple[float, float] = pydantic.Field((6.0, 12.0), description='Range (min, max) in meters the width of the intermediate rooms is drawn from.')
        small_height: tuple[float, float] = pydantic.Field((7.0, 14.0), description='Range (min, max) in meters the depth of the intermediate rooms is drawn from.')
        door_width: float = pydantic.Field(2.5, description='Width of each room door onto the hallway in meters.')

        @property
        def hallway_top(self) -> float:
            return self.height / 2 + self.hallway_height / 2

        @property
        def hallway_bottom(self) -> float:
            return self.height / 2 - self.hallway_height / 2

    config: Configuration

    def configure(self, configuration: dict):
        self.config = self.Configuration.model_validate(configuration)
        logger.info(self.config)

    def compute(self) -> World:
        level = Level()
        descriptions: dict[str, str] = {}
        self._impl("top", self.config.rooms_per_side, level, descriptions)
        self._impl("bottom", self.config.rooms_per_side, level, descriptions)
        return world_of(level, descriptions)

    def _impl(self, side: str, num_rooms: int, level: Level, descriptions: dict[str, str]) -> None:
        rooms: list[Zone] = []

        widths: list[float] = []
        heights: list[float] = []

        rooms.append(
            Zone(
                name=f"{side}_room_{len(rooms)}",
                polygon=[Position(x=0, y=self.config.hallway_bottom), Position(x=self.config.width, y=self.config.hallway_bottom), Position(x=self.config.width, y=self.config.hallway_top), Position(x=0, y=self.config.hallway_top)],
            )
        )
        descriptions[rooms[-1].name] = "hallway"
        level.walls.extend(
            [
                WallPath(path=[Position(x=self.config.wall_gap / 2, y=self.config.hallway_top), Position(x=self.config.wall_gap / 2, y=self.config.hallway_bottom)]),
                WallPath(path=[Position(x=self.config.width - self.config.wall_gap / 2, y=self.config.hallway_bottom), Position(x=self.config.width - self.config.wall_gap / 2, y=self.config.hallway_top)]),
            ]
        )

        for i in range(num_rooms):
            if i == 0 or i == num_rooms - 1:
                w = self.rng.uniform(*self.config.big_width)
                h = self.rng.uniform(*self.config.big_height)
            else:
                w = self.rng.uniform(*self.config.small_width)
                h = self.rng.uniform(*self.config.small_height)
            widths.append(w)
            heights.append(h)

        total_width = sum(widths)
        norm_factor = self.config.width / total_width
        widths = [w * norm_factor for w in widths]

        current_x = 0.0
        for i in range(num_rooms):
            w = round(widths[i], 1)
            h = round(heights[i], 1)
            if side == "top":
                y = self.config.hallway_top
                door_y = self.config.hallway_top
            else:
                y = self.config.hallway_bottom - h
                door_y = self.config.hallway_bottom
            if w > self.config.door_width:
                door_start = self.rng.uniform(current_x + self.config.door_width, current_x + w - self.config.door_width)
                door_end = door_start + self.config.door_width
            else:
                door_start, door_end = current_x, current_x + w
            x = round(current_x, 1)
            y = round(y, 1)

            door_start = round(door_start, 1)
            door_end = round(door_end, 1)

            room_polygon = shapely.Polygon(((x, y), (x + w, y), (x + w, y + h), (x, y + h))).buffer(-self.config.wall_gap / 2)

            door_height = self.config.resolution * 2
            door_polygon = shapely.Polygon(((door_start, door_y - door_height), (door_end, door_y - door_height), (door_end, door_y + door_height), (door_start, door_y + door_height)))

            room_walls = shapely.difference(room_polygon.exterior, door_polygon)

            level.doors.append(
                Door(
                    name=f"{side}_room_{len(rooms)}_door",
                    start=Position(x=door_start, y=door_y),
                    end=Position(x=door_end, y=door_y),
                )
            )
            rooms.append(Zone(name=f"{side}_room_{len(rooms)}", polygon=to_corners(room_polygon)))
            descriptions[rooms[-1].name] = f"{side} room {i + 1} of size {w: .1f} x {h: .1f}"
            level.walls.extend(to_walls(room_walls))

            current_x += w

        level.zones.extend(rooms)
