import logging
from collections.abc import Iterable

import shapely

from arena_simulation_setup.shared import Door, Position, Wall

from . import BaseConfiguration, LevelDescription, WorldGeneratorImpl
from .utils import to_corners, to_walls

logger = logging.getLogger(__name__)


class WorldGeneratorHallway(WorldGeneratorImpl):
    class Configuration(BaseConfiguration):
        width: float = 80.0
        height: float = 50.0

        # Hallway parameters (a horizontal band)
        hallway_height: float = 5.0

        # Room parameters (for each side)
        rooms_per_side: int = 7

        # For "big" rooms (first and last); (min, max)
        big_width: tuple[float, float] = (18.0, 28.0)
        big_height: tuple[float, float] = (12.0, 20.0)

        # For "small" rooms (intermediate ones); (min, max)
        small_width: tuple[float, float] = (6.0, 12.0)
        small_height: tuple[float, float] = (7.0, 14.0)

        # door width
        door_width: float = 2.5

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

    def compute(self) -> LevelDescription:
        # The hallway is shared by both sides and emitted once here, not per side: two coincident
        # slabs and walls interpenetrate in physics.
        hallway = self._hallway()
        top_rooms = self._impl("top", self.config.rooms_per_side)
        bottom_rooms = self._impl("bottom", self.config.rooms_per_side)

        return LevelDescription(zones=[hallway, *top_rooms, *bottom_rooms])

    def _hallway(self) -> LevelDescription.Zone:
        return LevelDescription.Zone(
            name="hallway",
            corners=[Position(x=0, y=self.config.hallway_bottom), Position(x=self.config.width, y=self.config.hallway_bottom), Position(x=self.config.width, y=self.config.hallway_top), Position(x=0, y=self.config.hallway_top)],
            walls=[
                Wall(
                    start=Position(x=self.config.wall_gap / 2, y=self.config.hallway_top),
                    end=Position(x=self.config.wall_gap / 2, y=self.config.hallway_bottom),
                ),
                Wall(
                    start=Position(x=self.config.width - self.config.wall_gap / 2, y=self.config.hallway_bottom),
                    end=Position(x=self.config.width - self.config.wall_gap / 2, y=self.config.hallway_top),
                ),
            ],
            description="hallway",
        )

    def _impl(self, side: str, num_rooms: int) -> Iterable[LevelDescription.Zone]:
        rooms: list[LevelDescription.Zone] = []

        widths: list[float] = []
        heights: list[float] = []

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

            rooms.append(
                LevelDescription.Zone(
                    name=f"{side}_room_{len(rooms)}",
                    corners=to_corners(room_polygon),
                    walls=to_walls(room_walls),
                    doors=[
                        Door(
                            name=f"{side}_room_{len(rooms)}_door",
                            start=Position(x=door_start, y=door_y),
                            end=Position(x=door_end, y=door_y),
                        )
                    ],
                    description=f"{side} room {i + 1} of size {w: .1f} x {h: .1f}",
                )
            )

            current_x += w

        return rooms
