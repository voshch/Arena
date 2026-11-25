"""Fakes for the robot task mode unit tests.

World draws positions inside the polygon it is handed, from the episode's seeded world stream, and refuses when a polygon holds too few spots, as the real world does. Level answers zone lookups the way
the compacted level does, stair groups included. node() is the ROS parameter node a mode reads its declarations from, with
the given values as task.mode.<name> overrides.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from types import SimpleNamespace

import shapely
from arena_runtime.env.constants.rng import EpisodeRng
from arena_runtime.env.task_layer import TaskConfiguration
from arena_runtime.shared import Position

BAY = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
GOAL_TOLERANCE = 0.5  # the goal tolerance radius: the disc a goal is reached in, from task.episode.goto_pose.tolerance.radius


class Level:
    """The compacted world, answering zone lookups the way the real level does: a stair group answers with its first flight."""

    def __init__(self, zones: dict[str, list[tuple[float, float]]], stairs: Sequence[object] = ()) -> None:
        self._zones = zones
        self.stairs = list(stairs)

    def lookup_zone_polygon(self, name: str) -> list[Position] | None:
        corners = self._zones.get(name)
        if corners is None:
            corners = next((stair.corners for stair in self.stairs if name in (stair.name, stair.group)), None)
        return None if corners is None else [Position(x, y) for x, y in corners]

    def zone_ref_names(self) -> list[str]:
        return [*self._zones, *(stair.name for stair in self.stairs), *dict.fromkeys(stair.group for stair in self.stairs if stair.group)]


class World:
    """Draws positions inside the polygon it is handed, without one from the whole map."""

    ATTEMPTS = 10_000

    def __init__(self, rng: EpisodeRng, zones: dict[str, list[tuple[float, float]]], stairs: Sequence[object] = ()) -> None:
        self._rng = rng
        self._level = Level(zones, stairs)

    def world_compacted(self) -> Level:
        return self._level

    def get_positions_on_map(
        self,
        n: int,
        safe_dist: float,
        forbidden_zones: object = None,
        forbid: bool = True,
        polygon: shapely.Polygon | None = None,
        level_id: str = "",
    ) -> list[Position]:
        del safe_dist, forbidden_zones, forbid, level_id
        if polygon is None:
            return [Position(-5.0 - index, -5.0) for index in range(n)]
        stream = self._rng.stream("world", "positions")
        min_x, min_y, max_x, max_y = polygon.bounds
        spots: list[Position] = []
        for _ in range(self.ATTEMPTS):
            if len(spots) == n:
                return spots
            x, y = stream.uniform(min_x, max_x), stream.uniform(min_y, max_y)
            if polygon.contains(shapely.Point(x, y)):
                spots.append(Position(x, y))
        if len(spots) == n:
            return spots
        raise RuntimeError(f"fewer than {n} positions fit in the polygon")


def ctx(world: World, names: list[str], rng: EpisodeRng) -> SimpleNamespace:
    value = SimpleNamespace(value=0.5)
    conf = TaskConfiguration(RNG=rng, HUMAN=value, SAFE_DIST=value, GOAL_TOLERANCE_RADIUS=SimpleNamespace(value=GOAL_TOLERANCE), TIMEOUT=value, KG_DERIVE=value)
    robots = {name: SimpleNamespace(name=name, safe_distance=0.3) for name in names}
    return SimpleNamespace(robots=robots, world_manager=world, conf=conf)


def node(**values: object) -> object:
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from rclpy.parameter import Parameter

    server = ROSParamServer(f"robot_mode_{uuid.uuid4().hex[:8]}", start_parameter_services=False, parameter_overrides=[Parameter(f"task.mode.{name}", value=value) for name, value in values.items()])
    server.sim_time = SimpleNamespace(sec=5.0)
    return server


def namespace() -> object:
    from arena_rclpy_mixins.shared import Namespace

    return Namespace("task.mode").ParamNamespace()
