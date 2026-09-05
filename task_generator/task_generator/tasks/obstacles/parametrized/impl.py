import math

import numpy as np
import shapely
from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.configs.parametrized import (
    ParametrizedConfig,
    ParametrizedIdentifier,
)

from task_generator.shared import DynamicObstacle, Obstacle, Orientation, Pose
from task_generator.tasks.obstacles import TM_Obstacles


class TM_Parametrized(TM_Obstacles):
    _config: ROSParamT[ParametrizedConfig]

    def _parse(self, config_name: str) -> ParametrizedConfig:
        return ParametrizedIdentifier(config_name).resolve_sync()

    def _zone_polygon(self, zone: str) -> shapely.Polygon | None:
        if not zone:
            return None
        vertices = self._ctx.world_manager.world_compacted().lookup_zone_polygon(zone)
        if vertices is None:
            raise ValueError(f"zone '{zone}' not found in world")
        return shapely.Polygon([(v.x, v.y) for v in vertices])

    def _get_pose(self, rng: np.random.Generator, polygon: shapely.Polygon | None = None) -> Pose:
        return Pose(
            self._ctx.world_manager.get_position_on_map(1, polygon=polygon),
            Orientation.from_yaw(rng.random() * 2 * math.pi),
        )

    def _get_points(self, n: int, polygon: shapely.Polygon | None = None) -> list[object]:
        return self._ctx.world_manager.get_positions_on_map(
            n=n,
            safe_dist=1.0,
            polygon=polygon,
        )

    async def reset(self, *, seed: int) -> tuple[list[Obstacle], list[DynamicObstacle]]:
        rng = self.node.conf.General.RNG.stream("obstacles", "parametrized")
        dynamic_obstacles: list[DynamicObstacle] = []
        obstacles: list[Obstacle] = []

        # Create static obstacles
        for config in self._config.value.STATIC:
            poly = self._zone_polygon(config.zone)
            for i in range(rng.integers(config.min, config.max, endpoint=True)):
                obstacle = Obstacle(
                    name=f"S_{config.model}_{i + 1}",
                    model=config.model,
                    pose=self._get_pose(rng, polygon=poly),
                )
                obstacle.extra["type"] = config.type
                obstacles.append(obstacle)

        # Create interactive obstacles
        for config in self._config.value.INTERACTIVE:
            poly = self._zone_polygon(config.zone)
            for i in range(rng.integers(config.min, config.max, endpoint=True)):
                obstacle = Obstacle(
                    name=f"S_{config.model}_{i + 1}",
                    model=config.model,
                    pose=self._get_pose(rng, polygon=poly),
                )
                obstacle.extra["type"] = config.type
                obstacles.append(obstacle)

        # Create dynamic obstacles
        for config in self._config.value.DYNAMIC:
            poly = self._zone_polygon(config.zone)
            for i in range(rng.integers(config.min, config.max, endpoint=True)):
                obstacle = DynamicObstacle(
                    name=f"S_{config.model}_{i + 1}",
                    model=config.model,
                    pose=self._get_pose(rng, polygon=poly),
                    waypoints=self._get_points(2, polygon=poly),
                )
                obstacle.extra["type"] = config.type
                dynamic_obstacles.append(obstacle)

        return obstacles, dynamic_obstacles

    def __init__(self, **kwargs: object) -> None:
        TM_Obstacles.__init__(self, **kwargs)

        self._config = self.node.ROSParam[ParametrizedConfig](self.namespace('file'), '', parse=self._parse)
