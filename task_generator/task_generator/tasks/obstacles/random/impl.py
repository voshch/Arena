from __future__ import annotations

import functools
import itertools
from collections.abc import Callable, Iterator

import attrs
import numpy as np
import rclpy
from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree import Identifier
from arena_simulation_setup.tree.assets.Human import HumanIdentifier
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from typing_extensions import Self

from task_generator.shared import DynamicObstacle, Obstacle, Orientation, Pose
from task_generator.tasks import identifier_to_available
from task_generator.tasks.obstacles import Obstacles, TM_Obstacles


@attrs.define()
class _Config:
    N_STATIC_OBSTACLES: ROSParamT[tuple[int, int]]
    N_INTERACTIVE_OBSTACLES: ROSParamT[tuple[int, int]]
    N_DYNAMIC_OBSTACLES: ROSParamT[tuple[int, int]]

    MODELS_STATIC_OBSTACLES: ROSParamT[list[str]]
    MODELS_INTERACTIVE_OBSTACLES: ROSParamT[list[str]]
    MODELS_DYNAMIC_OBSTACLES: ROSParamT[list[str]]


class TM_Random(TM_Obstacles):
    """
    Random task generator for obstacles.

    This class generates random obstacles for a task scenario.

    Attributes:
        _config (Config): Configuration object for obstacle generation.
    """

    _config: _Config

    async def reset(self, *, seed: int) -> Obstacles:
        rng = self.node.conf.General.RNG.stream("obstacles", "random")

        N_STATIC_OBSTACLES = int(rng.integers(*self._config.N_STATIC_OBSTACLES.value, endpoint=True))
        N_INTERACTIVE_OBSTACLES = int(rng.integers(*self._config.N_INTERACTIVE_OBSTACLES.value, endpoint=True))
        N_DYNAMIC_OBSTACLES = int(rng.integers(*self._config.N_DYNAMIC_OBSTACLES.value, endpoint=True))

        class ModelList(dict[str, float]):
            @classmethod
            def fromkeys(cls, *args: object, **kwargs: object) -> Self:
                result = cls(super().fromkeys(*args, **kwargs))
                if not len(result):
                    self._logger.warn('Empty model list passed. Defaulting to empty string.')
                    result[""] = 1.0
                return result

            @property
            def a(self) -> list[str]:
                return list(self.keys())

            @property
            def p(self) -> list[float]:
                _p = np.array(list(self.values()), dtype=float)
                _p /= _p.sum()
                return _p.tolist()

            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)

        MODELS_STATIC_OBSTACLES = ModelList.fromkeys(self._config.MODELS_STATIC_OBSTACLES.value, 1)
        MODELS_INTERACTIVE_OBSTACLES = ModelList.fromkeys(self._config.MODELS_INTERACTIVE_OBSTACLES.value, 1)
        MODELS_DYNAMIC_OBSTACLES = ModelList.fromkeys(self._config.MODELS_DYNAMIC_OBSTACLES.value, 1)

        def indexer() -> Callable[..., int]:
            indices: dict[str, Iterator[int]] = dict()

            def index(model: str) -> int:
                if model not in indices:
                    indices[model] = itertools.count(1)
                return next(indices[model])

            return index

        waypoints_per_ped = 2
        waypoint_points = self._ctx.world_manager.get_positions_on_map(
            n=N_DYNAMIC_OBSTACLES * waypoints_per_ped,
            safe_dist=0,
            forbid=False,
        )
        spawn_points = self._ctx.world_manager.get_positions_on_map(
            n=N_STATIC_OBSTACLES + N_INTERACTIVE_OBSTACLES + N_DYNAMIC_OBSTACLES,
            safe_dist=1,
        )

        positions = map(lambda pos: Pose(pos, orientation=Orientation.from_yaw(2 * np.pi * rng.random())), spawn_points)
        waypoints = iter(waypoint_points)

        obstacles: list[Obstacle] = []

        # Create static obstacles
        if N_STATIC_OBSTACLES:
            index = indexer()
            obstacles += [
                Obstacle(
                    name=f"S_{model}_{index(model)}",
                    model=model,
                    pose=next(positions),
                )
                for model in rng.choice(
                    a=MODELS_STATIC_OBSTACLES.a,
                    p=MODELS_STATIC_OBSTACLES.p,
                    size=N_STATIC_OBSTACLES,
                )
            ]

        # Create interactive obstacles
        if N_INTERACTIVE_OBSTACLES:
            index = indexer()

            obstacles += [
                Obstacle(
                    name=f"I_{model}_{index(model)}",
                    model=model,
                    pose=next(positions),
                )
                for model in rng.choice(
                    a=MODELS_INTERACTIVE_OBSTACLES.a,
                    p=MODELS_INTERACTIVE_OBSTACLES.p,
                    size=N_INTERACTIVE_OBSTACLES,
                )
            ]

        # Create dynamic obstacles

        dynamic_obstacles: list[DynamicObstacle] = []

        if N_DYNAMIC_OBSTACLES:
            index = indexer()

            dynamic_obstacles += [
                DynamicObstacle(
                    name=f"Pedestrian_{i}",
                    model=model,
                    waypoints=list(itertools.islice(waypoints, waypoints_per_ped)),
                    pose=next(positions),
                )
                for i, model in enumerate(
                    rng.choice(
                        a=MODELS_DYNAMIC_OBSTACLES.a,
                        p=MODELS_DYNAMIC_OBSTACLES.p,
                        size=N_DYNAMIC_OBSTACLES,
                    )
                )
            ]

        return obstacles, dynamic_obstacles

    def __init__(self, **kwargs: object) -> None:
        TM_Obstacles.__init__(self, **kwargs)

        def param_to_tuple(v: list[object]) -> tuple[int, int]:
            lo = int(v[0])  # type: ignore[arg-type]
            hi = int(v[1] if len(v) >= 2 else v[0])  # type: ignore[arg-type]
            lo, hi = min(lo, hi), max(lo, hi)
            return lo, hi

        def param_to_modellist(identifier: type[Identifier], v: list[str]) -> list[str]:
            if len(v):
                return v
            return list(identifier_to_available(identifier))

        STATIC = 'static'
        INTERACTIVE = 'interactive'
        DYNAMIC = 'dynamic'

        self._config = _Config(
            N_STATIC_OBSTACLES=self.node.ROSParam[tuple[int, int]](self.namespace(STATIC, 'n'), [5, 15], parse=param_to_tuple),
            N_INTERACTIVE_OBSTACLES=self.node.ROSParam[tuple[int, int]](self.namespace(INTERACTIVE, 'n'), [0, 0], parse=param_to_tuple),
            N_DYNAMIC_OBSTACLES=self.node.ROSParam[tuple[int, int]](self.namespace(DYNAMIC, 'n'), [1, 5], parse=param_to_tuple),
            MODELS_STATIC_OBSTACLES=self.node.ROSParam[list[str]](self.namespace(STATIC, 'models'), [], type_=rclpy.Parameter.Type.STRING_ARRAY, parse=functools.partial(param_to_modellist, ObjectIdentifier)),
            MODELS_INTERACTIVE_OBSTACLES=self.node.ROSParam[list[str]](self.namespace(INTERACTIVE, 'models'), [], type_=rclpy.Parameter.Type.STRING_ARRAY, parse=functools.partial(param_to_modellist, ObjectIdentifier)),
            MODELS_DYNAMIC_OBSTACLES=self.node.ROSParam[list[str]](self.namespace(DYNAMIC, 'models'), [], type_=rclpy.Parameter.Type.STRING_ARRAY, parse=functools.partial(param_to_modellist, HumanIdentifier)),
        )
