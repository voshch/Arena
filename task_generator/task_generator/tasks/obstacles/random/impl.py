from __future__ import annotations

import itertools
import typing
from collections.abc import Callable, Iterable, Iterator

import numpy as np
from arena_runtime.shared import DynamicObstacle, Obstacle, Orientation, Pose
from arena_simulation_setup.tree import Identifier
from arena_simulation_setup.tree.assets.Human import HumanIdentifier
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from typing_extensions import Self

from task_generator.tasks import identifier_to_available
from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.obstacles import Obstacles, TM_Obstacles
from task_generator.tasks.obstacles.random import RandomParams


def _models(identifier: type[Identifier], models: list[str]) -> list[str]:
    return models if models else list(identifier_to_available(identifier))


class TM_Random(TM_Obstacles):
    """
    Random task generator for obstacles.

    This class generates random obstacles for a task scenario.

    Attributes:
        _config (Config): Configuration object for obstacle generation.
    """

    _config: RandomParams

    async def reset(self, *, seed: int) -> Obstacles:
        rng = self._ctx.conf.RNG.stream("obstacles", "random")

        def count(bounds: tuple[float, float]) -> int:
            lo, hi = bounds
            return int(rng.integers(int(lo), int(hi), endpoint=True))

        N_STATIC_OBSTACLES = count(self._config.STATIC_N.value)
        N_INTERACTIVE_OBSTACLES = count(self._config.INTERACTIVE_N.value)
        N_DYNAMIC_OBSTACLES = count(self._config.DYNAMIC_N.value)

        class ModelList(dict[str, float]):
            @classmethod
            def from_models(cls, models: Iterable[str], value: float, /) -> Self:
                result = cls(dict.fromkeys(models, value))
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

        MODELS_STATIC_OBSTACLES = ModelList.from_models(_models(ObjectIdentifier, self._config.STATIC_MODELS.value), 1)
        MODELS_INTERACTIVE_OBSTACLES = ModelList.from_models(_models(ObjectIdentifier, self._config.INTERACTIVE_MODELS.value), 1)
        MODELS_DYNAMIC_OBSTACLES = ModelList.from_models(_models(HumanIdentifier, self._config.DYNAMIC_MODELS.value), 1)

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

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        TM_Obstacles.__init__(self, **kwargs)

        self._config = RandomParams(self.node, namespace=self.namespace())
