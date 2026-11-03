from collections.abc import Sequence

from arena_runtime.shared import DynamicObstacle, Obstacle
from arena_simulation_setup.shared import Light

from task_generator.tasks.mode import TaskMode

from . import environment, parametrized, prompt, random, scenario

Obstacles = tuple[list[Obstacle], list[DynamicObstacle]]


class TM_Obstacles(TaskMode):
    lights: Sequence[Light] = ()

    async def reset(self, *, seed: int) -> Obstacles:
        return [], []

    async def teardown(self) -> None:
        """Release everything this mode drives; called when the mode is replaced or the task ends."""


__all__ = ["TM_Obstacles", "Obstacles", "environment", "parametrized", "prompt", "random", "scenario"]
