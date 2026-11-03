"""Pure no-op backend for human:=none, every impl succeeds without side effects."""

from collections.abc import Mapping, Sequence

from arena_runtime.humans import BaseHumanSimulator
from arena_runtime.shared import Door, DynamicObstacle, Obstacle, Region, Robot, Wall


class NoopHumanSimulator(BaseHumanSimulator):
    async def _spawn_obstacles_impl(
        self,
        obstacles: Sequence[Obstacle],
    ) -> Sequence[Obstacle | None]:
        return obstacles

    async def _spawn_dynamic_obstacles_impl(
        self,
        obstacles: Sequence[DynamicObstacle],
    ) -> Sequence[DynamicObstacle | None]:
        return obstacles

    async def _remove_obstacles_impl(
        self,
        obstacles: Sequence[Obstacle],
    ) -> bool:
        return True

    async def _remove_pedestrians_impl(
        self,
    ) -> bool:
        return True

    async def _spawn_walls_impl(
        self,
        walls: Mapping[str, Wall],
    ) -> bool:
        return True

    async def _spawn_doors_impl(
        self,
        doors: Mapping[str, Door],
    ) -> bool:
        return True

    async def _remove_walls_impl(
        self,
        names: Sequence[str],
    ) -> bool:
        return True

    async def _remove_doors_impl(
        self,
        names: Sequence[str],
    ) -> bool:
        return True

    async def _spawn_robot_impl(
        self,
        robots: Sequence[Robot],
    ) -> Sequence[bool]:
        return (True,) * len(robots)

    async def _remove_robot_impl(
        self,
        robots: Sequence[Robot],
    ) -> Sequence[bool]:
        return (True,) * len(robots)

    async def _move_robot_impl(
        self,
        robots: Sequence[Robot],
    ) -> Sequence[bool]:
        return (True,) * len(robots)

    async def _add_regions_impl(
        self,
        regions: Sequence[Region],
    ) -> bool:
        return True

    async def _remove_regions_impl(
        self,
        regions: Sequence[Region],
    ) -> bool:
        return True
