"""
`DummyHost` is constructed once by `arena_node` and owns the singleton paused-state flag that gates the /clock loop.
`DummySimulator` is per-env on `task_generator_node` and is a no-op stub for the spawn/move/delete verbs.
"""

from __future__ import annotations

import asyncio
import typing
from collections.abc import Sequence

from arena_people_msgs.msg import Pedestrians
from task_generator.shared import Door, DynamicObstacle, Elevator, Entity, Floor, Obstacle, Robot, Wall

from arena_runtime.sim import BaseSim, SimLifecycle

T = typing.TypeVar('T')


class DummyHost(SimLifecycle):
    paused: bool = False

    async def pause(self) -> bool:
        self.paused = True
        return True

    async def unpause(self) -> bool:
        self.paused = False
        return True

    async def cleanup_namespace(self, prefix: str) -> int:
        del prefix
        return 0

    async def ensure_ready(self) -> None:
        return


class DummySimulator(BaseSim):
    """
    Does nothing.
    """

    SIM_NAME = 'dummy'

    # fake spawn
    @staticmethod
    async def _wrap_future(v: T) -> T:
        return v

    async def __spawn_entity(self, entities: Sequence[Entity]) -> Sequence[bool]:
        await asyncio.gather(*(self.safe_resolve(e.model) for e in entities))
        return tuple(True for _ in entities)

    async def obstacle_spawn(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        return await self.__spawn_entity(obstacles)

    async def pedestrian_spawn(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        return await self.__spawn_entity(pedestrians)

    async def robot_spawn(self, robots: Sequence[Robot]) -> Sequence[bool]:
        return await self.__spawn_entity(robots)

    # fake move
    def __move_entity(self, entities: Sequence[Entity]) -> Sequence[bool]:
        self._logger.debug(f"moving {len(entities)} entities")
        return tuple(True for _ in entities)

    async def obstacle_move(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        return self.__move_entity(obstacles)

    async def pedestrian_move(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        return self.__move_entity(pedestrians)

    async def robot_move(self, robots: Sequence[Robot]) -> Sequence[bool]:
        return self.__move_entity(robots)

    # fake delete
    def __delete_entity(self, entities: Sequence[Entity]) -> Sequence[bool]:
        self._logger.debug(f"deleting {len(entities)} entities")
        return tuple(True for _ in entities)

    async def obstacle_delete(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        return self.__delete_entity(obstacles)

    async def pedestrian_delete(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        return self.__delete_entity(pedestrians)

    async def robot_delete(self, robots: Sequence[Robot]) -> Sequence[bool]:
        return self.__delete_entity(robots)

    # assorted
    async def pedestrian_update(self, pedestrians: Pedestrians) -> Sequence[bool]:
        self._logger.debug(f'updating {len(pedestrians.pedestrians)} pedestrians')
        return tuple(True for _ in pedestrians.pedestrians)

    # world interface
    async def spawn_walls(self, walls: Sequence[Wall], clear_existing: bool = True) -> bool:
        async def resolve(wall: Wall):
            sub_walls_iter, obs_iter = await wall.assets()
            sub_walls = tuple(sub_walls_iter)
            obs = tuple(obs_iter)
            await asyncio.gather(
                self.__spawn_entity(obs),
                *(self.safe_resolve(w.material) for w in sub_walls),
            )

        await asyncio.gather(*map(resolve, walls))
        return True

    async def spawn_floors(self, floors: Sequence[Floor]) -> bool:
        await asyncio.gather(*(self.safe_resolve(floor.material) for floor in floors))
        return True

    async def remove_world(self) -> bool:
        self._logger.debug('removing all walls and doors')
        return True

    # mechanism interface
    async def spawn_doors(self, doors: Sequence[Door]) -> bool:
        await asyncio.gather(*(self.safe_resolve(door.material) for door in doors))
        return True

    async def remove_doors(self, names: Sequence[str]) -> bool:
        self._logger.debug(f'removing {len(names)} doors')
        return True

    async def spawn_elevators(self, elevators: Sequence[Elevator]) -> bool:
        await asyncio.gather(*(self.safe_resolve(elevator.material) for elevator in elevators))
        return True

    async def remove_elevators(self, names: Sequence[str]) -> bool:
        self._logger.debug(f'removing {len(names)} elevators')
        return True
