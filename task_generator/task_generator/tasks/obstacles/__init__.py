import enum
import uuid

from task_generator.shared import CustomDynamicObstacle, DynamicObstacle, Obstacle, Pose, Region
from task_generator.tasks.mode import TaskMode
from task_generator.tasks.obstacles._placement import random_placement

from . import edge_case, environment, parametrized, prompt, random, scenario

Obstacles = tuple[list[Obstacle], list[DynamicObstacle]]
CustomObstacles = tuple[list[Obstacle], list[CustomDynamicObstacle]]


@enum.unique
class ObstacleKind(enum.Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class TM_Obstacles(TaskMode):
    @property
    def pending_regions(self) -> list[Region]:
        """Flow regions (sources/sinks) this mode wants configured for the coming episode.

        A mode *fills* this during `reset` rather than pushing to the simulator itself, and
        `Task._reset_episode` applies it once afterwards. Two reasons:

        * A mode that pushes its own regions has already half-applied the population by the
          time `reset` returns, so anything wrapping it cannot see or change the flow -
          `tm_obstacles:=edge_case` could perturb agents but not the crowd that spawns them,
          which is exactly what Level B needs.
        * Nothing ever called `remove_all_regions`, so regions accumulated across scenario
          switches. Applying them from one place makes clearing-then-setting the normal path.

        Lazily created so modes need not touch `__init__`.
        """
        regions = getattr(self, "_pending_regions", None)
        if regions is None:
            regions = []
            self._pending_regions = regions
        return regions

    async def reset(self, *, seed: int) -> Obstacles:
        return [], []

    async def teardown(self) -> None:
        """Release everything this mode drives; called when the mode is replaced or the task ends."""

    async def extend(self, kind: ObstacleKind, model: str, pose: Pose | None = None, level_id: str = "") -> str:
        resolved_pose = pose if pose is not None else await random_placement(self._ctx, level_id=level_id)
        name = f"ext_{model}_{uuid.uuid4().hex[:6]}"

        if kind is ObstacleKind.STATIC:
            obstacle: Obstacle | DynamicObstacle = Obstacle(name=name, model=model, pose=resolved_pose, level_id=level_id)
            await self._ctx.environment_manager.spawn_obstacles([obstacle])
        else:
            waypoints = self._ctx.world_manager.get_positions_on_map(n=2, safe_dist=0, forbid=False, level_id=level_id)
            obstacle = DynamicObstacle(name=name, model=model, waypoints=waypoints, pose=resolved_pose, level_id=level_id)
            await self._ctx.environment_manager.spawn_dynamic_obstacles([obstacle])
        return obstacle.sim_path

    async def retract(self, entity_id: str) -> bool:
        """Remove one obstacle this mode added with :meth:`extend`. True if it was there.

        The counterpart `extend` never had. Without it the only way to remove an obstacle is
        `EnvironmentManager.reset`, which purges by layer at an episode boundary - so an
        object could appear mid-episode but never disappear, and "the blocked doorway
        clears" was not expressible. That case is the interesting half: whether a robot that
        gave up on a route ever retries it.

        Takes the id `extend` returned (the entity's `sim_path`). False means the id
        resolved to nothing, which is not an error - it is the state the caller wanted.
        """
        removed, _ = await self._ctx.environment_manager.remove_obstacles_by_id([entity_id])
        return bool(removed)


__all__ = ["TM_Obstacles", "ObstacleKind", "Obstacles", "CustomObstacles", "edge_case", "environment", "parametrized", "prompt", "random", "scenario"]
