from __future__ import annotations

import asyncio
import importlib.metadata
import typing
from collections.abc import Awaitable, Callable

import attrs

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from arena_simulation_setup.shared import Light

    from arena_runtime.acoustics import BaseAcousticsSimulator
    from arena_runtime.env.environment_manager import EnvironmentManager
    from arena_runtime.env.episode_node import EpisodeNode
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.env.robot_manager import RobotManager, RobotsManager
    from arena_runtime.env.world_manager.world_manager_ros import WorldManagerROS
    from arena_runtime.shared import DynamicObstacle, Obstacle, Pose
    from arena_runtime.sim import BaseSim

TASK_LAYER_GROUP = "arena_runtime.task_layer"


class TaskModeNames(typing.NamedTuple):
    robots: list[str]
    obstacles: list[str]
    modules: list[str]


@attrs.define
class TaskContext:
    """The env's parts a task mode works on."""

    environment_manager: EnvironmentManager
    robots_manager: RobotsManager
    world_manager: WorldManagerROS
    realizer: Realizer
    simulator: BaseSim
    acoustics_simulator: BaseAcousticsSimulator
    reset_lock: asyncio.Lock
    compromise: Callable[[str], None]
    abort_episode: Callable[[str], None]

    @property
    def robots(self) -> dict[str, RobotManager]:
        return self.robots_manager.managers


@attrs.define
class ResetPlan:
    """What the bound modes want spawned for the next episode."""

    obstacles: list[Obstacle]
    dynamic_obstacles: list[DynamicObstacle]
    lights: typing.Sequence[Light]


class TaskParams(typing.Protocol):
    """The names of the selected modes."""

    @property
    def robots(self) -> str: ...

    @property
    def obstacles(self) -> str: ...

    @property
    def modules(self) -> list[str]: ...


class RobotModes(typing.Protocol):
    @property
    def start_poses(self) -> dict[str, Pose]: ...


class EpisodeTask(typing.Protocol):
    """One env's bound task modes, as the episode loop drives them."""

    @property
    def done(self) -> Awaitable[bool]: ...

    @property
    def tm_robots(self) -> RobotModes: ...

    async def prepare_reset(self, seed: int) -> ResetPlan: ...

    def after_reset(self) -> None: ...

    async def teardown(self) -> None: ...


class TaskLayer(typing.Protocol):
    """What an env needs from the package that holds the task modes."""

    def declare_task_params(self, node: ROSParamServer) -> TaskParams: ...

    async def create_task(self, *, node: EpisodeNode, ctx: TaskContext) -> EpisodeTask: ...

    def mode_names(self) -> TaskModeNames: ...

    def mode_vocabulary(self) -> TaskModeNames: ...

    def declare_mode_schemas(self, node: ROSParamServer) -> None: ...

    def mode_namespaces(self) -> dict[str, dict[str, str]]: ...

    def fleet_only_robots(self) -> list[str]: ...


def task_layer_entry_point() -> importlib.metadata.EntryPoint:
    """The one entry point registered under arena_runtime.task_layer."""
    providers = list(importlib.metadata.entry_points(group=TASK_LAYER_GROUP))
    if len(providers) != 1:
        raise LookupError(f"an env needs exactly one package registering the {TASK_LAYER_GROUP} entry point, found {sorted(entry.name for entry in providers) or 'none'}")
    return providers[0]


def task_layer() -> TaskLayer:
    return task_layer_entry_point().load()
