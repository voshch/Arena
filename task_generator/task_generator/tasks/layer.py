"""The task layer arena_runtime loads through its arena_runtime.task_layer entry point."""

from __future__ import annotations

import enum
import typing

from arena_runtime.env.task_layer import TaskContext, TaskModeNames

from task_generator.modes import Modes
from task_generator.tasks.obstacles.prompt import register_prompt_mode
from task_generator.tasks.registry import MODULE_MODES, OBSTACLES_MODES, ROBOTS_MODES, TaskModeRegistry, walk_schemas
from task_generator.tasks.robots.composite import extra_tm_kinds
from task_generator.tasks.task import Task

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from arena_runtime.env.episode_node import EpisodeNode


_ALWAYS_ON = (Modes.TM_Module.CLEAR_FORBIDDEN_ZONES, Modes.TM_Module.RVIZ_UI)


def _modules(value: object) -> set[Modes.TM_Module]:
    if not isinstance(value, str):
        raise TypeError(f"expected a comma-separated string, got {value!r}")
    return {Modes.TM_Module(module) for module in value.split(",") if module != ""}


class ModeParams:
    def __init__(self, server: ROSParamServer) -> None:
        self.TM_ROBOTS = server.ROSParam[Modes.TM_Robots]("task.robots", Modes.TM_Robots.default().value, parse=Modes.TM_Robots)
        self.TM_OBSTACLES = server.ROSParam[Modes.TM_Obstacles]("task.obstacles", Modes.TM_Obstacles.default().value, parse=Modes.TM_Obstacles)
        self.TM_CONFIG = server.ROSParam[str]("task.config", "")
        self.TM_MODULES = server.ROSParam[set[Modes.TM_Module]]("task.modules", ",".join(module.value for module in Modes.TM_Module.default()), parse=_modules)

    @property
    def robots(self) -> str:
        return self.TM_ROBOTS.value.value if self.TM_ROBOTS.value else ""

    @property
    def obstacles(self) -> str:
        return self.TM_OBSTACLES.value.value if self.TM_OBSTACLES.value else ""

    @property
    def modules(self) -> list[str]:
        return [module.value for module in self.TM_MODULES.value]


def declare_task_params(node: ROSParamServer) -> ModeParams:
    return ModeParams(node)


async def create_task(*, node: EpisodeNode, ctx: TaskContext) -> Task:
    register_prompt_mode(node.conf.Arena.HUMAN.value)
    params = typing.cast("ModeParams", node.task_params)
    modules = params.TM_MODULES.value
    modules.update(_ALWAYS_ON)
    return await Task.create(node=node, ctx=ctx, params=params, modules=list(modules))


def mode_names() -> TaskModeNames:
    return TaskModeNames(
        robots=[key.value for key in ROBOTS_MODES.keys()],
        obstacles=[key.value for key in OBSTACLES_MODES.keys()],
        modules=[key.value for key in MODULE_MODES.keys()],
    )


def mode_vocabulary() -> TaskModeNames:
    return TaskModeNames(
        robots=[mode.value for mode in Modes.TM_Robots],
        obstacles=[mode.value for mode in Modes.TM_Obstacles],
        modules=[mode.value for mode in Modes.TM_Module],
    )


def declare_mode_schemas(node: ROSParamServer) -> None:
    walk_schemas(node)


def _namespaces[K: enum.Enum, V](registry: TaskModeRegistry[K, V]) -> dict[str, str]:
    return {key.value: str(registry.meta(key).namespace) for key in sorted(registry.keys(), key=lambda key: key.value)}


def mode_namespaces() -> dict[str, dict[str, str]]:
    return {"robots": _namespaces(ROBOTS_MODES), "obstacles": _namespaces(OBSTACLES_MODES), "modules": _namespaces(MODULE_MODES)}


def fleet_only_robots() -> list[str]:
    return extra_tm_kinds()
