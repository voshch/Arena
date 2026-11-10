"""The task layer arena_runtime loads through its arena_runtime.task_layer entry point."""

from __future__ import annotations

import enum
import typing
from collections.abc import Collection

from arena_rclpy_mixins.params import Choices, Param, declare
from arena_runtime.env.task_layer import EpisodeRecord, TaskContext, TaskModeOverrides, TaskModes, mode_param
from arena_runtime.env.task_layer import validate_modes as _validate_modes

from task_generator.modes import Modes
from task_generator.tasks.obstacles.prompt import register_prompt_mode
from task_generator.tasks.registry import MODULE_MODES, OBSTACLES_MODES, ROBOTS_MODES, TaskModeRegistry
from task_generator.tasks.robots.composite import extra_tm_kinds
from task_generator.tasks.task import Task

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from arena_runtime.env.task_layer import TaskParams


_REGISTRIES: dict[str, tuple[type[enum.Enum], TaskModeRegistry]] = {
    "robots": (Modes.TM_Robots, ROBOTS_MODES),
    "obstacles": (Modes.TM_Obstacles, OBSTACLES_MODES),
    "modules": (Modes.TM_Module, MODULE_MODES),
}
_ALWAYS_ON = (Modes.TM_Module.CLEAR_FORBIDDEN_ZONES, Modes.TM_Module.RVIZ_UI)
_VOCABULARY = {
    "robots": [mode.value for mode in Modes.TM_Robots],
    "obstacles": [mode.value for mode in Modes.TM_Obstacles],
    "modules": sorted(key.value for key in MODULE_MODES.keys()),
}
_MULTI_AXES = ("modules",)


def validate_modes(modes: TaskModes) -> TaskModes:
    """The listed axes' selection in canonical names, ValueError naming the first axis or mode that would not run."""
    return _validate_modes(modes, _VOCABULARY, _MULTI_AXES)


def multi_axes() -> Collection[str]:
    """The axes that take several modes at once."""
    return _MULTI_AXES


def _parse_axis(axis: str, value: object) -> list[str]:
    if not isinstance(value, str):
        raise TypeError(f"expected a comma-separated string, got {value!r}")
    return validate_modes({axis: [name for name in value.split(",") if name != ""]})[axis]


def _parse_robots(value: object) -> Modes.TM_Robots:
    return Modes.TM_Robots(_parse_axis("robots", value)[0])


def _parse_obstacles(value: object) -> Modes.TM_Obstacles:
    return Modes.TM_Obstacles(_parse_axis("obstacles", value)[0])


def _parse_modules(value: object) -> set[Modes.TM_Module]:
    return {Modes.TM_Module(name) for name in _parse_axis("modules", value)}


class TaskModeParams:
    def __init__(self, server: ROSParamServer) -> None:
        self._server = server
        defaults = {
            "robots": Modes.TM_Robots.default().value,
            "obstacles": Modes.TM_Obstacles.default().value,
            "modules": ",".join(module.value for module in Modes.TM_Module.default()),
        }
        self.TM_ROBOTS = declare(server, Param[Modes.TM_Robots](mode_param("robots"), defaults["robots"], "Robots task mode.", constraint=Choices(*_VOCABULARY["robots"], many=True, ignore_case=True), parse=_parse_robots))
        self.TM_OBSTACLES = declare(server, Param[Modes.TM_Obstacles](mode_param("obstacles"), defaults["obstacles"], "Obstacles task mode.", constraint=Choices(*_VOCABULARY["obstacles"], many=True, ignore_case=True), parse=_parse_obstacles))
        self.TM_CONFIG = declare(server, Param[str](mode_param("config"), "", "Task config file name."))
        self.TM_MODULES = declare(server, Param[set[Modes.TM_Module]](mode_param("modules"), defaults["modules"], "Comma-separated task modules.", constraint=Choices(*_VOCABULARY["modules"], many=True, ignore_case=True), parse=_parse_modules))
        self.resolve({})

    def _with_implied(self, selected: set[Modes.TM_Module]) -> list[Modes.TM_Module]:
        """The modules plus those every env runs and sounds when the env has acoustics or static sounds."""
        selected = selected | set(_ALWAYS_ON)
        rosparam = self._server.rosparam[str]
        static_sounds = rosparam.get_unsafe("auditory.static_sounds", "").strip() not in ("", "[]")
        if rosparam.get_unsafe("acoustics", "none") != "none" or static_sounds:
            selected.add(Modes.TM_Module.SOUNDS)
        return sorted(selected, key=lambda module: module.value)

    @property
    def modules(self) -> list[Modes.TM_Module]:
        return self._with_implied(set(self.TM_MODULES.value))

    def resolve(self, modes: TaskModes) -> TaskModes:
        live = {"robots": [self.TM_ROBOTS.value.value], "obstacles": [self.TM_OBSTACLES.value.value], "modules": [module.value for module in self.modules]}
        resolved: TaskModes = {}
        for axis in _REGISTRIES:
            names = modes.get(axis)
            if names is None:
                resolved[axis] = live[axis]
            elif axis not in _MULTI_AXES:
                resolved[axis] = list(names)
            else:
                resolved[axis] = [module.value for module in self._with_implied({Modes.TM_Module(name) for name in names})]
        return resolved

    def select(self, modes: TaskModes) -> None:
        live = self.resolve({})
        for axis, names in self.resolve(modes).items():
            if names != live[axis]:
                self._server.rosparam[str].set(mode_param(axis), ",".join(names))


def declare_task_params(node: ROSParamServer) -> TaskModeParams:
    return TaskModeParams(node)


async def create_task(*, node: ArenaMixinNode, ctx: TaskContext, params: TaskParams) -> Task:
    register_prompt_mode(ctx.conf.HUMAN.value)
    mode_params = typing.cast("TaskModeParams", params)
    return await Task.create(node=node, ctx=ctx, params=mode_params, modules=mode_params.modules)


def declare_mode_schema(node: ROSParamServer, axis: str, mode: str) -> None:
    _, registry = _REGISTRIES[axis]
    meta = registry.meta(next(key for key in registry.keys() if key.value == mode))
    if meta.schema is not None:
        meta.schema(node, namespace=meta.namespace)


def _namespaces[K: enum.Enum, V](registry: TaskModeRegistry[K, V]) -> dict[str, str]:
    return {key.value: str(registry.meta(key).namespace) for key in sorted(registry.keys(), key=lambda key: key.value)}


def mode_namespaces() -> dict[str, dict[str, str]]:
    return {axis: _namespaces(registry) for axis, (_, registry) in _REGISTRIES.items()}


def fleet_only_robots() -> list[str]:
    return extra_tm_kinds()


async def next_episode(*, previous: EpisodeRecord, queued: TaskModeOverrides) -> TaskModeOverrides:
    del previous
    return queued
