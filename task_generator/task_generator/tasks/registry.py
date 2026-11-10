import typing
from collections.abc import Callable, Iterable

import attrs
from arena_rclpy_mixins.params import ParamGroup
from arena_rclpy_mixins.registry import ClassRegistry
from arena_rclpy_mixins.shared import Namespace
from arena_simulation_setup.tree import Identifier

from task_generator.modes import Modes
from task_generator.tasks.mode import Namespaced

if typing.TYPE_CHECKING:
    from task_generator.tasks.modules import TM_Module
    from task_generator.tasks.obstacles import TM_Obstacles
    from task_generator.tasks.robots import TM_Robots

_REGISTRY_NAMESPACE: Namespace = Namespaced._namespace("task")
_RESERVED_NAMESPACE: Namespace = _REGISTRY_NAMESPACE("episode")


def identifier_to_available(identifier: type[Identifier], **kwargs: bool) -> Iterable[str]:
    yield from (identifier.shortname for identifier in identifier.listall(**kwargs))


async def identifier_to_available_async(identifier: type[Identifier], **kwargs: bool) -> list[str]:
    return [identifier.shortname for identifier in await identifier.listall_async(**kwargs)]


@attrs.frozen
class TaskModeMeta:
    namespace: Namespace
    schema: type[ParamGroup] | None = None


class TaskModeRegistry[K, V]:
    """Lazy class registry with eager per-key metadata. Meta is registry-side (not class-side like AdapterMeta/BringupMeta) because the layer reads it before any impl is imported."""

    def __init__(self) -> None:
        self._classes: ClassRegistry[K, V] = ClassRegistry()
        self._meta: dict[K, TaskModeMeta] = {}

    def register(self, key: K, *, namespace: Namespace, schema: type[ParamGroup] | None = None) -> Callable[[Callable[[], V]], Callable[[], V]]:
        if namespace == _RESERVED_NAMESPACE:
            raise ValueError(f"{namespace} holds the task.episode.* parameters, pick another mode name for {key}")
        meta = TaskModeMeta(namespace=namespace, schema=schema)

        def _dec(loader: Callable[[], V]) -> Callable[[], V]:
            self._classes.register(key)(loader)
            self._meta[key] = meta
            return loader

        return _dec

    def get(self, key: K) -> V:
        return self._classes.get(key)

    def meta(self, key: K) -> TaskModeMeta:
        return self._meta[key]

    def keys(self) -> typing.KeysView[K]:
        return self._classes.keys()

    def __contains__(self, key: object) -> bool:
        return key in self._classes


ROBOTS_MODES: TaskModeRegistry[Modes.TM_Robots, type["TM_Robots"]] = TaskModeRegistry()
OBSTACLES_MODES: TaskModeRegistry[Modes.TM_Obstacles, type["TM_Obstacles"]] = TaskModeRegistry()
MODULE_MODES: TaskModeRegistry[Modes.TM_Module, type["TM_Module"]] = TaskModeRegistry()
