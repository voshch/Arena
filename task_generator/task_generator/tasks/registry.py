import typing
from collections.abc import Callable, Iterable

import attrs
from arena_rclpy_mixins.registry import ClassRegistry
from arena_rclpy_mixins.shared import Namespace
from arena_simulation_setup.tree import Identifier
from arena_simulation_setup.tree.World import WorldIdentifier

from task_generator.constants import Constants
from task_generator.tasks.mode import Namespaced

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer

    from task_generator.tasks.modules import TM_Module
    from task_generator.tasks.obstacles import TM_Obstacles
    from task_generator.tasks.robots import TM_Robots

_REGISTRY_NAMESPACE: Namespace = Namespaced._namespace("task")


def identifier_to_available(identifier: type[Identifier], **kwargs: object) -> Iterable[str]:
    yield from (identifier.shortname for identifier in identifier.listall(**kwargs))


async def identifier_to_available_async(identifier: type[Identifier], **kwargs: object) -> list[str]:
    return [identifier.shortname for identifier in await identifier.listall_async(**kwargs)]


def default_scenario(world: str) -> str:
    """'default' when the world ships it, else its first scenario."""
    scenarios = list(identifier_to_available(WorldIdentifier(world).resolve_sync().scenario))
    if not scenarios:
        raise ValueError(f"No scenarios found in world {world}")
    return 'default' if 'default' in scenarios else scenarios[0]


_SchemaFn = Callable[["ROSParamServer", Namespace], None]


@attrs.frozen
class TaskModeMeta:
    namespace: Namespace
    schema: _SchemaFn | None = None


class TaskModeRegistry[K, V]:
    """Lazy class registry with eager per-key metadata. Meta is registry-side (not class-side like AdapterMeta/BringupMeta) because `walk_schemas` reads it before any impl is imported."""

    def __init__(self) -> None:
        self._classes: ClassRegistry[K, V] = ClassRegistry()
        self._meta: dict[K, TaskModeMeta] = {}

    def register(self, key: K, *, namespace: Namespace, schema: _SchemaFn | None = None) -> Callable[[Callable[[], V]], Callable[[], V]]:
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


ROBOTS_MODES: TaskModeRegistry[Constants.TaskMode.TM_Robots, type["TM_Robots"]] = TaskModeRegistry()
OBSTACLES_MODES: TaskModeRegistry[Constants.TaskMode.TM_Obstacles, type["TM_Obstacles"]] = TaskModeRegistry()
MODULE_MODES: TaskModeRegistry[Constants.TaskMode.TM_Module, type["TM_Module"]] = TaskModeRegistry()


def walk_schemas(node: "ROSParamServer") -> None:
    for reg in (MODULE_MODES, OBSTACLES_MODES, ROBOTS_MODES):
        for key in reg.keys():
            meta = reg.meta(key)
            if meta.schema is not None:
                meta.schema(node, meta.namespace)
