import typing

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("demo")


@ROBOTS_MODES.register(Modes.TM_Robots.DEMO, namespace=_NS)
def _load_demo() -> type["TM_Robots"]:
    from .impl import TM_Demo

    return TM_Demo
