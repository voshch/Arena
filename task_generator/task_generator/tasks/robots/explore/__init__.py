import typing

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("explore")


@ROBOTS_MODES.register(Modes.TM_Robots.EXPLORE, namespace=_NS)
def _load_explore() -> type["TM_Robots"]:
    from .impl import TM_Explore

    return TM_Explore
