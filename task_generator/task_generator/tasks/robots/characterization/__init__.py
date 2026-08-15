import typing

from task_generator.constants import Constants
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("characterization")


@ROBOTS_MODES.register(Constants.TaskMode.TM_Robots.CHARACTERIZATION, namespace=_NS)
def _load_characterization() -> type["TM_Robots"]:
    from .impl import TM_Characterization

    return TM_Characterization
