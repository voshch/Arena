import typing

from arena_rclpy_mixins.params import Catalog, Param, ParamGroup

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, OBSTACLES_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.obstacles import TM_Obstacles

_NS = _REGISTRY_NAMESPACE("environment")


class EnvironmentParams(ParamGroup):
    FILE = Param[str]("file", "office.yaml", "Environment config file name.", constraint=Catalog("environments"))


@OBSTACLES_MODES.register(Modes.TM_Obstacles.ENVIRONMENT, namespace=_NS, schema=EnvironmentParams)
def _load_environment() -> type["TM_Obstacles"]:
    from .impl import TM_Environment

    return TM_Environment
