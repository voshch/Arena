import typing

from arena_rclpy_mixins.params import Catalog, Param, ParamGroup
from arena_simulation_setup.tree.configs.parametrized import ParametrizedConfig, ParametrizedIdentifier

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, OBSTACLES_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.obstacles import TM_Obstacles

_NS = _REGISTRY_NAMESPACE("parametrized")


def _parse_config(name: object) -> ParametrizedConfig:
    return ParametrizedIdentifier(str(name)).resolve_sync()


class ParametrizedParams(ParamGroup):
    FILE = Param[ParametrizedConfig]("file", "default.xml", "Parametrized config file name.", constraint=Catalog("parametrizeds"), parse=_parse_config)


@OBSTACLES_MODES.register(Modes.TM_Obstacles.PARAMETRIZED, namespace=_NS, schema=ParametrizedParams)
def _load_parametrized() -> type["TM_Obstacles"]:
    from .impl import TM_Parametrized

    return TM_Parametrized
