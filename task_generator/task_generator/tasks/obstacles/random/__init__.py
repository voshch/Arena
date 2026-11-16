import typing

from arena_rclpy_mixins.params import Catalog, Pair, Param, ParamGroup, pair

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, OBSTACLES_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.obstacles import TM_Obstacles

_NS = _REGISTRY_NAMESPACE("random")


class RandomParams(ParamGroup):
    STATIC_N = Param[tuple[float, float]]("static.n", [0, 0], "[min, max] count of static obstacles.", constraint=Pair(), parse=pair)
    INTERACTIVE_N = Param[tuple[float, float]]("interactive.n", [0, 0], "[min, max] count of interactive obstacles.", constraint=Pair(), parse=pair)
    DYNAMIC_N = Param[tuple[float, float]]("dynamic.n", [0, 0], "[min, max] count of dynamic obstacles.", constraint=Pair(), parse=pair)
    STATIC_MODELS = Param[list[str]]("static.models", ["Warehouse/Object/Barrel_Plastic"], "Allowed static obstacle models (empty = all).", constraint=Catalog("objects"))
    INTERACTIVE_MODELS = Param[list[str]]("interactive.models", ["Warehouse/Object/Barrel_Plastic"], "Allowed interactive obstacle models (empty = all).", constraint=Catalog("objects"))
    DYNAMIC_MODELS = Param[list[str]]("dynamic.models", ["arenian"], "Allowed dynamic obstacle models (empty = all).", constraint=Catalog("pedestrians"))


@OBSTACLES_MODES.register(Modes.TM_Obstacles.RANDOM, namespace=_NS, schema=RandomParams)
def _load_random() -> type["TM_Obstacles"]:
    from .impl import TM_Random

    return TM_Random
