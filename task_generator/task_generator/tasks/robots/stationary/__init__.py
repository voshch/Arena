import math
import typing

from arena_rclpy_mixins.params import Param, ParamGroup

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("stationary")

_UNSET_DESCRIPTION = "NaN in pos_x or pos_y parks every robot at a free pose sampled per episode."


class StationaryParams(ParamGroup):
    POS_X = Param[float]("pos_x", math.nan, f"X of the shared parking pose. {_UNSET_DESCRIPTION}")
    POS_Y = Param[float]("pos_y", math.nan, f"Y of the shared parking pose. {_UNSET_DESCRIPTION}")
    POS_THETA = Param[float]("pos_theta", 0.0, "Yaw of the shared parking pose in radians. Applied only when pos_x and pos_y are both set.")


@ROBOTS_MODES.register(Modes.TM_Robots.STATIONARY, namespace=_NS, schema=StationaryParams)
def _load_stationary() -> type["TM_Robots"]:
    from .impl import TM_Stationary

    return TM_Stationary
