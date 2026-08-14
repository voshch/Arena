import math
import typing

from arena_rclpy_mixins.declarations import declare_double
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_rclpy_mixins.shared import Namespace

from task_generator.constants import Constants
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("stationary")

_UNSET_DESCRIPTION = "NaN in pos_x or pos_y leaves every robot at its configured start pose."


def _declare_schema(node: ROSParamServer, ns: Namespace) -> None:
    declare_double(node, ns("pos_x"), math.nan, label="Parking x", description=f"X of the shared parking pose. {_UNSET_DESCRIPTION}")
    declare_double(node, ns("pos_y"), math.nan, label="Parking y", description=f"Y of the shared parking pose. {_UNSET_DESCRIPTION}")
    declare_double(node, ns("pos_theta"), 0.0, label="Parking yaw", description="Yaw of the shared parking pose in radians. Applied only when pos_x and pos_y are both set.")


@ROBOTS_MODES.register(Constants.TaskMode.TM_Robots.STATIONARY, namespace=_NS, schema=_declare_schema)
def _load_stationary() -> type["TM_Robots"]:
    from .impl import TM_Stationary

    return TM_Stationary
