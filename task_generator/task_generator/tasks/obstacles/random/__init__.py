import typing

from arena_rclpy_mixins.declarations import declare_catalog_array, declare_int_pair
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_rclpy_mixins.shared import Namespace

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, OBSTACLES_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.obstacles import TM_Obstacles

_NS = _REGISTRY_NAMESPACE("random")


def _declare_schema(node: ROSParamServer, ns: Namespace) -> None:
    declare_int_pair(node, ns("static", "n"), [0, 0], label="Static count", description="[min, max] count of static obstacles.")
    declare_int_pair(node, ns("interactive", "n"), [0, 0], label="Interactive count", description="[min, max] count of interactive obstacles.")
    declare_int_pair(node, ns("dynamic", "n"), [0, 0], label="Dynamic count", description="[min, max] count of dynamic obstacles.")
    declare_catalog_array(node, ns("static", "models"), ["shelf"], catalog="objects", label="Static models", description="Allowed static obstacle models (empty = all).")
    declare_catalog_array(node, ns("interactive", "models"), ["shelf"], catalog="objects", label="Interactive models", description="Allowed interactive obstacle models (empty = all).")
    declare_catalog_array(node, ns("dynamic", "models"), ["arenian"], catalog="pedestrians", label="Dynamic models", description="Allowed dynamic obstacle models (empty = all).")


@OBSTACLES_MODES.register(Modes.TM_Obstacles.RANDOM, namespace=_NS, schema=_declare_schema)
def _load_random() -> type["TM_Obstacles"]:
    from .impl import TM_Random

    return TM_Random
