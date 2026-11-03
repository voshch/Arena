import typing

from arena_rclpy_mixins.declarations import declare_bool
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_rclpy_mixins.shared import Namespace

from task_generator.modes import Modes
from task_generator.tasks.obstacles.scenario import declare_schema
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("scenario")


def _declare_schema(node: ROSParamServer, ns: Namespace) -> None:
    declare_schema(node, ns)
    declare_bool(node, ns("linger_after_completion"), False, label="Linger after completion", description="Keep the episode running after every robot finished its phases, until the episode timeout.")
    declare_bool(node, ns("mechanism_map"), False, label="Mechanism map", description="Serve the planner map with every door blocked that no robot of this mode can open and that is not open right now.")


@ROBOTS_MODES.register(Modes.TM_Robots.SCENARIO, namespace=_NS, schema=_declare_schema)
def _load_scenario() -> type["TM_Robots"]:
    from .impl import TM_Scenario

    return TM_Scenario
