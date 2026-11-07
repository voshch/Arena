import typing

from arena_rclpy_mixins.declarations import declare_bool, declare_double, declare_string
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_rclpy_mixins.shared import Namespace

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("search")


def _declare_schema(node: ROSParamServer, ns: Namespace) -> None:
    declare_string(node, ns("target"), "", label="Target", description="Scenario human to find. Empty picks the scenario's only human.")
    declare_double(node, ns("radius"), 1.5, label="Found radius", description="Distance in m to the target at which the robot has found it.", lo=0.1, hi=10.0)
    declare_bool(node, ns("line_of_sight"), False, label="Line of sight", description="Count the target found only while no wall lies between the robot and it.")
    declare_double(node, ns("dwell_s"), 0.0, label="Dwell", description="Seconds the robot has to keep the target found before the episode succeeds, 0 = at once.", lo=0.0, hi=60.0)
    declare_bool(node, ns("linger_after_completion"), False, label="Linger after completion", description="Keep the episode running after every robot found the target, until the episode timeout.")
    declare_bool(node, ns("mechanism_map"), False, label="Mechanism map", description="Serve the planner map with every door blocked that no robot of this mode can open and that is not open right now.")


@ROBOTS_MODES.register(Modes.TM_Robots.SEARCH, namespace=_NS, schema=_declare_schema)
def _load_search() -> type["TM_Robots"]:
    from .impl import TM_Search

    return TM_Search
