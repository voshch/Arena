import typing

from arena_rclpy_mixins.declarations import declare_bool, declare_double, declare_string
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_rclpy_mixins.shared import Namespace

from task_generator.constants import Constants
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("patrol")

DEFAULT_CLIP = "beckon"


def declare_schema(node: ROSParamServer, ns: Namespace) -> None:
    """Patrol-with-service: the robot walks the scenario's checkpoints on a loop until a pedestrian's call reaches it."""
    declare_string(node, ns("service_agent"), "", label="Service agent", description="Name of the pedestrian whose call ends the episode when answered. Empty = the first pedestrian that beckons.")
    declare_string(node, ns("clip"), DEFAULT_CLIP, label="Call clip", description="Body clip that counts as a call for help.")
    declare_double(node, ns("service_radius"), 2.5, label="Service radius", description="The pedestrian counts as served once the robot holds within this distance for service_dwell.", lo=0.3, hi=10.0)
    declare_double(node, ns("service_dwell"), 0.0, label="Service dwell", description="Seconds the robot must hold inside service_radius. 0 = the call counts as answered the moment the robot gets there, which a pass at walking pace satisfies; a dwell can outlast the few seconds a pass spends inside the radius.", lo=0.0, hi=30.0)
    declare_double(node, ns("service_standoff"), 0.8, label="Service standoff", description="Distance in m from the pedestrian at which the service goal is placed.", lo=0.3, hi=5.0)
    declare_double(node, ns("sensor_range"), 12.0, label="Sensor range", description="Maximum range at which a call can be perceived, the platform's range-sensor reach.", lo=1.0, hi=50.0)
    declare_bool(node, ns("require_line_of_sight"), True, label="Require line of sight", description="A call behind a wall is not perceived. Off = the robot is told regardless of what it could see.")
    declare_double(node, ns("recognition_dwell"), 0.5, label="Recognition dwell", description="Seconds the call must stay continuously visible before the robot acts on it.", lo=0.0, hi=10.0)


@ROBOTS_MODES.register(Constants.TaskMode.TM_Robots.PATROL, namespace=_NS, schema=declare_schema)
def _load_patrol() -> type["TM_Robots"]:
    from .impl import TM_Patrol

    return TM_Patrol
