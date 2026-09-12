import typing

from arena_rclpy_mixins.declarations import declare_catalog, declare_double, declare_enum
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_rclpy_mixins.shared import Namespace

from task_generator.constants import Constants
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, OBSTACLES_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.obstacles import TM_Obstacles

_NS = _REGISTRY_NAMESPACE("scenario")

CONTACT_MODES = ("enabled", "locomotion_only")


def declare_schema(node: ROSParamServer, ns: Namespace) -> None:
    """Shared schema for the `scenario` task mode, used by both TM_Obstacles and TM_Robots."""
    declare_catalog(node, ns("file"), "default", catalog="scenarios", label="Scenario file", description="Scenario file name.")
    declare_enum(
        node,
        ns("contact_mode"),
        CONTACT_MODES[0],
        choices=CONTACT_MODES,
        label="Contact mode",
        description="locomotion_only = motion-matched control: contact interactions (hug, handshake) approach and hold at standing_distance, no contact clip.",
    )
    declare_double(node, ns("standing_distance"), 1.2, label="Standing distance", description="Pair separation in m for contact interactions under contact_mode=locomotion_only.", lo=0.3, hi=5.0)


@OBSTACLES_MODES.register(Constants.TaskMode.TM_Obstacles.SCENARIO, namespace=_NS, schema=declare_schema)
def _load_scenario() -> type["TM_Obstacles"]:
    from .impl import TM_Scenario

    return TM_Scenario
