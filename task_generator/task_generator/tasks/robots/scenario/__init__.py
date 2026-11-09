import typing

from arena_rclpy_mixins.params import Param

from task_generator.modes import Modes
from task_generator.tasks.obstacles.scenario import ScenarioParams
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("scenario")


class ScenarioRobotParams(ScenarioParams):
    LINGER_AFTER_COMPLETION = Param[bool]("linger_after_completion", False, "Keep the episode running after every robot finished its phases, until the episode timeout.")
    MECHANISM_MAP = Param[bool]("mechanism_map", False, "Serve the planner map with every door blocked that no robot of this mode can open and that is not open right now.")


@ROBOTS_MODES.register(Modes.TM_Robots.SCENARIO, namespace=_NS, schema=ScenarioRobotParams)
def _load_scenario() -> type["TM_Robots"]:
    from .impl import TM_Scenario

    return TM_Scenario
