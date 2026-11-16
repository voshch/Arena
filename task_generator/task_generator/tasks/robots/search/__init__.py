import typing

from arena_rclpy_mixins.params import Param, ParamGroup, Range

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("search")


class SearchParams(ParamGroup):
    TARGET = Param[str]("target", "", "Scenario human to find. Empty picks the scenario's only human.")
    RADIUS = Param[float]("radius", 1.5, "Distance in m to the target at which the robot has found it.", constraint=Range(0.1, 10.0))
    LINE_OF_SIGHT = Param[bool]("line_of_sight", False, "Count the target found only while no wall lies between the robot and it.")
    DWELL_S = Param[float]("dwell_s", 0.0, "Seconds the robot has to keep the target found before the episode succeeds, 0 = at once.", constraint=Range(0.0, 60.0))
    LINGER_AFTER_COMPLETION = Param[bool]("linger_after_completion", False, "Keep the episode running after every robot found the target, until the episode timeout.")
    MECHANISM_MAP = Param[bool]("mechanism_map", False, "Serve the planner map with every door blocked that no robot of this mode can open and that is not open right now.")


@ROBOTS_MODES.register(Modes.TM_Robots.SEARCH, namespace=_NS, schema=SearchParams)
def _load_search() -> type["TM_Robots"]:
    from .impl import TM_Search

    return TM_Search
