import math
import typing

from arena_rclpy_mixins.params import Param, ParamGroup, Range

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("random")


class RandomParams(ParamGroup):
    GOAL_ZONE = Param[str]("goal_zone", "", "World zone the goals are sampled from. Empty samples from the whole map unless a goal center is set.")
    GOAL_CENTER_X = Param[float]("goal_center_x", math.nan, "X of the goal sampling center. NaN leaves the center unset.")
    GOAL_CENTER_Y = Param[float]("goal_center_y", math.nan, "Y of the goal sampling center. NaN leaves the center unset.")
    GOAL_RADIUS = Param[float]("goal_radius", 1.5, "Radius in m of the disc the goals are sampled from around the goal center. Zones ignore it.", constraint=Range(0.1, 10.0))
    START_ZONE = Param[str]("start_zone", "", "World zone the robots start in. Empty starts on the whole map unless a start center is set.")
    START_CENTER_X = Param[float]("start_center_x", math.nan, "X of the start sampling center. NaN leaves the center unset.")
    START_CENTER_Y = Param[float]("start_center_y", math.nan, "Y of the start sampling center. NaN leaves the center unset.")
    START_RADIUS = Param[float]("start_radius", 1.5, "Radius in m of the disc the starts are sampled from around the start center. Zones ignore it.", constraint=Range(0.1, 10.0))
    HOLD_S = Param[float]("hold_s", 0.0, "Seconds each robot must stay in the target disc around its goal, the goal tolerance, after reaching the goal and before its arm phases. 0 turns the hold off.", constraint=Range(0.0, 3600.0))


@ROBOTS_MODES.register(Modes.TM_Robots.RANDOM, namespace=_NS, schema=RandomParams)
def _load_random() -> type["TM_Robots"]:
    from .impl import TM_Random

    return TM_Random
