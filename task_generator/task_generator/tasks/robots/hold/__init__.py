import math
import typing

from arena_rclpy_mixins.params import Param, ParamGroup, Range

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("hold")


class HoldParams(ParamGroup):
    ZONE = Param[str]("zone", "", "World zone the spots are sampled from. Empty uses the center and radius instead.")
    CENTER_X = Param[float]("center_x", math.nan, "X of the sampling center. NaN leaves the center unset, then zone must name a zone.")
    CENTER_Y = Param[float]("center_y", math.nan, "Y of the sampling center. NaN leaves the center unset, then zone must name a zone.")
    RADIUS = Param[float]("radius", 1.5, "Radius in m of the disc the spots are sampled from around the center. Zones ignore it.", constraint=Range(0.1, 10.0))
    HOLD_S = Param[float]("hold_s", 60.0, "Seconds the robot must stay in the target disc around its spot, the goal tolerance task.episode.goto_pose.tolerance.radius. The count starts when it enters the disc and restarts whenever it leaves.", constraint=Range(0.1, 3600.0))


@ROBOTS_MODES.register(Modes.TM_Robots.HOLD, namespace=_NS, schema=HoldParams)
def _load_hold() -> type["TM_Robots"]:
    from .impl import TM_Hold

    return TM_Hold
