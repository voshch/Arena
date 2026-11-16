import typing

from arena_rclpy_mixins.params import Catalog, Choices, Param, ParamGroup, Range

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, OBSTACLES_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.obstacles import TM_Obstacles

_NS = _REGISTRY_NAMESPACE("scenario")

CONTACT_MODES = ("enabled", "locomotion_only")
GESTURE_MODES = ("enabled", "disabled")


class ScenarioParams(ParamGroup):
    """Shared by the `scenario` task mode of TM_Obstacles and TM_Robots."""

    FILE = Param[str]("file", "default", "Scenario file name.", constraint=Catalog("scenarios"))
    CONTACT_MODE = Param[str]("contact_mode", CONTACT_MODES[0], "locomotion_only = motion-matched control: contact interactions (hug, handshake) approach and hold at standing_distance, no contact clip.", constraint=Choices(*CONTACT_MODES))
    STANDING_DISTANCE = Param[float]("standing_distance", 1.2, "Pair separation in m for contact interactions under contact_mode=locomotion_only.", constraint=Range(0.3, 5.0))
    GESTURE_MODE = Param[str]("gesture_mode", GESTURE_MODES[0], "disabled = the gesture-off control: agents behave identically but publish no gesture, so nothing renders and no consumer can read one.", constraint=Choices(*GESTURE_MODES))


@OBSTACLES_MODES.register(Modes.TM_Obstacles.SCENARIO, namespace=_NS, schema=ScenarioParams)
def _load_scenario() -> type["TM_Obstacles"]:
    from .impl import TM_Scenario

    return TM_Scenario
