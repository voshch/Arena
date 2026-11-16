import enum
import typing

from arena_rclpy_mixins.params import Param, ParamGroup, Range

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("demo")

_RANDOM_ALIASES = frozenset({"", "random", "<random>"})


def _parse_gesture(value: object) -> str:
    from arena_simulation_setup import ASS_DIR

    if value in _RANDOM_ALIASES:
        return "<random>"
    available = {p.stem for p in (ASS_DIR / "configs" / "gestures").glob("*.yaml")}
    if value not in available:
        raise ValueError(f"unknown gesture {value!r}; available: {sorted(available | {'<random>'})}")
    return str(value)


class DemoOrientation(enum.Enum):
    TANGENT = "tangent"
    RADIAL_IN = "radial_in"
    RADIAL_OUT = "radial_out"


class DemoParams(ParamGroup):
    VERTICES = Param[int]("vertices", 4, "Corners of the regular polygon each robot drives, at least 3.", constraint=Range(3, 20))
    RADIUS = Param[float]("radius", 1.5, "Distance in m from the polygon center to each corner.", constraint=Range(0.5, 10.0))
    GESTURE = Param[str]("gesture", "<random>", "Gesture played at every corner, <random> draws one per corner.", parse=_parse_gesture)
    ORIENTATION = Param[DemoOrientation]("orientation", DemoOrientation.RADIAL_IN, "Heading at each corner: tangent faces the next corner, radial_in the center, radial_out away from it.")


@ROBOTS_MODES.register(Modes.TM_Robots.DEMO, namespace=_NS, schema=DemoParams)
def _load_demo() -> type["TM_Robots"]:
    from .impl import TM_Demo

    return TM_Demo
