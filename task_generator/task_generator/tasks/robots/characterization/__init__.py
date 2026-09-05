import typing

from arena_rclpy_mixins.declarations import declare_bool, declare_double, declare_double_array
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_rclpy_mixins.shared import Namespace

from task_generator.constants import Constants
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

from .schedule import DURATION_DEFAULTS, MODES, SWEEP_DEFAULTS

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("characterization")

_MODE_DESCRIPTIONS = {
    "idle": "Standstill blocks measuring baseline standby draw.",
    "linear": "Longitudinal out-and-back sweep through the rated linear speeds.",
    "lateral": "Sideways sweep, holonomic robots only.",
    "arc": "Closed circular orbits at radii scaled from the footprint.",
    "ramps": "Acceleration and deceleration ramps to every linear step.",
    "brake": "Cruise at the rated speed, then a step command to zero.",
    "angular": "In-place pivot sweep through the rated angular rates.",
}

_DURATION_DESCRIPTIONS = {
    "idle_s": "Length of each standstill block. Pre-block idles run half as long.",
    "linear_dwell_s": "Capture per linear speed step, shortened when the excursion budget runs out.",
    "linear_settle_s": "Rest between directions, also used after lateral and arc maneuvers.",
    "lateral_dwell_s": "Capture per lateral speed step.",
    "angular_dwell_s": "Capture per in-place pivot rate.",
    "ramp_settle_s": "Hold at the ramp apex before decelerating.",
    "brake_approach_s": "Cruise at the rated speed before the step to zero.",
    "brake_dwell_s": "Capture after the step to zero.",
}

_SWEEP_DESCRIPTIONS = {
    "arc_speed_factors": "Arc speeds as fractions of the rated linear speed.",
    "arc_radius_factors": "Arc radii as multiples of the footprint radius.",
    "ramp_horizons_s": "Ramp horizons in seconds, one acceleration family per entry.",
}


def _declare_schema(node: ROSParamServer, ns: Namespace) -> None:
    for name in MODES:
        declare_bool(node, ns("modes", name), True, label=f"Sweep {name}", description=_MODE_DESCRIPTIONS[name])
    for name, default in DURATION_DEFAULTS.items():
        label = name.removesuffix("_s").replace("_", " ").capitalize()
        declare_double(node, ns(name), default, label=f"{label} (s)", description=_DURATION_DESCRIPTIONS[name])
    for name, default in SWEEP_DEFAULTS.items():
        label = name.removesuffix("_s").replace("_", " ").capitalize()
        declare_double_array(node, ns(name), default, label=label, description=_SWEEP_DESCRIPTIONS[name])


@ROBOTS_MODES.register(Constants.TaskMode.TM_Robots.CHARACTERIZATION, namespace=_NS, schema=_declare_schema)
def _load_characterization() -> type["TM_Robots"]:
    from .impl import TM_Characterization

    return TM_Characterization
