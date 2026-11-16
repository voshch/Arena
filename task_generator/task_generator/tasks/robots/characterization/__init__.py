import typing

from arena_rclpy_mixins.params import Param, ParamGroup
from arena_robots.characterization import DURATION_DEFAULTS, SWEEP_DEFAULTS
from rclpy import Parameter

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

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


class CharacterizationParams(ParamGroup):
    MODE_IDLE = Param[bool]("modes.idle", True, _MODE_DESCRIPTIONS["idle"])
    MODE_LINEAR = Param[bool]("modes.linear", True, _MODE_DESCRIPTIONS["linear"])
    MODE_LATERAL = Param[bool]("modes.lateral", True, _MODE_DESCRIPTIONS["lateral"])
    MODE_ARC = Param[bool]("modes.arc", True, _MODE_DESCRIPTIONS["arc"])
    MODE_RAMPS = Param[bool]("modes.ramps", True, _MODE_DESCRIPTIONS["ramps"])
    MODE_BRAKE = Param[bool]("modes.brake", True, _MODE_DESCRIPTIONS["brake"])
    MODE_ANGULAR = Param[bool]("modes.angular", True, _MODE_DESCRIPTIONS["angular"])
    IDLE_S = Param[float]("idle_s", DURATION_DEFAULTS["idle_s"], _DURATION_DESCRIPTIONS["idle_s"])
    LINEAR_DWELL_S = Param[float]("linear_dwell_s", DURATION_DEFAULTS["linear_dwell_s"], _DURATION_DESCRIPTIONS["linear_dwell_s"])
    LINEAR_SETTLE_S = Param[float]("linear_settle_s", DURATION_DEFAULTS["linear_settle_s"], _DURATION_DESCRIPTIONS["linear_settle_s"])
    LATERAL_DWELL_S = Param[float]("lateral_dwell_s", DURATION_DEFAULTS["lateral_dwell_s"], _DURATION_DESCRIPTIONS["lateral_dwell_s"])
    ANGULAR_DWELL_S = Param[float]("angular_dwell_s", DURATION_DEFAULTS["angular_dwell_s"], _DURATION_DESCRIPTIONS["angular_dwell_s"])
    RAMP_SETTLE_S = Param[float]("ramp_settle_s", DURATION_DEFAULTS["ramp_settle_s"], _DURATION_DESCRIPTIONS["ramp_settle_s"])
    BRAKE_APPROACH_S = Param[float]("brake_approach_s", DURATION_DEFAULTS["brake_approach_s"], _DURATION_DESCRIPTIONS["brake_approach_s"])
    BRAKE_DWELL_S = Param[float]("brake_dwell_s", DURATION_DEFAULTS["brake_dwell_s"], _DURATION_DESCRIPTIONS["brake_dwell_s"])
    ARC_SPEED_FACTORS = Param[list[float]]("arc_speed_factors", list(SWEEP_DEFAULTS["arc_speed_factors"]), _SWEEP_DESCRIPTIONS["arc_speed_factors"], type_=Parameter.Type.DOUBLE_ARRAY)
    ARC_RADIUS_FACTORS = Param[list[float]]("arc_radius_factors", list(SWEEP_DEFAULTS["arc_radius_factors"]), _SWEEP_DESCRIPTIONS["arc_radius_factors"], type_=Parameter.Type.DOUBLE_ARRAY)
    RAMP_HORIZONS_S = Param[list[float]]("ramp_horizons_s", list(SWEEP_DEFAULTS["ramp_horizons_s"]), _SWEEP_DESCRIPTIONS["ramp_horizons_s"], type_=Parameter.Type.DOUBLE_ARRAY)


@ROBOTS_MODES.register(Modes.TM_Robots.CHARACTERIZATION, namespace=_NS, schema=CharacterizationParams)
def _load_characterization() -> type["TM_Robots"]:
    from .impl import TM_Characterization

    return TM_Characterization
