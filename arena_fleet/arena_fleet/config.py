import pathlib
import types
import typing

import yaml


class Zone(typing.TypedDict):
    x: list[float]
    y: list[float]


class TaskConfig(typing.TypedDict):
    goal_zone: Zone
    min_separation_m: float
    clearance_m: float
    assign: typing.Literal["hungarian", "fixed"]


class WeightsConfig(typing.TypedDict):
    time: float
    encounter: float
    facility: float
    wait: float


class PlanConfig(typing.TypedDict):
    funnel_len_m: float
    funnel_k: float
    occupied_cost_per_m: float
    around_horizon_s: float


class TimeConfig(typing.TypedDict):
    v_nom: float


class EncounterConfig(typing.TypedDict):
    horizon_s: float


class FacilityConfig(typing.TypedDict):
    open_door_s: float
    robot_same_door_s: float
    person_at_door_s: float
    crowd_r_m: float


class WaitConfig(typing.TypedDict):
    cross_m: float


class PerceptionConfig(typing.TypedDict):
    range_m: float
    fov_deg: float
    memory_s: float


class GridConfig(typing.TypedDict):
    cell_m: float
    x_max_m: float
    y_max_m: float


class PeopleViewConfig(typing.TypedDict):
    period_s: float
    dt_s: float
    person_r_m: float
    robot_r_m: float
    grid: GridConfig


class FlowConfig(typing.TypedDict):
    period_s: float
    threshold: float
    footprint_r_m: float
    blend_tau_s: float
    sigma0_m: float
    sigma_per_s: float


class HataConfig(typing.TypedDict):
    w0: float
    w1: float
    delta: float


class CostConfig(typing.TypedDict):
    mode: typing.NotRequired[typing.Literal["fleet", "hata"]]
    hata: typing.NotRequired[HataConfig]
    task: TaskConfig
    weights: WeightsConfig
    plan: PlanConfig
    time: TimeConfig
    encounter: EncounterConfig
    facility: FacilityConfig
    wait: WaitConfig
    perception: PerceptionConfig
    people_view: PeopleViewConfig
    flow: FlowConfig


RULES = ("capacity", "queue", "yield_people", "plan_around", "slot_by_people", "reassign_door", "reassign_goal")


class Profile(typing.TypedDict):
    capacity: bool
    queue: bool
    yield_people: bool
    plan_around: bool
    slot_by_people: bool
    reassign_door: bool
    reassign_goal: bool


class CapacityConfig(typing.TypedDict):
    k: int
    hold_m: float
    through_m: float
    occupy_timeout_s: float | None  # None = held until the robot is through


class QueueConfig(typing.TypedDict):
    join_m: float
    head_m: list[float]
    lateral_m: float
    spacing_m: float
    stagger_m: float
    keep_slot: bool
    other_queue_m: float
    clearance_m: float
    slot_tol_m: float


class YieldConfig(typing.TypedDict):
    margin_m: float
    lookahead_s: float
    max_wait_s: float
    hold_ttc_s: float
    clear_s: float
    no_return_m: float
    grace_s: float
    regrace_s: float
    aside_radii_m: list[float]
    aside_check_s: float
    aside_clearance_m: float
    aside_tol_m: float


class ReassignDoorConfig(typing.TypedDict):
    period_s: float
    hysteresis_j: float
    min_dwell_s: float
    lock_m: float


class ReassignGoalConfig(typing.TypedDict):
    period_s: float
    margin_j: float
    min_dwell_s: float
    lock_m: float


class EpisodeConfig(typing.TypedDict):
    goal_tol_m: float
    stuck_s: float
    tail_s: float


class RulesConfig(typing.TypedDict):
    default_profile: str
    profiles: dict[str, Profile]
    capacity: CapacityConfig
    queue: QueueConfig
    yield_people: YieldConfig
    reassign_door: ReassignDoorConfig
    reassign_goal: ReassignGoalConfig
    episode: EpisodeConfig


def _is_typed_dict(kind: object) -> typing.TypeGuard[type]:
    return isinstance(kind, type) and typing.is_typeddict(kind)


def _check(value: object, kind: object, where: str) -> None:
    """Raise ValueError when a YAML value does not have the shape of its annotation."""
    origin, args = typing.get_origin(kind), typing.get_args(kind)
    if origin in (typing.NotRequired, typing.Required):
        _check(value, args[0], where)
    elif _is_typed_dict(kind):
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected a mapping, got {value!r}")
        hints = typing.get_type_hints(kind, include_extras=True)
        required = {key for key, hint in hints.items() if typing.get_origin(hint) is not typing.NotRequired}
        missing, unknown = sorted(required - set(value)), sorted(set(value) - set(hints))
        if missing or unknown:
            raise ValueError(f"{where}: missing keys {missing}, unknown keys {unknown}")
        for key, item in value.items():
            _check(item, hints[key], f"{where}.{key}")
    elif origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected a mapping, got {value!r}")
        for key, item in value.items():
            _check(item, args[1], f"{where}.{key}")
    elif origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{where}: expected a list, got {value!r}")
        for i, item in enumerate(value):
            _check(item, args[0], f"{where}[{i}]")
    elif origin is typing.Literal:
        if value not in args:
            raise ValueError(f"{where}: expected one of {args}, got {value!r}")
    elif origin in (typing.Union, types.UnionType):
        errors: list[str] = []
        for option in args:
            try:
                _check(value, option, where)
                return
            except ValueError as error:
                errors.append(str(error))
        raise ValueError("; ".join(errors))
    elif kind is type(None):
        if value is not None:
            raise ValueError(f"{where}: expected null, got {value!r}")
    elif kind is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{where}: expected a number, got {value!r}")
    elif kind is int or kind is bool or kind is str:
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise ValueError(f"{where}: expected {kind.__name__}, got {value!r}")


def _load(path: pathlib.Path) -> object:
    return yaml.safe_load(path.read_text())


def load_cost(path: pathlib.Path) -> CostConfig:
    data = _load(path)
    _check(data, CostConfig, str(path))
    return typing.cast(CostConfig, data)


def load_rules(path: pathlib.Path) -> RulesConfig:
    data = _load(path)
    _check(data, RulesConfig, str(path))
    rules = typing.cast(RulesConfig, data)
    if rules["default_profile"] not in rules["profiles"]:
        raise ValueError(f"{path}: default_profile {rules['default_profile']!r} is not a profile")
    return rules
