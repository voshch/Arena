"""Typed task phases and requests, authored in scenario YAML or built at runtime."""

from __future__ import annotations

import copy
import typing
from abc import ABC
from collections.abc import Iterable, Mapping, Sequence

import attrs

from arena_simulation_setup.shared.conditions import ConditionSource, EpisodeCondition, Node, OnFailure, parse_node
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import Pose


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (typing.SupportsFloat, str)):
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


def _until(value: object) -> ConditionSource | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    return str(value)


def via_list(value: object) -> list[str]:
    """`via:` elevator entries normalized to `"<name>"` or `"<name>><destination>"`."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out: list[str] = []
    for entry in typing.cast("Sequence[object]", value):
        if isinstance(entry, Mapping):
            unknown = set(entry) - {"elevator", "to"}
            if "elevator" not in entry or unknown:
                raise ValueError(f"via entry must be an elevator name or {{elevator, to}}; got {dict(entry)}")
            to = str(entry.get("to", "") or "")
            out.append(f"{entry['elevator']}>{to}" if to else str(entry["elevator"]))
        else:
            out.append(str(entry))
    return out


def _pose_dict(pose: Pose) -> list[float]:
    x, y, yaw = pose.to_2d()
    return [float(x), float(y), float(yaw)]


@attrs.define
class TaskPhase(ABC):
    """One typed step within a :class:`TaskRequest`."""

    kind: typing.ClassVar[str]

    on_failure: OnFailure = attrs.field(default="continue", kw_only=True)
    until: ConditionSource | None = attrs.field(default=None, kw_only=True)
    conditions: list[EpisodeCondition] = attrs.field(factory=list, kw_only=True)
    text: str = attrs.field(default="", kw_only=True)

    def __attrs_post_init__(self) -> None:
        if self.until is not None:
            parse_node(self.until)

    @property
    def until_node(self) -> Node | None:
        """The parsed `until`, None without one."""
        return None if self.until is None else parse_node(self.until)

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> TaskPhase:
        """Pick the phase type by key presence: `loop`, `goto`, `gesture`, `reach`, or a hold phase with none."""
        data = dict(value)
        if "loop" in data:
            loop_phases = data.pop("loop")
            if not isinstance(loop_phases, Iterable):
                raise TypeError(f"loop must be a list of phases, got {loop_phases!r}")
            loop = LoopPhase(phases=[TaskPhase.parse(p) for p in loop_phases], until=_until(data.pop("until", None)), text=str(data.pop("text", "")))
            if data:
                raise ValueError(f"unknown phase keys {sorted(data)} for LoopPhase")
            return loop
        until = _until(data.pop("until", None))
        instruction = str(data.pop("instruction", ""))
        conditions = data.pop("conditions", [])
        if not isinstance(conditions, Iterable):
            raise TypeError(f"conditions must be a list, got {conditions!r}")
        common = {
            "on_failure": str(data.pop("on_failure", "continue")),
            "until": until,
            "conditions": [EpisodeCondition.parse(c) for c in conditions],
            "text": str(data.pop("text", "")) or instruction,
        }
        if "gesture" in data:
            phase: TaskPhase = PlayGesturePhase(gesture=str(data.pop("gesture")), instance=str(data.pop("instance", "")), **common)
        elif "reach" in data:
            phase = ReachPhase._parse_body(data, common)
        else:
            phase = GoToPhase._parse_body(data, common)
        if data:
            raise ValueError(f"unknown phase keys {sorted(data)} for {type(phase).__name__}")
        return phase

    def serialize(self) -> dict:
        result: dict = {}
        if self.on_failure != "continue":
            result["on_failure"] = self.on_failure
        if self.until is not None:
            result["until"] = copy.deepcopy(self.until)
        if self.conditions:
            result["conditions"] = [c.serialize() for c in self.conditions]
        if self.text:
            result["text"] = self.text
        return result


@attrs.define
class GoToPhase(TaskPhase):
    """Navigate to a pose or a named target (judged on the target, `pose` is then the dispatch pose), hold in place with neither. With `dispatch` false the phase is judged but never sent to a planner."""

    kind: typing.ClassVar[str] = "goto_pose"

    pose: Pose | None = None
    target: str | None = None
    tolerance_radius: float | None = None
    tolerance_angle: float | None = None
    hold_time: float | None = None
    signal: str | None = None
    pose_zone: str = ""
    via: tuple[str, ...] = attrs.field(default=(), converter=tuple, kw_only=True)
    dispatch: bool = attrs.field(default=True, kw_only=True)

    @property
    def hold(self) -> bool:
        return self.pose is None and self.target is None

    @classmethod
    def _parse_body(cls, data: dict, common: dict) -> GoToPhase:
        goto = data.pop("goto", None)
        pose: Pose | None = None
        target: str | None = None
        pose_zone = ""
        if isinstance(goto, str):
            target = goto
            dispatch = data.pop("pose", None)
            if dispatch is not None:
                pose = Pose.parse(dispatch)
                pose_zone = dispatch if isinstance(dispatch, str) else ""
        elif goto is not None:
            pose = Pose.parse(goto)
        return cls(
            pose=pose,
            target=target,
            tolerance_radius=_optional_float(data.pop("tolerance_radius", None)),
            tolerance_angle=_optional_float(data.pop("tolerance_angle", None)),
            hold_time=_optional_float(data.pop("hold_time", None)),
            signal=None if data.get("signal") is None else str(data.pop("signal")),
            pose_zone=pose_zone,
            via=via_list(data.pop("via", None)),
            dispatch=bool(data.pop("dispatch", True)),
            **common,
        )

    def serialize(self) -> dict:
        result = super().serialize()
        if self.target is not None:
            result["goto"] = self.target
            if self.pose is not None:
                result["pose"] = _pose_dict(self.pose)
        elif self.pose is not None:
            result["goto"] = _pose_dict(self.pose)
        for key in ("tolerance_radius", "tolerance_angle", "hold_time"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.signal:
            result["signal"] = self.signal
        if self.via:
            result["via"] = list(self.via)
        if not self.dispatch:
            result["dispatch"] = False
        return result


@attrs.define
class ReachPhase(TaskPhase):
    """Move an arm to a pose in `frame` (a robot frame, an episode object or `<object>/<link>`), to a named configuration, or to a random workspace sample, ended early and canceled when `stop` holds."""

    kind: typing.ClassVar[str] = "reach_pose"
    MOTIONS: typing.ClassVar[tuple[str, ...]] = ("free", "linear")

    target: Pose | None = None
    frame: str = ""
    named_target: str | None = None
    random: bool = False
    position_tolerance: float | None = None
    orientation_tolerance: float | None = None
    planning_time: float | None = None
    instance: str = ""  # arm cap instance (mount name), empty = sole arm
    motion: str = "free"
    speed: float | None = None
    relative: bool = False
    stop: ConditionSource | None = None

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if sum([self.target is not None, self.named_target is not None, self.random]) != 1:
            raise ValueError("ReachPhase requires exactly one of target / named_target / random")
        if self.motion not in self.MOTIONS:
            raise ValueError(f"reach motion must be one of {list(self.MOTIONS)}, got {self.motion!r}")
        if self.target is None and (self.relative or self.motion != "free"):
            raise ValueError("reach relative and linear motion need a Cartesian target")
        if self.speed is not None and not 0.0 < self.speed <= 1.0:
            raise ValueError(f"reach speed must be in (0, 1], got {self.speed}")
        if self.stop is not None:
            parse_node(self.stop)

    @property
    def stop_node(self) -> Node | None:
        """The parsed `stop`, None without one."""
        return None if self.stop is None else parse_node(self.stop)

    @classmethod
    def _parse_body(cls, data: dict, common: dict) -> ReachPhase:
        reach = data.pop("reach")
        target: Pose | None = None
        named_target: str | None = None
        random = False
        if reach == "random":
            random = True
        elif isinstance(reach, str):
            named_target = reach
        else:
            target = Pose.parse(reach)
        return cls(
            target=target,
            frame=str(data.pop("frame", "")),
            named_target=named_target,
            random=random,
            position_tolerance=_optional_float(data.pop("position_tolerance", None)),
            orientation_tolerance=_optional_float(data.pop("orientation_tolerance", None)),
            planning_time=_optional_float(data.pop("planning_time", None)),
            instance=str(data.pop("instance", "")),
            motion=str(data.pop("motion", "free")),
            speed=_optional_float(data.pop("speed", None)),
            relative=bool(data.pop("relative", False)),
            stop=_until(data.pop("stop", None)),
            **common,
        )

    def serialize(self) -> dict:
        result = super().serialize()
        if self.random:
            result["reach"] = "random"
        elif self.named_target is not None:
            result["reach"] = self.named_target
        elif self.target is not None:
            orientation = self.target.orientation
            result["reach"] = [*map(float, self.target.position), orientation.w, orientation.x, orientation.y, orientation.z]
        if self.frame:
            result["frame"] = self.frame
        for key in ("position_tolerance", "orientation_tolerance", "planning_time", "speed"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.instance:
            result["instance"] = self.instance
        if self.motion != "free":
            result["motion"] = self.motion
        if self.relative:
            result["relative"] = True
        if self.stop is not None:
            result["stop"] = copy.deepcopy(self.stop)
        return result


@attrs.define
class PlayGesturePhase(TaskPhase):
    kind: typing.ClassVar[str] = "play_gesture"

    gesture: str | None = None  # None means random; adapter expands before dispatch
    instance: str = ""  # arm cap instance (mount name), empty = sole arm

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.gesture in ("", "random"):
            self.gesture = None

    def serialize(self) -> dict:
        result = super().serialize()
        result["gesture"] = self.gesture if self.gesture is not None else "random"
        if self.instance:
            result["instance"] = self.instance
        return result


@attrs.define
class LoopPhase(TaskPhase):
    """Phases repeated in order until `until` holds, which drops the running one and continues after the loop."""

    kind: typing.ClassVar[str] = "loop"

    phases: list[TaskPhase] = attrs.field(factory=list)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if not self.phases:
            raise ValueError("LoopPhase requires at least one phase")
        if any(isinstance(phase, LoopPhase) for phase in self.phases):
            raise ValueError("LoopPhase cannot nest another loop")

    def serialize(self) -> dict:
        result = super().serialize()
        result["loop"] = [p.serialize() for p in self.phases]
        return result


@attrs.define
class TaskRequest:
    """Typed sequence of phases submitted to a robot, with conditions over the whole request."""

    phases: list[TaskPhase]
    conditions: list[EpisodeCondition] = attrs.field(factory=list)

    @property
    def kind(self) -> str | None:
        """Single homogeneous kind of all phases, or None if mixed/empty."""
        if not self.phases:
            return None
        first = self.phases[0].kind
        for phase in self.phases[1:]:
            if phase.kind != first:
                return None
        return first

    @classmethod
    def parse(cls, value: Mapping[str, object] | Sequence[Mapping[str, object]]) -> TaskRequest:
        if isinstance(value, Mapping):
            data = dict(value)
            raw_phases = data.pop("phases", [])
            raw_conditions = data.pop("conditions", [])
            if not isinstance(raw_phases, Iterable) or not isinstance(raw_conditions, Iterable):
                raise TypeError(f"request phases and conditions must be lists, got {raw_phases!r} and {raw_conditions!r}")
            phases = [TaskPhase.parse(p) for p in raw_phases]
            conditions = [EpisodeCondition.parse(c) for c in raw_conditions]
            if data:
                raise ValueError(f"unknown request keys {sorted(data)}")
            return cls(phases=phases, conditions=conditions)
        return cls(phases=[TaskPhase.parse(p) for p in value])

    def serialize(self) -> dict:
        result: dict = {"phases": [p.serialize() for p in self.phases]}
        if self.conditions:
            result["conditions"] = [c.serialize() for c in self.conditions]
        return result


converter.register_structure_hook(TaskPhase, lambda v, _t: TaskPhase.parse(v))
converter.register_structure_hook(TaskRequest, lambda v, _t: TaskRequest.parse(v))

__all__ = [
    "GoToPhase",
    "LoopPhase",
    "PlayGesturePhase",
    "ReachPhase",
    "TaskPhase",
    "TaskRequest",
    "via_list",
]
