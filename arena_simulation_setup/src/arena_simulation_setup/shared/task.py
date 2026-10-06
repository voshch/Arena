"""Typed task phases and requests, authored in scenario YAML or built at runtime."""

from __future__ import annotations

import typing
from abc import ABC
from collections.abc import Mapping, Sequence

import attrs

from arena_simulation_setup.shared.conditions import EpisodeCondition, OnFailure, parse_atom
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import Pose


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _pose_dict(pose: Pose) -> list[float]:
    x, y, yaw = pose.to_2d()
    return [float(x), float(y), float(yaw)]


@attrs.define
class TaskPhase(ABC):
    """One typed step within a :class:`TaskRequest`."""

    kind: typing.ClassVar[str]

    on_failure: OnFailure = attrs.field(default="continue", kw_only=True)
    until: str | None = attrs.field(default=None, kw_only=True)
    conditions: list[EpisodeCondition] = attrs.field(factory=list, kw_only=True)
    text: str = attrs.field(default="", kw_only=True)

    def __attrs_post_init__(self) -> None:
        if self.until is not None:
            parse_atom(self.until)

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> TaskPhase:
        """Pick the phase type by key presence: `goto`, `gesture`, `reach`, or a hold phase with neither."""
        data = dict(value)
        until = data.pop("until", None)
        instruction = str(data.pop("instruction", ""))
        common = {
            "on_failure": str(data.pop("on_failure", "continue")),
            "until": None if until is None else str(until),
            "conditions": [EpisodeCondition.parse(c) for c in data.pop("conditions", [])],
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
            result["until"] = self.until
        if self.conditions:
            result["conditions"] = [c.serialize() for c in self.conditions]
        if self.text:
            result["text"] = self.text
        return result


@attrs.define
class GoToPhase(TaskPhase):
    """Navigate to a pose or a named target (judged on the target, `pose` is then the dispatch pose), hold in place with neither."""

    kind: typing.ClassVar[str] = "goto_pose"

    pose: Pose | None = None
    target: str | None = None
    tolerance_radius: float | None = None
    tolerance_angle: float | None = None
    hold_time: float | None = None
    signal: str | None = None

    @property
    def hold(self) -> bool:
        return self.pose is None and self.target is None

    @classmethod
    def _parse_body(cls, data: dict, common: dict) -> GoToPhase:
        goto = data.pop("goto", None)
        pose: Pose | None = None
        target: str | None = None
        if isinstance(goto, str):
            target = goto
            dispatch = data.pop("pose", None)
            if dispatch is not None:
                pose = Pose.parse(dispatch)
        elif goto is not None:
            pose = Pose.parse(goto)
        return cls(
            pose=pose,
            target=target,
            tolerance_radius=_optional_float(data.pop("tolerance_radius", None)),
            tolerance_angle=_optional_float(data.pop("tolerance_angle", None)),
            hold_time=_optional_float(data.pop("hold_time", None)),
            signal=None if data.get("signal") is None else str(data.pop("signal")),
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
        return result


@attrs.define
class ReachPhase(TaskPhase):
    """Move an arm to a pose in `frame`, to a named configuration, or to a random workspace sample."""

    kind: typing.ClassVar[str] = "reach_pose"

    target: Pose | None = None
    frame: str = ""
    named_target: str | None = None
    random: bool = False
    position_tolerance: float | None = None
    orientation_tolerance: float | None = None
    planning_time: float | None = None
    instance: str = ""  # arm cap instance (mount name), empty = sole arm

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if sum([self.target is not None, self.named_target is not None, self.random]) != 1:
            raise ValueError("ReachPhase requires exactly one of target / named_target / random")

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
            **common,
        )

    def serialize(self) -> dict:
        result = super().serialize()
        if self.random:
            result["reach"] = "random"
        elif self.named_target is not None:
            result["reach"] = self.named_target
        elif self.target is not None:
            result["reach"] = _pose_dict(self.target)
        if self.frame:
            result["frame"] = self.frame
        for key in ("position_tolerance", "orientation_tolerance", "planning_time"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.instance:
            result["instance"] = self.instance
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
    def parse(cls, value: Mapping[str, object] | Sequence[object]) -> TaskRequest:
        if isinstance(value, Mapping):
            data = dict(value)
            phases = [TaskPhase.parse(p) for p in data.pop("phases", [])]
            conditions = [EpisodeCondition.parse(c) for c in data.pop("conditions", [])]
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
    "PlayGesturePhase",
    "ReachPhase",
    "TaskPhase",
    "TaskRequest",
]
