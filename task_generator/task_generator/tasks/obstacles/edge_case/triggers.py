"""Firing a case on where the robot *is* rather than on a clock.

`formation_break` is "a queue that breaks **when the robot approaches**" - a conditional, not
a schedule. `at:` and `every:` cannot express it: a queue that scatters eight seconds in
scatters whether the robot came near it or not, and an arm that happens to route the robot
elsewhere produces a perturbation nobody encountered.

A trigger is a **false-to-true edge**, the same semantics scenario timelines already use for
`when: {entity, field, is}`. What was missing was a robot-distance predicate.

**The reference point is fixed, not tracked.** `robot_within` measures to an explicit point,
to a zone centroid, or - by default - to the centroid of the agents the plan owns *as placed*.
For the case this exists to serve that is exact: a queue is standing still, and where it was
placed is where it is. It is deliberately not live pedestrian tracking, which would make the
firing instant depend on crowd drift and stop the case being reproducible across arms.

Pure, and separate from the driver that consumes it, for the same reason :mod:`waypoints` is
separate from :mod:`waypoint_driver`: the predicate stays testable without a simulator.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import attrs

Point = tuple[float, float]

#: Keys a `trigger:` block may carry.
ALLOWED = frozenset({"robot_within", "robot_beyond", "of"})

#: Fields a `trigger:` block may NOT carry, and why. `then:` appears in the design sketch as a
#: knob to apply on firing; the pedestrian backend allocates a fresh agent id on every spawn
#: and has no in-place type replacement, so an agent's *kinematics* cannot be rewritten once
#: the episode is running. A trigger gates what is time-driven - a waypoint plan - and saying
#: so is better than accepting a `then:` that would never fire.
UNSUPPORTED: dict[str, str] = {
    "then": (
        "applying a knob on firing needs mid-episode agent-type replacement, which the "
        "pedestrian backend cannot do (every spawn allocates a new agent). Express the "
        "reaction as a 'waypoints:' plan, which a trigger can gate"
    ),
}


class TriggerError(Exception):
    """A trigger is malformed or cannot be evaluated."""


@attrs.define
class Trigger:
    """A robot-proximity predicate, latched on its first true.

    Exactly one of `robot_within` / `robot_beyond` is set: "near" and "far" as one condition
    would be a band, which no case here wants and which reads ambiguously.
    """

    #: Metres. Fires when the robot is closer than this to the reference point.
    robot_within: float | None = None
    #: Metres. Fires when the robot is further than this. The mirror case - a crowd that only
    #: acts once the robot has committed past it.
    robot_beyond: float | None = None
    #: Explicit reference, as written: a zone name or an `[x, y]` pair. `None` means "the
    #: centroid of the agents the plan owns", resolved by the caller, which is the half that
    #: knows who those are.
    of: Any = None

    @property
    def distance(self) -> float:
        """The threshold, whichever side it is on."""
        return float(self.robot_within if self.robot_within is not None else self.robot_beyond)

    def satisfied(self, distance: float | None) -> bool:
        """Whether the predicate holds at this robot distance.

        `None` - a respawn window, or a reference that could not be resolved - is *not*
        satisfied. A trigger that fired because it could not see the robot would report a
        case that ran when nothing had happened.
        """
        if distance is None or not math.isfinite(distance):
            return False
        if self.robot_within is not None:
            return distance <= self.robot_within
        return distance >= self.robot_beyond

    def describe(self, reference: Point | None = None) -> str:
        side = "within" if self.robot_within is not None else "beyond"
        where = f" of {reference[0]:.2f},{reference[1]:.2f}" if reference else ""
        return f"robot {side} {self.distance:g}m{where}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "robot_within": self.robot_within,
            "robot_beyond": self.robot_beyond,
            "of": list(self.of) if isinstance(self.of, (list, tuple)) else self.of,
        }


def parse(spec: Mapping[str, Any]) -> Trigger:
    """Validate a `trigger:` block.

    Strict, and for the usual reason: a trigger that silently ignored `robot_wihin: 3.0` would
    fire immediately and report a proximity case that was really a clock.
    """
    if not isinstance(spec, Mapping):
        raise TriggerError(f"trigger: expected a mapping, got {type(spec).__name__}")

    unsupported = sorted(k for k in spec if k in UNSUPPORTED)
    if unsupported:
        detail = "; ".join(f"{k!r}: {UNSUPPORTED[k]}" for k in unsupported)
        raise TriggerError(f"trigger: {detail}")

    unknown = sorted(set(spec) - ALLOWED)
    if unknown:
        raise TriggerError(f"trigger: unknown key(s) {unknown}; allowed: {sorted(ALLOWED)}")

    have = [k for k in ("robot_within", "robot_beyond") if spec.get(k) is not None]
    if not have:
        raise TriggerError(f"trigger: needs one of 'robot_within' or 'robot_beyond'; got {sorted(spec)}")
    if len(have) > 1:
        raise TriggerError(
            "trigger: set 'robot_within' or 'robot_beyond', not both - a band is not a "
            "proximity condition, and which edge fires would be ambiguous"
        )

    key = have[0]
    try:
        value = float(spec[key])
    except (TypeError, ValueError):
        raise TriggerError(f"trigger.{key}: {spec[key]!r} is not a number") from None
    if not math.isfinite(value) or value <= 0.0:
        raise TriggerError(f"trigger.{key}: must be a positive distance in metres, got {value!r}")

    return Trigger(**{key: value, "of": spec.get("of")})


def centroid(points: Sequence[Point]) -> Point | None:
    """Mean of the placed agent positions. `None` for an empty crowd."""
    if not points:
        return None
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def distance_to(robot: Point | None, reference: Point | None) -> float | None:
    """Robot-to-reference distance, or `None` if either is unknown."""
    if robot is None or reference is None:
        return None
    return math.hypot(robot[0] - reference[0], robot[1] - reference[1])
