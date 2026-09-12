"""What a route-rewrite plan means, without touching ROS.

An effect (`effects.py`) is what the block says; a plan is what the driver executes once the
mode has resolved zone names, scopes and route fractions against the world. Several plans run
in one episode - a rally over the crowd, a hold on the injected walker - each with its own
clock and its own outcome.

Split from :mod:`waypoint_driver` for the same reason :mod:`objects` is split from
:mod:`object_driver`: the geometry stays testable without a simulator.
"""

from __future__ import annotations

import enum
import math
from typing import Any

import attrs

Point = tuple[float, float]

#: Golden angle. Successive shuffles land well apart instead of walking round a circle.
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


class Mode(enum.StrEnum):
    """What the driver does to the routes it owns."""

    RALLY = "rally"
    """Send every owned agent to its target, once."""

    TRACK_ROBOT = "track_robot"
    """Re-issue the robot's current position as the goal."""

    SHUFFLE = "shuffle"
    """Re-issue a fresh point near where the agent started, every `period`."""

    HOLD = "hold"
    """Send each agent to its hold point, then release it to its resume point."""

    RETUNE = "retune"
    """Respawn each owned agent where it stands with new parameters, once."""


class WaypointError(Exception):
    """A plan cannot be executed."""


@attrs.define
class Plan:
    """A resolved route-rewrite plan: every point already in the abstract frame."""

    mode: Mode
    #: Agent name -> where it was placed. What `shuffle` throws its points around, and the
    #: default hold point.
    homes: dict[str, Point] = attrs.Factory(dict)
    #: Sim seconds after the clock starts before anything is rewritten.
    at: float = 0.0
    #: Sim seconds between re-issues. Ignored by `rally` and `hold`.
    period: float = 4.0
    #: Metres. How far a `shuffle` throws its next point.
    radius: float = 3.0
    #: `shuffle`: stop after the first issue.
    once: bool = False
    #: `rally`: agent name -> destination.
    targets: dict[str, Point] = attrs.Factory(dict)
    #: `hold`, `track_robot`: seconds before release. 0 means never.
    duration: float = 0.0
    #: `hold`, `track_robot`: agent name -> where it goes when released.
    resume: dict[str, Point] = attrs.Factory(dict)
    #: `hold`: agent name -> where it is pinned. Defaults to its home.
    points: dict[str, Point] = attrs.Factory(dict)
    #: `retune`: agent name -> the new `agent:` block (agent_type, desired_velocity).
    retune: dict[str, dict[str, Any]] = attrs.Factory(dict)
    label: str = ""

    def __attrs_post_init__(self) -> None:
        if not self.homes:
            raise WaypointError(f"{self.label or self.mode}: a plan owns no agents")
        if self.mode is Mode.RALLY and set(self.targets) != set(self.homes):
            raise WaypointError(f"{self.label or self.mode}: a rally needs a target for every owned agent")
        for name in self.resume:
            if name not in self.homes:
                raise WaypointError(f"{self.label or self.mode}: resume point for {name!r}, which the plan does not own")

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mode": str(self.mode),
            "agents": sorted(self.homes),
            "at": self.at,
            "period": self.period,
            "radius": self.radius,
            "once": self.once,
            "duration": self.duration,
            "targets": {k: list(v) for k, v in sorted(self.targets.items())},
            "retune": {k: dict(v) for k, v in sorted(self.retune.items())},
        }

    def shuffle_routes(self, step: int) -> dict[str, list[Point]]:
        """Deterministic in (agent index, issue count): a rerun shuffles identically."""
        out: dict[str, list[Point]] = {}
        for i, (name, (hx, hy)) in enumerate(sorted(self.homes.items())):
            angle = (step + i) * _GOLDEN_ANGLE
            out[name] = [(hx + self.radius * math.cos(angle), hy + self.radius * math.sin(angle))]
        return out

    def hold_routes(self) -> dict[str, list[Point]]:
        return {name: [self.points.get(name, home)] for name, home in self.homes.items()}

    def resume_routes(self) -> dict[str, list[Point]]:
        """Where released agents go. Agents with no resume point stay put."""
        return {name: [pt] for name, pt in self.resume.items()}


@attrs.define
class WaypointOutcome:
    """What the driver actually did with one plan, for the score row."""

    label: str = ""
    mode: str = ""
    agents: int = 0
    issued: int = 0
    #: The proximity condition the plan waited on, as text. Empty when it ran on a clock.
    trigger: str = ""
    #: Sim seconds at which the trigger latched. `None` with a non-empty `trigger` means the
    #: robot never came near enough - the effect did not run.
    t_trigger: float | None = None
    #: Sim seconds after the clock started at which the first rewrite went out. `None` means
    #: the episode ended before `at` elapsed.
    t_first: float | None = None
    #: Sim seconds at which a hold or a tracking plan released its agents.
    t_release: float | None = None
    failures: int = 0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)
