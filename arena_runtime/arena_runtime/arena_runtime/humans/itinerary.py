"""Pedestrian itineraries on top of a waypoint list: dwell stops and elevator rides."""

from __future__ import annotations

import enum
import math
import typing
from collections.abc import Callable, Sequence

import attrs

ARRIVAL_RADIUS = 0.5

STIMULUS_TELEPORTED = "teleported"


def semantic_atom(entity: str, field: str, value: str) -> str:
    """Humansim stimulus / event name of a semantic field value, the `conditions:` atom without spaces."""
    return f"{entity}.{field}=={value}"


@attrs.frozen
class Stop:
    """One itinerary point. `dwell > 0` holds the ped there, `ride` holds it until an elevator teleports it."""

    x: float
    y: float
    dwell: float = 0.0
    ride: bool = False
    elevator: str = ""  # authored boarding elevator
    target: str = ""  # destination token written to the boarding elevator's `target` on boarding, "" leaves it

    @property
    def holds(self) -> bool:
        return self.ride or self.dwell > 0.0

    def to_dict(self) -> dict[str, typing.Any]:
        return {"x": self.x, "y": self.y, "dwell": self.dwell, "ride": self.ride, "elevator": self.elevator, "target": self.target}

    @classmethod
    def from_dict(cls, raw: typing.Mapping[str, typing.Any]) -> Stop:
        return cls(
            x=float(raw["x"]),
            y=float(raw["y"]),
            dwell=float(raw.get("dwell", 0.0) or 0.0),
            ride=bool(raw.get("ride", False)),
            elevator=str(raw.get("elevator", "") or ""),
            target=str(raw.get("target", "") or ""),
        )


Hops = Callable[[str, str], Sequence[Stop]]


def plan_itinerary(
    points: Sequence[Stop],
    levels: Sequence[str],
    mode: str,
    start_level: str,
    hops: Hops,
) -> tuple[list[Stop], str]:
    """Insert ride stops from `hops(from_level, to_level)` between consecutive points on different levels."""
    pts = list(points)
    lvls = list(levels)
    if len(pts) != len(lvls):
        raise ValueError("points and levels must have the same length")
    mode = mode.lower()
    if mode == "reverse" and len(pts) > 1:
        pts = pts + pts[-2:0:-1]
        lvls = lvls + lvls[-2:0:-1]
        mode = "repeat"

    out: list[Stop] = []
    prev_level = start_level
    prev_ride = False
    for point, level in zip(pts, lvls, strict=True):
        if not prev_ride and prev_level and level and level != prev_level:
            out.extend(hops(prev_level, level))
        out.append(point)
        prev_level = level or prev_level
        prev_ride = point.ride
    if mode == "repeat" and len(pts) > 1 and not prev_ride:
        first = next((level for level in lvls if level), "")
        if prev_level and first and first != prev_level:
            out.extend(hops(prev_level, first))
    return out, mode


class Phase(enum.Enum):
    MOVING = "moving"  # walking a leg toward a holding stop
    DWELL = "dwell"  # standing at a dwell stop
    RIDE = "ride"  # in the cabin, waiting for the teleport
    FREE = "free"  # last leg handed over, nothing left to hold


@attrs.frozen
class Step:
    """Result of one driver update: a new leg for the human simulator and/or the ride stop just boarded."""

    waypoints: list[tuple[float, float]] | None = None
    boarded: Stop | None = None


class ItineraryDriver:
    """Leg-by-leg itinerary for one ped. Positions are in whatever frame the stops are in."""

    def __init__(self, stops: Sequence[Stop], mode: str = "once", arrival_radius: float = ARRIVAL_RADIUS) -> None:
        if not stops:
            raise ValueError("an itinerary needs at least one stop")
        self._stops = list(stops)
        self._cyclic = mode.lower() in ("repeat", "reverse", "random")
        self._radius = arrival_radius
        self._target = -1
        self._dwell_until = math.inf
        self.phase = Phase.FREE

    @property
    def stops(self) -> list[Stop]:
        return list(self._stops)

    @property
    def target(self) -> Stop | None:
        """Holding stop the current leg ends at, None once the ped is free."""
        return self._stops[self._target] if self.phase is not Phase.FREE else None

    def start(self) -> list[tuple[float, float]]:
        """First leg."""
        return self._leg(0)

    def _leg(self, start: int) -> list[tuple[float, float]]:
        n = len(self._stops)
        if not self._cyclic and start >= n:
            self.phase = Phase.FREE
            return []
        indices: list[int] = []
        for k in range(n):
            i = (start + k) % n if self._cyclic else start + k
            if i >= n:
                break
            indices.append(i)
            if self._stops[i].holds:
                self._target = i
                self.phase = Phase.MOVING
                return [(self._stops[j].x, self._stops[j].y) for j in indices]
        self.phase = Phase.FREE
        return [(self._stops[j].x, self._stops[j].y) for j in indices]

    def update(self, pos: tuple[float, float], now: float) -> Step:
        """Advance on arrival and dwell expiry. A ride stop only ends through `on_teleport`."""
        if self.phase is Phase.MOVING:
            stop = self._stops[self._target]
            if math.hypot(pos[0] - stop.x, pos[1] - stop.y) > self._radius:
                return Step()
            if stop.ride:
                self.phase = Phase.RIDE
                return Step(boarded=stop)
            self.phase = Phase.DWELL
            self._dwell_until = now + stop.dwell
            return Step()
        if self.phase is Phase.DWELL and now >= self._dwell_until:
            return Step(waypoints=self._leg(self._target + 1))
        return Step()

    def on_teleport(self) -> Step:
        """The elevator moved the ped: a pending ride is done, hand over the leg after it."""
        if self.phase in (Phase.MOVING, Phase.RIDE) and self._stops[self._target].ride:
            return Step(waypoints=self._leg(self._target + 1))
        return Step()
