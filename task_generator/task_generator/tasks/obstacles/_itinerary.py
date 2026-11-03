"""Scenario ped itineraries: waypoint `dwell` and `{elevator: <name>}` rides."""

from __future__ import annotations

from collections.abc import Callable

import attrs
from arena_runtime.env.task_api import Stop, plan_itinerary
from arena_runtime.shared import DynamicObstacle, Position
from arena_simulation_setup.shared.entities import Waypoint

LevelOrigin = Callable[[str], tuple[float, float]]
Placement = Callable[[str, str], str]
# (elevator, next waypoint or None) -> boarding stop, None if unknown
ElevatorRide = Callable[[str, Position | None], Stop | None]


def attach_itinerary(
    obs: DynamicObstacle,
    *,
    level_origin: LevelOrigin,
    placement: Placement,
    elevator_ride: ElevatorRide,
    warn: Callable[[str], object] = print,
) -> DynamicObstacle:
    """`placement(zone, level)` names the level a point of `obs` is local to, as the realizer places it."""
    waypoints = list(obs.waypoints)
    if not waypoints or not any(wp.dwell > 0.0 or wp.elevator for wp in waypoints):
        return obs

    def flat(x: float, y: float, zone: str, level: str) -> tuple[float, float]:
        ox, oy = level_origin(placement(zone, level))
        return x + ox, y + oy

    points = [flat(wp.x, wp.y, wp.zone, wp.level) for wp in waypoints]

    stops: list[Stop] = []
    for i, (wp, (x, y)) in enumerate(zip(waypoints, points, strict=True)):
        if not wp.elevator:
            stops.append(Stop(x=x, y=y, dwell=float(wp.dwell)))
            continue
        toward = Position(*points[i + 1], 0.0) if i + 1 < len(points) else None
        ride = elevator_ride(wp.elevator, toward)
        if ride is None:
            warn(f"{obs.name}: elevator {wp.elevator!r} unknown to the world manager, riding from the parsed cabin point")
            ride = Stop(x=x, y=y, ride=True, elevator=wp.elevator)
        stops.append(ride)

    planned, mode = plan_itinerary(stops, [""] * len(stops), obs.waypoint_mode, "", lambda _a, _b: [])
    if mode == "random":
        warn(f"{obs.name}: waypoint_mode random with dwell/elevator stops walks the waypoints in order")
    extra = {
        **obs.extra,
        "itinerary": [{k: v for k, v in stop.to_dict().items() if k not in ("x", "y")} for stop in planned],
    }
    ox, oy = level_origin(placement("", ""))
    return attrs.evolve(obs, waypoints=[Waypoint(x=stop.x - ox, y=stop.y - oy, z=0.0) for stop in planned], waypoint_mode=mode, extra=extra)
