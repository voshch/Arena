"""Robot triggers of a scenario timeline: the `robot within` atom and where along a route a release fires."""

from __future__ import annotations

import math
from collections.abc import Sequence

from arena_simulation_setup.shared.pathing import Point, point_at_arc, route_length

GO = "go"
# a `route_fraction` onset fires when the robot comes this close to that point of its route
ROUTE_TRIGGER_M = 1.5
TRIGGER_STEP_M = 0.05


def _num(value: float) -> str:
    return f"{round(float(value), 2):g}"


def point_literal(pt: Point, origin: Point = (0.0, 0.0)) -> str:
    """`x,y` as a condition atom names a point of the map frame: the level-frame `pt` shifted by the level's `origin`."""
    return f"{_num(pt[0] + origin[0])},{_num(pt[1] + origin[1])}"


def within(radius: float, pt: Point, origin: Point = (0.0, 0.0)) -> str:
    return f"robot within {_num(radius)} of {point_literal(pt, origin)}"


def release_point(route: Sequence[Point], at: float, radius: float = ROUTE_TRIGGER_M) -> tuple[Point, float]:
    """The route point whose `radius` circle the robot enters `at` meters along `route`, and that radius, less near the goal."""
    total = route_length(route)
    here = point_at_arc(route, at)[0]
    s = at
    while s < total and math.dist(point_at_arc(route, s)[0], here) < radius:
        s += TRIGGER_STEP_M
    if s < total:
        return point_at_arc(route, s)[0], radius
    end = point_at_arc(route, total)[0]
    return end, max(math.dist(end, here), TRIGGER_STEP_M)
