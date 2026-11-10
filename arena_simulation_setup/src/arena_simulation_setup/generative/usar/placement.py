"""Where the robot starts in a rooms world and where the person hides, by turns on the route between them."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import attrs
import shapely

from arena_simulation_setup.generative.egobench.encounter import CLEARANCE, PED_RADIUS, ROBOT_RADIUS, Rejected, summary
from arena_simulation_setup.generative.world.rooms import ENTRANCE
from arena_simulation_setup.shared.pathing import Point
from arena_simulation_setup.shared.route import TURN_RAD, Portal, crossings, portals
from arena_simulation_setup.tree.World import Level, World
from arena_simulation_setup.tree.World.membership import derive

from .manifest import Hiding

CANDIDATES_PER_ROOM = 12
SPOT_TRIES = 200
MIN_DISTANCE_M = 3.0  # person to robot start, so no task is found at spawn


@attrs.frozen
class Hideout:
    start: Point
    heading: float
    person: Point
    zone: str
    route: tuple[Point, ...]  # robot start, each door crossed, person
    corners: int
    misdirection: float  # rad between the call's direction and the route's first leg

    @property
    def doors(self) -> int:
        return len(self.route) - 2


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _heading(a: Point, b: Point) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def corners_of(route: Sequence[Point]) -> int:
    """Turns sharper than TURN_RAD along a polyline."""
    legs = [_heading(a, b) for a, b in zip(route, route[1:], strict=False) if math.dist(a, b) > 1e-6]
    return sum(abs(_wrap(b - a)) > TURN_RAD for a, b in zip(legs, legs[1:], strict=False))


def misdirection_of(route: Sequence[Point]) -> float:
    return abs(_wrap(_heading(route[0], route[1]) - _heading(route[0], route[-1])))


def start_of(level: Level) -> tuple[Point, float]:
    """The entrance room's centre, facing the building's centre."""
    shapes = {zone.name: zone.shape() for zone in level.zones if len(zone.polygon) >= 3}
    if ENTRANCE not in shapes:
        raise ValueError(f"the world has no {ENTRANCE} zone")
    start = shapes[ENTRANCE].representative_point()
    middle = shapely.union_all(list(shapes.values())).centroid
    heading = _heading((start.x, start.y), (middle.x, middle.y)) if start.distance(middle) > 1e-6 else 0.0
    return (start.x, start.y), heading


def candidates(shape: shapely.Polygon, rng: random.Random) -> list[Point]:
    """Points a standing person fits at, drawn inside a room."""
    inner = shape.buffer(-(PED_RADIUS + CLEARANCE))
    if inner.is_empty:
        return []
    minx, miny, maxx, maxy = inner.bounds
    out: list[Point] = []
    for _ in range(SPOT_TRIES):
        if len(out) >= CANDIDATES_PER_ROOM:
            break
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if inner.contains(shapely.Point(x, y)):
            out.append((round(x, 2), round(y, 2)))
    return out


def route_to(found: Sequence[Portal], start: Point, goal: Point, goal_zone: str) -> tuple[Point, ...] | None:
    chain = crossings(found, start, ENTRANCE, goal, goal_zone)
    if chain is None:
        return None
    return (start, *(portal.mid for portal, _ in chain), goal)


def hide(world: World, corners: int, hiding: Hiding, rng: random.Random) -> Hideout | Rejected:
    """The person's spot `corners` turns from the robot's start, at random or the one whose call misleads most."""
    level = next(iter(world.levels.values()))
    start, heading = start_of(level)
    shapes = {zone.name: zone.shape() for zone in level.zones if len(zone.polygon) >= 3}
    if not shapes[ENTRANCE].contains(shapely.Point(start).buffer(ROBOT_RADIUS + CLEARANCE)):
        return Rejected("the robot does not fit at the entrance's centre")
    found = portals(level, derive(world))
    matches: list[Hideout] = []
    reasons: list[str] = []
    for zone, shape in sorted(shapes.items()):
        for spot in candidates(shape, rng):
            if math.dist(spot, start) < MIN_DISTANCE_M:
                continue
            route = route_to(found, start, spot, zone)
            if route is None:
                reasons.append(f"{zone} is not reachable from the {ENTRANCE}")
                continue
            turns = corners_of(route)
            if turns == corners:
                matches.append(Hideout(start=start, heading=heading, person=spot, zone=zone, route=route, corners=turns, misdirection=misdirection_of(route)))
    if not matches:
        return Rejected(summary(reasons) or f"no spot lies {corners} corners from the {ENTRANCE}")
    if hiding is Hiding.ADVERSARIAL:
        return max(matches, key=lambda h: (round(h.misdirection, 3), h.doors))
    return rng.choice(matches)
