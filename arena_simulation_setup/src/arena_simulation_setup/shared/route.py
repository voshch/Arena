"""Walking directions between two points of a level, through its doors and wall openings."""

from __future__ import annotations

import heapq
import itertools
import math
import re
import typing

import attrs

from arena_simulation_setup.shared.render import Nouns, noun, render_phase

if typing.TYPE_CHECKING:
    import shapely

    from arena_simulation_setup.shared.task import GoToPhase
    from arena_simulation_setup.tree.World import LevelDescription

Point = tuple[float, float]

WORDINGS = ("goal", "route")

SNAP_M = 0.15
MIN_OPENING_M = 0.6
NEAR_M = 1.0
SHORT_HOP_M = 3.0
LANDMARK_RADIUS_M = 1.5
TURN_RAD = math.radians(30.0)
SIDE_RAD = math.radians(45.0)
AROUND_RAD = math.radians(150.0)

ORDINALS = ("first", "second", "third", "fourth", "fifth")
_INDEX = re.compile(r"(_([a-z]|[a-z]*[0-9][a-z0-9]*))+$")
_SPEAKABLE = re.compile(r"[A-Za-z0-9 '()-]+")


@attrs.frozen
class Portal:
    """A door or wall opening joining two zones, `start` to `end` along their shared boundary."""

    zones: tuple[str, str]
    start: Point
    end: Point
    door: bool

    @property
    def mid(self) -> Point:
        return (self.start[0] + self.end[0]) / 2.0, (self.start[1] + self.end[1]) / 2.0

    def other(self, zone: str) -> str:
        return self.zones[1] if zone == self.zones[0] else self.zones[0]


def level_nouns(level: LevelDescription, entity_prefix: str = "") -> dict[str, str]:
    """Spoken names of a level's zones, doors and static entities: `the` plus the zone description when it is a plain phrase, else the spaced name without `entity_prefix` and trailing indices."""
    nouns: dict[str, str] = {}
    for zone in level.zones:
        nouns[zone.name] = f"the {(zone.description if _SPEAKABLE.fullmatch(zone.description) else zone.name.replace('_', ' ')).lower()}"
        for named in (*zone.doors, *zone.entities.static):
            nouns[named.name] = f"the {_INDEX.sub('', named.name.removeprefix(entity_prefix)).replace('_', ' ')}"
    return nouns


def zone_shapes(level: LevelDescription) -> dict[str, shapely.Polygon]:
    """Floor polygon of every zone, keyed by name."""
    import shapely

    return {zone.name: shapely.Polygon([(c.x, c.y) for c in zone.corners]) for zone in level.zones if len(zone.corners) >= 3}


def portals(level: LevelDescription) -> list[Portal]:
    """Every door between two zones and every unwalled stretch of a shared zone boundary at least `MIN_OPENING_M` long."""
    import shapely

    shapes = zone_shapes(level)
    found: list[Portal] = []
    solid: list[shapely.LineString] = []
    for zone in level.zones:
        for door in zone.doors:
            line = shapely.LineString([(door.start.x, door.start.y), (door.end.x, door.end.y)])
            solid.append(line)
            beyond = [(shape.distance(line.centroid), name) for name, shape in shapes.items() if name != zone.name and shape.distance(line.centroid) < SNAP_M]
            if beyond:
                found.append(Portal((zone.name, min(beyond)[1]), line.coords[0], line.coords[1], True))
        for wall in zone.walls:
            if (wall.material is None or wall.material.name) and (wall.start.x, wall.start.y) != (wall.end.x, wall.end.y):
                solid.append(shapely.LineString([(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)]))
    blocked = shapely.unary_union(solid).buffer(SNAP_M)
    names = list(shapes)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = shapes[a].boundary.intersection(shapes[b].boundary.buffer(SNAP_M)).difference(blocked)
            for piece in shapely.get_parts(shapely.line_merge(shared)):
                if piece.geom_type == "LineString" and piece.length >= MIN_OPENING_M:
                    found.append(Portal((a, b), piece.coords[0], piece.coords[-1], False))
    return found


def crossings(found: typing.Sequence[Portal], start: Point, start_zone: str, goal: Point, goal_zone: str) -> list[tuple[Portal, str]] | None:
    """Portals on the shortest chain of straight hops from `start` to `goal`, each with the zone it leads into, None when the zones do not connect."""
    if start_zone == goal_zone:
        return []
    best: dict[tuple[int, str], float] = {}
    parent: dict[tuple[int, str], tuple[int, str] | None] = {}
    order = itertools.count()
    queue: list[tuple[float, int, int, str, tuple[int, str] | None]] = [(math.dist(start, portal.mid), next(order), i, portal.other(start_zone), None) for i, portal in enumerate(found) if start_zone in portal.zones]
    heapq.heapify(queue)
    arrival: tuple[float, tuple[int, str]] | None = None
    while queue:
        cost, _, i, zone, came_from = heapq.heappop(queue)
        if (i, zone) in best:
            continue
        best[(i, zone)] = cost
        parent[(i, zone)] = came_from
        if zone == goal_zone:
            total = cost + math.dist(found[i].mid, goal)
            if arrival is None or total < arrival[0]:
                arrival = (total, (i, zone))
            continue
        for j, portal in enumerate(found):
            if j != i and zone in portal.zones and (j, portal.other(zone)) not in best:
                heapq.heappush(queue, (cost + math.dist(found[i].mid, portal.mid), next(order), j, portal.other(zone), (i, zone)))
    if arrival is None:
        return None
    chain: list[tuple[Portal, str]] = []
    node: tuple[int, str] | None = arrival[1]
    while node is not None:
        chain.append((found[node[0]], node[1]))
        node = parent[node]
    return chain[::-1]


def describe_route(level: LevelDescription, start: Point, yaw: float, goal: Point, nouns: Nouns | None = None, found: typing.Sequence[Portal] | None = None) -> str | None:
    """Turn-by-turn walking directions from `start` facing `yaw` to `goal`, None when either lies outside the zones or they do not connect, `found` being the level's portals when already computed."""
    import shapely

    shapes = zone_shapes(level)
    names = {**level_nouns(level), **(nouns or {})}
    start_zone = _zone_at(shapes, start)
    goal_zone = _zone_at(shapes, goal)
    if start_zone is None or goal_zone is None:
        return None
    found = portals(level) if found is None else found
    chain = crossings(found, start, start_zone, goal, goal_zone)
    if chain is None:
        return None

    sentences: list[str] = []
    here, heading, zone = start, yaw, start_zone
    for portal, entered in chain:
        approach = _bearing(here, portal.mid)
        inward = _inward(portal, shapes[entered])
        room = noun(entered, names)
        if math.dist(here, portal.mid) < SHORT_HOP_M:
            sentences.append(f"{_cross(portal, approach - heading, room)}.")
        else:
            which = _ordinal([door for door in found if door.door and zone in door.zones], here, portal) if portal.door else ""
            sentences.append(f"{_turn(approach - heading)}walk through {noun(zone, names)}, then {_cross(portal, inward - approach, room, which)}.")
        here, heading, zone = portal.mid, inward, entered

    distance = math.dist(here, goal)
    stop = "stop"
    statics = next(z.entities.static for z in level.zones if z.name == goal_zone)
    reach = shapely.Point(goal).buffer(LANDMARK_RADIUS_M)
    near = [entity for entity in statics if entity.name.islower() and reach.covers(shapely.Point(entity.pose.position.x, entity.pose.position.y))]
    if near:
        landmark = min(near, key=lambda entity: math.dist(goal, (entity.pose.position.x, entity.pose.position.y)))
        stop = f"stop next to {noun(landmark.name, names)}"
    if distance < NEAR_M:
        sentences.append(f"{stop} just inside {noun(zone, names)}." if chain else f"{stop} here.")
    else:
        meters = round(distance)
        into = f" into {noun(zone, names)}" if chain else ""
        sentences.append(f"{_turn(_bearing(here, goal) - heading)}walk about {meters} meter{'' if meters == 1 else 's'}{into} and {stop}.")
    return " ".join(sentence[0].upper() + sentence[1:] for sentence in sentences)


def instruct(phase: GoToPhase, level: LevelDescription, here: tuple[float, float, float] | None, wording: str = "", entity_prefix: str = "", found: typing.Sequence[Portal] | None = None) -> dict[str, str]:
    """`source` and `text` of a goto's instruction from pose `here`: authored text, else walking directions (`route`) or the named target (`goal`) per `wording`, unset meaning route for a bare pose, else the goal coordinates."""
    if phase.text:
        return {"source": "authored", "text": phase.text}
    nouns = level_nouns(level, entity_prefix)
    if here is not None and phase.pose is not None and (wording == "route" or (phase.target is None and wording != "goal")):
        route = describe_route(level, here[:2], here[2], (phase.pose.position.x, phase.pose.position.y), nouns, found)
        if route is not None:
            return {"source": "route", "text": route}
    return {"source": "coordinates" if phase.target is None else "target", "text": render_phase(phase, nouns)}


def _zone_at(shapes: typing.Mapping[str, shapely.Polygon], point: Point) -> str | None:
    import shapely

    spot = shapely.Point(point)
    return next((name for name, shape in shapes.items() if shape.covers(spot)), None)


def _bearing(a: Point, b: Point) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _inward(portal: Portal, shape: shapely.Polygon) -> float:
    """Bearing across `portal` into `shape`."""
    import shapely

    along = _bearing(portal.start, portal.end)
    normal = along + math.pi / 2.0
    probe = (portal.mid[0] + 2.0 * SNAP_M * math.cos(normal), portal.mid[1] + 2.0 * SNAP_M * math.sin(normal))
    return normal if shape.covers(shapely.Point(probe)) else normal + math.pi


def _ordinal(doors: typing.Sequence[Portal], here: Point, target: Portal) -> str:
    """`first `, `second ` and so on for `target` among the doors on its side of a walk from `here` along its wall, empty when no other door lies ahead on that side."""
    wall = _bearing(target.start, target.end)

    def frame(point: Point) -> tuple[float, float]:
        dx, dy = point[0] - here[0], point[1] - here[1]
        return dx * math.cos(wall) + dy * math.sin(wall), dy * math.cos(wall) - dx * math.sin(wall)

    along, lateral = frame(target.mid)
    ahead = [a * along for a, side in (frame(door.mid) for door in doors if door is not target) if a * along > 0.0 and side * lateral > 0.0]
    passed = sum(a < along * along for a in ahead)
    return f"{ORDINALS[passed]} " if ahead and passed < len(ORDINALS) else ""


def _cross(portal: Portal, angle: float, room: str, which: str = "") -> str:
    """Clause for passing `portal` into `room` when it lies `angle` off the walking direction, `which` being its ordinal among the doors on that side."""
    angle = _wrap(angle)
    side = "left" if angle > 0 else "right"
    if abs(angle) > AROUND_RAD:
        return f"turn around and go through the door into {room}" if portal.door else f"turn around and continue into {room}"
    if abs(angle) < SIDE_RAD:
        return f"go through the door ahead into {room}" if portal.door else f"continue straight into {room}"
    return f"take the {which}door on your {side} into {room}" if portal.door else f"turn {side} into {room}"


def _turn(angle: float) -> str:
    angle = _wrap(angle)
    if abs(angle) < TURN_RAD:
        return ""
    if abs(angle) > AROUND_RAD:
        return "turn around and "
    return f"turn {'left' if angle > 0 else 'right'} and "


__all__ = ["ORDINALS", "WORDINGS", "Portal", "crossings", "describe_route", "instruct", "level_nouns", "portals", "zone_shapes"]
