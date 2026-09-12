"""Scripted object timelines: things that appear and disappear while the robot is driving.

Levels A-D all perturb *people*. This is the level that perturbs the *world*: a cart wheeled
into a doorway, a cleaning sign put out after the robot has committed to a corridor, an
obstruction that later clears. It is the static-object analogue of Level C — the interest is
not the object, it is **where and when** it meets the robot's route.

Why it needs its own timeline rather than a scenario `static:` block: a scenario places
objects at t=0, and an object present from the start is just furniture. The robot plans
around it and nothing is learned. The failure modes worth measuring all require the world to
change *after* the plan exists:

* a corridor narrows once the robot is inside it — does it replan or freeze?
* a blocked doorway **clears** — does a robot that gave up on a route ever retry it?
* an object appears behind the robot in a dead end — is backtracking even attempted?

Two coordinate conventions, and the choice matters:

``route_fraction``
    Position along the robot's *planned* route, by arc length. Portable: the same timeline
    is a valid case in every one of the twelve generated office worlds, because it never
    names a coordinate. This is the only way `arena_arena_002` gets object cases at all —
    its scenarios must use zone references, since only `open_work_area` and `reception`
    exist in all twelve, and hand-authored coordinates would be right in exactly one world
    and silently wrong in eleven.

``pose``
    Absolute map-frame coordinates, for a hand-authored world like `hospital_1` where the
    geometry is known and the point is a *specific* doorway.

Nothing here touches ROS or the filesystem beyond reading the timeline file, so the
arithmetic is testable without a simulator.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import attrs

from .pathing import point_at_arc, route_length

Point = tuple[float, float]

#: Actions a timeline entry may take.
SPAWN = "spawn"
DESPAWN = "despawn"
_ACTIONS = (SPAWN, DESPAWN)

_EVENT_KEYS = frozenset({"action", "entity", "model", "at", "place", "note"})
_AT_KEYS = frozenset({"t", "route_fraction"})
_PLACE_KEYS = frozenset({"route_fraction", "lateral_offset", "pose", "yaw"})


class TimelineError(Exception):
    """A timeline could not be read, or says something that cannot be executed."""


@attrs.frozen
class ObjectEvent:
    """One entry of a timeline, before it is resolved against a route."""

    action: str
    #: Timeline-local name. A `despawn` refers to the `spawn` that used the same one, so the
    #: author never has to know the `sim_path` the spawn will return.
    entity: str
    model: str = ""
    at_t: float | None = None
    at_fraction: float | None = None
    place_fraction: float | None = None
    lateral_offset: float = 0.0
    pose: Point | None = None
    yaw: float | None = None
    note: str = ""


@attrs.frozen
class ResolvedEvent:
    """An event with a concrete fire time and, for a spawn, a concrete pose.

    Both the requested and the achieved quantities travel with it, for the same reason
    `approach_angle_requested` / `_achieved` do on a case record: a timeline whose events
    silently landed elsewhere reports a factor that was never run.
    """

    event: ObjectEvent
    t: float
    pose: Point | None = None
    yaw: float = 0.0
    #: How far along the route the robot is when this fires, in metres. Combined with
    #: `pose`, this is what says whether the object appeared in front of the robot or on
    #: top of it.
    robot_arc: float = 0.0
    #: Distance between the robot and the spawned object at the moment it appears. Small
    #: values are not rejected - a cart wheeled out at 1 m is a real and nasty case - but
    #: they change what the episode measures, so they are never left implicit.
    reveal_distance_m: float | None = None

    @property
    def action(self) -> str:
        return self.event.action

    @property
    def entity(self) -> str:
        return self.event.entity


def _as_float(value: object, where: str) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise TimelineError(f"{where}: expected a number, got {value!r}") from None


def _parse_at(raw: object, where: str) -> tuple[float | None, float | None]:
    if not isinstance(raw, Mapping):
        raise TimelineError(f"{where}.at: expected a mapping with `t` or `route_fraction`")
    unknown = set(raw) - _AT_KEYS
    if unknown:
        raise TimelineError(f"{where}.at: unknown key(s) {sorted(unknown)}; allowed: {sorted(_AT_KEYS)}")
    if not raw:
        raise TimelineError(f"{where}.at: give either `t` (sim seconds) or `route_fraction`")
    if len(raw) > 1:
        raise TimelineError(f"{where}.at: give `t` or `route_fraction`, not both - they would disagree")

    if "t" in raw:
        t = _as_float(raw["t"], f"{where}.at.t")
        if t < 0:
            raise TimelineError(f"{where}.at.t: must not be negative, got {t}")
        return t, None

    fraction = _as_float(raw["route_fraction"], f"{where}.at.route_fraction")
    if not 0.0 <= fraction <= 1.0:
        raise TimelineError(f"{where}.at.route_fraction: must be within [0, 1], got {fraction}")
    return None, fraction


def _parse_place(raw: object, where: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TimelineError(f"{where}.place: expected a mapping")
    unknown = set(raw) - _PLACE_KEYS
    if unknown:
        raise TimelineError(f"{where}.place: unknown key(s) {sorted(unknown)}; allowed: {sorted(_PLACE_KEYS)}")

    out: dict[str, Any] = {}
    if "pose" in raw:
        pose = raw["pose"]
        if not isinstance(pose, Sequence) or isinstance(pose, (str, bytes)) or len(pose) < 2:
            raise TimelineError(f"{where}.place.pose: expected [x, y] or [x, y, yaw]")
        out["pose"] = (_as_float(pose[0], f"{where}.place.pose[0]"), _as_float(pose[1], f"{where}.place.pose[1]"))
        if len(pose) > 2:
            out["yaw"] = _as_float(pose[2], f"{where}.place.pose[2]")
    if "route_fraction" in raw:
        if "pose" in out:
            raise TimelineError(f"{where}.place: give `pose` or `route_fraction`, not both")
        fraction = _as_float(raw["route_fraction"], f"{where}.place.route_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise TimelineError(f"{where}.place.route_fraction: must be within [0, 1], got {fraction}")
        out["place_fraction"] = fraction
    if "lateral_offset" in raw:
        out["lateral_offset"] = _as_float(raw["lateral_offset"], f"{where}.place.lateral_offset")
    if "yaw" in raw:
        out["yaw"] = _as_float(raw["yaw"], f"{where}.place.yaw")
    return out


def parse_event(raw: object, index: int) -> ObjectEvent:
    where = f"events[{index}]"
    if not isinstance(raw, Mapping):
        raise TimelineError(f"{where}: expected a mapping, got {type(raw).__name__}")

    unknown = set(raw) - _EVENT_KEYS
    if unknown:
        raise TimelineError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(_EVENT_KEYS)}")

    action = str(raw.get("action", SPAWN)).strip().lower()
    if action not in _ACTIONS:
        raise TimelineError(f"{where}.action: expected one of {list(_ACTIONS)}, got {action!r}")

    entity = str(raw.get("entity", "") or "").strip()
    if not entity:
        raise TimelineError(f"{where}.entity: every event needs a name, so a despawn can refer to its spawn")

    model = str(raw.get("model", "") or "").strip()
    if action == SPAWN and not model:
        raise TimelineError(f"{where}.model: a spawn needs a model")
    if action == DESPAWN and model:
        raise TimelineError(f"{where}.model: a despawn names an entity, not a model")

    at_t, at_fraction = _parse_at(raw.get("at"), where)
    place = _parse_place(raw.get("place"), where)
    if action == DESPAWN and place:
        raise TimelineError(f"{where}.place: a despawn has no placement - it removes what the spawn placed")

    return ObjectEvent(
        action=action,
        entity=entity,
        model=model,
        at_t=at_t,
        at_fraction=at_fraction,
        note=str(raw.get("note", "") or ""),
        **place,
    )


def parse_timeline(document: object, *, source: str = "<timeline>") -> list[ObjectEvent]:
    """Validate a whole timeline. Ordering, references and duplicates are checked here."""
    if document is None:
        raise TimelineError(f"{source}: timeline is empty")
    if isinstance(document, Sequence) and not isinstance(document, (str, bytes)):
        document = {"events": document}
    if not isinstance(document, Mapping):
        raise TimelineError(f"{source}: timeline must be a mapping with `events:`, or a bare list of events")

    unknown = set(document) - {"events"}
    if unknown:
        raise TimelineError(f"{source}: unknown key(s) {sorted(unknown)}; a timeline holds only `events`")

    raw_events = document.get("events")
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
        raise TimelineError(f"{source}: `events` must be a list")
    if not raw_events:
        raise TimelineError(f"{source}: timeline has no events - omit the file rather than shipping an empty one")

    events = [parse_event(raw, i) for i, raw in enumerate(raw_events)]

    # Reference checking. A despawn of something never spawned is the mistake this catches,
    # and it is worth catching at parse time: at runtime it looks exactly like an object
    # that vanished on its own.
    spawned: set[str] = set()
    for i, event in enumerate(events):
        if event.action == SPAWN:
            if event.entity in spawned:
                raise TimelineError(f"events[{i}]: {event.entity!r} is spawned twice; use a distinct entity name")
            spawned.add(event.entity)
        elif event.entity not in spawned:
            raise TimelineError(f"events[{i}]: despawns {event.entity!r}, which no earlier event spawns")

    return events


#: In-package timelines, shipped so a case does not need an absolute path to be reproducible.
_TIMELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "timelines")


def resolve_timeline_path(name_or_path: str, search: Sequence[str] = ()) -> str:
    """Resolve a timeline name or path to a file.

    Same rule as `resolve_catalogue_path`, and deliberately so: `task.edge_case.catalogue`
    already takes "a name or a path", and having its sibling parameter take only a path
    would be a difference with no reason behind it.

    Tries, in order: an explicit path; each directory in `search`; the installed share
    directory; the in-package ``timelines/`` directory, which covers running straight from a
    source tree.

    `search` is how a timeline lives *beside the scenario it belongs to* -
    `objects: ./objects.yaml` next to a `scenario.yaml`, the same convention agent types
    already use with `agent_type: ./talker.yaml`. A hand-authored timeline is written against
    one world's coordinates and belongs with that world, not in a shared directory where it
    would look portable.
    """
    if os.path.isfile(name_or_path):
        return os.path.abspath(name_or_path)

    relative = name_or_path.lstrip("./") if name_or_path.startswith("./") else name_or_path
    for directory in search:
        candidate = os.path.join(directory, relative)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    base = os.path.basename(name_or_path)
    stem = base if os.path.splitext(base)[1] else f"{base}.yaml"

    try:
        from ament_index_python.packages import get_package_share_directory

        share = os.path.join(get_package_share_directory("task_generator"), "edge_case", "timelines", stem)
        if os.path.isfile(share):
            return share
    except Exception:  # noqa: BLE001 - ament is absent outside a sourced workspace
        pass

    local = os.path.join(_TIMELINE_DIR, stem)
    if os.path.isfile(local):
        return local

    known = ", ".join(available_timelines()) or "(none installed)"
    raise TimelineError(f"object timeline {name_or_path!r} not found; shipped timelines: {known}")


def available_timelines() -> list[str]:
    """Names of the shipped timelines, sorted."""
    if not os.path.isdir(_TIMELINE_DIR):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(_TIMELINE_DIR) if f.endswith((".yaml", ".yml")))


def load_timeline(name_or_path: str, search: Sequence[str] = ()) -> list[ObjectEvent]:
    """Read and validate a timeline, named or given as a path."""
    import yaml

    path = resolve_timeline_path(name_or_path, search)
    try:
        with open(path) as fh:
            document = yaml.safe_load(fh)
    except OSError as e:
        raise TimelineError(f"cannot read object timeline {path}: {e}") from e
    except yaml.YAMLError as e:
        raise TimelineError(f"object timeline {path} is not valid YAML: {e}") from e
    return parse_timeline(document, source=path)


def resolve(
    events: Sequence[ObjectEvent],
    route: Sequence[Point],
    *,
    robot_speed: float,
    delay: float = 0.0,
) -> list[ResolvedEvent]:
    """Turn route fractions into times and poses, and sort by fire time.

    `delay` shifts every fire time later by that many sim seconds - the block's `delay:` field.
    It is applied after route fractions are converted to times, so a route-relative event keeps
    its place in the route and simply happens later; the alternative, shifting the fraction,
    would move the object somewhere else entirely.

    `robot_speed` is the nominal speed the encounter geometry is designed against, not a
    measurement. A robot that is slower than nominal reaches a `route_fraction` later than
    the timeline fires, so the object appears further ahead than intended — which is why
    `reveal_distance_m` is recorded per event rather than assumed.
    """
    if len(route) < 2:
        raise TimelineError("resolving a route-relative timeline needs a route of at least two points")
    speed = max(float(robot_speed), 1e-6)
    total = route_length(route)
    shift = max(0.0, float(delay))

    resolved: list[ResolvedEvent] = []
    poses: dict[str, Point] = {}

    for event in events:
        if event.at_t is not None:
            t = event.at_t
        else:
            t = (event.at_fraction or 0.0) * total / speed

        # Geometry is computed from the UNSHIFTED time so a route-relative event keeps its
        # place on the route; only the moment it fires moves.
        robot_arc = min(total, max(0.0, t * speed))
        t += shift

        pose: Point | None = None
        yaw = event.yaw if event.yaw is not None else 0.0
        if event.action == SPAWN:
            if event.pose is not None:
                pose = event.pose
            else:
                fraction = event.place_fraction
                if fraction is None:
                    # Default to where the timing put it. Same point, so the object lands on
                    # the robot's own position at that moment - which is a legitimate thing
                    # to ask for (an object dropped in its path) and is exactly why
                    # reveal_distance_m is reported rather than assumed.
                    fraction = event.at_fraction if event.at_fraction is not None else robot_arc / total if total else 0.0
                point, heading = point_at_arc(route, fraction * total)
                pose = _offset(point, heading, event.lateral_offset)
                if event.yaw is None:
                    # Face back down the route, i.e. toward an oncoming robot.
                    yaw = math.atan2(-heading[1], -heading[0])
            poses[event.entity] = pose
        else:
            pose = poses.get(event.entity)

        reveal = None
        if event.action == SPAWN and pose is not None:
            robot_point, _ = point_at_arc(route, robot_arc)
            reveal = math.dist(robot_point, pose)

        resolved.append(
            ResolvedEvent(
                event=event,
                t=t,
                pose=pose if event.action == SPAWN else None,
                yaw=yaw,
                robot_arc=robot_arc,
                reveal_distance_m=reveal,
            )
        )

    # Stable sort: two events at the same time keep their authored order, so a despawn
    # written after its spawn cannot be reordered ahead of it.
    resolved.sort(key=lambda r: r.t)
    return resolved


def _offset(point: Point, heading: Point, lateral: float) -> Point:
    """`point` shifted `lateral` metres to the *left* of the direction of travel."""
    if not lateral:
        return point
    return (point[0] - heading[1] * lateral, point[1] + heading[0] * lateral)
