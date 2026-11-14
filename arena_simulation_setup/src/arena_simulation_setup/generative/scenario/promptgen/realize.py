"""The steps' design realized offline as intercepting pedestrians, behavior-tree switches and the timeline raising them."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import attrs

from arena_simulation_setup.generative.core.agents import STAND_STEP, WALK_PATIENCE_S, TypeFiles, agent_type_of
from arena_simulation_setup.generative.core.bodies import RUNTIME_SAFE_DIST
from arena_simulation_setup.generative.core.triggers import GO, ROUTE_TRIGGER_M, TRIGGER_STEP_M, release_point, within
from arena_simulation_setup.shared.pathing import Point, point_at_arc, route_length

from .encounter import Encounter, GeometryError, encounter_fractions, solve_encounter
from .step_a import StepError, is_tree, set_type, walk_as_tree
from .world import WorldContext, WorldError

FORM = "form"
BREAK = "break"
INJECT_PREFIX = "edge"
LEAD_IN_M = 4.0
STAND_TRIGGER_M = 3.0
MIN_RELEASE_S = 0.5
START_MARGIN_M = 0.25
FOLLOW_STEP_M = 2.0
ANGLE_SEARCH_DEG = 90.0
LEAD_OUT_M = 3.0


@attrs.define
class Realized:
    timeline: list[dict[str, Any]] = attrs.Factory(list)
    injected: list[str] = attrs.Factory(list)
    encounters: list[dict[str, Any]] = attrs.Factory(list)
    notes: list[str] = attrs.Factory(list)
    #: Step letter -> the injected pedestrians its intercepts were realized as.
    owned: dict[str, list[str]] = attrs.Factory(dict)


def _position(agent: Mapping[str, Any], world: WorldContext) -> Point | None:
    pose = agent.get("pose")
    if isinstance(pose, str):
        return world.centroids.get(pose)
    if isinstance(pose, (list, tuple)) and len(pose) >= 2:
        return (float(pose[0]), float(pose[1]))
    return None


def _centroid(points: Sequence[Point]) -> Point | None:
    if not points:
        return None
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def trigger(when: Mapping[str, Any], world: WorldContext, *, reference: Point | None = None, origin: Point = (0.0, 0.0), hold: float = 0.0) -> dict[str, Any]:
    """An onset's trigger: `at` seconds after the start `hold`, or the robot's distance to a level point moved by `origin`."""
    if when.get("route_fraction") is not None:
        pt, _heading = world.point_at(float(when["route_fraction"]))
        return {"when": within(ROUTE_TRIGGER_M, pt, origin)}
    for key, negate in (("robot_within", False), ("robot_beyond", True)):
        if when.get(key) is None:
            continue
        of = when.get("of")
        pt = (float(of[0]), float(of[1])) if of is not None else reference
        if pt is None:
            raise StepError("placement", f"a proximity onset {dict(when)} has no point to measure the robot against")
        atom = within(float(when[key]), pt, origin)
        return {"when": f"not {atom}" if negate else atom}
    return {"at": float(when.get("at") or 0.0) + hold}


def entered_at(route: Sequence[Point], pt: Point, radius: float) -> float | None:
    """Meters along `route` where the robot first comes within `radius` of `pt`, None when it never does."""
    total = route_length(route)
    steps = max(1, math.ceil(total / TRIGGER_STEP_M))
    return next((total * i / steps for i in range(steps + 1) if math.dist(point_at_arc(route, total * i / steps)[0], pt) <= radius), None)


def holds_at_start(world: WorldContext, pt: Point, radius: float) -> bool:
    """Whether `robot within radius of pt` already holds, give or take `START_MARGIN_M`, with the robot at its start."""
    return math.dist(world.start, pt) <= radius + START_MARGIN_M


def _go_to(pt: Point, *, theta: float = 0.0, duration: float | None = None) -> dict[str, Any]:
    step: dict[str, Any] = {"kind": "go_to", "target_pose": {"x": round(pt[0], 2), "y": round(pt[1], 2), "theta": round(theta, 3)}, "patience": {"mean": WALK_PATIENCE_S}}
    if duration is not None:
        step["duration"] = {"mean": duration}
    return step


def _heading(a: Point, b: Point) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _entry(trig: Mapping[str, Any], agents: Sequence[str], need: str) -> dict[str, Any]:
    return {**trig, "stimulus": [{"agents": list(agents), "need": need}]}


def _human(name: str, effect: Mapping[str, Any], ref: str, spawn: Point, yaw: float) -> dict[str, Any]:
    return {
        "name": name,
        "model": str(effect["model"]),
        "pose": [round(spawn[0], 2), round(spawn[1], 2), round(yaw, 3)],
        "agent": {"agent_type": ref, "desired_velocity": round(float(effect["speed"]), 2)},
        "waypoints": [[round(spawn[0], 2), round(spawn[1], 2), 0.0]],
        "waypoint_mode": "once",
    }


def _stand(effect: Mapping[str, Any], name: str, world: WorldContext, files: TypeFiles) -> tuple[dict[str, Any], list[dict[str, Any]], str, Point]:
    route = world.route
    total = world.route_length
    duration = float(effect.get("duration") or 0.0)
    for fraction in encounter_fractions(float(effect["route_fraction"])):
        spot, heading = point_at_arc(route, fraction * total)
        if world.valid(spot, radius=RUNTIME_SAFE_DIST, on_route=True) is None and not (duration > 0 and holds_at_start(world, spot, STAND_TRIGGER_M)):
            break
    else:
        raise StepError("placement", f"{name} has no free point on the robot's route to stand on {STAND_TRIGGER_M:g} m past its start")
    release = point_at_arc(route, min(total, fraction * total + float(effect.get("lead_out", LEAD_OUT_M))))[0]
    ref = files.standing(str(effect["profile"]), name)
    entries: list[dict[str, Any]] = []
    if duration > 0:
        steps = {"hold": {"duration": {"mean": duration}, "patience": {"mean": duration + STAND_STEP["patience"]["mean"]}}, "release": _go_to(release, theta=_heading(spot, release)), "stay": dict(STAND_STEP)}
        ref = files.switching(ref, need=GO, to="go", steps=steps)
        entries.append(_entry({"when": within(STAND_TRIGGER_M, spot, world.origin)}, [name], GO))
    human = _human(name, effect, ref, spot, math.atan2(-heading[1], -heading[0]))
    note = f"{effect.get('label')}: {name} stands on the route at fraction {fraction:.2f} ({spot[0]:.1f}, {spot[1]:.1f})" + (f", steps on {duration:g} s after the robot comes within {STAND_TRIGGER_M:g} m" if duration > 0 else "")
    return human, entries, note, spot


def _solved(effect: Mapping[str, Any], name: str, world: WorldContext, *, robot_speed: float) -> tuple[Encounter, float, Point, float]:
    """The encounter nearest the asked fraction whose release holds neither at the start nor early, with that release."""
    speed = float(effect["speed"])
    offset = float(effect.get("offset") or 0.0)
    total = world.route_length
    why = f"{name} has no encounter on the route"
    for fraction in encounter_fractions(float(effect["route_fraction"])):
        try:
            enc = solve_encounter(
                world.route,
                robot_speed=robot_speed,
                ped_speed=speed,
                approach_angle=float(effect.get("angle", 180.0)),
                encounter_at=fraction,
                co_arrival_offset=offset,
                lead_out=float(effect.get("lead_out", LEAD_OUT_M)),
                angle_search=ANGLE_SEARCH_DEG,
                is_valid=world.grid.clear_of(RUNTIME_SAFE_DIST) if world.grid is not None else None,
                spawn_valid=lambda pt: world.valid(pt, radius=RUNTIME_SAFE_DIST, on_route=True) is None,
                max_lead_in=LEAD_IN_M,
            )
        except GeometryError as exc:
            raise StepError("placement", f"{exc}") from None
        release = robot_speed * max(math.dist(enc.spawn, enc.encounter) / speed - offset, MIN_RELEASE_S)
        at = max(0.0, enc.encounter_at_achieved * total - release)
        point, radius = release_point(world.route, at)
        entered = entered_at(world.route, point, radius)
        if holds_at_start(world, point, radius):
            why = f"{name}'s release holds with the robot at its start"
        elif entered is not None and entered < at - 0.5:
            why = f"{name}'s release at {at:.1f} m along the route is within {radius:g} m of the route {at - entered:.1f} m earlier"
        else:
            return enc, release, point, radius
    raise StepError("placement", f"{why}")


def _intercept(effect: Mapping[str, Any], name: str, world: WorldContext, files: TypeFiles, *, robot_speed: float, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], str, Point]:
    """The pedestrian `name`, the timeline entry that releases it, a note, and the designed encounter point."""
    if effect.get("after") == "stand":
        return _stand(effect, name, world, files)
    speed = float(effect["speed"])
    enc, release, point, radius = _solved(effect, name, world, robot_speed=robot_speed)
    total = world.route_length
    _at, robot_heading = point_at_arc(world.route, enc.encounter_at_achieved * total)
    facing = math.atan2(-robot_heading[1], -robot_heading[0])
    after = str(effect.get("after") or "continue")
    duration = float(effect.get("duration") or 0.0)
    leave = _go_to(enc.goal, theta=_heading(enc.encounter, enc.goal))
    steps: dict[str, Any]
    if after == "stop_facing":
        steps = {"meet": _go_to(enc.encounter, theta=facing, duration=duration), "leave": leave}
    elif after == "follow":
        start = enc.encounter_at_achieved * total
        reach = min(total, start + duration * speed)
        count = max(1, math.ceil((reach - start) / FOLLOW_STEP_M)) if reach > start else 0
        follow = {f"follow_{k}": _go_to(point_at_arc(world.route, start + (reach - start) * k / count)[0]) for k in range(1, count + 1)}
        steps = {"meet": _go_to(enc.encounter), **follow, "leave": leave}
    elif after == "veer":
        target = str(effect["veer_to"])
        try:
            veer = world.zone_sample(target, random.Random(f"{world.name}:{seed}:{name}:veer:{target}"))
        except WorldError as exc:
            raise StepError("placement", f"{exc}") from None
        steps = {"meet": _go_to(enc.encounter), "veer": _go_to(veer, theta=_heading(enc.encounter, veer))}
    else:
        steps = {"cross": leave}
    steps["stay"] = dict(STAND_STEP)
    ref = files.switching(files.standing(str(effect["profile"]), name), need=GO, to="walk", steps=steps)
    human = _human(name, effect, ref, enc.spawn, _heading(enc.spawn, enc.encounter))
    note = (
        f"{effect.get('label')}: {name} waits at ({enc.spawn[0]:.1f}, {enc.spawn[1]:.1f}) and walks when the robot has {release:.1f} m of its route left to the encounter ({enc.encounter[0]:.1f}, {enc.encounter[1]:.1f}), "
        f"angle {enc.angle_achieved:g} (asked {enc.angle_requested:g}), fraction {enc.encounter_at_achieved:g} (asked {float(effect['route_fraction']):g}), then {after}"
    )
    return human, [_entry({"when": within(radius, point, world.origin)}, [name], GO)], note, enc.encounter


def _switch(doc: dict[str, Any], names: Sequence[str], files: TypeFiles, world: WorldContext, made: set[str], *, need: str, to: str, steps_of: Mapping[str, dict[str, Any]], seed: int) -> None:
    """Every named agent switched to its own `to` sequence on `need`, a plain walker baked into a walking tree first."""
    walk_as_tree(doc, files, world, [n for n in names if not is_tree(files, next(a for a in doc["humans"] if str(a["name"]) == n))], seed=seed)
    for agent in doc["humans"]:
        name = str(agent["name"])
        if name not in names:
            continue
        ref = agent_type_of(agent)
        if ref in made:
            files.transient.add(ref)
        shared = ref not in files.transient
        out = files.switching(ref, need=need, to=to, steps=steps_of[name], tag=name if shared else "")
        made.add(out)
        set_type(agent, out)


def _rally_steps(names: Sequence[str], targets: Sequence[Any], world: WorldContext, *, seed: int) -> dict[str, Point]:
    """Targets dealt to the agents in sorted-name order, a zone drawn once to a point clear of furniture."""
    out: dict[str, Point] = {}
    for i, name in enumerate(sorted(names)):
        target = targets[i % len(targets)]
        if isinstance(target, str):
            out[name] = world.zone_sample(target, random.Random(f"{world.name}:{seed}:{name}:{target}"))
        else:
            out[name] = (float(target[0]), float(target[1]))
    return out


def _scatter_points(names: Sequence[str], spots: Mapping[str, Point], radius: float, world: WorldContext, *, seed: int) -> tuple[dict[str, Point], list[str]]:
    out: dict[str, Point] = {}
    notes: list[str] = []
    for name in sorted(names):
        here = spots[name]
        angle = random.Random(f"{world.name}:scatter:{seed}:{name}").uniform(0.0, 2.0 * math.pi)
        target = (here[0] + radius * math.cos(angle), here[1] + radius * math.sin(angle))
        try:
            out[name], _why = world.resolve(target, radius=0.3, on_route=True, within=1.0)
        except WorldError as exc:
            out[name] = here
            notes.append(f"{name} stays where it stood: {exc}")
    return out, notes


def realize(doc: dict[str, Any], effects: Sequence[Mapping[str, Any]], world: WorldContext, files: TypeFiles, *, seed: int, robot_speed: float, hold: float = 0.0) -> Realized:
    """Add the injected pedestrians and every switch to `doc` and `files`, and return the timeline that drives them."""
    out = Realized()
    made: set[str] = set()
    spots: dict[str, Point] = {}
    doc.setdefault("humans", [])
    for effect in effects:
        try:
            _one(doc, effect, world, files, out, made, spots, seed=seed, robot_speed=robot_speed, hold=hold)
        except (StepError, WorldError) as exc:
            raise StepError(exc.kind if isinstance(exc, StepError) else "placement", str(exc), step=str(effect.get("label", "")).split(":", 1)[0]) from None
    return out


def _one(doc: dict[str, Any], effect: Mapping[str, Any], world: WorldContext, files: TypeFiles, out: Realized, made: set[str], spots: dict[str, Point], *, seed: int, robot_speed: float, hold: float) -> None:
    kind = effect["type"]
    if kind == "intercept":
        name = f"{INJECT_PREFIX}_{len(out.injected)}"
        human, entries, note, point = _intercept(effect, name, world, files, robot_speed=robot_speed, seed=seed)
        doc["humans"].append(human)
        out.injected.append(name)
        out.owned.setdefault(str(effect.get("label", "")).split(":", 1)[0], []).append(name)
        out.encounters.append({"agent": name, "point": [round(point[0], 2), round(point[1], 2)]})
        out.timeline.extend(entries)
        out.notes.append(note)
        return
    scope = [str(n) for n in effect["scope"]]
    placed = [p for a in doc["humans"] if str(a["name"]) in scope and (p := _position(a, world)) is not None]
    crowd = _centroid(placed)
    reference = None if crowd is None else world.point_at(world.fraction_of(crowd))[0]
    trig = trigger(effect["when"], world, reference=reference, origin=world.origin, hold=hold)
    if kind == "stimulus":
        out.timeline.append(_entry(trig, scope, str(effect["need"])))
        return
    forming = str(effect.get("label", "")).endswith(":form")
    if kind == "rally":
        targets = _rally_steps(scope, effect["targets"], world, seed=seed)
        if forming:
            spots.update(targets)
        to = FORM if forming else "disperse"
    else:
        homes = {n: spots.get(n) or next(p for a in doc["humans"] if str(a["name"]) == n and (p := _position(a, world)) is not None) for n in scope}
        targets, notes = _scatter_points(scope, homes, float(effect.get("radius", 2.0)), world, seed=seed)
        out.notes.extend(notes)
        to = "scatter"
    need = FORM if forming else BREAK
    _switch(doc, scope, files, world, made, need=need, to=to, steps_of={n: {"go": _go_to(pt), "stay": dict(STAND_STEP)} for n, pt in targets.items()}, seed=seed)
    out.timeline.append(_entry(trig, scope, need))
