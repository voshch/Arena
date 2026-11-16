"""Step 2, Type D: placed pedestrian interactions, unbroken or breaking formations."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arena_simulation_setup.generative.core.agents import STAND_STEP, TypeFiles, agent_type_of, entry, mesh_for, unique_names
from arena_simulation_setup.generative.core.profiles import profile, scaled_speed, tier_value, tune_block
from arena_simulation_setup.generative.core.triggers import ROUTE_TRIGGER_M
from arena_simulation_setup.shared.pathing import Point

from ..populate import for_people
from .ask import ask, system_prompt
from .llm import Oracle
from .spec import DDecision, Record, parse_d
from .step_a import StepError, set_type, walk_as_tree
from .world import WorldContext, WorldError

FORM = "form"

LINE_SPACING = 0.6
QUEUE_SPACING = 0.7
MIN_ROUTE_FOR_FORMATION_M = 4.0
FORM_AT_S = 4.0
BREAK_AFTER_FORM_S = 3.0
ROUTE_FRACTION_STEP = 0.05
FORMATION_RADIUS_M = 1.8
CIRCLE_SPACING_M = 0.9
HUDDLE_RADIUS_M = 0.35


def circle_radius(n: int) -> float:
    return max(HUDDLE_RADIUS_M, CIRCLE_SPACING_M / (2.0 * math.sin(math.pi / max(2, n))))


def population_summary(doc: Mapping[str, Any], world: WorldContext, owned: Mapping[str, Sequence[str]]) -> str:
    taken = {n for names in owned.values() for n in names}
    lines = []
    for agent in doc.get("humans") or []:
        if not isinstance(agent, Mapping):
            continue
        pose = agent.get("pose")
        where = pose if isinstance(pose, str) else (world.zone_of((float(pose[0]), float(pose[1]))) if isinstance(pose, (list, tuple)) else "?")
        kind = (agent.get("agent") or {}).get("agent_type", "adult") if isinstance(agent.get("agent"), Mapping) else agent.get("agent")
        flag = " (owned)" if str(agent.get("name")) in taken else ""
        lines.append(f"  - {agent.get('name')}: {kind} in {where}{flag}")
    return "population:\n" + ("\n".join(lines) if lines else "  (nobody)")


def user_prompt(prompt: str, doc: Mapping[str, Any], world: WorldContext, record: Record, *, who: str = "") -> str:
    parts = [world.context_for_model(), population_summary(doc, world, record.owned)]
    if who:
        parts.append(who)
    parts.append(f"prompt: {prompt}")
    if record.a:
        parts.append(f"already decided by Step A: {record.a.as_dict()}")
    return "\n\n".join(parts)


def decide(prompt: str, doc: Mapping[str, Any], world: WorldContext, record: Record, *, cache_dir: Path | None, model: str | None, oracle: Oracle | None, who: str = "") -> DDecision | None:
    """Ask the model (or the oracle) for the Type D decision, once more on a schema error."""
    return ask("D", system_prompt("d"), user_prompt(prompt, doc, world, record, who=who), record, lambda raw: None if str(raw.get("case", "")).lower() == "none" else parse_d(raw), cache_dir=cache_dir, model=model, oracle=oracle)


def _place(decision: DDecision, world: WorldContext) -> tuple[Point, Point, bool]:
    if decision.place.get("route_fraction") is not None:
        pt, heading = world.point_at(float(decision.place["route_fraction"]))
        return pt, heading, True
    zone = str(decision.place["zone"])
    if not for_people([zone]):
        raise StepError("placement", f"{zone!r} is a room nobody is placed in (wc/storage, sealed in the inflated grid)")
    pt = world.zone_point(zone, need=circle_radius(len(decision.members)))
    _rp, heading = world.point_at(world.fraction_of(pt))
    return pt, heading, world.route_clearance(pt) < 1.5


def _circle(center: Point, n: int, r: float | None = None) -> list[Point]:
    r = circle_radius(n) if r is None else r
    return [(center[0] + r * math.cos(2 * math.pi * i / n), center[1] + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def _circle_radius(world: WorldContext, center: Point, heading: Point, n: int) -> float:
    want = circle_radius(n)
    if world.grid is None:
        return want
    width, _l, _r = world.passage_width(center, heading)
    return max(HUDDLE_RADIUS_M, min(want, width / 2 - 0.35))


def _break_after_form(when: dict[str, Any], world: WorldContext, center: Point, robot_speed: float, *, floor: float = 0.0) -> list[str]:
    """Narrow a proximity break, or move a route-fraction one along, until the robot cannot reach it before the members form."""
    lead = robot_speed * (FORM_AT_S + BREAK_AFTER_FORM_S)
    if "robot_within" in when:
        latest = math.dist(world.start, center) - lead
        if float(when["robot_within"]) <= latest:
            return []
        if latest < floor:
            raise StepError("placement", f"the robot can come within {floor:.1f} m of the formation at ({center[0]:.1f}, {center[1]:.1f}) before it forms ({FORM_AT_S + BREAK_AFTER_FORM_S:g} s at {robot_speed:g} m/s)")
        when["robot_within"] = round(latest, 2)
        return [f"break trigger narrowed to {when['robot_within']} m so the members form before the robot is near"]
    fraction = float(when["route_fraction"])
    asked = fraction
    while math.dist(world.start, world.point_at(fraction)[0]) - ROUTE_TRIGGER_M < lead:
        fraction += ROUTE_FRACTION_STEP
        if fraction > 0.95:
            raise StepError("placement", f"no point of the route is far enough from the start to break the formation after it forms ({FORM_AT_S + BREAK_AFTER_FORM_S:g} s at {robot_speed:g} m/s)")
    if fraction == asked:
        return []
    when["route_fraction"] = round(fraction, 3)
    return [f"break moved from route fraction {asked:g} to {when['route_fraction']:g} so the members form before the robot is near"]


def _take_reuse(doc: dict[str, Any], names: Sequence[str], owned: set[str]) -> list[dict[str, Any]]:
    wanted = [n for n in names if n not in owned]
    kept, taken = [], []
    for agent in doc.get("humans") or []:
        (taken if isinstance(agent, dict) and str(agent.get("name")) in wanted else kept).append(agent)
    doc["humans"] = kept
    return [copy.deepcopy(a) for a in taken]


def apply(doc: dict[str, Any], decision: DDecision, world: WorldContext, files: TypeFiles, *, seed: int, owned: set[str], robot_speed: float) -> tuple[list[dict[str, Any]], list[str], list[str], dict[str, Any]]:
    """Realize a Type D decision in place, a stubborn walker named when realized, `info` holding the formation."""
    rng = random.Random(f"{world.name}:D:{seed}")
    notes: list[str] = []
    label = f"D:{decision.case}"
    info: dict[str, Any] = {}

    if decision.case == "stubborn":
        prof = profile(decision.profile)
        ref = files.derived(prof.base, {"repulsion": "low"}, prof=prof)
        effect = {
            "type": "intercept",
            "label": label,
            "profile": ref,
            "model": mesh_for(prof, rng),
            "angle": 180.0,
            "route_fraction": decision.route_fraction,
            "speed": scaled_speed(prof, decision.speed),
            "after": "continue",
            "lead_out": 6.0,
            "waypoint_mode": "once",
        }
        notes.append(f"stubborn: a {prof.name} walks head-on at route fraction {decision.route_fraction:g}, not yielding")
        return [effect], [], notes, info

    if decision.place.get("route_fraction") is not None and world.route_length < MIN_ROUTE_FOR_FORMATION_M:
        raise StepError("applicability", f"the robot's first leg is {world.route_length:.1f} m, too short to place a formation on (needs {MIN_ROUTE_FOR_FORMATION_M:g} m)")
    try:
        center, heading, on_route = _place(decision, world)
    except WorldError as exc:
        raise StepError("placement", f"{exc}") from None
    if on_route and decision.variant != "blocking":
        need = circle_radius(len(decision.members)) if decision.variant != "line" else QUEUE_SPACING * len(decision.members) / 2
        fit = world.fit_along_route(world.fraction_of(center), need) or world.fit_along_route(world.fraction_of(center), HUDDLE_RADIUS_M)
        if fit is not None:
            fitted, heading, fraction = fit
            if math.dist(fitted, center) > 0.3:
                notes.append(f"placed {math.dist(fitted, center):.1f} m along the route from the requested point, where the formation fits (fraction {fraction:.2f})")
            center = fitted
        else:
            room = world.zone_of(center) or world.exit_zone()
            if room is None:
                raise StepError("placement", "no room to stand the formation in")
            try:
                center = world.zone_point(room, need=need)
            except WorldError as exc:
                raise StepError("placement", f"{exc}") from None
            on_route = world.route_clearance(center) < 1.5
            notes.append(f"no stretch of the {world.route_length:.1f} m route fits the formation, placed {'on' if on_route else 'beside'} it in {room}")

    reused = _take_reuse(doc, decision.reuse, owned) if decision.reuse else []
    profiles = [profile(m) for m in decision.members]
    variant = decision.variant

    if decision.case == "interaction_unbroken" and variant == "blocking":
        pt, heading, width = world.narrowest_passage(world.fraction_of(center)) if on_route else (center, heading, world.passage_width(center, heading)[0])
        n = min(8, max(2, int(round(width / LINE_SPACING))))
        if width > n * LINE_SPACING + 0.8:
            notes.append(f"blocking: passage is {width:.1f} m wide, {n} members leave a gap")
        profiles = (profiles * 8)[:n]
        normal = (-heading[1], heading[0])
        points = [(pt[0] + normal[0] * (i - (n - 1) / 2) * LINE_SPACING, pt[1] + normal[1] * (i - (n - 1) / 2) * LINE_SPACING) for i in range(n)]
        center = pt
    elif decision.case == "interaction_unbroken" and variant == "meet":
        points = _circle(center, len(profiles), _circle_radius(world, center, heading, len(profiles)))
    elif decision.case == "formation_break" and variant == "line":
        points = [(center[0] - heading[0] * i * QUEUE_SPACING, center[1] - heading[1] * i * QUEUE_SPACING) for i in range(len(profiles))]
    else:
        points = _circle(center, len(profiles), _circle_radius(world, center, heading, len(profiles)))

    slots = [reused[i] if i < len(reused) else None for i in range(len(profiles))]
    resolved: list[Point] = []
    kept_profiles: list[Any] = []
    kept_slots: list[dict[str, Any] | None] = []
    for p, prof_i, slot in zip(points, profiles, slots, strict=True):
        radius = 0.2 if variant == "blocking" else 0.3
        try:
            q, why = world.resolve(p, radius=radius, on_route=on_route or variant == "blocking", within=1.0)
        except WorldError as exc:
            notes.append(f"dropped a member at ({p[0]:.2f}, {p[1]:.2f}): {exc}")
            continue
        if why:
            notes.append(f"member {why}")
        resolved.append(q)
        kept_profiles.append(prof_i)
        kept_slots.append(slot)
    if len(resolved) < 2:
        raise StepError("placement", f"fewer than two members could stand at ({center[0]:.2f}, {center[1]:.2f})")
    profiles = kept_profiles

    doc["humans"] = [*doc.get("humans", []), *(a for a in reused if all(a is not s for s in kept_slots))]
    fresh = iter(unique_names(doc, "d", sum(s is None for s in kept_slots)))
    names = [str(s["name"]) if s is not None else next(fresh) for s in kept_slots]
    here = world.zone_of(center) or ""
    elsewhere = world.other_zones(exclude=[here])
    homes = [z for z in elsewhere if z in world.homes] or elsewhere or world.other_zones()
    if not homes:
        raise StepError("placement", "no zone for the members to walk in from")
    members: list[dict[str, Any]] = []
    meet = decision.case == "interaction_unbroken" and variant == "meet"
    for i, prof in enumerate(profiles):
        home = homes[i % len(homes)]
        if meet:
            member = entry(names[i], prof, home, [home], speed=scaled_speed(prof, "mid"), mode="once", agent_type=prof.base, rng=rng)
        else:
            away = [z for z in world.other_zones(exclude=[home]) if z != here] or [home]
            member = entry(names[i], prof, home, [home, away[(i + 1) % len(away)]], speed=scaled_speed(prof, "mid"), mode="reverse", agent_type=prof.base, rng=rng)
        if (kept := kept_slots[i]) is not None:
            member["model"] = kept.get("model", member["model"])
        members.append(member)
    doc["humans"] = [*doc.get("humans", []), *members]

    spot = {name: pt for name, pt in zip(names, resolved, strict=True)}
    ordered = sorted(names)
    targets = [[round(spot[n][0], 2), round(spot[n][1], 2)] for n in ordered]
    info["formation"] = {"center": [round(center[0], 2), round(center[1], 2)], "radius": FORMATION_RADIUS_M, "variant": variant}
    effects: list[dict[str, Any]] = []
    gated = decision.when.robot_within is not None or decision.when.route_fraction is not None
    if meet or (decision.case == "interaction_unbroken" and gated):
        form_when: dict[str, Any] = decision.when.as_effect()
        if "robot_within" in form_when:
            form_when["of"] = info["formation"]["center"]
    else:
        form_when = {"at": FORM_AT_S if decision.case == "formation_break" else max(FORM_AT_S, decision.when.at or 0.0)}
    if decision.case == "interaction_unbroken":
        ends = decision.duration != "high"
        stand = {**STAND_STEP, "duration": {"mean": tier_value("duration", decision.duration)}, "satisfies": {FORM: -100.0}} if ends else dict(STAND_STEP)
        walk_as_tree(doc, files, world, ordered, seed=seed)
        for member, prof in zip(members, profiles, strict=True):
            x, y = spot[str(member["name"])]
            face = (-heading[0], -heading[1]) if variant == "blocking" else (center[0] - x, center[1] - y)
            steps = {"form": {"kind": "go_to", "target_pose": {"x": round(x, 2), "y": round(y, 2), "theta": round(math.atan2(face[1], face[0]), 3)}, "patience": {"mean": 300.0}}, "stand": dict(stand)}
            set_type(member, files.switching(agent_type_of(member), need=FORM, to="form", steps=steps, tune=tune_block({"repulsion": "low"}, prof=prof), resume=ends))
        effects.append({"type": "stimulus", "label": f"{label}:form", "scope": ordered, "need": FORM, "when": form_when})
    else:
        effects.append({"type": "rally", "label": f"{label}:form", "scope": ordered, "targets": targets, "when": form_when})
    info["formation"]["at"] = form_when
    if decision.case == "formation_break":
        when = decision.when.as_effect()
        if "robot_within" in when:
            when["of"] = info["formation"]["center"]
            if not on_route:
                offset = world.route_clearance(center)
                if offset + 1.0 > float(when["robot_within"]):
                    when["robot_within"] = round(offset + 1.0, 2)
                    notes.append(f"trigger widened to {when['robot_within']} m: the formation stands {offset:.1f} m off the route")
            ring = max(math.dist(center, p) for p in resolved)
            notes.extend(_break_after_form(when, world, center, robot_speed, floor=max(world.route_clearance(center), ring) + 1.0))
        elif "route_fraction" in when:
            notes.extend(_break_after_form(when, world, center, robot_speed))
        elif "at" in when:
            when["at"] = max(float(when["at"]), FORM_AT_S + BREAK_AFTER_FORM_S)
        if decision.break_kind == "scatter":
            effects.append({"type": "scatter", "label": label, "scope": ordered, "radius": decision.radius, "when": when})
        else:
            away_zones = [z for z in world.other_zones(exclude=[world.zone_of(center) or ""])][: len(names)] or [z for z in (world.exit_zone(),) if z]
            if not away_zones:
                raise StepError("placement", "formation_break: no zone to disperse to")
            effects.append({"type": "rally", "label": label, "scope": ordered, "targets": away_zones, "when": when})
        notes.append(f"formation_break: {len(names)} walk, form a {variant} at {form_when}, then {decision.break_kind} {when}")
    else:
        notes.append(f"interaction_unbroken/{variant}: {len(names)} member(s) {'walk in and meet' if variant == 'meet' else 'walk, then form'} at ({center[0]:.1f}, {center[1]:.1f}), {'on' if on_route else 'off'} the route")
    return effects, names, notes, info
