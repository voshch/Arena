"""Step 3 - Type B (tune some pedestrians) and Type C (one pedestrian against the robot)."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs

from arena_simulation_setup.generative.core.agents import TypeFiles, agent_type_of, mesh_for, profile_of_agent
from arena_simulation_setup.generative.core.profiles import BT_ONLY, MODELS, TUNABLE, ProfileError, is_behavior_tree, profile, scaled_speed, tier_value

from ..populate import for_people, speed_for
from .ask import ask, system_prompt
from .lexicon import when_in
from .llm import Oracle
from .spec import A_CASES, BDecision, CDecision, ConflictError, Record, Selection, conflict_m, conflicts, parse_b, parse_c
from .step_a import StepError, bake, set_type, tune_trees, walk_as_tree, walks
from .step_d import population_summary
from .world import WorldContext, WorldError

ERRATIC = "erratic"
ERRATIC_STOPS = 6
# a wanderer starting this close to the route meets the robot, fewer than ERRATIC_NEAR_MIN such are topped up beside it
ERRATIC_NEAR_M = 4.0
ERRATIC_NEAR_MIN = 3
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def user_prompt(prompt: str, doc: Mapping[str, Any], world: WorldContext, record: Record, *, taken_fractions: Sequence[float], who: str = "") -> str:
    parts = [world.context_for_model(), population_summary(doc, world, record.owned)]
    if who:
        parts.append(who)
    parts.append(f"prompt: {prompt}")
    decided = {k: v for k, v in record.decisions().items() if v and k in ("A", "D")}
    if decided:
        parts.append(f"already decided: {decided}")
    if taken_fractions:
        parts.append("route fractions already used by a placement (keep C at least 2 m away): " + ", ".join(f"{f:.2f}" for f in taken_fractions))
    return "\n\n".join(parts)


def decide(prompt: str, doc: Mapping[str, Any], world: WorldContext, record: Record, *, taken_fractions: Sequence[float], cache_dir: Path | None, model: str | None, oracle: Oracle | None, who: str = "") -> tuple[BDecision | None, CDecision | None]:
    """Ask the model (or the oracle) for the Type B and C decisions, once more on a schema error or a conflict with Type D."""

    def parse(raw: dict[str, Any]) -> tuple[BDecision | None, CDecision | None]:
        b = parse_b(raw.get("B"))
        if b is not None and not (isinstance(raw.get("B"), Mapping) and raw["B"].get("when")):
            spoken = when_in(prompt, exclude_cases=tuple(A_CASES), lexicon=None)
            if spoken is not None:
                b = attrs.evolve(b, when=spoken)
                record.warnings.append(f"B onset read from the prompt: {spoken.as_dict()}")
        c = parse_c(raw.get("C"))
        if c is not None:
            gap = conflict_m(world.route_length)
            for f in taken_fractions:
                if conflicts(f, c.route_fraction, world.route_length):
                    if oracle is None:
                        raise ConflictError(f"conflict: C.route_fraction {c.route_fraction:g} is within {gap:g} m of a Type D placement at {f:.2f}")
                    shift = (gap + 0.3) / max(world.route_length, 1e-9)
                    nudged = f + shift if c.route_fraction >= f else f - shift
                    if not 0.1 <= nudged <= 0.9:
                        nudged = f - shift if c.route_fraction >= f else f + shift
                    c = attrs.evolve(c, route_fraction=round(min(0.9, max(0.1, nudged)), 3))
                    record.warnings.append(f"C moved to route fraction {c.route_fraction:g} to clear the Type D placement at {f:.2f}")
        return b, c

    return ask("BC", system_prompt("bc"), user_prompt(prompt, doc, world, record, taken_fractions=taken_fractions, who=who), record, parse, cache_dir=cache_dir, model=model, oracle=oracle)


def _agents(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [a for a in (doc.get("humans") or []) if isinstance(a, dict)]


def _zone_of_agent(agent: Mapping[str, Any], world: WorldContext) -> str:
    pose = agent.get("pose")
    if isinstance(pose, str):
        return pose
    if isinstance(pose, (list, tuple)) and len(pose) >= 2:
        return world.zone_of((float(pose[0]), float(pose[1]))) or ""
    return ""


def select(doc: Mapping[str, Any], sel: Selection, world: WorldContext, owned: set[str], files: TypeFiles, *, walking_only: bool = False) -> list[dict[str, Any]]:
    """The unowned agents a Type B selection names, in scenario order, with `walking_only` only those that walk."""
    pool = [a for a in _agents(doc) if str(a.get("name")) not in owned and (not walking_only or walks(a, files))]
    if sel.names:
        by_name = {str(a.get("name")): a for a in pool}
        missing = [n for n in sel.names if n not in by_name]
        if missing:
            raise StepError("applicability", f"{missing} are not unowned {'walking ' if walking_only else ''}agents of this scenario")
        return [by_name[n] for n in sel.names]
    if sel.profile:
        pool = [a for a in pool if (p := profile_of_agent(a)) is not None and p.name == sel.profile]
    if sel.zone:
        pool = [a for a in pool if _zone_of_agent(a, world) == sel.zone]
    if sel.amount == "half":
        pool = pool[: max(1, math.ceil(len(pool) / 2))]
    elif sel.amount == "all_but_one":
        pool = pool[:-1]
    if sel.n:
        pool = pool[: sel.n]
    if not pool:
        raise StepError("applicability", f"selection {sel.as_dict()} matched nobody among the unowned {'walking ' if walking_only else ''}agents")
    return pool


def _home(agent: Mapping[str, Any], world: WorldContext, *, seed: int) -> tuple[float, float]:
    pose = agent.get("pose")
    if pose is None:
        raise WorldError(f"agent {agent['name']!r} has no pose to wander from")
    return world.zone_sample(pose, random.Random(f"{world.name}:{seed}:{agent['name']}:{ERRATIC}")) if isinstance(pose, str) else (float(pose[0]), float(pose[1]))


def _beside_route(world: WorldContext, fraction: float, side: float) -> tuple[float, float]:
    (cx, cy), (hx, hy) = world.point_at(fraction)
    return cx - hy * side, cy + hx * side


def wanderers_beside_route(doc: dict[str, Any], world: WorldContext, n: int, *, seed: int) -> list[str]:
    """Up to `n` pedestrians standing beside the middle of the route."""
    rng = random.Random(f"{world.name}:{seed}:wander")
    taken = {str(a.get("name")) for a in doc.get("humans") or [] if isinstance(a, Mapping)}
    added: list[dict[str, Any]] = []
    for i in range(4 * n):
        if len(added) == n:
            break
        fraction = 0.2 + 0.6 * ((i * 0.618) % 1.0)
        found = next((r[0] for side in (2.0, -2.0, 3.0, -3.0) if (r := _resolved(world, _beside_route(world, fraction, side))) is not None), None)
        if found is None:
            continue
        name = next(f"wander_{j}" for j in itertools.count(1) if f"wander_{j}" not in taken)
        taken.add(name)
        spot = [round(found[0], 2), round(found[1], 2), 0.0]
        added.append({"name": name, "model": MODELS[len(added) % len(MODELS)], "pose": spot, "agent": {"agent_type": "adult", "desired_velocity": speed_for("adult", rng)}, "waypoints": [spot]})
    doc["humans"] = [*doc.get("humans", []), *added]
    return [str(a["name"]) for a in added]


def _resolved(world: WorldContext, pt: tuple[float, float]) -> tuple[tuple[float, float], str] | None:
    try:
        return world.resolve(pt, within=1.5)
    except WorldError:
        return None


def erratic_trees(doc: dict[str, Any], files: TypeFiles, world: WorldContext, names: Sequence[str], *, period: float, radius: float, around: str, label: str, when: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Each agent switched on a stimulus to its own loop of `erratic` points, each given up after `period`."""
    walk_as_tree(doc, files, world, names, seed=seed)
    agents = [a for a in doc.get("humans") or [] if isinstance(a, dict) and str(a["name"]) in names]
    for i, agent in enumerate(agents):
        home = _home(agent, world, seed=seed)
        steps: dict[str, Any] = {}
        for k in range(ERRATIC_STOPS):
            angle = (k + i) * _GOLDEN_ANGLE
            if around == "route":
                target = _beside_route(world, 0.15 + 0.7 * ((k * 0.618 + i * 0.29) % 1.0), radius * math.sin(angle))
            else:
                target = (home[0] + radius * math.cos(angle), home[1] + radius * math.sin(angle))
            try:
                (x, y), _why = world.resolve(target, radius=0.3, on_route=True, within=1.0)
            except WorldError:
                continue
            steps[f"to_{len(steps) + 1}"] = {"kind": "go_to", "target_pose": {"x": round(x, 2), "y": round(y, 2), "theta": 0.0}, "patience": {"mean": period}}
        if len(steps) < 2:
            raise StepError("placement", f"{label}: no room around {agent['name']}'s start for an erratic route")
        ref = agent_type_of(agent)
        shared = ref not in files.transient
        set_type(agent, files.switching(ref, need=ERRATIC, to=ERRATIC, steps=steps, tag=str(agent["name"]) if shared else "", advance=True))
    return [{"type": "stimulus", "label": label, "scope": list(names), "need": ERRATIC, "when": dict(when)}]


def apply_b(doc: dict[str, Any], decision: BDecision, world: WorldContext, files: TypeFiles, *, owned: set[str], seed: int) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    tops_up = decision.case == "erratic_route" and decision.around == "route"
    if tops_up and all(str(a.get("name")) in owned for a in _agents(doc)):
        chosen = []
    else:
        chosen = select(doc, decision.select, world, owned, files, walking_only=decision.case == "intrinsic")
    names = [str(a["name"]) for a in chosen]
    notes: list[str] = []
    effects: list[dict[str, Any]] = []
    label = f"B:{decision.case}"

    if decision.case == "erratic_route":
        if tops_up and (near := sum(world.route_clearance(_home(a, world, seed=seed)) <= ERRATIC_NEAR_M for a in chosen)) < ERRATIC_NEAR_MIN:
            added = wanderers_beside_route(doc, world, ERRATIC_NEAR_MIN - near, seed=seed)
            names += added
            notes.append(f"erratic_route: {near} selected agent(s) start near the route, added {len(added)} beside it")
        if not names:
            raise StepError("applicability", f"selection {decision.select.as_dict()} matched nobody and no wanderer fits beside the route")
        effects.extend(erratic_trees(doc, files, world, names, period=decision.period, radius=decision.radius, around=decision.around, label=label, when=decision.when.as_effect(), seed=seed))
        notes.append(f"erratic_route: {len(names)} agent(s) switch to a new point around the {decision.around} every {decision.period:g} s")
        return effects, names, notes

    if decision.case == "erratic_behavior":
        bt, plain = [], []
        for agent in chosen:
            (bt if is_behavior_tree(files.base_doc(agent_type_of(agent))) else plain).append(agent)
        if bt:
            bake(doc, {"need_decay": "high"}, files, names=[str(a["name"]) for a in bt])
        if plain:
            effects.extend(erratic_trees(doc, files, world, [str(a["name"]) for a in plain], period=decision.period, radius=decision.radius, around=decision.around, label=label, when=decision.when.as_effect(), seed=seed))
            notes.append(f"erratic_behavior: {len(plain)} plain walker(s) fell back to erratic_route")
        if bt:
            notes.append(f"erratic_behavior: {len(bt)} behavior-tree agent(s) with need_decay high from the start")
        return effects, names, notes

    params = dict(decision.params)
    by_usable: dict[tuple[tuple[str, str], ...], list[str]] = {}
    for agent in chosen:
        ref = agent_type_of(agent)
        try:
            base = files.base_doc(ref)
        except ProfileError as exc:
            raise StepError("applicability", f"intrinsic: {exc}") from None
        usable = {k: v for k, v in params.items() if k not in BT_ONLY or is_behavior_tree(base)}
        dropped = sorted(set(params) - set(usable))
        if dropped:
            notes.append(f"intrinsic: {dropped} skipped on {agent['name']} (not a behavior-tree type)")
        if usable:
            by_usable.setdefault(tuple(sorted(usable.items())), []).append(str(agent["name"]))
    for key, members in by_usable.items():
        usable = dict(key)
        fixed = {k: v for k, v in usable.items() if k not in TUNABLE}
        if fixed:
            try:
                bake(doc, fixed, files, names=members)
            except StepError as exc:
                raise StepError(exc.kind, f"intrinsic: {exc}") from None
            notes.append(f"intrinsic: {fixed} on {members} from the start")
        tunable = {k: v for k, v in usable.items() if k in TUNABLE}
        if tunable:
            walk_as_tree(doc, files, world, members, seed=seed)
            if tunable.get("speed") == "high":
                tunable["max_speed"] = "high"
            effects.extend(tune_trees(doc, tunable, files, need=decision.case, label=label, when=decision.when.as_effect(), names=members))
    notes.append(f"intrinsic: {params} on {len(names)} agent(s) at {decision.when.as_dict()}")
    return effects, names, notes


def apply_c(decision: CDecision, world: WorldContext, files: TypeFiles, *, seed: int, taken_fractions: Sequence[float]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """The intercept effect of a Type C decision, its pedestrian named when realized."""
    rng = random.Random(f"{world.name}:C:{seed}")
    clash = next((f for f in taken_fractions if conflicts(f, decision.route_fraction, world.route_length)), None)
    if clash is not None:
        raise StepError("conflict", f"route fraction {decision.route_fraction:g} is within {conflict_m(world.route_length):g} m of a Type D placement at {clash:.2f}")
    prof = profile(decision.profile)
    ref = files.derived(prof.base, {}, prof=prof)
    label = f"C:{decision.case}"
    effect: dict[str, Any] = {"type": "intercept", "label": label, "profile": ref, "model": mesh_for(prof, rng), "route_fraction": decision.route_fraction}
    if decision.case == "blocking":
        if decision.mode == "stand":
            effect.update({"after": "stand", "duration": decision.duration, "speed": scaled_speed(prof, "mid")})
            note = f"blocking/stand: a {prof.name} stands on the route at {decision.route_fraction:g} for {decision.duration:g} s"
        else:
            effect.update({"angle": 0.0, "speed": tier_value("speed", "low", prof=prof), "after": "continue", "lead_out": 6.0, "waypoint_mode": "once"})
            note = f"blocking/lead: a {prof.name} crawls ahead of the robot from {decision.route_fraction:g}"
    else:
        effect.update({"angle": decision.angle, "speed": scaled_speed(prof, decision.speed), "after": decision.after, "duration": decision.duration, "offset": decision.offset, "waypoint_mode": "once"})
        if decision.after == "veer":
            if decision.veer_to not in world.polygons:
                raise StepError("placement", f"approach/veer: {decision.veer_to!r} is not a zone of this world")
            if not for_people([decision.veer_to]):
                raise StepError("placement", f"approach/veer: {decision.veer_to!r} is a room nobody walks into (wc/storage)")
            effect["veer_to"] = decision.veer_to
        note = f"approach: a {prof.name} at {decision.angle:g} deg, fraction {decision.route_fraction:g}, then {decision.after}"
    return [effect], [], [note]
