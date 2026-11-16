"""Prompt, world and base scenario to `scenarios/<base>__<prompt_id>/`: steps A, D and B+C, then the offline realization."""

from __future__ import annotations

import copy
import json
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import attrs
import yaml

from arena_simulation_setup.generative.core.agents import TYPES_DIR, TypeFiles, agent_type_of
from arena_simulation_setup.generative.core.frames import shift_humans
from arena_simulation_setup.generative.core.generated import GENERATED_MARKER

from ..formats import PROVENANCE, scenario_text
from ..materialize import WouldOverwrite, is_generated
from ..route import ROBOT_SPEED_MPS
from .lexicon import match as lexicon_match
from .llm import Oracle, Unavailable
from .prompts import Prompt
from .realize import realize, trigger
from .spec import Failure, FailureKind, Record, SchemaError
from .step_a import StepError
from .step_a import apply as apply_a
from .step_bc import apply_b, apply_c
from .step_bc import decide as decide_bc
from .step_d import apply as apply_d
from .step_d import decide as decide_d
from .world import ROBOT_CLEAR, WorldContext, WorldError

#: Type A cases that run on an empty base, the others need a crowd.
A_ON_EMPTY = frozenset({"busy"})
GENERATOR_VERSION = "promptgen-2"
_STEP_OF = {"A": "a", "B": "b", "C": "c", "D": "d"}


@attrs.frozen(kw_only=True)
class RobotPlan:
    """The robot a case is designed for: release speed, traversals, start hold, goal tolerance and route clearance."""

    speed: float = ROBOT_SPEED_MPS
    traversals: int = 1
    hold: float = 0.0
    tolerance: float = 0.0
    radius: float = attrs.field(default=ROBOT_CLEAR, validator=attrs.validators.gt(0.0))

    def as_provenance(self) -> dict[str, Any]:
        """The `robot` block of `provenance.yaml`, a tolerance of 0 (the runtime's own) as null."""
        return {"speed_mps": self.speed, "hold_s": self.hold, "traversals": self.traversals, "tolerance_m": self.tolerance if self.tolerance > 0 else None, "radius_m": self.radius}

    def phases(self, start: list[float], goal: list[float]) -> list[dict[str, Any]]:
        """A hold phase when `hold` is set, then `2N - 1` alternating legs to the goal and back to the start."""
        if self.traversals < 1:
            raise ValueError(f"traversals must be >= 1, got {self.traversals}")
        tolerance = {"tolerance_radius": self.tolerance} if self.tolerance > 0 else {}
        legs = [{"goto": list(goal if i % 2 == 0 else start), **tolerance} for i in range(2 * self.traversals - 1)]
        return [{"hold_time": self.hold}, *legs] if self.hold > 0 else legs


def denied_on_empty(targets: tuple[str, ...]) -> list[str]:
    """Targets that cannot apply to an empty base: Type B and every Type A but `busy`."""
    return [t for t in targets if t.startswith("B:") or (t.startswith("A:") and t.split(":", 1)[1] not in A_ON_EMPTY)]


@attrs.define
class Result:
    record: Record
    doc: dict[str, Any]
    files: TypeFiles
    effects: list[dict[str, Any]]
    scenario_id: str
    prompt: Prompt | None = None
    timeline: list[dict[str, Any]] = attrs.Factory(list)
    injected: list[str] = attrs.Factory(list)
    encounters: list[dict[str, Any]] = attrs.Factory(list)
    robot: RobotPlan = attrs.Factory(RobotPlan)
    written: Path | None = None

    @property
    def provenance(self) -> dict[str, Any]:
        """What `provenance.yaml` records beside the scenario, never read by the runtime."""
        r = self.record
        return {
            "id": self.scenario_id,
            "prompt_id": r.prompt_id,
            "steps": r.steps,
            "note": r.prompt,
            "decisions": r.decisions(),
            "owned": {"C": list(r.owned.get("C", [])), "D": list(r.owned.get("D", [])), "injected": list(self.injected)},
            **({"formation": dict(r.formation)} if r.formation else {}),
            "encounters": [dict(e) for e in self.encounters],
            "robot": self.robot.as_provenance(),
            "provenance": {**r.provenance, "generator": GENERATOR_VERSION, "status": r.status, "dedup_key": r.dedup_key()},
        }


def _fail(record: Record, step: str, exc: Exception) -> None:
    kind: FailureKind = exc.kind if isinstance(exc, StepError) else "schema" if isinstance(exc, SchemaError) else "placement" if isinstance(exc, WorldError) else "unavailable" if isinstance(exc, Unavailable) else "infeasible"
    record.failures.append(Failure(step=step, kind=kind, reason=str(exc)))


def _oracle_for(prompt: Prompt, key: str) -> Oracle:
    """The prompt's own slots as the model's answer."""
    if key == "D":
        return Oracle(prompt.oracle.get("D") or {"case": "none"})
    return Oracle({"B": prompt.oracle.get("B") or "none", "C": prompt.oracle.get("C") or "none"})


def pin_robot_route(doc: dict[str, Any], world: WorldContext, record: Record, plan: RobotPlan) -> None:
    """Pin the robot's start and goal to the planned route's endpoints and drive them as `plan` says."""
    robots = doc.get("robots") or []
    if not robots or not isinstance(robots[0], dict) or world.grid is None or len(world.route) < 2:
        return
    (sx, sy), (gx, gy) = world.start, world.goal
    nx, ny = world.route[1]
    yaw = math.atan2(ny - sy, nx - sx)
    start = [round(sx, 2), round(sy, 2), round(yaw, 3)]
    robots[0] = {**robots[0], "start": start, "level": world.level, "goal_level": world.level, "phases": plan.phases(start, [round(gx, 2), round(gy, 2)])}
    doc["robots"] = robots
    record.warnings.append(f"robot route pinned to the planned one: ({sx:.1f}, {sy:.1f}) -> ({gx:.1f}, {gy:.1f})")


def generate(
    prompt: Prompt,
    *,
    world: WorldContext,
    base_id: str,
    base_doc: Mapping[str, Any],
    base_dir: Path,
    seed: int = 0,
    cache_dir: Path | None = None,
    model: str | None = None,
    oracle: bool = False,
    robot: RobotPlan | None = None,
) -> Result:
    robot = RobotPlan() if robot is None else robot
    if robot.radius != world.robot_radius:
        raise ValueError(f"the robot plan's radius {robot.radius:g} m differs from the {world.robot_radius:g} m the world's route was planned for")
    doc = copy.deepcopy(dict(base_doc))
    files = TypeFiles(base_dir=Path(base_dir), origin=world.origin)
    record = Record(prompt_id=prompt.id, prompt=prompt.text, world=world.name, base=base_id, targets=tuple(prompt.targets))
    pin_robot_route(doc, world, record, robot)
    effects: list[dict[str, Any]] = []
    owned: set[str] = set()
    taken_fractions: list[float] = []
    populated = bool(doc.get("humans"))

    denied = denied_on_empty(prompt.targets)
    if not populated and denied and len(denied) == len(prompt.targets):
        record.failures.append(Failure(step="base", kind="denied", reason=f"empty base: {', '.join(denied)} need a crowd and nothing else is targeted"))
        return Result(record=record, doc=doc, files=files, effects=[], scenario_id=f"{base_id}__{prompt.id}", prompt=prompt, robot=robot)
    shift_humans(doc, -world.origin[0], -world.origin[1])

    a = lexicon_match(prompt.text)
    who = ""
    if oracle is False:
        from . import step_a_llm

        try:
            a_llm, who_block, notes = step_a_llm.decide(prompt.text, record, cache_dir=cache_dir, model=model)
            record.warnings.extend(notes)
            if who_block:
                record.provenance["who"] = who_block
                who = step_a_llm.who_line(who_block)
            if a is None and a_llm is not None:
                record.warnings.append(f"A read by the model (lexicon silent): {a_llm.case}")
                a = a_llm
            elif a is not None and a_llm is None:
                record.warnings.append(f"lexicon reads A:{a.case}, the model reads none, keeping the lexicon")
            elif a is not None and a_llm is not None and a_llm.case != a.case:
                record.warnings.append(f"A disagreement: lexicon {a.case}, model {a_llm.case}, taking the model")
                a = a_llm
        except (StepError, Unavailable) as exc:
            record.warnings.append(f"A model unavailable, lexicon only: {exc}")
    wanted_a = [t for t in prompt.targets if t.startswith("A:")]
    if a is None and wanted_a:
        record.failures.append(Failure(step="A", kind="schema", reason=f"prompt targets {wanted_a[0]} but the lexicon matched nothing"))
    if a is not None and wanted_a and f"A:{a.case}" != wanted_a[0]:
        record.warnings.append(f"lexicon matched {a.case}, prompt targets {wanted_a[0]}")
    if a is not None and not populated and a.case not in A_ON_EMPTY:
        record.failures.append(Failure(step="A", kind="denied", reason=f"{a.case} needs a crowd and the base is empty"))
        a = None
    if a is not None:
        try:
            work, work_files = copy.deepcopy(doc), copy.deepcopy(files)
            new, names, notes = apply_a(work, a, world, work_files, seed=seed)
            doc, files = work, work_files
            effects.extend(new)
            owned.update(names)
            record.owned["A"] = list(names)
            record.a = a
            record.warnings.extend(notes)
        except StepError as exc:
            _fail(record, "A", exc)

    try:
        d = decide_d(prompt.text, doc, world, record, cache_dir=cache_dir, model=model, oracle=_oracle_for(prompt, "D") if oracle else None, who=who)
    except (StepError, Unavailable) as exc:
        _fail(record, "D", exc)
        d = None
    if d is not None:
        try:
            work, work_files = copy.deepcopy(doc), copy.deepcopy(files)
            new, names, notes, info = apply_d(work, d, world, work_files, seed=seed, owned=owned, robot_speed=robot.speed)
            doc, files = work, work_files
            effects.extend(new)
            owned.update(names)
            record.owned["D"] = list(names)
            record.d = d
            record.warnings.extend(notes)
            if info.get("formation"):
                record.formation = dict(info["formation"])
            if d.fraction is not None:
                taken_fractions.append(d.fraction)
        except StepError as exc:
            _fail(record, "D", exc)

    try:
        b, c = decide_bc(prompt.text, doc, world, record, taken_fractions=taken_fractions, cache_dir=cache_dir, model=model, oracle=_oracle_for(prompt, "BC") if oracle else None, who=who)
    except (StepError, Unavailable) as exc:
        _fail(record, "BC", exc)
        b = c = None
    if b is not None:
        if not populated:
            record.failures.append(Failure(step="B", kind="denied", reason="empty base: nobody to tune"))
        elif not (b.case == "erratic_route" and b.around == "route") and not [x for x in (doc.get("humans") or []) if isinstance(x, dict) and str(x.get("name")) not in owned]:
            record.failures.append(Failure(step="B", kind="applicability", reason="no unowned pedestrian to tune"))
        else:
            try:
                work, work_files = copy.deepcopy(doc), copy.deepcopy(files)
                new, names, notes = apply_b(work, b, world, work_files, owned=owned, seed=seed)
                doc, files = work, work_files
                effects.extend(new)
                owned.update(names)
                record.owned["B"] = list(names)
                record.b = b
                record.warnings.extend(notes)
            except StepError as exc:
                _fail(record, "B", exc)
    if c is not None:
        try:
            work_files = copy.deepcopy(files)
            new, names, notes = apply_c(c, world, work_files, seed=seed, taken_fractions=taken_fractions)
            files = work_files
            effects.extend(new)
            owned.update(names)
            record.owned["C"] = list(names)
            record.c = c
            record.warnings.extend(notes)
        except StepError as exc:
            _fail(record, "C", exc)

    return _realized(Result(record=record, doc=doc, files=files, effects=effects, scenario_id=f"{base_id}__{prompt.id}", prompt=prompt, robot=robot), world, seed=seed)


def _realized(result: Result, world: WorldContext, *, seed: int) -> Result:
    """The steps' design realized offline, a step whose realization fails recorded as failed and dropped."""
    record, robot = result.record, result.robot
    while True:
        doc, files = copy.deepcopy(result.doc), copy.deepcopy(result.files)
        try:
            done = realize(doc, result.effects, world, files, seed=seed, robot_speed=robot.speed, hold=robot.hold)
        except StepError as exc:
            step = exc.step
            if step not in _STEP_OF:
                raise
            _fail(record, step, exc)
            setattr(record, _STEP_OF[step], None)
            record.owned.pop(step, None)
            if step == "D":
                record.formation = None
            result.effects = [e for e in result.effects if not str(e.get("label", "")).startswith(f"{step}:")]
            continue
        break
    if record.formation and record.formation.get("at") is not None:
        record.formation["at"] = trigger(record.formation["at"], world)
    record.warnings.extend(done.notes)
    for step, names in done.owned.items():
        record.owned.setdefault(step, []).extend(names)
    shift_humans(doc, *world.origin)
    files.to_map()
    result.doc, result.files, result.timeline, result.injected, result.encounters = doc, files, done.timeline, done.injected, done.encounters
    return result


def _header(result: Result) -> str:
    r = result.record
    return "\n".join(
        (
            f"# Prompt scenario: {result.scenario_id}",
            f"# prompt: {r.prompt}",
            f"# base: {r.base}, steps: {', '.join(r.steps) or 'none'}, status: {r.status}",
            "#",
            GENERATED_MARKER,
            f"# The timeline raises the case's needs, {PROVENANCE} records the decisions.",
            "",
            "",
        )
    )


def write_result(result: Result, world_root: Path, *, keep_partial: bool = True) -> Path | None:
    """Write `scenarios/<base>__<prompt_id>/` over a generated one only, None when infeasible, denied or a dropped partial."""
    if result.record.status in ("infeasible", "denied") or (result.record.status == "partial" and not keep_partial):
        return None
    out = Path(world_root) / "scenarios" / result.scenario_id
    existing = out / "scenario.yaml"
    if existing.is_file() and not is_generated(existing):
        raise WouldOverwrite(f"{existing} already exists and was not generated by this tool. Move it away or give the prompt another id.")
    for agent in result.doc.get("humans") or []:
        ref = agent_type_of(agent)
        if ref.startswith("./") and ref not in result.files.docs:
            result.files.base_doc(ref)
    doc = {**result.doc, **({"timeline": result.timeline} if result.timeline else {})}
    with tempfile.TemporaryDirectory() as stage:
        result.files.write(Path(stage))
        try:
            text = scenario_text(doc, world_root=Path(world_root), types_dir=Path(stage), header=_header(result))
        except ValueError as exc:
            raise ValueError(f"{out}: {exc}") from exc
    for stale in sorted((out / TYPES_DIR).glob("*.yaml")):
        if f"./{TYPES_DIR}/{stale.name}" not in result.files.docs:
            stale.unlink()
    result.files.write(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scenario.yaml").write_text(text)
    (out / PROVENANCE).write_text(_header(result) + yaml.safe_dump(result.provenance, sort_keys=False, default_flow_style=None, allow_unicode=False))
    (out / "record.json").write_text(json.dumps(result.record.as_dict(), indent=1, sort_keys=True, default=str))
    result.written = out
    return out
