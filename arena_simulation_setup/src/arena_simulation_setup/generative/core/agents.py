"""Cast helpers: naming, meshes, per-agent entries and the derived type files beside a scenario (`types/<name>.yaml`)."""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs
import yaml

from arena_simulation_setup.shared.pathing import Point

from .frames import shift_targets
from .profiles import PROFILES, Profile, ProfileError, derive, derived_name, profile

TYPES_DIR = "types"
#: Seconds a walking tree may take to reach one of its route points before it starts the route again.
WALK_PATIENCE_S = 300.0
STAND_STEP: dict[str, Any] = {"duration": {"mean": 600.0}, "patience": {"mean": 900.0}}


def agent_type_of(agent: Mapping[str, Any]) -> str:
    block = agent.get("agent")
    if isinstance(block, str):
        return block
    if isinstance(block, Mapping):
        return str(block.get("agent_type", "adult"))
    return "adult"


def type_stem(ref: str) -> str:
    """The base type an agent type ref derives from: `./types/adult__walk_x.yaml` is `adult`."""
    return ref.rsplit("/", 1)[-1].removesuffix(".yaml").split("__", 1)[0]


def unique_names(doc: Mapping[str, Any], prefix: str, n: int) -> list[str]:
    taken = {str(a.get("name")) for a in (doc.get("humans") or []) if isinstance(a, Mapping)}
    out, i = [], 1
    while len(out) < n:
        name = f"{prefix}_{i}"
        if name not in taken:
            out.append(name)
            taken.add(name)
        i += 1
    return out


def mesh_for(prof: Profile, rng: random.Random) -> str:
    return rng.choice(list(prof.meshes))


def entry(name: str, prof: Profile, pose: Point | str, waypoints: Sequence[Point | str], *, speed: float, mode: str, agent_type: str, rng: random.Random) -> dict[str, Any]:
    def pt(p: Point | str) -> str | list[float]:
        return p if isinstance(p, str) else [round(p[0], 2), round(p[1], 2), 0.0]

    return {
        "name": name,
        "model": mesh_for(prof, rng),
        "pose": pose if isinstance(pose, str) else [round(pose[0], 2), round(pose[1], 2), 0.0],
        "agent": {"agent_type": agent_type, "desired_velocity": round(speed, 2)},
        "waypoints": [pt(p) for p in waypoints],
        "waypoint_mode": mode,
    }


@attrs.define
class TypeFiles:
    base_dir: Path
    #: The working level's shift into the map frame, a scenario-local file's targets read back into the level frame.
    origin: Point = (0.0, 0.0)
    docs: dict[str, dict[str, Any]] = attrs.Factory(dict)
    #: Files copied verbatim from the base scenario directory (a `./talker.yaml`).
    copies: dict[str, Path] = attrs.Factory(dict)
    #: Walking trees (`walking`) that exist only for the derivation built on them, which replaces them.
    transient: set[str] = attrs.Factory(set)

    def base_doc(self, ref: str) -> dict[str, Any]:
        """The document a derivation starts from: one derived earlier in this run, `{extends: builtin}`, or a scenario-local file."""
        if ref in self.docs:
            return copy.deepcopy(self.docs[ref])
        if ref.startswith("./"):
            path = self.base_dir / ref[2:]
            if not path.is_file():
                raise ProfileError(f"{ref}: not found beside the base scenario ({path})")
            self.copies.setdefault(ref, path)
            doc = yaml.safe_load(path.read_text()) or {}
            shift_targets(doc, -self.origin[0], -self.origin[1])
            return doc
        return {"extends": ref}

    def derived(self, ref: str, params: Mapping[str, str], *, prof: Profile | None = None) -> str:
        """The ref of `ref` with `params` pinned, writing the document once per distinct combination."""
        if not params:
            return ref
        name = derived_name(ref, params)
        out = f"./{TYPES_DIR}/{name}.yaml"
        if out not in self.docs:
            base = self.base_doc(ref)
            base.pop("name", None)
            self.docs[out] = derive(base, params, name=name, prof=prof)
        return out

    def walking(self, agent: Mapping[str, Any], route: Sequence[Point]) -> str:
        """`agent`'s own type as a behavior tree looping a `walk` sequence over `route`, one per walker."""
        base = agent_type_of(agent)
        name = f"{type_stem(base)}__walk_{agent['name']}"
        out = f"./{TYPES_DIR}/{name}.yaml"
        doc = self.base_doc(base)
        steps = {f"to_{i + 1}": {"kind": "go_to", "target_pose": {"x": round(x, 2), "y": round(y, 2), "theta": 0.0}, "patience": {"mean": WALK_PATIENCE_S}} for i, (x, y) in enumerate(route)}
        doc.update(name=name, mode="behavior_tree", initial_sequence="walk", sequences={"walk": {"steps": steps, "then": "walk", "on_failure": "walk"}})
        self.docs[out] = doc
        self.transient.add(out)
        return out

    def standing(self, ref: str, name: str) -> str:
        """`ref` as a behavior tree whose `wait` sequence stands where the agent spawned, one per agent `name`."""
        type_name = f"{type_stem(ref)}__wait_{name}"
        out = f"./{TYPES_DIR}/{type_name}.yaml"
        doc = self.base_doc(ref)
        doc.update(name=type_name, mode="behavior_tree", initial_sequence="wait", sequences={"wait": {"steps": {"stand": dict(STAND_STEP)}, "then": "wait"}})
        self.docs[out] = doc
        self.transient.add(out)
        return out

    def _start(self, ref: str) -> dict[str, Any]:
        """`base_doc(ref)`, dropping a walking tree from the files once a derivation has started from it."""
        doc = self.base_doc(ref)
        if ref in self.transient:
            self.transient.discard(ref)
            self.docs.pop(ref)
        return doc

    def switching(self, ref: str, *, need: str, to: str, steps: Mapping[str, Any], tune: Mapping[str, Any] | None = None, tag: str = "", advance: bool = False, resume: bool = False) -> str:
        """`ref` with a `to` sequence of `steps` that every other sequence preempts into once `need` rises above 50."""
        name = f"{derived_name(ref, {})}__{to}" + (f"_{tag}" if tag else "")
        out = f"./{TYPES_DIR}/{name}.yaml"
        if out not in self.docs:
            doc = self._start(ref)
            doc["name"] = name
            doc.setdefault("needs", {})[need] = {"initial": 0.0, "decay_rate": 0.0}
            for sequence in doc["sequences"].values():
                sequence.setdefault("transitions", []).append({"when": {need: {"above": 50.0}}, "goto": to})
            held = {"tune": dict(tune)} if tune else {}
            if advance:
                chain = [to, *(f"{to}_{k}" for k in range(2, len(steps) + 1))]
                for i, (key, step) in enumerate(steps.items()):
                    doc["sequences"][chain[i]] = {"steps": {key: step}, "then": chain[(i + 1) % len(chain)], "on_failure": chain[(i + 1) % len(chain)], **held}
            else:
                doc["sequences"][to] = {"steps": dict(steps), "then": doc["initial_sequence"] if resume else to, **held}
            self.docs[out] = doc
        return out

    def tuned(self, ref: str, *, need: str, tune: Mapping[str, Any], params: Mapping[str, str]) -> str:
        """`ref` holding `tune` through a type-level `tunes:` entry once `need` rises above 50."""
        name = f"{derived_name(ref, params)}__{need}"
        out = f"./{TYPES_DIR}/{name}.yaml"
        if out not in self.docs:
            doc = self._start(ref)
            doc["name"] = name
            doc.setdefault("needs", {})[need] = {"initial": 0.0, "decay_rate": 0.0}
            doc.setdefault("tunes", []).append({"when": {need: {"above": 50.0}}, "tune": dict(tune)})
            self.docs[out] = doc
        return out

    def to_map(self) -> None:
        """Move every derived document's `go_to` targets from the level frame into the map frame."""
        for doc in self.docs.values():
            shift_targets(doc, *self.origin)

    def write(self, scenario_dir: Path) -> list[Path]:
        written = []
        for ref, doc in self.docs.items():
            path = scenario_dir / ref[2:]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))
            written.append(path)
        for ref, src in self.copies.items():
            dst = scenario_dir / ref[2:]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text())
            written.append(dst)
        return written


def profile_of_agent(agent: Mapping[str, Any]) -> Profile | None:
    """The library profile an existing agent most resembles, by agent type refined by mesh."""
    ref = agent_type_of(agent)
    try:
        base = profile(type_stem(ref))
    except ProfileError:
        return None
    mesh = str(agent.get("model") or "")
    for staff in ("nurse", "doctor"):
        prof = PROFILES[staff]
        if base.name == prof.base and mesh in prof.meshes:
            return prof
    return base
