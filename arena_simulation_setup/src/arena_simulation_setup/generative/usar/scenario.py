"""The scenario of one USAR task: a standing person calling on and off, a robot given only its start."""

from __future__ import annotations

import math
import random
from pathlib import Path

import yaml

from arena_simulation_setup.generative.egobench.encounter import Generated
from arena_simulation_setup.generative.scenario.materialize import GENERATED_MARKER
from arena_simulation_setup.generative.scenario.promptgen.agents import TypeFiles, mesh_for
from arena_simulation_setup.generative.scenario.promptgen.profiles import profile
from arena_simulation_setup.tree.World.Scenario import SCENARIO_FORMAT, parse_scenario

from .manifest import Manifest
from .placement import Hideout
from .sampler import Task

PERSON = "caller"
CALL = "call"
CALL_ONSET_S = 1.0  # first call this long into the episode


def _pose(point: tuple[float, float], theta: float) -> list[float]:
    return [round(point[0], 3), round(point[1], 3), round(theta, 3)]


def _sounding(on: bool, offset: float, period: float) -> dict[str, object]:
    return {"every": period, "offset": round(offset, 3), "set": [{"entity": CALL, "field": "sounding", "value": "true" if on else "false"}]}


def build(manifest: Manifest, task: Task, hideout: Hideout, split: str) -> Generated:
    """The scenario, agent types and provenance of one placed task."""
    rng = random.Random(f"{task.world_key}:{task.spot_key}")
    files = TypeFiles(base_dir=Path())
    ref = files.standing(manifest.agent_type, PERSON)
    person = {
        "name": PERSON,
        "model": mesh_for(profile(manifest.agent_type), rng),
        "pose": _pose(hideout.person, rng.uniform(-math.pi, math.pi)),
        "agent": {"agent_type": ref},
        "waypoints": [[hideout.person[0], hideout.person[1], 0.0]],
        "waypoint_mode": "once",
    }
    call = {
        "name": CALL,
        "asset_id": manifest.call_asset,
        "position": [hideout.person[0], hideout.person[1]],
        "loop": True,
        "semantics": [{"preset": "sound", "params": {"sounding": False, "volume_db": manifest.call_db[task.call_level]}}],
    }
    period = manifest.call_period_s
    scenario: dict[str, object] = {
        "format": SCENARIO_FORMAT,
        "humans": [person],
        "sounds": [call],
        "robots": [{"start": _pose(hideout.start, hideout.heading)}],
        "timeline": [_sounding(True, CALL_ONSET_S, period), _sounding(False, CALL_ONSET_S + manifest.call_on_s, period)],
    }
    parse_scenario(scenario)
    provenance: dict[str, object] = {
        "task": task.to_dict(),
        "split": split,
        "design": {
            "robot_start": _pose(hideout.start, hideout.heading),
            "person": [hideout.person[0], hideout.person[1]],
            "person_zone": hideout.zone,
            "route": [[round(x, 3), round(y, 3)] for x, y in hideout.route],
            "corners": hideout.corners,
            "doors": hideout.doors,
            "misdirection_rad": round(hideout.misdirection, 3),
            "call_db": manifest.call_db[task.call_level],
            "call_period_s": period,
            "call_on_s": manifest.call_on_s,
        },
        "run": {"task.robots": "search", "task.obstacles": "scenario", "task.scenario.file": task.scenario_key},
    }
    return Generated(scenario=scenario, types={path[2:]: doc for path, doc in files.docs.items()}, provenance=provenance)


def _yaml(doc: object, marked: bool) -> bytes:
    text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=None)
    return ((GENERATED_MARKER + "\n" + text) if marked else text).encode("utf-8")


def pack(task: Task, generated: Generated) -> dict[str, bytes]:
    """The scenario directory of one task, by path relative to its world."""
    root = f"scenarios/{task.scenario_key}"
    files = {f"{root}/scenario.yaml": _yaml(generated.scenario, marked=True), f"{root}/provenance.yaml": _yaml(generated.provenance, marked=False)}
    files.update({f"{root}/{path}": _yaml(doc, marked=True) for path, doc in generated.types.items()})
    return files
