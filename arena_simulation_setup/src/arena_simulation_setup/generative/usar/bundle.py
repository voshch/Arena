"""USAR suite bundles: each split of a generated study as one benchmark suite directory carrying its worlds, index and coverage table."""

from __future__ import annotations

import copy
import hashlib
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from arena_simulation_setup.generative.egobench.encounter import CLEARANCE, ROBOT_RADIUS
from arena_simulation_setup.generative.world.rooms import WorldGeneratorRooms
from arena_simulation_setup.generative.world.world_generator import save_generated
from arena_simulation_setup.tree.World import WorldView

from .manifest import CallLevel, Hiding, Manifest
from .sampler import TEST, TUNE

if TYPE_CHECKING:
    from .cli import Outcome, Report

SPLITS = (TEST, TUNE)  # the order the bundles are written in
SEARCH: dict[str, object] = {"radius": 1.0, "line_of_sight": True, "dwell_s": 0.0}
STAGE: dict[str, object] = {"robot": "auto", "modes": {"robots": "search", "obstacles": "scenario"}, "episodes": 3, "timeout": "300s"}


def _dump(doc: object) -> str:
    return yaml.safe_dump(doc, sort_keys=False)


def _index_name(bundle: Path) -> str:
    return f"{bundle.name}.index.yaml"


def stages(outcomes: Sequence[Outcome]) -> list[dict[str, object]]:
    """One suite stage per task, in world then scenario order."""
    keys = sorted((outcome.task.world_key, outcome.task.scenario_key) for outcome in outcomes)
    return [{"name": f"{world}__{scenario}", "map": world, **copy.deepcopy(STAGE), "config": {"scenario": {"file": scenario}, "search": dict(SEARCH)}} for world, scenario in keys]


def checksum(bundle: Path) -> str:
    """SHA-256 over every file of a bundle but its index, by relative path."""
    digest = hashlib.sha256()
    files = sorted((path.relative_to(bundle).as_posix(), path) for path in bundle.rglob("*") if path.is_file())
    for name, path in files:
        if name == _index_name(bundle):
            continue
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _by_world(outcomes: Sequence[Outcome]) -> dict[str, list[Outcome]]:
    worlds: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        worlds.setdefault(outcome.task.world_key, []).append(outcome)
    return dict(sorted(worlds.items()))


def _configuration(outcome: Outcome) -> WorldGeneratorRooms.Configuration:
    return WorldGeneratorRooms.Configuration.model_validate(outcome.task.world)


def coverage(outcomes: Sequence[Outcome]) -> str:
    """The coverage table of a split, one row per world."""
    lines = ["| world | corners | call level | hiding | rooms |", "|---|---|---|---|---|"]
    for key, members in _by_world(outcomes).items():
        corners = ", ".join(str(count) for count in sorted({member.task.corners for member in members}))
        levels = ", ".join(level.value for level in CallLevel if any(member.task.call_level is level for member in members))
        hiding = ", ".join(kind.value for kind in Hiding if any(member.task.hiding is kind for member in members))
        config = _configuration(members[0])
        lines.append(f"| {key} | {corners} | {levels} | {hiding} | {config.rows} x {config.cols} = {config.rows * config.cols} |")
    return "\n".join(lines)


def _readme(bundle: str, manifest: Manifest, split: str, outcomes: Sequence[Outcome]) -> str:
    worlds = _by_world(outcomes)
    door = min(_configuration(members[0]).door_width for members in worlds.values())
    return "\n".join(
        [
            f"# {bundle}",
            "",
            f"The {split} split of the USAR study `{manifest.name}` on seed {manifest.seed}: {len(worlds)} worlds, {len(outcomes)} stages, each hiding one calling person for the `search` robots mode.",
            "",
            "```bash",
            f"arena evaluation benchmark --suite {bundle} --contest <contest>",
            "```",
            "",
            f"The stages name no robot (`robot: auto`). The narrowest door is {door:g} m wide, and the start is placed for a robot of up to {ROBOT_RADIUS:g} m radius with {CLEARANCE:g} m to the walls.",
            "",
            coverage(outcomes),
            "",
        ]
    )


def write(manifest: Manifest, report: Report, root: Path, name: str) -> list[Path]:
    """Write `root/<name>_<split>` for every split with worlds, the test split first, each replacing the bundle already there."""
    rejected = len(report.outcomes) - report.placed
    if rejected:
        raise ValueError(f"{rejected} of {len(report.outcomes)} tasks rejected, a suite needs every one placed")
    written: list[Path] = []
    for split in SPLITS:
        outcomes = [outcome for outcome in report.outcomes if outcome.split == split]
        if not outcomes:
            continue
        bundle = root / f"{name}_{split}"
        if bundle.exists():
            shutil.rmtree(bundle)
        worlds = _by_world(outcomes)
        for key in worlds:
            built = report.built[key]
            save_generated(WorldView(bundle / "worlds" / key), built.world, built.files)
        suite = stages(outcomes)
        (bundle / "suite.yaml").write_text(_dump({"stages": suite}))
        (bundle / "README.md").write_text(_readme(bundle.name, manifest, split, outcomes))
        index = {"manifest": manifest.name, "seed": manifest.seed, "split": split, "worlds": list(worlds), "stages": len(suite), "checksum": checksum(bundle), "tasks": [outcome.to_dict() for outcome in outcomes]}
        (bundle / _index_name(bundle)).write_text(_dump(index))
        written.append(bundle)
    return written
