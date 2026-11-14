"""USAR suite bundles: each split of a generated study as one benchmark suite directory carrying its worlds, index and coverage table."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

from arena_simulation_setup.generative.core.bodies import CLEARANCE, ROBOT_RADIUS
from arena_simulation_setup.generative.study import bundle
from arena_simulation_setup.generative.study.bundle import by_world
from arena_simulation_setup.generative.study.runner import Outcome, Report
from arena_simulation_setup.generative.world.rooms import WorldGeneratorRooms

from .manifest import CallLevel, Hiding, Manifest
from .sampler import TEST, TUNE, Task

SPLITS = (TEST, TUNE)  # the order the bundles are written in
SEARCH: dict[str, object] = {"radius": 1.0, "line_of_sight": True, "dwell_s": 0.0}
STAGE: dict[str, object] = {"robot": "auto", "modes": {"robots": "search", "obstacles": "scenario"}, "episodes": 3, "timeout": "300s"}


def stages(outcomes: Sequence[Outcome[Task]]) -> list[dict[str, object]]:
    """One suite stage per task, in world then scenario order."""
    keys = sorted((outcome.item.world_key, outcome.item.scenario_key) for outcome in outcomes)
    return [{"name": f"{world}__{scenario}", "map": world, **copy.deepcopy(STAGE), "config": {"scenario": {"file": scenario}, "search": dict(SEARCH)}} for world, scenario in keys]


def _configuration(outcome: Outcome[Task]) -> WorldGeneratorRooms.Configuration:
    return WorldGeneratorRooms.Configuration.model_validate(outcome.item.world)


def coverage(outcomes: Sequence[Outcome[Task]]) -> str:
    """The coverage table of a split, one row per world."""
    lines = ["| world | corners | call level | hiding | rooms |", "|---|---|---|---|---|"]
    for key, members in by_world(outcomes).items():
        corners = ", ".join(str(count) for count in sorted({member.item.corners for member in members}))
        levels = ", ".join(level.value for level in CallLevel if any(member.item.call_level is level for member in members))
        hiding = ", ".join(kind.value for kind in Hiding if any(member.item.hiding is kind for member in members))
        config = _configuration(members[0])
        lines.append(f"| {key} | {corners} | {levels} | {hiding} | {config.rows} x {config.cols} = {config.rows * config.cols} |")
    return "\n".join(lines)


def _readme(manifest: Manifest, name: str, split: str, outcomes: Sequence[Outcome[Task]]) -> str:
    worlds = by_world(outcomes)
    door = min(_configuration(members[0]).door_width for members in worlds.values())
    return "\n".join(
        [
            f"# {name}",
            "",
            f"The {split} split of the USAR study `{manifest.name}` on seed {manifest.seed}: {len(worlds)} worlds, {len(outcomes)} stages, each hiding one calling person for the `search` robots mode.",
            "",
            "```bash",
            f"arena evaluation benchmark --suite {name} --contest <contest>",
            "```",
            "",
            f"The stages name no robot (`robot: auto`). The narrowest door is {door:g} m wide, and the start is placed for a robot of up to {ROBOT_RADIUS:g} m radius with {CLEARANCE:g} m to the walls.",
            "",
            coverage(outcomes),
            "",
        ]
    )


def plan(manifest: Manifest, report: Report[Task], root: Path, name: str) -> dict[Path, list[Outcome[Task]]]:
    """The bundle directory of every split with its tasks, the test split first, refused for a rejected task or a directory of another manifest or seed."""
    return bundle.plan(report, root, name, manifest=manifest.name, seed=manifest.seed, splits=SPLITS)


def write(manifest: Manifest, report: Report[Task], root: Path, name: str) -> dict[Path, list[Outcome[Task]]]:
    """Write the bundle of every split, replacing or removing only bundles of this manifest and seed."""
    return bundle.write(
        report,
        root,
        name,
        manifest=manifest.name,
        seed=manifest.seed,
        splits=SPLITS,
        stages=stages,
        readme=lambda bundle_name, split, outcomes: _readme(manifest, bundle_name, split, outcomes),
    )
