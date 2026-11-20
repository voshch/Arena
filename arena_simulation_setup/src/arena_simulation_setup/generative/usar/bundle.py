"""USAR suite bundles: each split of a generated study as one benchmark suite directory carrying its worlds, index and coverage table."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from arena_simulation_setup.generative.core.bodies import CLEARANCE, ROBOT_RADIUS
from arena_simulation_setup.generative.study import bundle
from arena_simulation_setup.generative.study.bundle import by_world
from arena_simulation_setup.generative.study.runner import Outcome, Report
from arena_simulation_setup.generative.world.rooms import WorldGeneratorRooms

from .manifest import CallLevel, Hiding, Manifest, Suite
from .sampler import TEST, TUNE, Task

SPLITS = (TEST, TUNE)  # the order the bundles are written in
MODES: dict[str, str] = {"robots": "search", "obstacles": "scenario"}


def stage(world: str, scenario: str, suite: Suite) -> dict[str, object]:
    """The suite stage of one task, running the suite settings of its manifest."""
    return {
        "name": f"{world}__{scenario}",
        "map": world,
        "robot": suite.robot,
        "modes": dict(MODES),
        "episodes": suite.episodes,
        "timeout": suite.timeout,
        "config": {"scenario": {"file": scenario}, "search": suite.search.model_dump()},
    }


def stages(outcomes: Sequence[Outcome[Task]], suite: Suite) -> list[dict[str, object]]:
    """One suite stage per task, in world then scenario order."""
    keys = sorted((outcome.item.world_key, outcome.item.scenario_key) for outcome in outcomes)
    return [stage(world, scenario, suite) for world, scenario in keys]


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


def _robot_sentence(robot: str) -> str:
    return "The stages name no robot (`robot: auto`)." if robot == "auto" else f"Every stage runs `robot: {robot}`."


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
            f"{_robot_sentence(manifest.suite.robot)} The narrowest door is {door:g} m wide, and the start is placed for a robot of up to {ROBOT_RADIUS:g} m radius with {CLEARANCE:g} m to the walls.",
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
        stages=lambda outcomes: stages(outcomes, manifest.suite),
        readme=lambda bundle_name, split, outcomes: _readme(manifest, bundle_name, split, outcomes),
    )
