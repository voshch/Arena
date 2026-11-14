"""`generate_usar <manifest>`: rooms worlds with their hidden-caller search tasks, split into tuning and test worlds, and an index of what came out."""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import attrs
import yaml

from arena_simulation_setup.generative.core.generated import Rejected, pack
from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
from arena_simulation_setup.generative.world.world_generator import save_generated
from arena_simulation_setup.tree import imported_root
from arena_simulation_setup.tree.World import World, WorldIdentifier, WorldView

from . import bundle
from .manifest import Manifest
from .placement import Hideout, hide
from .sampler import TEST, TUNE, Task, held_out, tasks
from .scenario import build


@attrs.frozen
class Outcome:
    task: Task
    split: str
    reason: str | None  # None = placed

    def to_dict(self) -> dict[str, object]:
        status = {"status": "ok"} if self.reason is None else {"status": "rejected", "reason": self.reason}
        return {"world": self.task.world_key, "scenario": self.task.scenario_key, "split": self.split, **status, "task": self.task.to_dict()}


@attrs.frozen
class Built:
    world: World
    files: dict[str, bytes]  # by path relative to the world


@attrs.frozen
class Report:
    outcomes: list[Outcome]
    worlds: dict[str, Path]  # world key -> where it was written, empty on a dry run
    built: dict[str, Built]  # world key -> every world with a placed task, also on a dry run

    @property
    def placed(self) -> int:
        return sum(o.reason is None for o in self.outcomes)

    def index(self, manifest: Manifest) -> dict[str, object]:
        splits = {split: sorted({o.task.world_key for o in self.outcomes if o.split == split}) for split in (TUNE, TEST)}
        return {"manifest": manifest.name, "placed": self.placed, "rejected": len(self.outcomes) - self.placed, "worlds": splits, "tasks": [o.to_dict() for o in self.outcomes]}


def _view(key: str, out: Path | None) -> WorldView:
    return WorldView(out / key) if out is not None else WorldIdentifier(key).resolve_write_sync()


def run(manifest: Manifest, *, out: Path | None = None, limit: int = 0, write: bool = False) -> Report:
    """Generate every task, one world per world configuration and seed, writing only with `write`."""
    items = tasks(manifest)
    test = held_out(manifest)
    if limit:
        items = items[: limit * len(manifest.call_levels)]
    groups: dict[str, list[Task]] = {}
    for item in items:
        groups.setdefault(item.world_key, []).append(item)
    outcomes: list[Outcome] = []
    worlds: dict[str, Path] = {}
    built: dict[str, Built] = {}
    for key, members in groups.items():
        split = TEST if key in test else TUNE
        first = members[0]
        generator = WorldGenerator(WorldGeneratorType.ROOMS, dict(first.world), first.world_seed)
        try:
            world = generator.compute()
        except ValueError as exc:
            outcomes.extend(Outcome(task=m, split=split, reason=f"world: {exc}") for m in members)
            continue
        files: dict[str, bytes] = dict(generator.files())
        placed = False
        hideouts: dict[str, Hideout | Rejected] = {}
        for member in members:
            if member.spot_key not in hideouts:
                hideouts[member.spot_key] = hide(world, member.corners, member.hiding, random.Random(f"{key}:{member.spot_key}"))
            hideout = hideouts[member.spot_key]
            if isinstance(hideout, Rejected):
                outcomes.append(Outcome(task=member, split=split, reason=hideout.reason))
                continue
            files.update(pack(member.scenario_key, build(manifest, member, hideout, split)))
            outcomes.append(Outcome(task=member, split=split, reason=None))
            placed = True
        if placed:
            built[key] = Built(world=world, files=files)
        if write and placed:
            worlds[key] = save_generated(_view(key, out), world, files)
    return Report(outcomes=outcomes, worlds=worlds, built=built)


def _index_path(report: Report, manifest: Manifest, out: Path | None) -> Path | None:
    if out is not None:
        return out / f"{manifest.name}.index.yaml"
    first = next(iter(report.worlds.values()), None)
    return first.parent / f"{manifest.name}.index.yaml" if first is not None else None


SUITE_NAME = re.compile(r"[a-z0-9_]+")
SUITES = "suites"


def _suite(manifest: Manifest, report: Report, name: str, write: bool) -> int:
    """Write the bundles of a fully placed report, or name them on a dry run."""
    root = imported_root() / SUITES
    try:
        planned = bundle.write(manifest, report, root, name) if write else bundle.plan(manifest, report, root, name)
    except ValueError as error:
        print(f"{error}, nothing written")
        return 1
    for path, outcomes in planned.items():
        if outcomes:
            print(f"{'bundle written to' if write else 'would write'} {path} ({len({o.task.world_key for o in outcomes})} worlds, {len(outcomes)} stages)")
    if not write:
        print("dry run, pass --write to save")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generate_usar", description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, help="directory to write the worlds and the index into, default the writable world root")
    parser.add_argument("--limit", type=int, default=0, help="generate only the first N spots, each at every call level")
    parser.add_argument("--write", action="store_true", help="write the worlds, a dry run without it")
    parser.add_argument("--suite", help="write one benchmark suite bundle <SUITE>_<split> per split into the imported suites root instead of bare worlds")
    args = parser.parse_args(argv)
    if args.suite is not None:
        if SUITE_NAME.fullmatch(args.suite) is None:
            parser.error(f"a suite name is lowercase letters, digits and underscores, got {args.suite!r}")
        if args.out is not None:
            parser.error("--suite takes no --out, its bundles go to the imported suites root")
        if args.limit:
            parser.error("--suite takes no --limit, a bundle holds every task of its split")

    manifest = Manifest.load(args.manifest)
    report = run(manifest, out=args.out, limit=args.limit, write=args.write and args.suite is None)
    for outcome in report.outcomes:
        if outcome.reason is not None:
            print(f"rejected {outcome.task.world_key}/{outcome.task.scenario_key}: {outcome.reason}")
    placed_worlds = {o.task.world_key for o in report.outcomes if o.reason is None}
    print(f"{report.placed} of {len(report.outcomes)} scenarios placed in {len(placed_worlds)} worlds")
    if args.suite is not None:
        return _suite(manifest, report, args.suite, args.write)
    if args.write:
        index = _index_path(report, manifest, args.out)
        if index is not None:
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_text(yaml.safe_dump(report.index(manifest), sort_keys=False))
            print(f"index written to {index}")
    else:
        print("dry run, pass --write to save")
    return 0 if report.placed else 1


if __name__ == "__main__":
    sys.exit(main())
