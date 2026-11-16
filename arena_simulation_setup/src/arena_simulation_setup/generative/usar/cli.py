"""`generate_usar <manifest>`: rooms worlds with their hidden-caller search tasks, split into tuning and test worlds, and an index of what came out."""

from __future__ import annotations

import random
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from arena_simulation_setup.generative.core.generated import Generated, Rejected
from arena_simulation_setup.generative.study import cli
from arena_simulation_setup.generative.study.runner import Place, Placer, Report
from arena_simulation_setup.generative.study.runner import run as run_study
from arena_simulation_setup.generative.world.layout import Layout
from arena_simulation_setup.tree import imported_root
from arena_simulation_setup.tree.World import World

from . import bundle
from .manifest import Manifest
from .placement import Hideout, hide
from .sampler import TEST, TUNE, Task, held_out, tasks
from .scenario import build


def _placer(manifest: Manifest, split: Callable[[str], str]) -> Placer[Task]:
    def placer(key: str, world: World, _layout: Layout | None) -> Place[Task]:
        hideouts: dict[str, Hideout | Rejected] = {}

        def place(task: Task) -> Generated | Rejected:
            if task.spot_key not in hideouts:
                hideouts[task.spot_key] = hide(world, task.corners, task.hiding, random.Random(f"{key}:{task.spot_key}"))
            hideout = hideouts[task.spot_key]
            return hideout if isinstance(hideout, Rejected) else build(manifest, task, hideout, split(key))

        return place

    return placer


def run(manifest: Manifest, *, out: Path | None = None, limit: int = 0, write: bool = False) -> Report[Task]:
    """Generate every task, one world per world configuration and seed, writing only with `write`."""
    items = tasks(manifest)
    if limit:
        items = items[: limit * len(manifest.call_levels)]
    test = held_out(manifest)

    def split(key: str) -> str:
        return TEST if key in test else TUNE

    return run_study(items, _placer(manifest, split), label="task", out=out, write=write, split=split, splits=(TUNE, TEST))


SUITE_NAME = re.compile(r"[a-z0-9_]+")
SUITES = "suites"


def _suite(manifest: Manifest, report: Report[Task], name: str, write: bool) -> int:
    """Write the bundles of a fully placed report, or name them on a dry run."""
    root = imported_root() / SUITES
    try:
        planned = bundle.write(manifest, report, root, name) if write else bundle.plan(manifest, report, root, name)
    except ValueError as error:
        print(f"{error}, nothing written")
        return 1
    for path, outcomes in planned.items():
        if outcomes:
            print(f"{'bundle written to' if write else 'would write'} {path} ({len({o.item.world_key for o in outcomes})} worlds, {len(outcomes)} stages)")
    if not write:
        print("dry run, pass --write to save")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.parser("generate_usar", __doc__, "generate only the first N spots, each at every call level")
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
    cli.summarize(report, "scenarios")
    if args.suite is not None:
        return _suite(manifest, report, args.suite, args.write)
    return cli.finish(report, manifest.name, args.out, args.write)


if __name__ == "__main__":
    sys.exit(main())
