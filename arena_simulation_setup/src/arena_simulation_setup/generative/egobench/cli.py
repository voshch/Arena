"""`generate_egobench <manifest>`: every sample of a manifest as worlds with their encounter scenarios, and an index of what came out."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from arena_simulation_setup.generative.core.generated import Generated, Rejected
from arena_simulation_setup.generative.study import cli
from arena_simulation_setup.generative.study.runner import Place, Report
from arena_simulation_setup.generative.study.runner import run as run_study
from arena_simulation_setup.generative.world.layout import Layout
from arena_simulation_setup.tree.World import World

from .manifest import Manifest
from .meta import SCENARIO_GENERATORS
from .sampler import Sample, sample


def _placer(_key: str, _world: World, layout: Layout | None) -> Place[Sample]:
    def place(item: Sample) -> Generated | Rejected:
        return SCENARIO_GENERATORS[item.space](layout, item) if layout is not None else Rejected("the world generator leaves no layout")

    return place


def run(manifest: Manifest, *, out: Path | None = None, limit: int = 0, write: bool = False) -> Report[Sample]:
    """Generate every sample, one world per distinct world configuration, writing only with `write`."""
    samples = sample(manifest)
    if limit:
        samples = samples[:limit]
    return run_study(samples, _placer, label="sample", out=out, write=write)


def main(argv: Sequence[str] | None = None) -> int:
    args = cli.parser("generate_egobench", __doc__, "generate only the first N samples").parse_args(argv)
    manifest = Manifest.load(args.manifest)
    report = run(manifest, out=args.out, limit=args.limit, write=args.write)
    cli.summarize(report, "samples")
    return cli.finish(report, manifest.name, args.out, args.write)


if __name__ == "__main__":
    sys.exit(main())
