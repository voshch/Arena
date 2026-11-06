"""`generate_egobench <manifest>`: every sample of a manifest as worlds with their encounter scenarios, and an index of what came out."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import attrs
import yaml

from arena_simulation_setup.generative.world import WorldGenerator
from arena_simulation_setup.generative.world.world_generator import save_generated
from arena_simulation_setup.tree.World import WorldIdentifier, WorldView

from .encounter import Rejected
from .manifest import Manifest
from .meta import SCENARIO_GENERATORS, pack
from .sampler import Sample, sample


@attrs.frozen
class Outcome:
    sample: Sample
    reason: str | None  # None = placed

    def to_dict(self) -> dict[str, object]:
        status = {"status": "ok"} if self.reason is None else {"status": "rejected", "reason": self.reason}
        return {"world": self.sample.world_key, "scenario": self.sample.scenario_key, **status, "sample": self.sample.to_dict()}


@attrs.frozen
class Report:
    outcomes: list[Outcome]
    worlds: dict[str, Path]  # world key -> where it was written, empty on a dry run

    @property
    def placed(self) -> int:
        return sum(o.reason is None for o in self.outcomes)

    def index(self, manifest: Manifest) -> dict[str, object]:
        return {"manifest": manifest.name, "placed": self.placed, "rejected": len(self.outcomes) - self.placed, "samples": [o.to_dict() for o in self.outcomes]}


def _view(key: str, out: Path | None) -> WorldView:
    return WorldView(out / key) if out is not None else WorldIdentifier(key).resolve_write_sync()


def run(manifest: Manifest, *, out: Path | None = None, limit: int = 0, write: bool = False) -> Report:
    """Generate every sample, one world per distinct world configuration, writing only with `write`."""
    samples = sample(manifest)
    if limit:
        samples = samples[:limit]
    groups: dict[str, list[Sample]] = {}
    for item in samples:
        groups.setdefault(item.world_key, []).append(item)
    outcomes: list[Outcome] = []
    worlds: dict[str, Path] = {}
    for key, members in groups.items():
        first = members[0]
        generator = WorldGenerator(first.space.generator, dict(first.world), first.world_seed)
        try:
            world = generator.compute()
        except ValueError as exc:
            outcomes.extend(Outcome(sample=m, reason=f"world: {exc}") for m in members)
            continue
        layout = generator.layout()
        files: dict[str, bytes] = dict(generator.files())
        for member in members:
            result = SCENARIO_GENERATORS[member.space](layout, member) if layout is not None else Rejected("the world generator leaves no layout")
            if isinstance(result, Rejected):
                outcomes.append(Outcome(sample=member, reason=result.reason))
                continue
            files.update(pack(member, result))
            outcomes.append(Outcome(sample=member, reason=None))
        if write and any(o.reason is None for o in outcomes if o.sample.world_key == key):
            worlds[key] = save_generated(_view(key, out), world, files)
    return Report(outcomes=outcomes, worlds=worlds)


def _index_path(report: Report, manifest: Manifest, out: Path | None) -> Path | None:
    if out is not None:
        return out / f"{manifest.name}.index.yaml"
    first = next(iter(report.worlds.values()), None)
    return first.parent / f"{manifest.name}.index.yaml" if first is not None else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generate_egobench", description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, help="directory to write the worlds and the index into, default the writable world root")
    parser.add_argument("--limit", type=int, default=0, help="generate only the first N samples")
    parser.add_argument("--write", action="store_true", help="write the worlds, a dry run without it")
    args = parser.parse_args(argv)

    manifest = Manifest.load(args.manifest)
    report = run(manifest, out=args.out, limit=args.limit, write=args.write)
    for outcome in report.outcomes:
        if outcome.reason is not None:
            print(f"rejected {outcome.sample.world_key}/{outcome.sample.scenario_key}: {outcome.reason}")
    print(f"{report.placed} of {len(report.outcomes)} samples placed in {len({o.sample.world_key for o in report.outcomes if o.reason is None})} worlds")
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
