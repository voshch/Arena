"""Suite bundles: each split of a generated study as one benchmark suite directory carrying its worlds, index and README, swapped in whole."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

from arena_simulation_setup.generative.world.world_generator import save_generated
from arena_simulation_setup.tree.World import WorldView

from .runner import Item, Outcome, Report

type Stages[I: Item] = Callable[[Sequence[Outcome[I]]], list[dict[str, object]]]
type Readme[I: Item] = Callable[[str, str, Sequence[Outcome[I]]], str]


def _dump(doc: object) -> str:
    return yaml.safe_dump(doc, sort_keys=False)


def _index_name(bundle: Path) -> str:
    return f"{bundle.name}.index.yaml"


def checksum(bundle: Path) -> str:
    """SHA-256 over every file of a bundle but its index and the dot files a cache adds, by relative path."""
    digest = hashlib.sha256()
    files = sorted((path.relative_to(bundle).as_posix(), path) for path in bundle.rglob("*") if path.is_file())
    for name, path in files:
        if name == _index_name(bundle) or path.name.startswith("."):
            continue
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def by_world[I: Item](outcomes: Sequence[Outcome[I]]) -> dict[str, list[Outcome[I]]]:
    """The outcomes of each world, worlds in key order."""
    worlds: dict[str, list[Outcome[I]]] = {}
    for outcome in outcomes:
        worlds.setdefault(outcome.item.world_key, []).append(outcome)
    return dict(sorted(worlds.items()))


def _own(bundle: Path, manifest: str, seed: int) -> bool:
    """Whether the directory is a bundle written from this manifest on its seed."""
    index = bundle / _index_name(bundle)
    if not index.is_file():
        return False
    doc = yaml.safe_load(index.read_text())
    return isinstance(doc, dict) and (doc.get("manifest"), doc.get("seed")) == (manifest, seed)


def plan[I: Item](report: Report[I], root: Path, name: str, *, manifest: str, seed: int, splits: Sequence[str]) -> dict[Path, list[Outcome[I]]]:
    """The bundle directory of every split with its outcomes, in `splits` order, refused for a rejected item or a directory of another manifest or seed."""
    rejected = len(report.outcomes) - report.placed
    if rejected:
        raise ValueError(f"{rejected} of {len(report.outcomes)} {report.label}s rejected, a suite needs every one placed")
    planned = {root / f"{name}_{split}": [outcome for outcome in report.outcomes if outcome.split == split] for split in splits}
    foreign = [bundle for bundle, outcomes in planned.items() if outcomes and bundle.exists() and not _own(bundle, manifest, seed)]
    if foreign:
        raise ValueError(f"{foreign[0]} was not written from the manifest {manifest} on seed {seed}, remove it or name the suite differently")
    return planned


def _fill[I: Item](bundle: Path, report: Report[I], outcomes: Sequence[Outcome[I]], *, manifest: str, seed: int, stages: Stages[I], readme: Readme[I]) -> None:
    worlds = by_world(outcomes)
    for key in worlds:
        built = report.built[key]
        save_generated(WorldView(bundle / "worlds" / key), built.world, built.files)
    suite = stages(outcomes)
    split = outcomes[0].split or ""
    (bundle / "suite.yaml").write_text(_dump({"stages": suite}))
    (bundle / "README.md").write_text(readme(bundle.name, split, outcomes))
    index = {
        "manifest": manifest,
        "seed": seed,
        "split": split,
        "worlds": list(worlds),
        "stages": len(suite),
        "checksum": checksum(bundle),
        f"{report.label}s": [outcome.to_dict(report.label) for outcome in outcomes],
    }
    (bundle / _index_name(bundle)).write_text(_dump(index))


def write[I: Item](report: Report[I], root: Path, name: str, *, manifest: str, seed: int, splits: Sequence[str], stages: Stages[I], readme: Readme[I]) -> dict[Path, list[Outcome[I]]]:
    """Build every split's bundle in a staging directory under `root`, then swap each in, replacing or removing only bundles of this manifest and seed."""
    planned = plan(report, root, name, manifest=manifest, seed=seed, splits=splits)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=root))
    try:
        for bundle, outcomes in planned.items():
            if outcomes:
                _fill(staging / "new" / bundle.name, report, outcomes, manifest=manifest, seed=seed, stages=stages, readme=readme)
        (staging / "old").mkdir()
        for bundle, outcomes in planned.items():
            if bundle.exists() and _own(bundle, manifest, seed):
                bundle.rename(staging / "old" / bundle.name)
            if outcomes:
                (staging / "new" / bundle.name).rename(bundle)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {bundle: outcomes for bundle, outcomes in planned.items() if outcomes}
