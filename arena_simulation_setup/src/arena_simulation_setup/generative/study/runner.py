"""One study run: every item on the world it names, one world per configuration and seed, placed, packed and written."""

from __future__ import annotations

import typing
from collections.abc import Callable, Sequence
from pathlib import Path

import attrs
import yaml

from arena_simulation_setup.generative.core.axes import Scalar
from arena_simulation_setup.generative.core.generated import Generated, Rejected, pack
from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
from arena_simulation_setup.generative.world.layout import Layout
from arena_simulation_setup.generative.world.world_generator import save_generated
from arena_simulation_setup.tree.World import World, WorldIdentifier, WorldView


class Item(typing.Protocol):
    """One scenario of a study on the world its configuration and seed generate."""

    @property
    def generator(self) -> WorldGeneratorType: ...
    @property
    def world(self) -> dict[str, Scalar]: ...
    @property
    def world_seed(self) -> int: ...
    @property
    def world_key(self) -> str: ...
    @property
    def scenario_key(self) -> str: ...
    def to_dict(self) -> dict[str, Scalar | dict[str, Scalar]]: ...


type Place[I] = Callable[[I], Generated | Rejected]
type Placer[I] = Callable[[str, World, Layout | None], Place[I]]


@attrs.frozen
class Outcome[I: Item]:
    item: I
    split: str | None
    reason: str | None  # None = placed

    def to_dict(self, label: str) -> dict[str, object]:
        entry: dict[str, object] = {"world": self.item.world_key, "scenario": self.item.scenario_key}
        if self.split is not None:
            entry["split"] = self.split
        entry.update({"status": "ok"} if self.reason is None else {"status": "rejected", "reason": self.reason})
        entry[label] = self.item.to_dict()
        return entry


@attrs.frozen
class Built:
    world: World
    files: dict[str, bytes]  # by path relative to the world


@attrs.frozen
class Report[I: Item]:
    label: str
    outcomes: list[Outcome[I]]
    worlds: dict[str, Path]  # world key -> where it was written, empty on a dry run
    built: dict[str, Built]  # world key -> every world with a placed item, also on a dry run
    splits: tuple[str, ...] = ()

    @property
    def placed(self) -> int:
        return sum(o.reason is None for o in self.outcomes)

    def placed_worlds(self) -> set[str]:
        return {o.item.world_key for o in self.outcomes if o.reason is None}

    def index(self, name: str) -> dict[str, object]:
        doc: dict[str, object] = {"manifest": name, "placed": self.placed, "rejected": len(self.outcomes) - self.placed}
        if self.splits:
            doc["worlds"] = {split: sorted({o.item.world_key for o in self.outcomes if o.split == split}) for split in self.splits}
        doc[f"{self.label}s"] = [o.to_dict(self.label) for o in self.outcomes]
        return doc


def view(key: str, out: Path | None) -> WorldView:
    """Where world `key` is written: under `out`, else the writable world root."""
    return WorldView(out / key) if out is not None else WorldIdentifier(key).resolve_write_sync()


def run[I: Item](
    items: Sequence[I],
    placer: Placer[I],
    *,
    label: str,
    out: Path | None = None,
    write: bool = False,
    split: Callable[[str], str] | None = None,
    splits: tuple[str, ...] = (),
) -> Report[I]:
    """Generate every item, one world per world key, writing a world with a placed item only with `write`."""
    groups: dict[str, list[I]] = {}
    for item in items:
        groups.setdefault(item.world_key, []).append(item)
    outcomes: list[Outcome[I]] = []
    worlds: dict[str, Path] = {}
    built: dict[str, Built] = {}
    for key, members in groups.items():
        where = split(key) if split is not None else None
        first = members[0]
        generator = WorldGenerator(first.generator, dict(first.world), first.world_seed)
        try:
            world = generator.compute()
        except ValueError as exc:
            outcomes.extend(Outcome(item=m, split=where, reason=f"world: {exc}") for m in members)
            continue
        files: dict[str, bytes] = dict(generator.files())
        place = placer(key, world, generator.layout())
        for member in members:
            result = place(member)
            if isinstance(result, Rejected):
                outcomes.append(Outcome(item=member, split=where, reason=result.reason))
                continue
            files.update(pack(member.scenario_key, result))
            outcomes.append(Outcome(item=member, split=where, reason=None))
        if any(o.reason is None for o in outcomes if o.item.world_key == key):
            built[key] = Built(world=world, files=files)
            if write:
                worlds[key] = save_generated(view(key, out), world, files)
    return Report(label=label, outcomes=outcomes, worlds=worlds, built=built, splits=splits)


def index_path[I: Item](report: Report[I], name: str, out: Path | None) -> Path | None:
    """`<name>.index.yaml` under `out`, else beside the first written world, None when nothing was written."""
    if out is not None:
        return out / f"{name}.index.yaml"
    first = next(iter(report.worlds.values()), None)
    return first.parent / f"{name}.index.yaml" if first is not None else None


def write_index[I: Item](report: Report[I], name: str, out: Path | None) -> Path | None:
    path = index_path(report, name, out)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(report.index(name), sort_keys=False))
    return path
