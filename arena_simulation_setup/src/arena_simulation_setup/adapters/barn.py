"""BARN challenge worlds by number, built by the barn world generator from the upstream world file."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

from arena_assets.importers import IMPORTERS, Attribution, Importer, import_root
from arena_assets.importers.providers import Http, Provider
from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
from arena_simulation_setup.generative.world.barn import BASE_URL, WORLDS, attribution, fetch_world, world_name
from arena_simulation_setup.generative.world.world_generator import save_generated
from arena_simulation_setup.tree.World import WorldView, derive


def world_index(item: str) -> int:
    if not item.isdigit() or int(item) >= WORLDS:
        raise ValueError(f"no BARN world '{item}', items are the challenge world numbers 0 to {WORLDS - 1}")
    return int(item)


def write_world(index: int, source: Path, out: Path) -> WorldGenerator:
    """Build challenge world index from source/world_<index>.world as out/barn_<index> and return its generator."""
    generator = WorldGenerator(WorldGeneratorType.BARN, {"index": index, "source": str(source)})
    built = generator.compute()
    built.kg = derive(built)
    save_generated(WorldView(out / world_name(index)), built, generator.files())
    return generator


class BARNWorld(Importer):
    meta = IMPORTERS.meta("world", "barn")

    @classmethod
    def default_provider(cls) -> Provider:
        return Http(BASE_URL)

    @classmethod
    def default_out(cls) -> Path:
        return import_root() / "worlds"

    def items(self) -> Iterator[str]:
        return (str(index) for index in range(WORLDS))

    def fetch(self, item: str) -> Path:
        return fetch_world(world_index(item), self.cache, self.provider)

    def convert(self, item: str, src: Path, out: Path) -> str:
        index = world_index(item)
        target = out / world_name(index)
        if target.exists():
            shutil.rmtree(target)
        write_world(index, src.parent, out)
        return world_name(index)

    def attribution(self, item: str) -> Attribution:
        world_index(item)
        return attribution()
