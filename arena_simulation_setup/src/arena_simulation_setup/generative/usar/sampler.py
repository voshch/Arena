"""Tasks of a USAR manifest: every world on the grid with its tasks cycling through the task axes."""

from __future__ import annotations

import hashlib
import itertools
import json

import attrs

from arena_simulation_setup.generative.egobench.manifest import Scalar, axis_values
from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType

from .manifest import CallLevel, Hiding, Manifest


@attrs.frozen
class Task:
    world: dict[str, Scalar]
    world_seed: int
    index: int
    corners: int
    call_level: CallLevel
    hiding: Hiding

    @property
    def world_key(self) -> str:
        """Name shared by every task on the same generated world."""
        digest = hashlib.sha1(json.dumps([self.world, self.world_seed], sort_keys=True).encode()).hexdigest()[:8]
        return f"usar_rooms_{digest}"

    @property
    def scenario_key(self) -> str:
        return f"t{self.index:02d}__c{self.corners}__{self.call_level.value}__{self.hiding.value}"

    def to_dict(self) -> dict[str, Scalar | dict[str, Scalar]]:
        return {
            "world": dict(self.world),
            "world_seed": self.world_seed,
            "index": self.index,
            "corners": self.corners,
            "call_level": self.call_level.value,
            "hiding": self.hiding.value,
        }


def tasks(manifest: Manifest) -> list[Task]:
    """Every task of the manifest, `tasks_per_world` per world, the task axes taken in turn across worlds, corners fastest."""
    names = list(manifest.world)
    combos = [(corners, level, hiding) for hiding, level, corners in itertools.product(manifest.hiding, manifest.call_levels, manifest.corners)]
    turn = itertools.cycle(combos)
    out: list[Task] = []
    for values in itertools.product(*(axis_values(manifest.world[name]) for name in names)):
        world = dict(zip(names, values, strict=True))
        WorldGenerator.config_model(WorldGeneratorType.ROOMS).model_validate(world)
        for variant in range(manifest.world_variants):
            for index in range(manifest.tasks_per_world):
                corners, level, hiding = next(turn)
                out.append(Task(world=world, world_seed=manifest.seed * 1000 + variant, index=index, corners=corners, call_level=level, hiding=hiding))
    return out


def held_out(keys: list[str], count: int) -> set[str]:
    """The world keys held out for testing: the first `count` in key order, so the split follows from the keys alone."""
    return set(sorted(set(keys))[:count])
