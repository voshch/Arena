"""Tasks of a USAR manifest: every world on the grid with its spots cycling through the spot axes, each spot once per call level."""

from __future__ import annotations

import itertools

import attrs

from arena_simulation_setup.generative.core.axes import Scalar
from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType

from .manifest import CallLevel, Hiding, Manifest, world_key

TEST = "test"
TUNE = "tune"


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
        return world_key(self.world, self.world_seed)

    @property
    def spot_key(self) -> str:
        """Name shared by the call levels of one hiding spot."""
        return f"t{self.index:02d}__c{self.corners}__{self.hiding.value}"

    @property
    def scenario_key(self) -> str:
        return f"{self.spot_key}__{self.call_level.value}"

    def to_dict(self) -> dict[str, Scalar | dict[str, Scalar]]:
        return {
            "world": dict(self.world),
            "world_seed": self.world_seed,
            "index": self.index,
            "spot": self.spot_key,
            "corners": self.corners,
            "call_level": self.call_level.value,
            "hiding": self.hiding.value,
        }


def tasks(manifest: Manifest) -> list[Task]:
    """Every task of the manifest: `spots_per_world` spots per world, each group taking its spot axes in turn across its worlds, corners fastest, each spot once per call level."""
    out: list[Task] = []
    for group in manifest.worlds:
        turn = itertools.cycle([(corners, hiding) for hiding, corners in itertools.product(manifest.hiding, group.corners)])
        for world in group.configurations():
            WorldGenerator.config_model(WorldGeneratorType.ROOMS).model_validate(world)
            for variant in range(group.world_variants):
                for index in range(manifest.spots_per_world):
                    corners, hiding = next(turn)
                    out.extend(Task(world=world, world_seed=manifest.world_seed(variant), index=index, corners=corners, call_level=level, hiding=hiding) for level in manifest.call_levels)
    return out


def held_out(manifest: Manifest) -> set[str]:
    """The world keys held out for testing: of each group its first `test_worlds` in key order, so the split follows from the keys alone."""
    return {key for group in manifest.worlds for key in sorted(manifest.keys(group))[: group.test_worlds]}
