"""What every study manifest shares: its name and seed, the world seeds and keys it derives, and world axes checked against their generator."""

from __future__ import annotations

import hashlib
import itertools
import json
import typing
from collections.abc import Iterable, Mapping
from pathlib import Path

import pydantic
import yaml

from arena_simulation_setup.generative.core.axes import Axis, Scalar, axis_values
from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType

VARIANTS = 1000


def digest(*parts: object) -> str:
    """Eight hex digits naming what `parts` generate."""
    return hashlib.sha1(json.dumps(list(parts), sort_keys=True).encode()).hexdigest()[:8]


def check_world_fields(generator: WorldGeneratorType, keys: Iterable[str]) -> None:
    """Raise on a key that is no configuration field of the world generator."""
    fields = WorldGenerator.config_model(generator).model_fields
    unknown = sorted(set(keys) - set(fields))
    if unknown:
        raise ValueError(f"the {generator.value} generator has no configuration field {', '.join(unknown)}, it has {', '.join(sorted(fields))}")


def configurations(axes: Mapping[str, Axis]) -> list[dict[str, Scalar]]:
    """Every configuration on the grid of the axes, the last axis fastest."""
    names = list(axes)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*(axis_values(axes[name]) for name in names))]


class StudyManifest(pydantic.BaseModel):
    """A study's name and seed."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str
    seed: int = pydantic.Field(0, ge=0)

    def world_seed(self, variant: int) -> int:
        return self.seed * VARIANTS + variant

    @classmethod
    def load(cls, path: Path) -> typing.Self:
        return cls.model_validate(yaml.safe_load(path.read_text()))
