"""USAR manifest: the rooms worlds of a study, the hiding spots placed on each and how the person calls."""

from __future__ import annotations

import enum
import hashlib
import itertools
import json
from collections.abc import Mapping
from pathlib import Path

import pydantic
import yaml

from arena_simulation_setup.generative.egobench.manifest import Axis, Scalar, Span, axis_values
from arena_simulation_setup.generative.scenario.promptgen.profiles import PROFILES
from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType


class CallLevel(enum.Enum):
    WEAK = "weak"
    LOUD = "loud"


class Hiding(enum.Enum):
    """How the person's spot is picked among those at the requested corner count."""

    RANDOM = "random"
    ADVERSARIAL = "adversarial"  # the call comes from where the route does not lead first


SHORT_FORM = ("world", "world_variants", "corners", "test_worlds")


def world_key(world: Mapping[str, Scalar], world_seed: int) -> str:
    """Name of the world generated from one rooms configuration and seed."""
    digest = hashlib.sha1(json.dumps([dict(world), world_seed], sort_keys=True).encode()).hexdigest()[:8]
    return f"usar_rooms_{digest}"


class Group(pydantic.BaseModel):
    """One grid of rooms worlds with the corner counts asked of them and how many of them are held out."""

    model_config = pydantic.ConfigDict(extra="forbid")

    world: dict[str, Axis]  # rooms generator configuration axes
    world_variants: int = pydantic.Field(1, ge=1)  # world seeds per world configuration
    corners: list[int] = pydantic.Field(min_length=1)  # turns on the route from the robot to the person
    test_worlds: int = pydantic.Field(0, ge=0)  # worlds held out from tuning and training

    @pydantic.model_validator(mode="after")
    def _consistent(self) -> Group:
        fields = WorldGenerator.config_model(WorldGeneratorType.ROOMS).model_fields
        unknown = sorted(set(self.world) - set(fields))
        if unknown:
            raise ValueError(f"the rooms generator has no configuration field {', '.join(unknown)}, it has {', '.join(sorted(fields))}")
        for name, axis in self.world.items():
            if isinstance(axis, Span) and axis.step is None:
                raise ValueError(f"world axis {name} needs a step on span [{axis.min}, {axis.max}]")
        if any(count < 0 for count in self.corners):
            raise ValueError(f"corner counts are 0 or more, got {self.corners}")
        if self.test_worlds > self.world_count:
            raise ValueError(f"{self.test_worlds} test worlds exceed the {self.world_count} worlds of the group")
        return self

    def configurations(self) -> list[dict[str, Scalar]]:
        """Every rooms configuration on the grid of the axes, the last axis fastest."""
        names = list(self.world)
        return [dict(zip(names, values, strict=True)) for values in itertools.product(*(axis_values(self.world[name]) for name in names))]

    @property
    def world_count(self) -> int:
        return len(self.configurations()) * self.world_variants


class Manifest(pydantic.BaseModel):
    """The dimensions of one USAR study; world axes are a value, a list of values or a stepped span."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str
    seed: int = 0
    worlds: list[Group] = pydantic.Field(min_length=1)
    spots_per_world: int = pydantic.Field(5, ge=1)
    call_levels: list[CallLevel] = pydantic.Field(default_factory=lambda: [CallLevel.WEAK, CallLevel.LOUD], min_length=1)
    hiding: list[Hiding] = pydantic.Field(default_factory=lambda: [Hiding.RANDOM], min_length=1)
    call_db: dict[CallLevel, float] = pydantic.Field(default_factory=lambda: {CallLevel.WEAK: 50.0, CallLevel.LOUD: 70.0})
    call_asset: str = "greeting"
    call_period_s: float = pydantic.Field(8.0, gt=0)
    call_on_s: float = pydantic.Field(2.0, gt=0)
    agent_type: str = "adult"

    @pydantic.model_validator(mode="before")
    @classmethod
    def _short_form(cls, data: object) -> object:
        """A manifest that gives one group's fields at the top level is that one group."""
        if not isinstance(data, dict):
            return data
        short = {key: data[key] for key in SHORT_FORM if key in data}
        if not short:
            return data
        if "worlds" in data:
            raise ValueError("give the worlds either as the short form world or as the list worlds, not both")
        return {**{key: value for key, value in data.items() if key not in SHORT_FORM}, "worlds": [short]}

    @pydantic.model_validator(mode="after")
    def _consistent(self) -> Manifest:
        if len(set(self.call_levels)) != len(self.call_levels):
            raise ValueError(f"call_levels repeats a level, got {', '.join(level.value for level in self.call_levels)}")
        missing = sorted(level.value for level in self.call_levels if level not in self.call_db)
        if missing:
            raise ValueError(f"call_db has no level for {', '.join(missing)}")
        if self.call_on_s >= self.call_period_s:
            raise ValueError(f"call_on_s {self.call_on_s} must be shorter than call_period_s {self.call_period_s}")
        if self.agent_type not in PROFILES:
            raise ValueError(f"no pedestrian profile {self.agent_type}, there are {', '.join(PROFILES)}")
        seen: set[str] = set()
        for group in self.worlds:
            listed = self.keys(group)
            keys = set(listed)
            if len(keys) < len(listed):
                raise ValueError(f"world {sorted(key for key in keys if listed.count(key) > 1)[0]} is twice in a group, an axis repeats a value")
            repeated = sorted(seen & keys)
            if repeated:
                raise ValueError(f"world {repeated[0]} is in two groups")
            seen |= keys
        return self

    def world_seed(self, variant: int) -> int:
        return self.seed * 1000 + variant

    def keys(self, group: Group) -> list[str]:
        """The world keys of a group, configurations in grid order and each at every variant."""
        return [world_key(world, self.world_seed(variant)) for world in group.configurations() for variant in range(group.world_variants)]

    @classmethod
    def load(cls, path: Path) -> Manifest:
        return cls.model_validate(yaml.safe_load(path.read_text()))
