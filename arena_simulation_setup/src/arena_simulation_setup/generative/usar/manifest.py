"""USAR manifest: the rooms worlds of a study, the hiding spots placed on each and how the person calls."""

from __future__ import annotations

import enum
from pathlib import Path

import pydantic
import yaml

from arena_simulation_setup.generative.egobench.manifest import Axis, Span, axis_values
from arena_simulation_setup.generative.scenario.promptgen.profiles import PROFILES
from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType


class CallLevel(enum.Enum):
    WEAK = "weak"
    LOUD = "loud"


class Hiding(enum.Enum):
    """How the person's spot is picked among those at the requested corner count."""

    RANDOM = "random"
    ADVERSARIAL = "adversarial"  # the call comes from where the route does not lead first


class Manifest(pydantic.BaseModel):
    """The dimensions of one USAR study; world axes are a value, a list of values or a stepped span."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str
    seed: int = 0
    world: dict[str, Axis]  # rooms generator configuration axes
    world_variants: int = pydantic.Field(1, ge=1)  # world seeds per world configuration
    spots_per_world: int = pydantic.Field(5, ge=1)
    test_worlds: int = pydantic.Field(0, ge=0)  # worlds held out from tuning and training
    corners: list[int] = pydantic.Field(min_length=1)  # turns on the route from the robot to the person
    call_levels: list[CallLevel] = pydantic.Field(default_factory=lambda: [CallLevel.WEAK, CallLevel.LOUD], min_length=1)
    hiding: list[Hiding] = pydantic.Field(default_factory=lambda: [Hiding.RANDOM], min_length=1)
    call_db: dict[CallLevel, float] = pydantic.Field(default_factory=lambda: {CallLevel.WEAK: 50.0, CallLevel.LOUD: 70.0})
    call_asset: str = "greeting"
    call_period_s: float = pydantic.Field(8.0, gt=0)
    call_on_s: float = pydantic.Field(2.0, gt=0)
    agent_type: str = "adult"

    @pydantic.model_validator(mode="after")
    def _consistent(self) -> Manifest:
        fields = WorldGenerator.config_model(WorldGeneratorType.ROOMS).model_fields
        unknown = sorted(set(self.world) - set(fields))
        if unknown:
            raise ValueError(f"the rooms generator has no configuration field {', '.join(unknown)}, it has {', '.join(sorted(fields))}")
        for name, axis in self.world.items():
            if isinstance(axis, Span) and axis.step is None:
                raise ValueError(f"world axis {name} needs a step on span [{axis.min}, {axis.max}]")
        if any(count < 0 for count in self.corners):
            raise ValueError(f"corner counts are 0 or more, got {self.corners}")
        if len(set(self.call_levels)) != len(self.call_levels):
            raise ValueError(f"call_levels repeats a level, got {', '.join(level.value for level in self.call_levels)}")
        missing = sorted(level.value for level in self.call_levels if level not in self.call_db)
        if missing:
            raise ValueError(f"call_db has no level for {', '.join(missing)}")
        if self.call_on_s >= self.call_period_s:
            raise ValueError(f"call_on_s {self.call_on_s} must be shorter than call_period_s {self.call_period_s}")
        if self.agent_type not in PROFILES:
            raise ValueError(f"no pedestrian profile {self.agent_type}, there are {', '.join(PROFILES)}")
        if self.test_worlds >= self.world_count:
            raise ValueError(f"{self.test_worlds} test worlds leave none of the {self.world_count} worlds for tuning")
        return self

    @property
    def world_count(self) -> int:
        count = self.world_variants
        for axis in self.world.values():
            count *= len(axis_values(axis))
        return count

    @classmethod
    def load(cls, path: Path) -> Manifest:
        return cls.model_validate(yaml.safe_load(path.read_text()))
