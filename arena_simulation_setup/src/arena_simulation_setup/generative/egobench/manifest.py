"""EgoBench manifest: the dimensions a dataset spans, their value ranges and how they are sampled."""

from __future__ import annotations

import enum
import math
import random
import typing
from pathlib import Path

import pydantic
import yaml

from arena_simulation_setup.generative.scenario.promptgen.profiles import PROFILES
from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType

Scalar = bool | int | float | str


class Space(enum.Enum):
    """World type a sample is generated in, named after its world generator."""

    CORRIDOR = "corridor"
    OPEN = "open"

    @property
    def generator(self) -> WorldGeneratorType:
        return WorldGeneratorType(self.value)


class Encounter(enum.Enum):
    """Relative motion of the pedestrian with respect to the robot."""

    ONCOMING = "oncoming"
    CROSSING_LEFT = "crossing_left"
    CROSSING_RIGHT = "crossing_right"
    OVERTAKE = "overtake"  # robot passes a pedestrian walking its way
    OVERTAKEN = "overtaken"  # pedestrian passes the robot from behind
    FOLLOW = "follow"  # robot walks behind the pedestrian
    LEAD = "lead"  # pedestrian walks behind the robot
    STANDING = "standing"


class Pedestrian(enum.Enum):
    """How the pedestrian walks its route."""

    FIXED = "fixed"  # ignores the robot and starts on the clock, the same route under every planner
    REACTIVE = "reactive"  # avoids the robot under the social force model and starts as the robot nears


class Span(pydantic.BaseModel):
    """A numeric range, gridded by `step` or drawn uniformly without one."""

    model_config = pydantic.ConfigDict(extra="forbid")

    min: float
    max: float
    step: float | None = pydantic.Field(None, gt=0)

    @pydantic.model_validator(mode="after")
    def _ordered(self) -> Span:
        if self.max < self.min:
            raise ValueError(f"span max {self.max} is below its min {self.min}")
        return self

    def values(self) -> list[float]:
        if self.step is None:
            raise ValueError(f"span [{self.min}, {self.max}] has no step, so it cannot be gridded")
        count = math.floor((self.max - self.min) / self.step + 1e-9) + 1
        return [round(self.min + i * self.step, 9) for i in range(count)]

    def draw(self, rng: random.Random) -> float:
        if self.step is None:
            return rng.uniform(self.min, self.max)
        return rng.choice(self.values())


Axis = Span | list[Scalar] | Scalar


def axis_values(axis: Axis) -> list[Scalar]:
    """Every value of an axis on a grid."""
    if isinstance(axis, Span):
        return list(axis.values())
    if isinstance(axis, list):
        return list(axis)
    return [axis]


def axis_draw(axis: Axis, rng: random.Random) -> Scalar:
    """One value of an axis drawn at random."""
    if isinstance(axis, Span):
        return axis.draw(rng)
    if isinstance(axis, list):
        return rng.choice(axis)
    return axis


def check_world_fields(space: Space, keys: typing.Iterable[str]) -> None:
    """Raise on a key that is no configuration field of the space's world generator."""
    fields = WorldGenerator.config_model(space.generator).model_fields
    unknown = sorted(set(keys) - set(fields))
    if unknown:
        raise ValueError(f"{space.value} has no configuration field {', '.join(unknown)}, it has {', '.join(sorted(fields))}")


class Manifest(pydantic.BaseModel):
    """The dimensions of one dataset; every axis is a value, a list of values or a span."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str
    seed: int = pydantic.Field(0, ge=0)
    sampling: typing.Literal["grid", "random"] = "grid"
    samples: int = pydantic.Field(0, ge=0)  # 0 = the whole grid, required above 0 for random
    world_variants: int = pydantic.Field(1, ge=1, le=1000)  # world seeds per world configuration
    worlds: dict[Space, dict[str, Axis]]
    encounters: list[Encounter] = pydantic.Field(min_length=1)
    pedestrians: list[Pedestrian] = pydantic.Field(default_factory=lambda: [Pedestrian.FIXED], min_length=1)
    lateral_offset: Axis = 1.2  # m between the robot's route and the pedestrian's lane
    robot_speed: Axis = 0.5  # m/s
    ped_speed: Axis = 1.2  # m/s
    agent_type: Axis = "adult"

    @pydantic.model_validator(mode="after")
    def _consistent(self) -> Manifest:
        if not self.worlds:
            raise ValueError("a manifest needs at least one world type")
        if self.sampling == "random" and self.samples == 0:
            raise ValueError("random sampling needs a sample count above 0")
        for space, axes in self.worlds.items():
            check_world_fields(space, axes)
        for label, axis in (("robot_speed", self.robot_speed), ("ped_speed", self.ped_speed)):
            lowest = axis.min if isinstance(axis, Span) else min(float(v) for v in axis_values(axis))
            if lowest <= 0:
                raise ValueError(f"{label} must be above 0, got {lowest}")
        if isinstance(self.agent_type, Span):
            raise ValueError("agent_type names profiles, it cannot be a span")
        unknown_types = sorted({str(t) for t in axis_values(self.agent_type)} - set(PROFILES))
        if unknown_types:
            raise ValueError(f"no pedestrian profile {', '.join(unknown_types)}, there are {', '.join(PROFILES)}")
        if self.sampling == "grid":
            for axis in self._axes():
                if isinstance(axis, Span) and axis.step is None:
                    raise ValueError(f"grid sampling needs a step on span [{axis.min}, {axis.max}]")
        return self

    def _axes(self) -> list[Axis]:
        return [*(axis for axes in self.worlds.values() for axis in axes.values()), self.lateral_offset, self.robot_speed, self.ped_speed, self.agent_type]

    @classmethod
    def load(cls, path: Path) -> Manifest:
        return cls.model_validate(yaml.safe_load(path.read_text()))
