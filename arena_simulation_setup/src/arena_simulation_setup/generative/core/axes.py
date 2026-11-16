"""Manifest axes: a value, a list of values or a numeric span, gridded or drawn."""

from __future__ import annotations

import math
import random

import pydantic

Scalar = bool | int | float | str


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
