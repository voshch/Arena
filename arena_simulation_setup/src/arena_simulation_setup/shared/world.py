from __future__ import annotations

import typing

import attrs
import numpy as np

from arena_simulation_setup.tree.assets.Material import Material, MaterialIdentifier
from arena_simulation_setup.utils.geometry import Position

from .entities import Named
from .semantics import SemanticCfg, parse_semantics


def _activation_distance_converter(x: float | typing.Sequence[float]) -> tuple[float, float]:
    if isinstance(x, (int, float)):
        return (float(x), float(x))
    a, b = x
    return (float(a), float(b))


@attrs.define
class Elevator(Named):
    position: Position = attrs.field(converter=Position.converter)
    size: list[float] = attrs.field(factory=lambda: [2.0, 2.0, 2.0])
    door_side: typing.Literal['+x', '-x', '+y', '-y'] = '+x'
    material: MaterialIdentifier = attrs.field(converter=MaterialIdentifier.converter, default=Material.default('elevator'))
    destination: str = attrs.field(default="")
    # must cover v_max * transition_time plus latency so the door is fully open before arrival
    activation_distance: float = 3.0
    transition_time: float = 1.0
    hold_time: float = 2.0
    travel_time: float = 3.0
    accept_outside_calls: bool = True
    recall_on: str | None = None
    semantics: list[SemanticCfg] = attrs.field(factory=list, converter=parse_semantics)

    def cabin_corners(self) -> list[Position]:
        cx, cy = self.position.x, self.position.y
        hw, hh = self.size[0] / 2.0, self.size[1] / 2.0
        return [
            Position(cx - hw, cy - hh),
            Position(cx + hw, cy - hh),
            Position(cx + hw, cy + hh),
            Position(cx - hw, cy + hh),
        ]


@attrs.define
class Door(Named):
    start: Position = attrs.field(converter=Position.converter)
    end: Position = attrs.field(converter=Position.converter)
    kind: typing.Literal['sliding', 'hinged', 'teleport', 'sliding_top'] = 'sliding'
    width: float = 0.1
    height: float = attrs.field(default=2.0)
    material: MaterialIdentifier = attrs.field(converter=MaterialIdentifier.converter, default=Material.default('door'))
    activation_distance: tuple[float, float] = attrs.field(
        converter=_activation_distance_converter,
        default=(3.0, 3.0),
    )
    transition_time: float = 1.0
    hold_time: float = 2.0
    semantics: list[SemanticCfg] = attrs.field(factory=list, converter=parse_semantics)

    @property
    def corners(self) -> list[Position]:
        direction = np.array(list(self.end)) - np.array(list(self.start))
        direction = direction / np.linalg.norm(direction)
        perp = np.array([-direction[1], direction[0], 0])
        projected_half_width = Position(*(self.width / 2 * perp))
        return [
            self.start + projected_half_width,
            self.start - projected_half_width,
            self.end - projected_half_width,
            self.end + projected_half_width,
        ]


@attrs.define
class Floor(Named):
    pos: Position = attrs.field(converter=Position.converter)
    x_length: float = attrs.field(converter=float, default=20.0)
    y_length: float = attrs.field(converter=float, default=20.0)
    material: MaterialIdentifier = attrs.field(converter=MaterialIdentifier.converter, default=Material.default('floor'))


@attrs.define
class Ceiling(Named):
    pos: Position = attrs.field(converter=Position.converter)
    x_length: float = attrs.field(converter=float, default=20.0)
    y_length: float = attrs.field(converter=float, default=20.0)
    z: float = attrs.field(converter=float, default=2.0)
    cast_shadows: bool = attrs.field(default=False)
    material: MaterialIdentifier = attrs.field(converter=MaterialIdentifier.converter, default=Material.default('ceiling'))


@attrs.define
class Schedule(Named):
    """Standalone time-windowed semantic entity (kind = schedule), no geometry."""

    semantics: list[SemanticCfg] = attrs.field(factory=list, converter=parse_semantics)


@attrs.define
class Signal(Named):
    """Standalone cycling-phase semantic entity (kind = signal), no geometry."""

    semantics: list[SemanticCfg] = attrs.field(factory=list, converter=parse_semantics)
