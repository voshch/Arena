from __future__ import annotations

import warnings
from pathlib import Path
from types import NotImplementedType
from typing import Union

import attrs
import cattrs
from typing_extensions import Self

from arena_simulation_setup.tree.assets.Human import HumanIdentifier
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.utils.cattrs import (
    ArenaConverter,
    Parseable,
    Serializable,
    converter,
)
from arena_simulation_setup.utils.geometry import Pose, Position, Scale
from arena_simulation_setup.utils.resolution import resolve_zone_point


@attrs.define(auto_attribs=True, kw_only=True)
class Waypoint(Position):
    level_id: str | None = None

    @property
    def position(self) -> Position:
        return self

    @position.setter
    def position(self, value: Position) -> None:
        self.x = value.x
        self.y = value.y
        self.z = value.z

    @classmethod
    def from_any(cls, obj: Union[Position, dict, str, 'Waypoint']) -> 'Waypoint':
        if isinstance(obj, str):
            return cls.from_position(pos=resolve_zone_point(obj))
        if isinstance(obj, Waypoint):
            return obj
        if isinstance(obj, Position):
            return cls.from_position(pos=obj)
        if isinstance(obj, dict):
            # Accept {"position": ..., "level_id": ...}, a flat waypoint dict,
            # or just a Position dict.
            if 'position' in obj:
                pos = obj['position']
                if not isinstance(pos, Position):
                    pos = Position.converter(pos)
                return cls.from_position(pos=pos, level_id=obj.get('level_id'))
            pos = Position.converter({k: obj[k] for k in ('x', 'y', 'z') if k in obj})
            return cls.from_position(pos=pos, level_id=obj.get('level_id'))
        raise TypeError(f"Cannot convert {obj!r} to Waypoint")

    @classmethod
    def from_position(cls, pos: Position, level_id: str | None = None) -> 'Waypoint':
        return cls(x=pos.x, y=pos.y, z=pos.z, level_id=level_id)

    def __add__(self, other: Position) -> Waypoint | NotImplementedType:
        if isinstance(other, Position):
            return type(self).from_position(pos=super().__add__(other), level_id=self.level_id)
        return NotImplemented

    def __radd__(self, other: Position) -> Waypoint | NotImplementedType:
        if isinstance(other, Position):
            return type(self).from_position(pos=other + Position(x=self.x, y=self.y, z=self.z), level_id=self.level_id)
        return NotImplemented


@attrs.define(kw_only=True)
class Named(Parseable, Serializable):
    name: str
    extra: dict = attrs.field(factory=dict)

    @property
    def sim_path(self) -> str:
        return self.extra.get('sim_path', self.name)

    @sim_path.setter
    def sim_path(self, value: str) -> None:
        self.extra['sim_path'] = str(value)

    @classmethod
    def parse(cls, value: dict) -> Self:
        if 'pos' in value:
            value['pose'] = value['pos']
            del value['pos']
        value['extra'] = {**value}
        return ArenaConverter.current().structure_attrs_fromdict(value, cls)

    def serialize(self) -> dict:
        result = cattrs.gen.make_dict_unstructure_fn(type(self), converter, _cattrs_omit_if_default=True)(self)
        for k in attrs.fields(type(self)):
            result.get('extra', {}).pop(k.name, None)
        if not result.get('extra', {}):
            result.pop('extra', None)
        return result


@attrs.define(kw_only=True)
class Entity(Named, Parseable, Serializable):
    pose: Pose = attrs.field(converter=Pose.converter)
    model: ObjectIdentifier = attrs.field(converter=ObjectIdentifier.converter)

    included_from: Path | None = attrs.field(default=None, repr=False)

    def asdict(self, expand_extra: bool = True) -> dict:
        if expand_extra:
            return {
                **self.extra,
                **attrs.asdict(self, filter=lambda a, v: a.name != 'extra'),
            }
        return attrs.asdict(self)


@attrs.define
class Obstacle(Entity):
    scale: Scale | None = None
    level_id: str | None = None


def _waypoints_converter(value: object) -> list[Waypoint]:
    if not isinstance(value, list):
        raise TypeError("waypoints must be a list")
    return [Waypoint.from_any(wp) for wp in value]


@attrs.define
class DynamicObstacle(Entity):
    model: HumanIdentifier = attrs.field(converter=HumanIdentifier.converter)
    waypoints: list[Waypoint] = attrs.field(factory=list, converter=_waypoints_converter)
    velocity: float = attrs.field(converter=float, default=1.0)  # m/s
    level_id: str = ""


@attrs.define
class CustomDynamicObstacle(DynamicObstacle):
    """
    DynamicObstacles but with properties can be define in runtime
    """

    def __getattr__(self, name: str) -> object:
        """
        Allow access to dynamic attributes "attr_name" via self.attr_name
        """
        if name in self.extra:
            return self.extra[name]
        raise AttributeError(f"{name} not found")

    @classmethod
    def parse(cls, value: object) -> Self:
        known_fields = set(f.name for f in attrs.fields(cls))

        if 'pos' in value:
            value['pose'] = value['pos']
            del value['pos']

        known_values = {k: v for k, v in value.items() if k in known_fields}
        custom_fields = {k: v for k, v in value.items() if k not in known_fields}

        warnings.warn("CustomDynamicObstacle.parse is deprecated and will be removed in a future release. Call the constructor directly, e.g., CustomDynamicObstacle(**value).", FutureWarning, stacklevel=2)

        obj = converter.structure_attrs_fromdict(known_values, cls)
        obj.extra.update(custom_fields)
        return obj
