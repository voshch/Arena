from __future__ import annotations

import typing
from collections.abc import Mapping
from pathlib import Path
from types import NotImplementedType
from typing import TYPE_CHECKING, Union

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
from arena_simulation_setup.utils.geometry import Pose, Position, Scale, Vector3
from arena_simulation_setup.utils.resolution import resolve_zone_point

from .light_state import ObjectLightSettings

if TYPE_CHECKING:
    import geometry_msgs.msg
    import shapely

    from arena_simulation_setup.utils.affordance import AffordanceSlot


_WAYPOINT_SOURCES = ('zone', 'elevator')
_WAYPOINT_KEYS = frozenset({'x', 'y', 'z', *_WAYPOINT_SOURCES, 'level', 'dwell'})


def _resolve_elevator_point(name: str) -> Position:
    """A point in the named elevator's cabin, `<level_id>.<name>` accepted."""
    try:
        return resolve_zone_point(name)
    except ValueError:
        if '.' not in name:
            raise
        return resolve_zone_point(name.split('.', 1)[1])


@attrs.define(auto_attribs=True, kw_only=True)
class Waypoint(Position):
    level: str = ''
    zone: str = ''
    dwell: float = attrs.field(default=0.0, converter=float, validator=attrs.validators.ge(0.0))
    elevator: str | None = None

    @property
    def position(self) -> Position:
        return self

    @position.setter
    def position(self, value: Position) -> None:
        self.x = value.x
        self.y = value.y
        self.z = value.z

    @classmethod
    def parse(cls, value: Union[geometry_msgs.msg.Point, Position, Mapping, str, typing.Sequence[float]]) -> Self:
        return cls.from_any(value)

    @classmethod
    def from_any(cls, obj: Union[geometry_msgs.msg.Point, Position, Mapping, str, typing.Sequence[float]]) -> Self:
        """A waypoint from a zone name, a position, `[x, y[, z]]` or a mapping with x/y, zone or elevator."""
        if isinstance(obj, str):
            return cls.from_position(pos=resolve_zone_point(obj), zone=obj)
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, Position):
            return cls.from_position(pos=obj)
        if isinstance(obj, Mapping):
            unknown = set(obj) - _WAYPOINT_KEYS
            if unknown:
                raise ValueError(f'waypoint has unknown keys {sorted(unknown)}')
            sources = [key for key in _WAYPOINT_SOURCES if key in obj] + (['x/y'] if {'x', 'y'} & set(obj) else [])
            if len(sources) != 1:
                raise ValueError(f'waypoint needs exactly one of zone, elevator or x/y, got {dict(obj)!r}')
            zone = ''
            elevator = None
            if 'zone' in obj:
                zone = str(obj['zone'])
                pos = resolve_zone_point(zone)
            elif 'elevator' in obj:
                elevator = str(obj['elevator'])
                pos = _resolve_elevator_point(elevator)
            else:
                pos = Position.parse({k: obj[k] for k in ('x', 'y', 'z') if k in obj})
            return cls(x=pos.x, y=pos.y, z=pos.z, level=str(obj.get('level', '')), zone=zone, dwell=obj.get('dwell', 0.0), elevator=elevator)
        return cls.from_position(pos=Position.parse(obj))

    @classmethod
    def from_position(cls, pos: Position, level: str = '', zone: str = '') -> Self:
        return cls(x=pos.x, y=pos.y, z=pos.z, level=level, zone=zone)

    def _moved(self, pos: Position) -> Waypoint:
        return attrs.evolve(self, x=pos.x, y=pos.y, z=pos.z)

    def __add__(self, other: Vector3) -> Waypoint | NotImplementedType:
        if isinstance(other, Position):
            return self._moved(Position(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z))
        return NotImplemented

    def __radd__(self, other: Position) -> Waypoint | NotImplementedType:
        if isinstance(other, Position):
            return self._moved(other + Position(x=self.x, y=self.y, z=self.z))
        return NotImplemented


@attrs.define(kw_only=True)
class Named(Parseable, Serializable):
    name: str
    extra: dict = attrs.field(factory=dict, eq=False)

    @property
    def sim_path(self) -> str:
        return self.extra.get('sim_path', self.name)

    @sim_path.setter
    def sim_path(self, value: str) -> None:
        self.extra['sim_path'] = str(value)

    @property
    def placement_zone(self) -> str:
        """The zone name a parsed `pose` or `position` was sampled from, its point then lies in the map frame."""
        return self.extra.get('placement_zone', '')

    @classmethod
    def parse(cls, value: dict) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError(f'{cls.__name__} must be a mapping, got {value!r}')
        unknown = set(value) - {field.name for field in attrs.fields(cls)} | ({'extra'} & set(value))
        if unknown:
            raise ValueError(f"{cls.__name__.lower()} {value.get('name')!r} has unknown keys {sorted(unknown)}")
        result = ArenaConverter.current().structure_attrs_fromdict(dict(value), cls)
        zones = [value[key] for key in ('pose', 'position') if isinstance(value.get(key), str)]
        if zones:
            result.extra['placement_zone'] = zones[0]
        return result

    def serialize(self) -> dict:
        result = cattrs.gen.make_dict_unstructure_fn(type(self), converter, _cattrs_omit_if_default=True)(self)
        result.pop('extra', None)
        return result


@attrs.define(kw_only=True)
class Entity(Named, Parseable, Serializable):
    pose: Pose = attrs.field(converter=Pose.converter)
    model: ObjectIdentifier = attrs.field(converter=ObjectIdentifier.converter)
    level: str = ''

    included_from: Path | None = attrs.field(default=None, repr=False)

    @classmethod
    def parse(cls, value: dict) -> Self:
        if isinstance(value, Mapping) and 'included_from' in value:
            raise ValueError(f"{cls.__name__.lower()} {value.get('name')!r} has unknown keys ['included_from']")
        return super().parse(value)


HUMANSIM_OBJECT_KEYS = frozenset({'type', 'capacity', 'satisfies', 'interaction_radius', 'formation', 'seats'})


def _humansim_object(value: Mapping[str, object] | None) -> dict:
    value = dict(value or {})
    unknown = set(value) - HUMANSIM_OBJECT_KEYS
    if unknown:
        raise ValueError(f'humansim block has unknown keys {sorted(unknown)}, expected some of {sorted(HUMANSIM_OBJECT_KEYS)}')
    return value


@attrs.define
class Obstacle(Entity):
    scale: Scale | None = None
    light: ObjectLightSettings | None = None
    humansim: dict = attrs.field(factory=dict, converter=_humansim_object)

    async def footprint(self) -> shapely.Polygon | None:
        """Map-frame 2D footprint from the asset's annotated bounds, None when unannotated."""
        import shapely
        import shapely.affinity

        try:
            view = await self.model.resolve()
        except FileNotFoundError:
            return None
        bounds = view.bounds
        if bounds is None:
            return None
        poly = shapely.Polygon(bounds)
        poly = shapely.affinity.rotate(poly, self.pose.orientation.to_yaw(), origin=(0, 0), use_radians=True)
        return shapely.affinity.translate(poly, xoff=self.pose.position.x, yoff=self.pose.position.y)

    async def affordance_slots(self) -> list[AffordanceSlot]:
        """Placed affordance slots of the asset in the obstacle-local frame, empty when unannotated."""
        try:
            view = await self.model.resolve()
        except FileNotFoundError:
            return []
        return view.affordances


converter.register_structure_hook_func(lambda t: t == dict | str | None, lambda v, _t: v)


def _waypoints_converter(value: object) -> list[Waypoint]:
    if not isinstance(value, list):
        raise TypeError("waypoints must be a list")
    return [Waypoint.from_any(wp) for wp in value]


WAYPOINT_MODES = ('repeat', 'reverse', 'once', 'random')


@attrs.define
class DynamicObstacle(Entity):
    model: HumanIdentifier = attrs.field(converter=HumanIdentifier.converter)
    waypoints: list[Waypoint] = attrs.field(factory=list, converter=_waypoints_converter)
    waypoint_mode: str = attrs.field(default='repeat', validator=attrs.validators.in_(WAYPOINT_MODES))
    velocity: float = attrs.field(converter=float, default=1.0)  # m/s
    agent: dict | str | None = None
    hunav: dict = attrs.field(factory=dict)
    spawn_delay: float = attrs.field(default=0.0, converter=float, validator=attrs.validators.ge(0.0))
    semantic_alias: dict[str, str] = attrs.field(factory=dict)
    track: str = ''
    prespawn: bool = False


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
