from __future__ import annotations

import math
import typing

import attrs
import cattrs
import numpy as np

from arena_simulation_setup.tree.assets.Material import Material, MaterialIdentifier
from arena_simulation_setup.utils.cattrs import Parseable, Serializable, converter
from arena_simulation_setup.utils.geometry import Position

from .entities import Named
from .light_state import ObjectLightSettings, check_light_state
from .semantics import SemanticCfg, parse_semantics

CABIN_DOOR_INSET = 0.05
CABIN_DOOR_WIDTH = 0.05


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

    def boarding_radius(self, agent_radius: float) -> float:
        """Largest distance from the cabin center at which a disc of agent_radius stays inside the cabin and clear of its door."""
        return min(self.size[0], self.size[1]) / 2.0 - CABIN_DOOR_INSET - CABIN_DOOR_WIDTH / 2.0 - agent_radius

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


def _optional_position(value: object) -> Position | None:
    if value is None:
        return None
    return Position.instance_or(Position.parse)(value)


@attrs.define
class Sound(Named):
    """Standalone audible semantic entity (kind = sound), placed in the world."""

    asset_id: str = attrs.field(converter=lambda value: str(value).strip())
    position: Position | None = attrs.field(converter=_optional_position, default=None)
    entity_ref: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    frame: str = attrs.field(converter=lambda value: str(value).strip().strip('/'), default='')
    offset: Position = attrs.field(converter=Position.converter, factory=lambda: Position(0.0, 0.0, 0.0))
    level: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    loop: bool = True
    reference_distance_m: float = attrs.field(converter=float, default=1.0)
    semantics: list[SemanticCfg] = attrs.field(factory=list, converter=parse_semantics)

    def __attrs_post_init__(self) -> None:
        if not self.name or ':' in self.name:
            raise ValueError("sound name must be non-empty and contain no ':'")
        if not self.asset_id or ':' in self.asset_id:
            raise ValueError(f"sound {self.name!r} asset_id must be non-empty and contain no ':'")
        if sum((bool(self.entity_ref), self.position is not None, bool(self.frame))) != 1:
            raise ValueError(f"sound {self.name!r} requires exactly one of position, entity_ref or frame")
        if self.level and self.position is None:
            raise ValueError(f"sound {self.name!r} takes level only with a direct position")
        if self.reference_distance_m <= 0.0:
            raise ValueError(f"sound {self.name!r} reference_distance_m must be positive")


@attrs.frozen
class LightFixture:
    """Emitter shape of one fixture type."""

    shape: str
    x_length: float = 0.0
    y_length: float = 0.0
    radius: float = 0.0
    cone_deg: float = 180.0

    @property
    def ambient(self) -> bool:
        return self.shape in ('dome', 'distant')

    @property
    def area(self) -> float:
        if self.shape == 'rect':
            return self.x_length * self.y_length
        if self.shape == 'disk':
            return math.pi * self.radius**2
        if self.shape == 'sphere':
            return 4.0 * math.pi * self.radius**2
        return 0.0


LIGHT_FIXTURES: dict[str, LightFixture] = {
    'panel': LightFixture(shape='rect', x_length=0.6, y_length=0.6),
    'tube': LightFixture(shape='rect', x_length=1.2, y_length=0.1),
    'downlight': LightFixture(shape='disk', radius=0.1),
    'spot': LightFixture(shape='disk', radius=0.05, cone_deg=40.0),
    'bulb': LightFixture(shape='sphere', radius=0.03),
    'dome': LightFixture(shape='dome'),
    'sun': LightFixture(shape='distant'),
}

_LIGHT_DOWN = (0.0, 0.0, -1.0)
_UNDIRECTED_FIXTURES = ('bulb', 'dome')


def cct_to_rgb(cct_K: float) -> tuple[float, float, float]:
    """Blackbody color of a color temperature in kelvin, each channel in 0..1."""
    t = min(max(float(cct_K), 1000.0), 40000.0) / 100.0
    if t <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        red = 329.698727446 * (t - 60.0) ** -0.1332047592
        green = 288.1221695283 * (t - 60.0) ** -0.0755148492
    if t >= 66.0:
        blue = 255.0
    elif t <= 19.0:
        blue = 0.0
    else:
        blue = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    return (
        min(max(red, 0.0), 255.0) / 255.0,
        min(max(green, 0.0), 255.0) / 255.0,
        min(max(blue, 0.0), 255.0) / 255.0,
    )


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _direction_converter(value: typing.Sequence[float]) -> tuple[float, float, float]:
    x, y, z = value
    return (float(x), float(y), float(z))


@attrs.define
class Light(Named):
    """Standalone light semantic entity (kind = light): one fixture, an ambient light or a ceiling rig."""

    fixture: str = attrs.field(converter=lambda value: str(value).strip())
    position: Position | None = attrs.field(converter=_optional_position, default=None)
    entity_ref: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    frame: str = attrs.field(converter=lambda value: str(value).strip().strip('/'), default='')
    offset: Position = attrs.field(converter=Position.converter, factory=lambda: Position(0.0, 0.0, 0.0))
    direction: tuple[float, float, float] = attrs.field(converter=_direction_converter, default=_LIGHT_DOWN)
    lumens: float | None = attrs.field(converter=_optional_float, default=None)
    lux: float | None = attrs.field(converter=_optional_float, default=None)
    cct_K: float = attrs.field(converter=float, default=6500.0)
    cone_deg: float | None = attrs.field(converter=_optional_float, default=None)
    cast_shadows: bool = False
    light_on: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    lit: bool | None = None
    level: float = attrs.field(converter=float, default=1.0)
    dead_fraction: float = attrs.field(converter=float, default=0.0)
    intrinsic: bool = False
    owner: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    glow: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    rig: bool = False
    fixtures: list[Position] = attrs.field(factory=list)
    fixture_ranks: list[float] = attrs.field(factory=list)

    def __attrs_post_init__(self) -> None:
        if not self.name or ':' in self.name:
            raise ValueError(f"light name {self.name!r} must be non-empty and contain no ':'")
        if self.fixture not in LIGHT_FIXTURES:
            raise ValueError(f'light {self.name!r} has unknown fixture {self.fixture!r}, expected one of {sorted(LIGHT_FIXTURES)}')
        anchors = sum((self.position is not None, bool(self.entity_ref), bool(self.frame)))
        if self.spec.ambient:
            if anchors:
                raise ValueError(f'light {self.name!r} ({self.fixture}) takes no position, entity_ref or frame')
            if self.rig:
                raise ValueError(f'light {self.name!r} ({self.fixture}) cannot be a rig')
            if self.lumens is not None:
                raise ValueError(f'light {self.name!r} ({self.fixture}) takes lux, not lumens')
            if self.lux is None or not self.lux >= 0.0:
                raise ValueError(f'light {self.name!r} ({self.fixture}) requires lux >= 0')
        else:
            if self.lux is not None:
                raise ValueError(f'light {self.name!r} ({self.fixture}) takes lumens, not lux')
            if self.lumens is None or not self.lumens >= 0.0:
                raise ValueError(f'light {self.name!r} ({self.fixture}) requires lumens >= 0')
            if self.rig:
                if anchors:
                    raise ValueError(f'rig light {self.name!r} takes no position, entity_ref or frame')
                if not self.fixtures:
                    raise ValueError(f'rig light {self.name!r} requires at least one fixture position')
                if len(self.fixture_ranks) != len(self.fixtures):
                    raise ValueError(f'rig light {self.name!r} requires one fixture rank per fixture position')
            elif anchors != 1:
                raise ValueError(f'light {self.name!r} requires exactly one of position, entity_ref or frame')
        if not self.rig and (self.fixtures or self.fixture_ranks):
            raise ValueError(f'light {self.name!r} takes fixture positions only as a rig')
        x, y, z = self.direction
        norm = math.sqrt(x * x + y * y + z * z)
        if not norm > 0.0:
            raise ValueError(f'light {self.name!r} direction must be non-zero')
        self.direction = (x / norm, y / norm, z / norm)
        if self.direction != _LIGHT_DOWN and self.fixture in _UNDIRECTED_FIXTURES:
            raise ValueError(f'light {self.name!r} ({self.fixture}) takes no direction')
        if self.cone_deg is not None and (self.fixture != 'spot' or not 0.0 < self.cone_deg < 180.0):
            raise ValueError(f'light {self.name!r} takes cone_deg only as a spot, within 0..180 exclusive')
        check_light_state(f'light {self.name!r}', self.cct_K, self.level, self.dead_fraction, self.light_on, self.lit)
        if self.dead_fraction and not self.rig:
            raise ValueError(f'light {self.name!r} takes dead_fraction only as a rig')

    @property
    def spec(self) -> LightFixture:
        spec = LIGHT_FIXTURES[self.fixture]
        return spec if self.cone_deg is None else attrs.evolve(spec, cone_deg=self.cone_deg)

    @property
    def semantics(self) -> list[SemanticCfg]:
        lit = None if self.light_on else (True if self.lit is None else self.lit)
        result = [
            SemanticCfg(role='predicate', name='lit', value=lit, params={'light_on': self.light_on} if self.light_on else {}),
            SemanticCfg(role='state', name='level', value=self.level),
        ]
        if self.rig:
            result.append(SemanticCfg(role='state', name='dead_fraction', value=self.dead_fraction))
        return result

    def alive(self, dead_fraction: float) -> list[bool]:
        """Per rig fixture, whether it survives the given dead fraction."""
        return [rank >= dead_fraction for rank in self.fixture_ranks]


_OBJECT_LIGHT_KEYS = frozenset({'name', 'fixture', 'offset', 'direction', 'lumens', 'cct_K', 'cone_deg', 'glow'})


def object_light(owner: str, entry: dict, settings: ObjectLightSettings | None) -> Light:
    """Light an object annotation entry declares, named after its owning entity and offset in the object frame."""
    unknown = set(entry) - _OBJECT_LIGHT_KEYS
    if unknown:
        raise ValueError(f'object light of {owner!r} has unknown keys {sorted(unknown)}')
    fixture = str(entry.get('fixture', 'bulb'))
    if fixture not in LIGHT_FIXTURES or LIGHT_FIXTURES[fixture].ambient:
        raise ValueError(f'object light of {owner!r} has fixture {fixture!r}, expected one of {sorted(name for name, spec in LIGHT_FIXTURES.items() if not spec.ambient)}')
    x, y, z = entry.get('offset', (0.0, 0.0, 0.0))
    settings = settings if settings is not None else ObjectLightSettings()
    return Light(
        name=f"{owner}_{entry.get('name', 'light')}",
        fixture=fixture,
        entity_ref=owner,
        offset=Position(x, y, z),
        direction=_LIGHT_DOWN if fixture in _UNDIRECTED_FIXTURES else entry.get('direction', _LIGHT_DOWN),
        lumens=entry.get('lumens'),
        cct_K=entry.get('cct_K', 6500.0),
        cone_deg=entry.get('cone_deg') if fixture == 'spot' else None,
        light_on=settings.light_on,
        lit=settings.lit,
        level=settings.level,
        intrinsic=True,
        glow=entry.get('glow') or '',
    )


@attrs.define
class CeilingLights(Parseable, Serializable):
    """Zone ceiling rig, expanded into one rig Light per zone."""

    fixture: str = attrs.field(converter=lambda value: str(value).strip(), default='panel')
    spacing: float = attrs.field(converter=float, default=2.4)
    lumens: float = attrs.field(converter=float, default=3600.0)
    cct_K: float = attrs.field(converter=float, default=6500.0)
    cast_shadows: bool = False
    light_on: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    lit: bool | None = None
    level: float = attrs.field(converter=float, default=1.0)
    dead_fraction: float = attrs.field(converter=float, default=0.0)

    def __attrs_post_init__(self) -> None:
        if self.fixture not in LIGHT_FIXTURES or LIGHT_FIXTURES[self.fixture].ambient:
            raise ValueError(f'ceiling_lights fixture {self.fixture!r} must be one of {sorted(name for name, spec in LIGHT_FIXTURES.items() if not spec.ambient)}')
        if not self.spacing > 0.0:
            raise ValueError('ceiling_lights spacing must be positive')
        if not self.lumens >= 0.0:
            raise ValueError('ceiling_lights requires lumens >= 0')
        check_light_state('ceiling_lights', self.cct_K, self.level, self.dead_fraction, self.light_on, self.lit)

    def serialize(self) -> dict:
        return cattrs.gen.make_dict_unstructure_fn(type(self), converter, _cattrs_omit_if_default=True)(self)
