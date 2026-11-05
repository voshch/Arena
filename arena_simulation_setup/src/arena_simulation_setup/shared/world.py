from __future__ import annotations

import math
import typing

import attrs
import cattrs
import numpy as np

from arena_simulation_setup.tree.assets.Material import Material, MaterialIdentifier
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.utils.cattrs import Parseable, Serializable, converter
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position, Vector3

from .entities import Named
from .light_state import ObjectLightSettings, check_light_state
from .semantics import SemanticCfg, parse_semantics
from .walls import Wall

CABIN_DOOR_INSET = 0.05
CABIN_DOOR_WIDTH = 0.05
CABIN_DOOR_THRESHOLD = 0.6


ActivationDistance = typing.NewType('ActivationDistance', tuple[float, float])


_POCKET_OVERLAP_M = 0.01


def _activation_distance_converter(x: float | typing.Sequence[float]) -> ActivationDistance:
    if isinstance(x, (int, float)):
        return ActivationDistance((float(x), float(x)))
    a, b = x
    return ActivationDistance((float(a), float(b)))


converter.register_structure_hook(ActivationDistance, lambda value, _: _activation_distance_converter(value))


def _call_mode_validator(_instance: object, _attribute: object, value: str) -> None:
    if value not in ('level', 'edge'):
        raise ValueError(f"elevator call_mode must be 'level' or 'edge', got {value!r}")


def _optional_capacity(value: object) -> int | None:
    if value is None:
        return None
    capacity = int(value)  # type: ignore[call-overload]
    if capacity < 1:
        raise ValueError(f'elevator capacity must be >= 1 or null, got {value!r}')
    return capacity


def _swing_direction_converter(value: object) -> int:
    direction = int(value)  # type: ignore[call-overload]
    if direction not in (1, -1):
        raise ValueError(f'door swing_direction must be 1 or -1, got {value!r}')
    return direction


@attrs.define
class Elevator(Named):
    position: Position = attrs.field(converter=Position.converter)
    size: list[float] = attrs.field(factory=lambda: [2.0, 2.0, 2.0])
    door_side: typing.Literal['+x', '-x', '+y', '-y'] | None = None
    material: MaterialIdentifier = attrs.field(converter=MaterialIdentifier.converter, default=Material.default('elevator'))
    destinations: list[str] = attrs.field(factory=list, converter=lambda value: [str(name).strip() for name in value])
    # must cover v_max * transition_time plus latency so the door is fully open before arrival
    activation_distance: float = 3.0
    transition_time: float = 1.0
    hold_time: float = 2.0
    travel_time: float = 3.0
    accept_outside_calls: bool = True
    recall_on: str | None = None
    # None = unlimited
    capacity: int | None = attrs.field(default=None, converter=_optional_capacity)
    call_mode: typing.Literal['level', 'edge'] = attrs.field(default='level', validator=_call_mode_validator)
    semantics: list[SemanticCfg] = attrs.field(factory=list, converter=parse_semantics)

    @property
    def destination(self) -> str:
        """The cabin a ride goes to, the first of the destinations."""
        return self.destinations[0] if self.destinations else ''

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


STAIRS_TREAD = 0.28
STAIRS_RAIL = 1.0
STAIRS_SEAM_INSET = 0.8
STAIRS_JOINED_INSET = -0.05
STAIRS_SEAM_REACH = 1.0
STAIRS_LANDING_BACKOFF = 0.1
STAIRS_PLATE = 0.05
STAIRS_GAP = 0.001
STAIRS_SINK = 0.001
STAIRS_CLEARANCE = 0.06
STAIRS_PLANK = 0.03
STAIRS_BASE = 0.15
STAIRS_FLAT = 0.8
STAIRS_HEADROOM = 2.1
STAIRS_HANDRAIL = 0.9
STAIRS_HANDRAIL_BAR = 0.05
STAIRS_HANDRAIL_INSET = 0.08
STAIRS_NOSING = 0.02
STAIRS_BRUSH = 0.03
STAIRS_BRUSH_LIFT = 0.03
STAIRS_BRUSH_DEPTH = 0.05
STAIRS_SURFACES = ('steps', 'risers', 'landings', 'fixture', 'brushes')
_SURFACE_DEFAULTS = {'brushes': MaterialIdentifier('Carpet_Charcoal')}
STAIRS_MODEL_SURFACES = ('steps', 'risers', 'landings', 'fixture')
_BOX_SURFACES = {'step': 'steps', 'riser': 'risers', 'plate': 'landings', 'base': 'fixture'}
_SIDE_AXES = {'+x': (1.0, 0.0), '-x': (-1.0, 0.0), '+y': (0.0, 1.0), '-y': (0.0, -1.0)}

StairsBox = tuple[tuple[float, float, float], Pose]


def _optional_material(value: object) -> MaterialIdentifier | None:
    return None if value is None else MaterialIdentifier.converter(value)


@attrs.define
class StairsSide:
    """One side wall of a stair, `height` above the walking surface, in a wall `kind` or a `material`, the stair's own when neither is set."""

    kind: str = ''
    material: MaterialIdentifier | None = attrs.field(default=None, converter=_optional_material)
    height: float = STAIRS_RAIL


@attrs.define
class StairsRail:
    """A handrail along a stair `across` meters left of its axis following the walking surface, a bar at `height` in a `material` or the parts of a wall `kind`, blocking the floor plan when `solid`."""

    across: float = 0.0
    kind: str = ''
    material: MaterialIdentifier | None = attrs.field(default=None, converter=_optional_material)
    height: float = STAIRS_HANDRAIL
    solid: bool = False


def _model_at(value: object) -> tuple[float, float, float]:
    along, across, z = (float(v) for v in typing.cast(typing.Iterable[float], value))
    return along, across, z


@attrs.define
class StairsModel:
    """An object dressing a staircase, `at` [along, across, z] from the middle of its mouth on the lower floor, along running up the whole staircase, turned `yaw` from its axis."""

    model: ObjectIdentifier = attrs.field(converter=ObjectIdentifier.converter)
    at: tuple[float, float, float] = attrs.field(default=(0.0, 0.0, 0.0), converter=_model_at)
    yaw: float = 0.0


def _on_top(box: StairsBox) -> Pose:
    """The middle of a box's top face, in the box's orientation."""
    (_length, _width, height), pose = box
    return Pose(position=pose.position + pose.orientation * Position(0.0, 0.0, height / 2.0), orientation=pose.orientation)


def _at_height(pose: Pose, z: float) -> Pose:
    return Pose(position=Position(pose.position.x, pose.position.y, z), orientation=pose.orientation)


def _default_sides() -> dict[str, StairsSide]:
    return {'left': StairsSide(), 'right': StairsSide()}


@attrs.define
class Stairs(Named):
    """Half of a stair or escalator: open at the mouth, cut at the seam, where a crossing lands on the `to` half of another level. A ramp is whole and has no seam: it stays on its level, its surface rising `rise` from `position.z` at the mouth."""

    position: Position = attrs.field(converter=Position.converter)
    kind: typing.Literal['stairs', 'escalator', 'ramp'] = 'stairs'
    size: list[float] = attrs.field(factory=lambda: [2.6, 1.4])
    mouth_side: typing.Literal['+x', '-x', '+y', '-y'] = '-x'
    to: str = ''
    top: bool = False
    rise: float = 3.0
    speed: float = 0.5
    style: typing.Literal['steps', 'belt'] = 'steps'
    tread: float = STAIRS_TREAD
    landing_length: float = STAIRS_FLAT
    plank: float = STAIRS_PLANK
    headroom: float = STAIRS_HEADROOM
    material: MaterialIdentifier = attrs.field(converter=MaterialIdentifier.converter, default=Material.default('wall'))
    surfaces: dict[str, MaterialIdentifier] = attrs.field(factory=dict)
    sides: dict[str, StairsSide] = attrs.field(factory=_default_sides)
    rails: list[StairsRail] = attrs.field(factory=list)
    surface_models: dict[str, ObjectIdentifier] = attrs.field(factory=dict)
    models: list[StairsModel] = attrs.field(factory=list)
    preset: str = ''
    authored: frozenset[str] = attrs.field(factory=frozenset)
    group: str = ''
    overhang: float = 0.0
    stacked: bool = False
    floor: bool = True
    joined: bool = attrs.field(default=False, init=False)
    whole: bool = attrs.field(default=False, init=False)

    @property
    def length(self) -> float:
        return self.size[0]

    @property
    def width(self) -> float:
        return self.size[1]

    @property
    def axis(self) -> tuple[float, float]:
        """Unit vector from the mouth towards the seam."""
        x, y = _SIDE_AXES[self.mouth_side]
        return -x, -y

    @property
    def yaw(self) -> float:
        return math.atan2(self.axis[1], self.axis[0])

    @property
    def inset(self) -> float:
        """Distance of the seam trigger line before the seam, just past it on a joined half, where a crossing lands on the very spot it left."""
        return STAIRS_JOINED_INSET if self.joined else STAIRS_SEAM_INSET

    @property
    def boards(self) -> bool:
        """Whether a crossing of this half's seam lands on another level."""
        return bool(self.to)

    @property
    def belt(self) -> float:
        """Belt speed along the axis, 0 on stairs, towards the seam on a boarding escalator half and away from it on an exit half."""
        if self.kind != 'escalator':
            return 0.0
        return self.speed if self.boards else -self.speed

    def local(self, along: float, across: float, z: float = 0.0) -> Position:
        """Level-frame point `along` meters from the mouth and `across` meters to the left of the axis."""
        ax, ay = self.axis
        x0 = self.position.x - ax * self.length / 2.0
        y0 = self.position.y - ay * self.length / 2.0
        return Position(x0 + ax * along - ay * across, y0 + ay * along + ax * across, self.position.z + z)

    def along[T: (float, np.ndarray)](self, x: T, y: T) -> T:
        """Distance of a level-frame point from the mouth along the axis."""
        ax, ay = self.axis
        mouth = self.mouth
        return (x - mouth.x) * ax + (y - mouth.y) * ay

    def across[T: (float, np.ndarray)](self, x: T, y: T) -> T:
        """Distance of a level-frame point to the left of the axis."""
        ax, ay = self.axis
        mouth = self.mouth
        return -(x - mouth.x) * ay + (y - mouth.y) * ax

    @property
    def reach(self) -> float:
        """How far past the seam agents still stand on a half, up to a trigger line past it."""
        return max(0.0, -2.0 * self.inset)

    def contains(self, x: float, y: float, beyond: float = 0.0) -> bool:
        return 0.0 <= self.along(x, y) <= self.length + beyond and abs(self.across(x, y)) <= self.width / 2.0

    def crossed(self, x: float, y: float) -> bool:
        """Whether a point stands past the seam trigger line of this half."""
        return self.contains(x, y) and self.along(x, y) >= self.length - self.inset

    def past_seam(self, x: float, y: float) -> bool:
        """Whether a point stands just beyond the seam, within `STAIRS_SEAM_REACH` of it, where a joined partner continues the stair."""
        return 0.0 < self.along(x, y) - self.length <= STAIRS_SEAM_REACH and abs(self.across(x, y)) <= self.width / 2.0

    def landing(self, dest: Stairs, x: float, y: float, yaw: float, travel: float = 0.0, dest_travel: float = 0.0) -> tuple[float, float, float, float]:
        """Pose on `dest` matching a seam crossing at x, y heading yaw as x, y, yaw and the height change: past its seam trigger, on the matching spot of a tread, the side offset kept relative to the heading."""
        along = self.along(x, y)
        limit = dest.length - dest.inset - STAIRS_LANDING_BACKOFF
        target = limit
        spans = self.spans(travel)
        if spans and dest.style == 'steps':
            start, _length = min(spans, key=lambda span: 0.0 if span[0] <= along <= span[0] + span[1] else abs(span[0] + span[1] / 2.0 - along))
            offset = along - start
            fits = [start + length - offset for start, length in dest.spans(dest_travel) if 0.0 <= start + length - offset <= limit]
            if fits:
                target = max(fits)
        point = dest.local(target, -self.across(x, y))
        dz = dest.surface(target, dest_travel) - self.surface(along, travel)
        return point.x, point.y, math.remainder(yaw - self.yaw + dest.yaw + math.pi, math.tau), dz

    @property
    def mouth(self) -> Position:
        return self.local(0.0, 0.0)

    @property
    def seam(self) -> Position:
        return self.local(self.length, 0.0)

    def corners(self, beyond: float = 0.0) -> list[Position]:
        """Footprint corners, reaching `beyond` meters past the seam."""
        half = self.width / 2.0
        end = self.length + beyond
        return [self.local(0.0, -half), self.local(end, -half), self.local(end, half), self.local(0.0, half)]

    @property
    def flat(self) -> float:
        """Length of the level landing at the mouth, where escalator treads and planks run flush with the floor."""
        return self.landing_length if self.kind == 'escalator' else 0.0

    @property
    def ramp(self) -> tuple[float, float]:
        """Start and end of the ramp along the axis, between the landing at the mouth and, on a whole stair, the one at the far end."""
        return self.flat, self.length - (self.flat if self.whole else 0.0)

    def height(self, along: float) -> float:
        """Ramp height `along` meters from the mouth, level over the landings, rising to half the rise at the seam, sinking on a top half, a ramp's rising its whole rise."""
        start, end = self.ramp
        t = min(max((along - start) / (end - start), 0.0), 1.0) * (self.rise if self.kind == 'ramp' else self.rise / 2.0)
        return -t if self.top else t

    @property
    def slope(self) -> float:
        """Signed incline angle of the ramp along the axis."""
        start, end = self.ramp
        return math.atan2(self.height(self.length), end - start)

    def material_of(self, surface: str) -> MaterialIdentifier:
        """Material of one of STAIRS_SURFACES, the brushes near black and every other the stair's unless set."""
        return self.surfaces.get(surface, _SURFACE_DEFAULTS.get(surface, self.material))

    def _pieces(self) -> list[tuple[float, float]]:
        """Along-axis stretches over which the walking surface runs straight: the landing at the mouth and the ramp."""
        start, end = self.ramp
        return [(a0, a1) for a0, a1 in ((0.0, start), (start, end), (end, self.length)) if a1 - a0 > 1e-9]

    def _along_surface(self, across: float, style: tuple[str, MaterialIdentifier | None], *, top: float | None = None, bottom: float | None = None, height: float | None = None) -> list[Wall]:
        """Walls `across` meters left of the axis in a (kind, material) style whose line follows the walking surface, their heights measured from it."""
        kind, material = style
        return [Wall(start=self.local(a0, across, self.height(a0)), end=self.local(a1, across, self.height(a1)), kind=kind, material=material, top=top, bottom=bottom, height=height) for a0, a1 in self._pieces()]

    def _style(self, kind: str, material: MaterialIdentifier | None) -> tuple[str, MaterialIdentifier | None]:
        return (kind, None) if kind else ('', material if material is not None else self.material)

    def walls(self) -> list[Wall]:
        """Walls that block the floor plan: each side's lining below the walking surface in the fixture material, reaching under the floor, and its wall above it, the seam wall, a top half's guards around its overhang, and the solid rails."""
        half = self.width / 2.0
        fixture = self.material_of('fixture')
        low = (-self.rise if self.stacked else -self.rise / 2.0) if self.top else 0.0
        walls: list[Wall] = []
        for name, across in (('left', half), ('right', -half)):
            side = self.sides[name]
            if self.top:
                walls.append(Wall(start=self.local(0.0, across), end=self.local(self.length, across), material=fixture, top=0.0, bottom=low))
                if side.height > 0.0:
                    kind, material = self._style(side.kind, side.material)
                    walls.append(Wall(start=self.local(0.0, across), end=self.local(self.length, across), kind=kind, material=material, height=side.height))
                continue
            if self.kind != 'ramp':
                walls.extend(self._along_surface(across, ('', fixture), top=0.0, bottom=-(abs(self.height(self.length)) + self.drop + STAIRS_BASE)))
            if side.height > 0.0:
                walls.extend(self._along_surface(across, self._style(side.kind, side.material), height=side.height))
        if not self.joined and self.kind != 'ramp':
            top = self.height(self.length) + max(side.height for side in self.sides.values())
            walls.append(Wall(start=self.local(self.length, -half), end=self.local(self.length, half), material=fixture, top=top, bottom=low))
        if self.top and self.overhang > 0.0:
            end = self.length + self.overhang
            guard = self.sides['left']
            for name, across in (('left', half), ('right', -half)):
                side = self.sides[name]
                kind, material = self._style(side.kind, side.material)
                walls.append(Wall(start=self.local(self.length, across), end=self.local(end, across), kind=kind, material=material, height=side.height))
            kind, material = self._style(guard.kind, guard.material)
            walls.append(Wall(start=self.local(end, -half), end=self.local(end, half), kind=kind, material=material, height=guard.height))
        return [*walls, *(wall for rail in self.rails if rail.solid for wall in self._rail(rail))]

    def open_walls(self) -> list[Wall]:
        """Walls that do not block the floor plan: the open rails and an escalator's skirt brushes, a strip along each inner side just above the treads."""
        walls = [wall for rail in self.rails if not rail.solid for wall in self._rail(rail)]
        if self.kind == 'escalator':
            lift = (self.riser if self.style == 'steps' else self._sag) + STAIRS_BRUSH_LIFT
            inner = self.width / 2.0 - STAIRS_BRUSH_DEPTH
            for across in (inner, -inner):
                walls.extend(self._along_surface(across, ('', self.material_of('brushes')), top=lift + STAIRS_BRUSH, bottom=lift))
        return walls

    def _rail(self, rail: StairsRail) -> list[Wall]:
        if rail.kind:
            return self._along_surface(rail.across, (rail.kind, None), height=rail.height)
        return self._along_surface(rail.across, self._style('', rail.material), top=rail.height, bottom=rail.height - STAIRS_HANDRAIL_BAR)

    @property
    def tread_count(self) -> int:
        return max(1, round(self.length / self.tread))

    @property
    def pitch(self) -> float:
        """Length of one tread, the static steps' or the moving treads' between the mouth gap and the seam wall."""
        if self.kind == 'stairs':
            return self.length / self.tread_count
        return self.run / self.tread_count

    @property
    def run(self) -> float:
        """Length the moving treads cover, from the mouth gap to the seam wall, or to the seam of a joined half."""
        return self.length - STAIRS_GAP - (STAIRS_GAP if self.joined or self.whole else STAIRS_CLEARANCE)

    @property
    def riser(self) -> float:
        """Ramp rise over one tread."""
        start, end = self.ramp
        return self.rise / 2.0 * self.pitch / (end - start)

    def spans(self, travel: float = 0.0) -> list[tuple[float, float]]:
        """Treads on the half as (start along the axis, length): the static steps, or the moving treads after the belt moved `travel` meters that stand clear of the landing plates."""
        n, pitch = self.tread_count, self.pitch
        if self.kind == 'stairs':
            return [(i * pitch, pitch) for i in range(n)]
        if self.style == 'belt' or self.kind == 'ramp':
            return []
        return [(start, pitch) for start in self.tread_starts(travel) if not self.under_plate(start)]

    def tread_starts(self, travel: float) -> list[float]:
        """Start of every moving tread along the axis after the belt moved `travel` meters, a tread past the far end of the run back at the mouth, sliding out from under its plate."""
        run, pitch = self.run, self.pitch
        lag = pitch / 2.0 if self.whole else 0.0
        starts: list[float] = []
        for k in range(self.tread_count):
            start = STAIRS_GAP + (k * pitch + travel) % run
            starts.append(start - run if start + pitch - (STAIRS_GAP + run) > lag + 1e-9 else start)
        return starts

    @property
    def plate_zones(self) -> list[tuple[float, float]]:
        """Along-axis ranges the landing plates of an escalator cover: the pit before each mouth, and on a whole stair half a tread into the run, where treads wrap out of sight."""
        if self.kind != 'escalator':
            return []
        reach = self.pitch + STAIRS_GAP
        if not self.whole:
            return [(-reach, STAIRS_GAP)]
        return [(-reach, STAIRS_GAP + self.pitch / 2.0), (self.length - STAIRS_GAP - self.pitch / 2.0, self.length + reach)]

    def under_plate(self, start: float) -> bool:
        """Whether a tread starting at `start` reaches under a landing plate."""
        return any(start < end - 1e-9 and start + self.pitch > begin + 1e-9 for begin, end in self.plate_zones)

    @property
    def drop(self) -> float:
        """Depth of the base slabs' top under the ramp line, clear of the lowest tread or plank, a ramp's slab being its walking surface."""
        if self.kind == 'ramp':
            return 0.0
        if self.style == 'belt':
            return STAIRS_SINK + 2.0 * STAIRS_GAP + self.plank / math.cos(self.slope) + self._sag
        return self.riser + 2.0 * STAIRS_GAP

    @property
    def _sag(self) -> float:
        """Most a belt plank's chord strays from the ramp line, over a knee of the ramp."""
        return self.pitch * abs(math.tan(self.slope)) / 4.0

    def _plate_level(self, start: float) -> float | None:
        """Top of the landing plate the tread starting at `start` reaches under, None when it reaches under none."""
        if not self.under_plate(start):
            return None
        return self.height(0.0) if start < self.length / 2.0 else self.height(self.length)

    def _plank_top(self, along: float, start: float) -> float:
        """Top of the belt plank starting at `start` at `along`, on the ramp line, capped `STAIRS_SINK` under a landing plate it reaches under."""
        level = self._plate_level(start)
        return self.height(along) if level is None else min(self.height(along), level) - STAIRS_SINK

    def _plank_climb(self, along: float, start: float) -> float:
        """Rate at which the top of the belt plank starting at `start` rises at `along` per metre of travel."""
        ramp_start, ramp_end = self.ramp
        level = self._plate_level(start)
        if not ramp_start < along < ramp_end or (level is not None and self.height(along) >= level):
            return 0.0
        return self.height(self.length) / (ramp_end - ramp_start)

    def surface(self, along: float, travel: float = 0.0) -> float:
        """Walking surface height `along` meters from the mouth: the top of the tread there, the belt or ramp, or the base between treads."""
        if self.kind == 'ramp' or (self.kind == 'escalator' and self.style == 'belt'):
            return self.height(along)
        for start, length in self.spans(travel):
            if start <= along <= start + length:
                return max(self.height(start), self.height(start + length))
        if self.kind == 'stairs':
            return self.height(along)
        return self.height(along) - self.riser - 2.0 * STAIRS_GAP

    def steps(self) -> list[StairsBox]:
        """Static step boxes of a stair half, a solid stair rising to the seam or a well sinking to it."""
        floor = -self.rise / 2.0 - 0.05
        orientation = Orientation.from_yaw(self.yaw)
        boxes: list[StairsBox] = []
        for start, length in self.spans():
            top = max(self.height(start), self.height(start + length))
            bottom = floor if self.top else 0.0
            if top - bottom <= 1e-3:
                continue
            center = self.local(start + length / 2.0, 0.0, (top + bottom) / 2.0)
            boxes.append(((length, self.width, top - bottom), Pose(position=center, orientation=orientation)))
        return boxes

    def base(self) -> list[StairsBox]:
        """Static slabs of an escalator half just under the moving treads: one along the ramp, one under each landing and the pit before it."""
        drop = self.drop
        start, end = self.ramp
        span = (end - start) / math.cos(self.slope)
        middle = (start + end) / 2.0
        center = self.local(middle + STAIRS_BASE / 2.0 * math.sin(self.slope), 0.0, self.height(middle) - drop - STAIRS_BASE / 2.0 * math.cos(self.slope))
        slabs = [((span, self.width, STAIRS_BASE), Pose(position=center, orientation=Orientation.from_euler((0.0, -self.slope, self.yaw))))]
        if self.flat:
            reach = self.pitch + STAIRS_GAP
            landings = [(-reach, self.flat, 0.0)] + ([(self.length - self.flat, self.length + reach, self.height(self.length))] if self.whole else [])
            for begin, end, level in landings:
                landing = self.local((begin + end) / 2.0, 0.0, level - drop - STAIRS_BASE / 2.0)
                slabs.append(((end - begin, self.width, STAIRS_BASE), Pose(position=landing, orientation=Orientation.from_yaw(self.yaw))))
        return slabs

    def plates(self) -> list[StairsBox]:
        """Static landing plates of an escalator, flush with the floor over `plate_zones`, combs the treads pass through."""
        boxes: list[StairsBox] = []
        for begin, end in self.plate_zones:
            level = self.height(0.0) if begin < 0.0 else self.height(self.length)
            boxes.append(((end - begin, self.width, STAIRS_PLATE), Pose(position=self.local((begin + end) / 2.0, 0.0, level - STAIRS_PLATE / 2.0), orientation=Orientation.from_yaw(self.yaw))))
        return boxes

    def static_models(self) -> list[tuple[str, ObjectIdentifier, Pose]]:
        """Objects dressing a half that stay put as (name, model, pose): one on top of every static box of a surface with a model, and the share of `models` on this half, all of them on a whole staircase."""
        placed = [(f'model_{name}', self.surface_models[surface], _on_top(box)) for name, _material, box in self.static_boxes() if (surface := _BOX_SURFACES[name.split('_')[0]]) in self.surface_models]
        for index, item in enumerate(self.models):
            along, across, z = item.at
            yaw = item.yaw
            if self.top and not self.whole:
                if along <= self.length:
                    continue
                along, across, z, yaw = 2.0 * self.length - along, -across, z - self.rise, yaw + math.pi
            elif not self.whole and along > self.length:
                continue
            placed.append((f'model_{index}', item.model, Pose(position=self.local(along, across, z), orientation=Orientation.from_yaw(self.yaw + yaw))))
        return placed

    def tread_models(self, travel: float) -> list[Pose]:
        """Pose of the `steps` model on top of every moving tread after the belt moved `travel` meters, none without one."""
        if self.kind != 'escalator' or 'steps' not in self.surface_models:
            return []
        return [_on_top(box) for box in self.treads(travel)]

    def static_boxes(self) -> list[tuple[str, MaterialIdentifier, StairsBox]]:
        """Static boxes of a half as (name, material, box): stair steps, each a tread plate on a riser body when `risers` is set, an escalator's base slabs and landing plates, or a ramp's slab unless the world already has a floor there."""
        if self.kind == 'ramp':
            return [(f'base_{i}', self.material_of('steps'), box) for i, box in enumerate(self.base())] if self.floor else []
        if self.kind == 'escalator':
            return [*((f'base_{i}', self.material_of('fixture'), box) for i, box in enumerate(self.base())), *((f'plate_{i}', self.material_of('landings'), box) for i, box in enumerate(self.plates()))]
        steps = self.material_of('steps')
        boxes: list[tuple[str, MaterialIdentifier, StairsBox]] = []
        for i, ((length, width, height), pose) in enumerate(self.steps()):
            if 'risers' not in self.surfaces or height <= STAIRS_NOSING:
                boxes.append((f'step_{i}', steps, ((length, width, height), pose)))
                continue
            top = pose.position.z + height / 2.0
            boxes.append((f'step_{i}', steps, ((length, width, STAIRS_NOSING), _at_height(pose, top - STAIRS_NOSING / 2.0))))
            boxes.append((f'riser_{i}', self.material_of('risers'), ((length, width, height - STAIRS_NOSING), _at_height(pose, top - (STAIRS_NOSING + height) / 2.0))))
        return boxes

    def tread_velocities(self, travel: float) -> list[tuple[float, float, float]]:
        """World-frame velocity of every moving tread after the belt moved `travel` meters: the belt speed along the axis, climbing or sinking with the tread top."""
        ax, ay = self.axis
        ramp_start, ramp_end = self.ramp
        rate = self.height(self.length) / (ramp_end - ramp_start)
        velocities: list[tuple[float, float, float]] = []
        for start in self.tread_starts(travel):
            if self.style == 'belt':
                end = start + self.pitch - STAIRS_GAP
                velocities.append((self.belt * ax, self.belt * ay, self.belt * (self._plank_climb(start, start) + self._plank_climb(end, start)) / 2.0))
                continue
            edge = start if self.top else start + self.pitch
            climbing = ramp_start < edge < ramp_end
            velocities.append((self.belt * ax, self.belt * ay, self.belt * rate if climbing else 0.0))
        return velocities

    def _rect(self, start: float, end: float) -> tuple[float, float, float, float]:
        corners = [self.local(along, side) for along in (start, end) for side in (-self.width / 2.0, self.width / 2.0)]
        return min(c.x for c in corners), min(c.y for c in corners), max(c.x for c in corners), max(c.y for c in corners)

    def floor_openings(self) -> list[tuple[float, float, float, float]]:
        """Rectangles (x_min, y_min, x_max, y_max) the half cuts out of its level's floor: a top half's well with its overhang, an escalator's pit."""
        pit = -(self.pitch + STAIRS_GAP) if self.kind == 'escalator' else 0.0
        if self.top:
            return [self._rect(pit, self.length + self.overhang)]
        if self.kind == 'escalator':
            return [self._rect(pit, self.length)]
        return []

    def ceiling_openings(self) -> list[tuple[float, float, float, float]]:
        """Rectangles a stacked bottom half cuts out of its level's ceilings, from where headroom runs short over it to the end of the top half."""
        if self.top or not self.stacked:
            return []
        return [self._rect(self.length - self.overhang, 2.0 * self.length)]

    def whole_stair(self) -> Stairs:
        """The stair a joined bottom half and its partner make together, in the bottom half's frame, as one body from mouth to mouth."""
        body = attrs.evolve(self, position=self.local(self.length, 0.0), size=[2.0 * self.length, self.width], rise=2.0 * self.rise, overhang=0.0)
        body.joined = True
        body.whole = True
        return body

    def treads(self, travel: float) -> list[StairsBox]:
        """Moving escalator treads after the belt moved `travel` meters along the axis: step blocks, or planks of a belt tilted along the chord of the ramp line under them, sunk under a landing plate while they reach under it."""
        pitch, width = self.pitch, self.width - 2.0 * STAIRS_CLEARANCE
        length = pitch - STAIRS_GAP
        boxes: list[StairsBox] = []
        for start in self.tread_starts(travel):
            middle = start + length / 2.0
            if self.style == 'belt':
                boxes.append(self._plank(start, width))
                continue
            top = max(self.height(start), self.height(start + pitch)) - (STAIRS_SINK if self.under_plate(start) else 0.0)
            thickness = self.riser - STAIRS_GAP
            center = self.local(middle, 0.0, top - thickness / 2.0)
            boxes.append(((length, width, thickness), Pose(position=center, orientation=Orientation.from_yaw(self.yaw))))
        return boxes

    def _plank_tilt(self, start: float) -> float:
        length = self.pitch - STAIRS_GAP
        return math.atan2(self._plank_top(start + length, start) - self._plank_top(start, start), length)

    def _plank(self, start: float, width: float) -> StairsBox:
        """The belt plank starting at `start`: its top face from the ramp line there along the chord, cut short where the next plank tilts flatter so their faces never cross."""
        length = self.pitch - STAIRS_GAP
        low = self._plank_top(start, start)
        tilt = self._plank_tilt(start)
        cut = self.plank * math.tan(max(0.0, tilt - self._plank_tilt(start + self.pitch)))
        span = length / math.cos(tilt) - cut
        along = start + span / 2.0 * math.cos(tilt) + self.plank / 2.0 * math.sin(tilt)
        z = low + span / 2.0 * math.sin(tilt) - self.plank / 2.0 * math.cos(tilt)
        return (span, width, self.plank), Pose(position=self.local(along, 0.0, z), orientation=Orientation.from_euler((0.0, -tilt, self.yaw)))


def stair_height(stairs: typing.Iterable[Stairs], x: float, y: float) -> float:
    """Walking surface height at a map point on the first stair half or ramp holding it, 0 off every one."""
    return next((stair.position.z + stair.height(stair.along(x, y)) for stair in stairs if stair.contains(x, y, stair.reach)), 0.0)


STOREY = 3.0


@attrs.define
class Placement:
    """Maps the abstract frame, where levels lie side by side, into the simulator frame, where each level takes its own offset."""

    origins: dict[str, tuple[float, float]] = attrs.field(factory=dict)
    offsets: dict[str, tuple[float, float, float]] = attrs.field(factory=dict)
    bounds: dict[str, tuple[float, float, float, float]] = attrs.field(factory=dict)
    env: tuple[float, float] = (0.0, 0.0)

    @property
    def stacked(self) -> bool:
        return any(self.shift(level) != (0.0, 0.0, 0.0) for level in self.offsets)

    def shift(self, level: str) -> tuple[float, float, float]:
        """Simulator minus abstract coordinates on `level`, zero for an unknown one."""
        if level not in self.offsets:
            return 0.0, 0.0, 0.0
        ox, oy = self.origins.get(level, (0.0, 0.0))
        sx, sy, sz = self.offsets[level]
        return sx - ox, sy - oy, sz

    def level_at(self, x: float, y: float) -> str:
        """The level whose abstract bounds hold the point, '' outside every level."""
        for level, (x0, y0, x1, y1) in self.bounds.items():
            ox, oy = self.origins.get(level, (0.0, 0.0))
            lx, ly = x - self.env[0] - ox, y - self.env[1] - oy
            if x0 <= lx <= x1 and y0 <= ly <= y1:
                return level
        return ''

    def place(self, position: Position, level: str | None = None) -> Position:
        """A point of the abstract frame in the simulator frame, on `level` or the level holding it."""
        dx, dy, dz = self.shift(self.level_at(position.x, position.y) if level is None else level)
        return Position(position.x + dx, position.y + dy, position.z + dz)

    def unplace(self, position: Position, level: str) -> Position:
        """A point of the simulator frame on `level` back in the abstract frame."""
        dx, dy, dz = self.shift(level)
        return Position(position.x - dx, position.y - dy, position.z - dz)


@attrs.define
class Door(Named):
    start: Position = attrs.field(converter=Position.converter)
    end: Position = attrs.field(converter=Position.converter)
    kind: typing.Literal['sliding', 'hinged', 'teleport', 'sliding_top'] = 'sliding'
    width: float = 0.1
    height: float = attrs.field(default=2.0)
    material: MaterialIdentifier = attrs.field(converter=MaterialIdentifier.converter, default=Material.default('door'))
    activation_distance: ActivationDistance = attrs.field(
        converter=_activation_distance_converter,
        default=ActivationDistance((3.0, 3.0)),
    )
    transition_time: float = 1.0
    hold_time: float = 2.0
    # hinged only: +1 swings towards the left normal of start->end, -1 towards the right.
    swing_direction: int = attrs.field(default=1, converter=_swing_direction_converter)
    semantics: list[SemanticCfg] = attrs.field(factory=list, converter=parse_semantics)

    @classmethod
    def parse(cls, value: dict) -> Door:
        value = dict(value)
        if 'start' in value or 'end' in value:
            raise ValueError(f"door {value.get('name')!r} takes span: [[x, y], [x, y]], not start and end")
        span = value.pop('span', None)
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError(f"door {value.get('name')!r} requires span: [[x, y], [x, y]]")
        if isinstance(value.get('activation_distance'), (int, float)):
            value['activation_distance'] = [value['activation_distance']] * 2
        return super().parse({**value, 'start': span[0], 'end': span[1]})

    def serialize(self) -> dict:
        result = super().serialize()
        result['span'] = [result.pop('start'), result.pop('end')]
        return result

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


def _leaf_rests_on(start: Position, end: Position, door: Door, others: typing.Sequence[Door]) -> bool:
    """Whether a leaf sliding from `start` past `end` by its own length covers the span of a collinear door among `others`."""
    length = math.hypot(end.x - start.x, end.y - start.y)
    if length == 0.0:
        return False
    ux, uy = (end.x - start.x) / length, (end.y - start.y) / length
    for other in others:
        if other is door:
            continue
        along = sorted((p.x - start.x) * ux + (p.y - start.y) * uy for p in (other.start, other.end))
        across = max(abs((p.y - start.y) * ux - (p.x - start.x) * uy) for p in (other.start, other.end))
        if across <= door.width and min(along[1], 2.0 * length) - max(along[0], length) > _POCKET_OVERLAP_M:
            return True
    return False


def clear_door_pockets(doors: typing.Sequence[Door]) -> list[Door]:
    """Keep open sliding leaves off other doors' spans: such a door slides the other way, or teleports open when that side is taken too."""
    result = list(doors)
    for index, door in enumerate(result):
        if door.kind != 'sliding' or not _leaf_rests_on(door.start, door.end, door, result):
            continue
        result[index] = attrs.evolve(door, kind='teleport') if _leaf_rests_on(door.end, door.start, door, result) else attrs.evolve(door, start=door.end, end=door.start)
    return result


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
    """Standalone cycling-phase semantic entity (kind = signal), no geometry, optional stop-line params `position` and `radius`."""

    semantics: list[SemanticCfg] = attrs.field(factory=list, converter=parse_semantics)

    def __attrs_post_init__(self) -> None:
        position = self._param('position')
        if position is not None:
            if not isinstance(position, (list, tuple)) or len(position) != 2 or not all(isinstance(v, (int, float)) for v in position):
                raise ValueError(f"signal {self.name!r} position must be [x, y], got {position!r}")
        radius = self._param('radius')
        if radius is not None and float(radius) <= 0.0:
            raise ValueError(f"signal {self.name!r} radius must be positive, got {radius!r}")

    def _param(self, key: str) -> typing.Any:  # noqa: ANN401 - untyped YAML semantics params
        for cfg in self.semantics:
            if key in cfg.params:
                return cfg.params[key]
        return None

    @property
    def stop_position(self) -> tuple[float, float] | None:
        """Stop-line position, None when not given."""
        position = self._param('position')
        return None if position is None else (float(position[0]), float(position[1]))

    @property
    def stop_radius(self) -> float | None:
        """Stop-line radius, None when not given."""
        radius = self._param('radius')
        return None if radius is None else float(radius)


def _optional_position(value: object) -> Position | None:
    if value is None:
        return None
    return Position.instance_or(Position.parse)(value)


@attrs.define
class Sound(Named):
    """Standalone audible semantic entity (kind = sound), placed in the world."""

    asset_id: str = attrs.field(converter=lambda value: str(value).strip())
    position: Position | None = attrs.field(converter=_optional_position, default=None)
    attach: str = attrs.field(converter=lambda value: str(value).strip(), default='')
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
        if sum((bool(self.attach), self.position is not None, bool(self.frame))) != 1:
            raise ValueError(f"sound {self.name!r} requires exactly one of position, attach or frame")
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
    def directed(self) -> bool:
        return self.shape not in ('dome', 'sphere')

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

_LIGHT_AXIS = Vector3(0.0, 0.0, -1.0)


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


def _pose(value: object) -> Pose:
    return Pose.instance_or(lambda raw: converter.structure(raw, Pose))(value)


def _optional_pose(value: object) -> Pose | None:
    if value is None:
        return None
    return _pose(value)


@attrs.define
class Light(Named):
    """Standalone light semantic entity (kind = light): one fixture, an ambient light or a ceiling rig."""

    fixture: str = attrs.field(converter=lambda value: str(value).strip())
    pose: Pose | None = attrs.field(converter=_optional_pose, default=None)
    attach: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    frame: str = attrs.field(converter=lambda value: str(value).strip().strip('/'), default='')
    offset: Pose = attrs.field(converter=_pose, factory=Pose)
    lumens: float | None = attrs.field(converter=_optional_float, default=None)
    lux: float | None = attrs.field(converter=_optional_float, default=None)
    cct_K: float = attrs.field(converter=float, default=6500.0)
    cone_deg: float | None = attrs.field(converter=_optional_float, default=None)
    cast_shadows: bool = False
    light_on: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    lit: bool | None = None
    dim: float = attrs.field(converter=float, default=1.0)
    dead_fraction: float = attrs.field(converter=float, default=0.0)
    level: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    intrinsic: bool = False
    owner: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    link: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    glow: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    rig: bool = False
    fixtures: list[Position] = attrs.field(factory=list)
    fixture_ranks: list[float] = attrs.field(factory=list)

    def __attrs_post_init__(self) -> None:
        if not self.name or ':' in self.name:
            raise ValueError(f"light name {self.name!r} must be non-empty and contain no ':'")
        if self.fixture not in LIGHT_FIXTURES:
            raise ValueError(f'light {self.name!r} has unknown fixture {self.fixture!r}, expected one of {sorted(LIGHT_FIXTURES)}')
        ambient = self.spec.ambient
        anchors = sum((self.pose is not None and not ambient, bool(self.attach), bool(self.frame)))
        if self.offset != Pose() and not (self.attach or self.frame):
            raise ValueError(f'light {self.name!r} takes an offset only with attach or frame')
        if self.link and not (self.attach or self.owner):
            raise ValueError(f'light {self.name!r} takes a link only with attach or owner')
        for placed in (self.pose, self.offset):
            if placed is not None and not math.isclose(math.sqrt(sum(component * component for component in placed.orientation)), 1.0, abs_tol=1e-3):
                raise ValueError(f'light {self.name!r} orientation must be a unit quaternion')
        if ambient:
            if anchors:
                raise ValueError(f'light {self.name!r} ({self.fixture}) takes no attach or frame')
            if self.pose is not None and any(self.pose.position):
                raise ValueError(f'light {self.name!r} ({self.fixture}) takes no position, a pose only aims it')
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
                    raise ValueError(f'rig light {self.name!r} takes no pose, attach or frame')
                if not self.fixtures:
                    raise ValueError(f'rig light {self.name!r} requires at least one fixture position')
                if len(self.fixture_ranks) != len(self.fixtures):
                    raise ValueError(f'rig light {self.name!r} requires one fixture rank per fixture position')
            elif anchors != 1:
                raise ValueError(f'light {self.name!r} requires exactly one of pose, attach or frame')
        if not self.rig and (self.fixtures or self.fixture_ranks):
            raise ValueError(f'light {self.name!r} takes fixture positions only as a rig')
        if not self.spec.directed and self._placement.orientation != Orientation.identity():
            raise ValueError(f'light {self.name!r} ({self.fixture}) takes no orientation')
        if self.cone_deg is not None and (self.fixture != 'spot' or not 0.0 < self.cone_deg < 180.0):
            raise ValueError(f'light {self.name!r} takes cone_deg only as a spot, within 0..180 exclusive')
        check_light_state(f'light {self.name!r}', self.cct_K, self.dim, self.dead_fraction, self.light_on, self.lit)
        if self.dead_fraction and not self.rig:
            raise ValueError(f'light {self.name!r} takes dead_fraction only as a rig')

    @property
    def _placement(self) -> Pose:
        return self.pose if self.pose is not None else self.offset

    @property
    def position(self) -> Position:
        """Where the fixture sits: in the level frame, or in the frame of its anchor while it sits on one."""
        return self._placement.position

    @property
    def direction(self) -> tuple[float, float, float]:
        """Unit vector the fixture emits along, the local -z axis of its pose, or of its offset while it sits on an anchor."""
        along = self._placement.orientation * _LIGHT_AXIS
        return (along.x, along.y, along.z)

    @property
    def spec(self) -> LightFixture:
        spec = LIGHT_FIXTURES[self.fixture]
        return spec if self.cone_deg is None else attrs.evolve(spec, cone_deg=self.cone_deg)

    @property
    def semantics(self) -> list[SemanticCfg]:
        lit = None if self.light_on else (True if self.lit is None else self.lit)
        result = [
            SemanticCfg(role='predicate', name='lit', value=lit, params={'light_on': self.light_on} if self.light_on else {}),
            SemanticCfg(role='state', name='dim', value=self.dim),
        ]
        if self.rig:
            result.append(SemanticCfg(role='state', name='dead_fraction', value=self.dead_fraction))
        return result

    def alive(self, dead_fraction: float) -> list[bool]:
        """Per rig fixture, whether it survives the given dead fraction."""
        return [rank >= dead_fraction for rank in self.fixture_ranks]


_OBJECT_LIGHT_KEYS = frozenset({'name', 'fixture', 'pose', 'lumens', 'cct_K', 'cone_deg', 'glow', 'link'})


def object_light(owner: str, entry: dict, settings: ObjectLightSettings | None) -> Light:
    """Light an object annotation entry declares, named after its owning entity and offset in the object frame."""
    unknown = set(entry) - _OBJECT_LIGHT_KEYS
    if unknown:
        raise ValueError(f'object light of {owner!r} has unknown keys {sorted(unknown)}')
    fixture = str(entry.get('fixture', 'bulb'))
    if fixture not in LIGHT_FIXTURES or LIGHT_FIXTURES[fixture].ambient:
        raise ValueError(f'object light of {owner!r} has fixture {fixture!r}, expected one of {sorted(name for name, spec in LIGHT_FIXTURES.items() if not spec.ambient)}')
    offset = _pose(entry['pose']) if 'pose' in entry else Pose()
    settings = settings if settings is not None else ObjectLightSettings()
    return Light(
        name=f"{owner}_{entry.get('name', 'light')}",
        fixture=fixture,
        attach=owner,
        offset=offset if LIGHT_FIXTURES[fixture].directed else Pose(position=offset.position),
        lumens=entry.get('lumens'),
        cct_K=entry.get('cct_K', 6500.0),
        cone_deg=entry.get('cone_deg') if fixture == 'spot' else None,
        light_on=settings.light_on,
        lit=settings.lit,
        dim=settings.dim,
        intrinsic=True,
        glow=entry.get('glow') or '',
        link=entry.get('link') or '',
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
    dim: float = attrs.field(converter=float, default=1.0)
    dead_fraction: float = attrs.field(converter=float, default=0.0)

    def __attrs_post_init__(self) -> None:
        if self.fixture not in LIGHT_FIXTURES or LIGHT_FIXTURES[self.fixture].ambient:
            raise ValueError(f'ceiling_lights fixture {self.fixture!r} must be one of {sorted(name for name, spec in LIGHT_FIXTURES.items() if not spec.ambient)}')
        if not self.spacing > 0.0:
            raise ValueError('ceiling_lights spacing must be positive')
        if not self.lumens >= 0.0:
            raise ValueError('ceiling_lights requires lumens >= 0')
        check_light_state('ceiling_lights', self.cct_K, self.dim, self.dead_fraction, self.light_on, self.lit)

    def serialize(self) -> dict:
        return cattrs.gen.make_dict_unstructure_fn(type(self), converter, _cattrs_omit_if_default=True)(self)
