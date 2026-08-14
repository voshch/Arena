"""Runtime semantic kinds: door/elevator FSM facts plus scripted and position-derived M2 kinds."""

from __future__ import annotations

import abc
import math
import typing
from collections.abc import Callable, Sequence
from typing import ClassVar

import attrs

from ._mechanism_shim import _DoorState, _near_door_segment, _reset_door

if typing.TYPE_CHECKING:
    from arena_simulation_setup.shared.semantics import SemanticCfg

    from ._interface import MechanismITF
    from ._mechanism_shim import _DoorRuntime, _ElevatorRuntime


@attrs.define
class SemanticEntitySnapshot:
    entity: str
    kind: str
    discrete: dict[str, str]
    continuous: dict[str, float]
    predicates: dict[str, bool]
    members: list[str] = attrs.field(factory=list)


@attrs.define
class SemanticChange:
    entity: str
    kind: str
    field_kind: str  # "state" | "predicate"
    field: str
    previous: str
    current: str


_SEMANTIC_KINDS: dict[str, type[SemanticKind]] = {}


def semantic_kind(name: str) -> Callable[[type[SemanticKind]], type[SemanticKind]]:
    """Register a SemanticKind subclass under name and set cls.KIND."""

    def register(cls: type[SemanticKind]) -> type[SemanticKind]:
        cls.KIND = name
        _SEMANTIC_KINDS[name] = cls
        return cls

    return register


def _stringify(value: str | float | bool | None) -> str:
    """Stringify a semantic value for the event stream. Missing prior value renders empty."""
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def _merge_params(cfgs: Sequence[SemanticCfg]) -> dict:
    """Shallow-merge the params dicts of a kind's cfg list into one config dict."""
    params: dict = {}
    for cfg in cfgs:
        params.update(cfg.params)
    return params


def _agents_xy(mech: MechanismITF) -> list[tuple[str, tuple[float, float]]]:
    """Every tracked agent (robots plus peds) as (name, (x, y)), mirroring the _tick scan."""
    agents = [(name, xy) for name, xy, _radius in mech.robot_discs()]
    if mech._human_simulator is not None:
        agents.extend((name, xy) for name, xy, _radius in mech._human_simulator.pedestrian_discs())
    return agents


def _in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """Ray-cast point-in-polygon for a flat xy ring."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class SemanticAttachError(Exception):
    """Raised by a kind's attach when the entity cannot back it, so the manager skips it."""


class SemanticKind(abc.ABC):
    KIND: ClassVar[str]
    SUPPORTED_SIMS: ClassVar[frozenset[str]]

    DISCRETE: ClassVar[tuple[str, ...]] = ()
    CONTINUOUS: ClassVar[tuple[str, ...]] = ()
    PREDICATES: ClassVar[tuple[str, ...]] = ()

    # Per-continuous-field emit quantum (F2). Fields absent here default to 1.0.
    QUANTA: ClassVar[dict[str, float]] = {}
    # Writable field names for the M2 write path. Empty means read-only.
    WRITABLE: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, entity: str, discrete: tuple[str, ...], continuous: tuple[str, ...], predicates: tuple[str, ...]) -> None:
        self._entity = entity
        self._discrete = discrete
        self._continuous = continuous
        self._predicates = predicates
        self._now = -math.inf

    @classmethod
    def _classify(cls, cfgs: Sequence[SemanticCfg]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Split requested cfg names into this kind's discrete/continuous/predicate vocabulary."""
        discrete: list[str] = []
        continuous: list[str] = []
        predicates: list[str] = []
        for cfg in cfgs:
            if cfg.name in cls.DISCRETE:
                discrete.append(cfg.name)
            elif cfg.name in cls.CONTINUOUS:
                continuous.append(cfg.name)
            elif cfg.name in cls.PREDICATES:
                predicates.append(cfg.name)
            else:
                raise ValueError(f"{cls.KIND!r} semantics: unknown field {cfg.name!r}")
        return tuple(discrete), tuple(continuous), tuple(predicates)

    def quantum(self, field: str) -> float:
        """Emit quantum for a continuous field."""
        return self.QUANTA.get(field, 1.0)

    @classmethod
    @abc.abstractmethod
    def attach(cls, mech: MechanismITF, entity: str, cfgs: Sequence[SemanticCfg], *, polygon: Sequence[tuple[float, float]] | None = None) -> SemanticKind:
        """Bind to a spawned entity, ValueError on cfg names outside the kind vocabulary."""

    def step(self, now: float) -> None:
        self._now = now

    @abc.abstractmethod
    def reset(self) -> None: ...

    @abc.abstractmethod
    def snapshot(self) -> SemanticEntitySnapshot: ...

    def set_value(self, field: str, value: str) -> None:
        """Read-only kinds reject every write."""
        raise ValueError('read-only')

    def asserts_regime(self, name: str) -> bool:
        """True when this instance currently asserts the named regime. Non-scripted kinds never do."""
        return False


def _coerce(inst: SemanticKind, field: str, value: str) -> str | float | bool:
    """Coerce a stringified write to the field's type, ValueError('malformed value') on failure."""
    if field in inst.CONTINUOUS:
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ValueError('malformed value') from None
    if field in inst.PREDICATES:
        token = value.strip().lower()
        if token in ('true', 'false'):
            return token == 'true'
        raise ValueError('malformed value')
    return value


@semantic_kind('door')
class DoorSemantics(SemanticKind):
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac'})

    DISCRETE = ('state',)
    CONTINUOUS = ('progress',)
    PREDICATES = ('open', 'in_transit', 'triggered')
    QUANTA = {'progress': 0.1}

    def __init__(
        self,
        entity: str,
        runtime: _DoorRuntime,
        discrete: tuple[str, ...],
        continuous: tuple[str, ...],
        predicates: tuple[str, ...],
    ) -> None:
        super().__init__(entity, discrete, continuous, predicates)
        self._runtime = runtime

    @classmethod
    def attach(cls, mech: MechanismITF, entity: str, cfgs: Sequence[SemanticCfg], *, polygon: Sequence[tuple[float, float]] | None = None) -> SemanticKind:
        return cls(entity, mech._door_runtime[entity], cls.DISCRETE, cls.CONTINUOUS, cls.PREDICATES)

    def reset(self) -> None:
        _reset_door(self._runtime)

    def snapshot(self) -> SemanticEntitySnapshot:
        rt = self._runtime
        discrete: dict[str, str] = {}
        continuous: dict[str, float] = {}
        predicates: dict[str, bool] = {}
        if 'state' in self._discrete:
            discrete['state'] = rt.state.value
        if 'progress' in self._continuous:
            continuous['progress'] = rt.progress
        if 'open' in self._predicates:
            predicates['open'] = rt.state == _DoorState.OPEN
        if 'in_transit' in self._predicates:
            predicates['in_transit'] = rt.state in (_DoorState.OPENING, _DoorState.CLOSING)
        if 'triggered' in self._predicates:
            predicates['triggered'] = (self._now - rt.last_trigger_sim_time) <= rt.door.hold_time
        return SemanticEntitySnapshot(self._entity, self.KIND, discrete, continuous, predicates)


@semantic_kind('elevator')
class ElevatorSemantics(SemanticKind):
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac'})

    DISCRETE = ('cabin_door',)
    CONTINUOUS = ('arriving_eta', 'occupants', 'cabin_door_progress')
    PREDICATES = ('departing', 'in_transit', 'dispatched', 'just_arrived')
    QUANTA = {'cabin_door_progress': 0.1}

    def __init__(
        self,
        entity: str,
        runtime: _ElevatorRuntime,
        cabin_door: _DoorRuntime | None,
        discrete: tuple[str, ...],
        continuous: tuple[str, ...],
        predicates: tuple[str, ...],
    ) -> None:
        super().__init__(entity, discrete, continuous, predicates)
        self._runtime = runtime
        self._cabin_door = cabin_door

    @classmethod
    def attach(cls, mech: MechanismITF, entity: str, cfgs: Sequence[SemanticCfg], *, polygon: Sequence[tuple[float, float]] | None = None) -> SemanticKind:
        cabin = mech._door_runtime.get(f"{entity}/door")
        return cls(entity, mech._elevator_runtime[entity], cabin, cls.DISCRETE, cls.CONTINUOUS, cls.PREDICATES)

    def reset(self) -> None:
        rt = self._runtime
        rt.arriving_eta = -math.inf
        rt.pending_occupants = ()
        rt.just_arrived = {}
        rt.departing = False
        rt.dispatched = set()
        cabin = self._cabin_door
        if cabin is not None:
            _reset_door(cabin)

    def _members(self) -> list[str]:
        rt = self._runtime
        return [*rt.dispatched, *(name for name, _ in rt.pending_occupants), *rt.just_arrived]

    def snapshot(self) -> SemanticEntitySnapshot:
        rt = self._runtime
        discrete: dict[str, str] = {}
        continuous: dict[str, float] = {}
        predicates: dict[str, bool] = {}
        if 'arriving_eta' in self._continuous:
            continuous['arriving_eta'] = max(0.0, rt.arriving_eta - self._now) if rt.arriving_eta > -math.inf else -1.0
        if 'occupants' in self._continuous:
            continuous['occupants'] = float(len(rt.dispatched) + len(rt.pending_occupants) + len(rt.just_arrived))
        if 'departing' in self._predicates:
            predicates['departing'] = rt.departing
        if 'in_transit' in self._predicates:
            predicates['in_transit'] = rt.arriving_eta > -math.inf
        if 'dispatched' in self._predicates:
            predicates['dispatched'] = bool(rt.dispatched)
        if 'just_arrived' in self._predicates:
            predicates['just_arrived'] = bool(rt.just_arrived)
        cabin = self._cabin_door
        if 'cabin_door' in self._discrete:
            discrete['cabin_door'] = cabin.state.value if cabin is not None else _DoorState.CLOSED.value
        if 'cabin_door_progress' in self._continuous:
            continuous['cabin_door_progress'] = cabin.progress if cabin is not None else 0.0
        return SemanticEntitySnapshot(self._entity, self.KIND, discrete, continuous, predicates, self._members())


@attrs.define
class _Phase:
    name: str
    duration: float


@semantic_kind('signal')
class SignalSemantics(SemanticKind):
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac', 'dummy'})

    DISCRETE = ('state',)
    CONTINUOUS = ('phase_remaining',)
    PREDICATES = ('stop',)
    QUANTA = {'phase_remaining': 1.0}
    WRITABLE = frozenset({'state'})

    def __init__(
        self,
        entity: str,
        phases: list[_Phase],
        stop_phases: frozenset[str],
        regime: str | None,
        discrete: tuple[str, ...],
        continuous: tuple[str, ...],
        predicates: tuple[str, ...],
    ) -> None:
        super().__init__(entity, discrete, continuous, predicates)
        self._phases = phases
        self._stop_phases = stop_phases
        self._regime = regime
        self._t0: float | None = None
        self._pending_offset: float | None = None
        self._stop = self._phase(0.0)[0] in stop_phases if phases else False

    @classmethod
    def attach(cls, mech: MechanismITF, entity: str, cfgs: Sequence[SemanticCfg], *, polygon: Sequence[tuple[float, float]] | None = None) -> SemanticKind:
        discrete, continuous, predicates = cls._classify(cfgs)
        params = _merge_params(cfgs)
        phases = [_Phase(str(p['name']), float(p['duration'])) for p in params.get('phases', [])]
        stops = params.get('stop_phases')
        stop_phases = frozenset(stops) if stops is not None else (frozenset({phases[-1].name}) if phases else frozenset())
        return cls(entity, phases, stop_phases, params.get('regime'), discrete, continuous, predicates)

    def _cycle(self) -> float:
        return sum(p.duration for p in self._phases)

    def _phase(self, now: float) -> tuple[str, float]:
        """Active phase name and seconds to the next boundary at now."""
        if not self._phases:
            return '', -1.0
        cycle = self._cycle()
        t0 = self._t0 if self._t0 is not None else now
        elapsed = (now - t0) % cycle if cycle > 0.0 else 0.0
        acc = 0.0
        for phase in self._phases:
            if elapsed < acc + phase.duration:
                return phase.name, acc + phase.duration - elapsed
            acc += phase.duration
        last = self._phases[-1]
        return last.name, last.duration

    def step(self, now: float) -> None:
        super().step(now)
        if self._t0 is None:
            self._t0 = now - (self._pending_offset or 0.0)
            self._pending_offset = None
        self._stop = self._phase(now)[0] in self._stop_phases

    def reset(self) -> None:
        self._t0 = None
        self._pending_offset = None
        self._stop = (self._phases[0].name in self._stop_phases) if self._phases else False

    def set_value(self, field: str, value: str) -> None:
        if field not in self.WRITABLE:
            raise ValueError('field not writable')
        token = _coerce(self, field, value)
        offset = 0.0
        for phase in self._phases:
            if phase.name == token:
                if math.isfinite(self._now):
                    self._t0 = self._now - offset
                else:
                    self._pending_offset = offset
                    self._t0 = None
                self._stop = phase.name in self._stop_phases
                return
            offset += phase.duration
        raise ValueError('malformed value')

    def asserts_regime(self, name: str) -> bool:
        return self._regime == name and self._stop

    def snapshot(self) -> SemanticEntitySnapshot:
        name, remaining = self._phase(self._now)
        discrete: dict[str, str] = {}
        continuous: dict[str, float] = {}
        predicates: dict[str, bool] = {}
        if 'state' in self._discrete:
            discrete['state'] = name
        if 'phase_remaining' in self._continuous:
            continuous['phase_remaining'] = remaining
        if 'stop' in self._predicates:
            predicates['stop'] = name in self._stop_phases
        return SemanticEntitySnapshot(self._entity, self.KIND, discrete, continuous, predicates)


@attrs.define
class _Window:
    start: float
    end: float
    value: str


@semantic_kind('schedule')
class ScheduleSemantics(SemanticKind):
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac', 'dummy'})

    DISCRETE = ('state',)
    CONTINUOUS = ('window_remaining',)
    PREDICATES = ('active',)
    QUANTA = {'window_remaining': 1.0}
    WRITABLE = frozenset({'state', 'active'})

    def __init__(
        self,
        entity: str,
        windows: list[_Window],
        default: str,
        regime: str | None,
        discrete: tuple[str, ...],
        continuous: tuple[str, ...],
        predicates: tuple[str, ...],
    ) -> None:
        super().__init__(entity, discrete, continuous, predicates)
        self._windows = windows
        self._default = default
        self._regime = regime
        self._t0: float | None = None
        self._override_state: str | None = None
        self._override_active: bool | None = None
        self._last_natural_active = False
        self._active = False

    @classmethod
    def attach(cls, mech: MechanismITF, entity: str, cfgs: Sequence[SemanticCfg], *, polygon: Sequence[tuple[float, float]] | None = None) -> SemanticKind:
        discrete, continuous, predicates = cls._classify(cfgs)
        params = _merge_params(cfgs)
        windows = [_Window(float(w['start']), float(w['end']), str(w['value'])) for w in params.get('windows', [])]
        return cls(entity, windows, str(params.get('default', '')), params.get('regime'), discrete, continuous, predicates)

    def _episode_seconds(self, now: float) -> float:
        return now - (self._t0 if self._t0 is not None else now)

    def _natural(self, now: float) -> tuple[bool, str, float]:
        """Natural (active, state, window_remaining) ignoring overrides."""
        sec = self._episode_seconds(now)
        for window in self._windows:
            if window.start <= sec < window.end:
                return True, window.value, window.end - sec
        return False, self._default, -1.0

    def step(self, now: float) -> None:
        super().step(now)
        if self._t0 is None:
            self._t0 = now
        natural_active, _, _ = self._natural(now)
        if natural_active != self._last_natural_active:
            self._override_state = None
            self._override_active = None
        self._last_natural_active = natural_active
        self._active = self._override_active if self._override_active is not None else natural_active

    def reset(self) -> None:
        self._t0 = None
        self._override_state = None
        self._override_active = None
        self._last_natural_active = False
        self._active = False

    def set_value(self, field: str, value: str) -> None:
        if field not in self.WRITABLE:
            raise ValueError('field not writable')
        coerced = _coerce(self, field, value)
        if field == 'active':
            self._override_active = bool(coerced)
            self._active = self._override_active
        else:
            self._override_state = str(coerced)

    def asserts_regime(self, name: str) -> bool:
        return self._regime == name and self._active

    def snapshot(self) -> SemanticEntitySnapshot:
        natural_active, natural_state, remaining = self._natural(self._now)
        active = self._override_active if self._override_active is not None else natural_active
        state = self._override_state if self._override_state is not None else natural_state
        discrete: dict[str, str] = {}
        continuous: dict[str, float] = {}
        predicates: dict[str, bool] = {}
        if 'state' in self._discrete:
            discrete['state'] = state
        if 'window_remaining' in self._continuous:
            continuous['window_remaining'] = remaining
        if 'active' in self._predicates:
            predicates['active'] = active
        return SemanticEntitySnapshot(self._entity, self.KIND, discrete, continuous, predicates)


@semantic_kind('gate')
class GateSemantics(SemanticKind):
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac'})

    PREDICATES = ('locked', 'blocked')
    WRITABLE = frozenset({'locked'})

    def __init__(
        self,
        mech: MechanismITF,
        entity: str,
        door: object,
        authorized: frozenset[str],
        initial_locked: bool,
        unlock_on: str | None,
        discrete: tuple[str, ...],
        continuous: tuple[str, ...],
        predicates: tuple[str, ...],
    ) -> None:
        super().__init__(entity, discrete, continuous, predicates)
        self._mech = mech
        self._door = door
        self._authorized = authorized
        self._locked = initial_locked
        self._initial_locked = initial_locked
        self._unlock_on = unlock_on

    @classmethod
    def attach(cls, mech: MechanismITF, entity: str, cfgs: Sequence[SemanticCfg], *, polygon: Sequence[tuple[float, float]] | None = None) -> SemanticKind:
        discrete, continuous, predicates = cls._classify(cfgs)
        params = _merge_params(cfgs)
        authorized = frozenset(str(a) for a in params.get('authorized', []))
        initial = True
        for cfg in cfgs:
            if cfg.name == 'locked' and cfg.value is not None:
                initial = bool(cfg.value)
        runtime = mech._door_runtime.get(entity)
        if runtime is None:
            raise SemanticAttachError(f"gate {entity!r} has no door runtime")
        return cls(mech, entity, runtime.door, authorized, initial, params.get('unlock_on'), discrete, continuous, predicates)

    def _authorized_match(self, name: str) -> bool:
        return name in self._authorized or name.rsplit('/', 1)[-1] in self._authorized

    def _at_door(self) -> list[str]:
        radius = max(self._door.activation_distance)
        if radius <= 0.0:
            return []
        return [name for name, xy in _agents_xy(self._mech) if _near_door_segment(self._door, [xy], radius)]

    def _effective_locked(self) -> bool:
        if self._unlock_on is not None and self._mech._semantics.regime(self._unlock_on):
            return False
        return self._locked

    def trigger_allowed(self, now: float) -> bool:
        """False only while effectively locked and no authorized agent is at the door."""
        if not self._effective_locked():
            return True
        return any(self._authorized_match(name) for name in self._at_door())

    def reset(self) -> None:
        self._locked = self._initial_locked

    def set_value(self, field: str, value: str) -> None:
        if field not in self.WRITABLE:
            raise ValueError('field not writable')
        self._locked = bool(_coerce(self, field, value))

    def snapshot(self) -> SemanticEntitySnapshot:
        locked = self._effective_locked()
        at_door = self._at_door()
        unauthorized = [name for name in at_door if not self._authorized_match(name)]
        predicates: dict[str, bool] = {}
        if 'locked' in self._predicates:
            predicates['locked'] = locked
        if 'blocked' in self._predicates:
            predicates['blocked'] = locked and bool(unauthorized)
        return SemanticEntitySnapshot(self._entity, self.KIND, {}, {}, predicates, unauthorized)


@semantic_kind('pressure_plate')
class PressurePlateSemantics(SemanticKind):
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac'})

    PREDICATES = ('pressed',)
    WRITABLE = frozenset({'pressed'})

    def __init__(
        self,
        mech: MechanismITF,
        entity: str,
        position: tuple[float, float],
        radius: float,
        drives: str | None,
        latch: float,
        press_on: str | None,
        regime: str | None,
        discrete: tuple[str, ...],
        continuous: tuple[str, ...],
        predicates: tuple[str, ...],
    ) -> None:
        super().__init__(entity, discrete, continuous, predicates)
        self._mech = mech
        self._position = position
        self._radius = radius
        self._drives = drives
        self._latch = latch
        self._press_on = press_on
        self._regime = regime
        self._last_contact = -math.inf
        self._forced = False
        self._pressed = False

    @classmethod
    def attach(cls, mech: MechanismITF, entity: str, cfgs: Sequence[SemanticCfg], *, polygon: Sequence[tuple[float, float]] | None = None) -> SemanticKind:
        discrete, continuous, predicates = cls._classify(cfgs)
        params = _merge_params(cfgs)
        pos = params.get('position', [0.0, 0.0])
        position = (float(pos[0]), float(pos[1]))
        return cls(
            mech,
            entity,
            position,
            float(params.get('radius', 0.0)),
            params.get('drives'),
            float(params.get('latch', 0.0)),
            params.get('press_on'),
            params.get('regime'),
            discrete,
            continuous,
            predicates,
        )

    def _contact(self) -> bool:
        px, py = self._position
        r2 = self._radius * self._radius
        return any((x - px) ** 2 + (y - py) ** 2 <= r2 for _n, (x, y) in _agents_xy(self._mech))

    def _compute_pressed(self, now: float, contact: bool) -> bool:
        if self._forced:
            return True
        if self._press_on is not None and self._mech._semantics.regime(self._press_on):
            return True
        if contact:
            return True
        return self._latch > 0.0 and (now - self._last_contact) <= self._latch

    def step(self, now: float) -> None:
        super().step(now)
        contact = self._contact()
        if contact:
            self._last_contact = now
        self._pressed = self._compute_pressed(now, contact)

    def apply_drive(self, mech: MechanismITF, now: float) -> None:
        """Refresh the driven door's trigger while pressed (HOOK-B)."""
        if not self._pressed or self._drives is None:
            return
        driven = mech._door_runtime.get(self._drives)
        if driven is not None:
            driven.last_trigger_sim_time = now

    def reset(self) -> None:
        self._last_contact = -math.inf
        self._forced = False
        self._pressed = False

    def set_value(self, field: str, value: str) -> None:
        if field not in self.WRITABLE:
            raise ValueError('field not writable')
        self._forced = bool(_coerce(self, field, value))
        self._pressed = self._pressed or self._forced

    def asserts_regime(self, name: str) -> bool:
        return self._regime == name and self._pressed

    def snapshot(self) -> SemanticEntitySnapshot:
        predicates = {'pressed': self._pressed} if 'pressed' in self._predicates else {}
        return SemanticEntitySnapshot(self._entity, self.KIND, {}, {}, predicates)


@semantic_kind('occupancy_cap')
class OccupancyCapSemantics(SemanticKind):
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac'})

    CONTINUOUS = ('occupancy', 'cap')
    PREDICATES = ('over_cap',)
    QUANTA = {'occupancy': 1.0, 'cap': 1.0}
    WRITABLE = frozenset({'cap'})

    def __init__(
        self,
        mech: MechanismITF,
        entity: str,
        polygon: list[tuple[float, float]],
        cap: float,
        discrete: tuple[str, ...],
        continuous: tuple[str, ...],
        predicates: tuple[str, ...],
    ) -> None:
        super().__init__(entity, discrete, continuous, predicates)
        self._mech = mech
        self._polygon = polygon
        self._cap = cap
        self._initial_cap = cap

    @classmethod
    def attach(cls, mech: MechanismITF, entity: str, cfgs: Sequence[SemanticCfg], *, polygon: Sequence[tuple[float, float]] | None = None) -> SemanticKind:
        discrete, continuous, predicates = cls._classify(cfgs)
        params = _merge_params(cfgs)
        ring = [(float(x), float(y)) for x, y in (polygon or [])]
        return cls(mech, entity, ring, float(params.get('cap', 0)), discrete, continuous, predicates)

    def _inside(self) -> list[str]:
        return [name for name, (x, y) in _agents_xy(self._mech) if _in_polygon(x, y, self._polygon)]

    def reset(self) -> None:
        self._cap = self._initial_cap

    def set_value(self, field: str, value: str) -> None:
        if field not in self.WRITABLE:
            raise ValueError('field not writable')
        self._cap = float(_coerce(self, field, value))

    def snapshot(self) -> SemanticEntitySnapshot:
        inside = self._inside()
        occupancy = float(len(inside))
        continuous: dict[str, float] = {}
        predicates: dict[str, bool] = {}
        if 'occupancy' in self._continuous:
            continuous['occupancy'] = occupancy
        if 'cap' in self._continuous:
            continuous['cap'] = self._cap
        if 'over_cap' in self._predicates:
            predicates['over_cap'] = occupancy > self._cap
        return SemanticEntitySnapshot(self._entity, self.KIND, {}, continuous, predicates, inside)


def _diff(prev: SemanticEntitySnapshot | None, snap: SemanticEntitySnapshot, quantum: Callable[[str], float]) -> list[SemanticChange]:
    """Emit one SemanticChange per changed field. Continuous fields use the per-field quantum (F2)."""
    prev_discrete = prev.discrete if prev is not None else {}
    prev_continuous = prev.continuous if prev is not None else {}
    prev_predicates = prev.predicates if prev is not None else {}
    changes: list[SemanticChange] = []
    for name, value in snap.discrete.items():
        old = prev_discrete.get(name)
        if old != value:
            changes.append(SemanticChange(snap.entity, snap.kind, 'state', name, _stringify(old), _stringify(value)))
    for name, fvalue in snap.continuous.items():
        fold = prev_continuous.get(name)
        q = quantum(name)
        if fold is None or round(fvalue / q) != round(fold / q):
            changes.append(SemanticChange(snap.entity, snap.kind, 'state', name, _stringify(fold), _stringify(fvalue)))
    for name, bvalue in snap.predicates.items():
        bold = prev_predicates.get(name)
        if bold != bvalue:
            changes.append(SemanticChange(snap.entity, snap.kind, 'predicate', name, _stringify(bold), _stringify(bvalue)))
    return changes


class SemanticsManager:
    def __init__(self, mech: MechanismITF) -> None:
        self._mech = mech
        self._sim: str | None = None
        self._instances: dict[str, list[SemanticKind]] = {}
        self._last: dict[tuple[str, str], SemanticEntitySnapshot] = {}
        self._recall: dict[str, str] = {}
        self._now = -math.inf
        self._callback: Callable[[Sequence[SemanticChange]], None] | None = None

    def set_sim(self, name: str) -> None:
        self._sim = name

    def attach(self, kind: str, entity: str, cfgs: Sequence[SemanticCfg] = (), *, polygon: Sequence[tuple[float, float]] | None = None) -> bool:
        """Instantiate the kind for entity, skipping unsupported sims. True when an instance was created."""
        kind_cls = _SEMANTIC_KINDS.get(kind)
        if kind_cls is None:
            self._mech.node.get_logger().info(f"semantics: unknown kind {kind!r} for {entity!r}, skipping")
            return False
        if self._sim not in kind_cls.SUPPORTED_SIMS:
            self._mech.node.get_logger().info(f"semantics: kind {kind!r} unsupported on sim {self._sim!r}, skipping {entity!r}")
            return False
        try:
            instance = kind_cls.attach(self._mech, entity, cfgs, polygon=polygon)
        except SemanticAttachError as e:
            self._mech.node.get_logger().warning(f"semantics: {e}, skipping")
            return False
        self._instances.setdefault(entity, []).append(instance)
        return True

    def set_recall(self, entity: str, regime: str | None) -> None:
        """Register (or clear) the regime name that recalls an elevator's cabin (HOOK-C)."""
        if regime is None:
            self._recall.pop(entity, None)
        else:
            self._recall[entity] = regime

    def detach(self, entity: str) -> None:
        self._instances.pop(entity, None)
        self._recall.pop(entity, None)
        for key in [k for k in self._last if k[0] == entity]:
            del self._last[key]

    def step(self, now: float) -> None:
        """Step every instance, diff against the last snapshot, emit changes as one batch."""
        self._now = now
        changes: list[SemanticChange] = []
        for entity, instances in self._instances.items():
            for instance in instances:
                instance.step(now)
                snap = instance.snapshot()
                key = (entity, instance.KIND)
                changes.extend(_diff(self._last.get(key), snap, instance.quantum))
                self._last[key] = snap
        if changes and self._callback is not None:
            self._callback(changes)

    def reset(self) -> None:
        for instances in self._instances.values():
            for instance in instances:
                instance.reset()
        self._last.clear()

    def snapshot(self) -> list[SemanticEntitySnapshot]:
        return [instance.snapshot() for instances in self._instances.values() for instance in instances]

    def set_change_callback(self, cb: Callable[[Sequence[SemanticChange]], None]) -> None:
        self._callback = cb

    def set_value(self, entity: str, field: str, value: str) -> bool:
        """Apply a write to the owning instance, re-snapshot, diff, fire the callback. False if entity unknown."""
        instances = self._instances.get(entity)
        if not instances:
            return False
        target: SemanticKind | None = None
        for instance in instances:
            if field in instance.DISCRETE or field in instance.CONTINUOUS or field in instance.PREDICATES:
                target = instance
                break
        if target is None:
            raise ValueError('field not writable')
        key = (entity, target.KIND)
        prev = self._last.get(key)
        target.set_value(field, value)
        snap = target.snapshot()
        self._last[key] = snap
        changes = _diff(prev, snap, target.quantum)
        if changes and self._callback is not None:
            self._callback(changes)
        return True

    def regime(self, name: str) -> bool:
        """OR of every scripted instance currently asserting the named regime."""
        return any(instance.asserts_regime(name) for instances in self._instances.values() for instance in instances)

    def trigger_allowed(self, door_name: str, now: float) -> bool:
        """HOOK-A: True with no gate on this door, else gate clearance."""
        gates = [i for i in self._instances.get(door_name, []) if isinstance(i, GateSemantics)]
        return all(gate.trigger_allowed(now) for gate in gates)

    def apply_plate_drives(self, now: float) -> None:
        """HOOK-B: refresh every driven door's trigger for each pressed plate."""
        for instances in self._instances.values():
            for instance in instances:
                if isinstance(instance, PressurePlateSemantics):
                    instance.apply_drive(self._mech, now)

    def elevator_recalled(self, elev_name: str, now: float) -> bool:
        """HOOK-C: True when a recall regime registered for this elevator currently asserts."""
        recall = self._recall.get(elev_name)
        return recall is not None and self.regime(recall)
