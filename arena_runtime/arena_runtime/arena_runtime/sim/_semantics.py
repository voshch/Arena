"""Runtime semantic kinds: door/elevator/stair FSM facts plus scripted and position-derived M2 kinds."""

from __future__ import annotations

import abc
import math
import typing
from collections.abc import Callable, Sequence
from typing import ClassVar

import attrs
from arena_simulation_setup.utils.articulation import STATE_TOLERANCE

from ._mechanism_shim import _DoorState, _near_door_segment, _reset_door, _reset_elevator, _reset_stair, stair_partner

if typing.TYPE_CHECKING:
    from arena_simulation_setup.shared.semantics import SemanticCfg

    from arena_runtime.shared import Door

    from ._articulation import ArticulatedObject
    from ._interface import MechanismITF
    from ._mechanism_shim import _DoorRuntime, _ElevatorRuntime, _StairsRuntime


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


def semantic_kind[K: type[SemanticKind]](name: str) -> Callable[[K], K]:
    """Register a SemanticKind subclass under name and set cls.KIND."""

    def register(cls: K) -> K:
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
    RANGES: ClassVar[dict[str, tuple[float, float]]] = {}

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

    def settle(self) -> None:  # noqa: B027 (deliberate no-op default, not abstract)
        """Post-step hook for logic that consults other instances, run once all clocks advanced."""

    @abc.abstractmethod
    def reset(self) -> None: ...

    @abc.abstractmethod
    def snapshot(self) -> SemanticEntitySnapshot: ...

    def has_field(self, field: str) -> bool:
        """Whether field is in this kind's vocabulary, the owner of a write to it."""
        return field in self.DISCRETE or field in self.CONTINUOUS or field in self.PREDICATES

    @classmethod
    def is_continuous(cls, field: str) -> bool:
        """Whether a state field of this kind is continuous."""
        return field in cls.CONTINUOUS

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
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac', 'mujoco', 'flatsim'})

    DISCRETE = ('state',)
    CONTINUOUS = ('progress',)
    PREDICATES = ('open', 'in_transit', 'triggered', 'blocked_close')
    QUANTA = {'progress': 0.1}

    def __init__(
        self,
        entity: str,
        runtime: _DoorRuntime,
        discrete: tuple[str, ...],
        continuous: tuple[str, ...],
        predicates: tuple[str, ...],
        mech: MechanismITF | None = None,
    ) -> None:
        super().__init__(entity, discrete, continuous, predicates)
        self._runtime = runtime
        self._mech = mech

    @classmethod
    def attach(cls, mech: MechanismITF, entity: str, cfgs: Sequence[SemanticCfg], *, polygon: Sequence[tuple[float, float]] | None = None) -> SemanticKind:
        return cls(entity, mech._door_runtime[entity], cls.DISCRETE, cls.CONTINUOUS, cls.PREDICATES, mech)

    def reset(self) -> None:
        _reset_door(self._runtime)

    def _members(self) -> list[str]:
        """Agents (robots and peds) within the door's activation radius of its segment."""
        door = self._runtime.door
        radius = max(door.activation_distance)
        if self._mech is None or radius <= 0.0:
            return []
        return [name for name, xy in _agents_xy(self._mech) if _near_door_segment(door, [xy], radius)]

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
        if 'blocked_close' in self._predicates:
            predicates['blocked_close'] = rt.close_blocked
        return SemanticEntitySnapshot(self._entity, self.KIND, discrete, continuous, predicates, self._members())


@semantic_kind('elevator')
class ElevatorSemantics(SemanticKind):
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac', 'mujoco', 'flatsim'})

    DISCRETE = ('cabin_door', 'target')
    CONTINUOUS = ('arriving_eta', 'occupants', 'cabin_door_progress', 'live_occupants', 'capacity')
    PREDICATES = ('departing', 'in_transit', 'dispatched', 'just_arrived', 'full')
    QUANTA = {'cabin_door_progress': 0.1}
    WRITABLE = frozenset({'target'})

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
        _reset_elevator(self._runtime)
        cabin = self._cabin_door
        if cabin is not None:
            _reset_door(cabin)

    def _members(self) -> list[str]:
        """Live occupants: agents (robots and peds) inside the cabin footprint at the last step."""
        return list(self._runtime.live_occupants)

    def _target(self) -> str:
        rt = self._runtime
        return rt.current_destination() if len(rt.destinations) > 1 else ''

    def set_value(self, field: str, value: str) -> None:
        """Write `target`: an authored or realized destination name of this elevator, '' = round-robin."""
        if field not in self.WRITABLE:
            raise ValueError('field not writable')
        token = str(_coerce(self, field, value)).strip()
        rt = self._runtime
        rt.target_override = rt.resolve_target(token) if token else None

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
        if 'live_occupants' in self._continuous:
            continuous['live_occupants'] = float(len(rt.live_occupants))
        if 'capacity' in self._continuous:
            continuous['capacity'] = float(rt.elevator.capacity) if rt.elevator.capacity is not None else -1.0
        if 'full' in self._predicates:
            predicates['full'] = rt.full
        if 'target' in self._discrete:
            discrete['target'] = self._target()
        cabin = self._cabin_door
        if 'cabin_door' in self._discrete:
            discrete['cabin_door'] = cabin.state.value if cabin is not None else _DoorState.CLOSED.value
        if 'cabin_door_progress' in self._continuous:
            continuous['cabin_door_progress'] = cabin.progress if cabin is not None else 0.0
        return SemanticEntitySnapshot(self._entity, self.KIND, discrete, continuous, predicates, self._members())


class _StairsSemantics(SemanticKind):
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac', 'mujoco', 'flatsim'})

    CONTINUOUS = ('occupants',)
    PREDICATES = ('just_crossed',)

    def __init__(
        self,
        mech: MechanismITF,
        entity: str,
        runtime: _StairsRuntime,
        discrete: tuple[str, ...],
        continuous: tuple[str, ...],
        predicates: tuple[str, ...],
    ) -> None:
        super().__init__(entity, discrete, continuous, predicates)
        self._mech = mech
        self._runtime = runtime

    @classmethod
    def attach(cls, mech: MechanismITF, entity: str, cfgs: Sequence[SemanticCfg], *, polygon: Sequence[tuple[float, float]] | None = None) -> SemanticKind:
        return cls(mech, entity, mech._stair_runtime[entity], cls.DISCRETE, cls.CONTINUOUS, cls.PREDICATES)

    def reset(self) -> None:
        _reset_stair(self._runtime)

    def _discrete_values(self) -> dict[str, str]:
        return {}

    def snapshot(self) -> SemanticEntitySnapshot:
        rt = self._runtime
        return SemanticEntitySnapshot(
            self._entity,
            self.KIND,
            self._discrete_values(),
            {'occupants': float(len(rt.members))},
            {'just_crossed': rt.just_crossed},
            list(rt.members),
        )


@semantic_kind('stairs')
class StairsSemantics(_StairsSemantics):
    pass


@semantic_kind('ramp')
class RampSemantics(_StairsSemantics):
    pass


@semantic_kind('escalator')
class EscalatorSemantics(_StairsSemantics):
    DISCRETE = ('running', 'direction')
    WRITABLE = frozenset({'running'})

    def set_value(self, field: str, value: str) -> None:
        """Write `running` on both halves of the pair."""
        if field not in self.WRITABLE:
            raise ValueError('field not writable')
        token = value.strip().lower()
        if token not in ('true', 'false'):
            raise ValueError('malformed value')
        partner = stair_partner(self._mech, self._runtime)
        for rt in (self._runtime, partner):
            if rt is not None:
                rt.running = token == 'true'

    def _direction(self) -> str:
        stair = self._runtime.stair
        partner = stair_partner(self._mech, self._runtime)
        boarding = stair if stair.boards else partner.stair if partner is not None and partner.stair.boards else None
        if boarding is None:
            return ''
        return 'down' if boarding.top else 'up'

    def _discrete_values(self) -> dict[str, str]:
        return {'running': _stringify(self._runtime.running), 'direction': self._direction()}


@attrs.define
class _Phase:
    name: str
    duration: float


@semantic_kind('signal')
class SignalSemantics(SemanticKind):
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac', 'mujoco', 'dummy', 'flatsim'})

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
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac', 'mujoco', 'dummy', 'flatsim'})

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

    def _effective_now(self) -> float:
        return self._now if math.isfinite(self._now) else 0.0

    def asserts_regime(self, name: str) -> bool:
        if self._regime != name:
            return False
        natural_active, _, _ = self._natural(self._effective_now())
        return self._override_active if self._override_active is not None else natural_active

    def snapshot(self) -> SemanticEntitySnapshot:
        natural_active, natural_state, remaining = self._natural(self._effective_now())
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


_GATE_DEFAULT_LOGGED: set[str] = set()


@semantic_kind('gate')
class GateSemantics(SemanticKind):
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac', 'mujoco', 'flatsim'})

    PREDICATES = ('locked', 'blocked')
    WRITABLE = frozenset({'locked'})

    def __init__(
        self,
        mech: MechanismITF,
        entity: str,
        door: Door,
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
        initial: bool | None = None
        for cfg in cfgs:
            if cfg.name == 'locked' and cfg.value is not None:
                initial = bool(cfg.value)
        if initial is None:
            initial = True
            if entity not in _GATE_DEFAULT_LOGGED:
                _GATE_DEFAULT_LOGGED.add(entity)
                mech.node.get_logger().info(f"gate {entity!r} authored without `locked`, starting locked (default). Set params.locked to be explicit.")
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
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac', 'mujoco', 'flatsim'})

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
        self._warned_missing = False

    @classmethod
    def attach(cls, mech: MechanismITF, entity: str, cfgs: Sequence[SemanticCfg], *, polygon: Sequence[tuple[float, float]] | None = None) -> SemanticKind:
        discrete, continuous, predicates = cls._classify(cfgs)
        params = _merge_params(cfgs)
        pos = params.get('position', [0.0, 0.0])
        position = (float(pos[0]), float(pos[1]))
        press_on = params.get('press_on')
        regime = params.get('regime')
        if press_on is not None and press_on.removeprefix('!') == regime:
            raise ValueError(f"pressure_plate {entity!r}: press_on cannot equal regime (self-latch)")
        return cls(
            mech,
            entity,
            position,
            float(params.get('radius', 0.0)),
            params.get('drives'),
            float(params.get('latch', 0.0)),
            press_on,
            regime,
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
        elif not self._warned_missing:
            self._warned_missing = True
            mech.node.get_logger().warning(f"pressure_plate {self._entity!r}: drives {self._drives!r} matches no spawned door")

    def drives_door(self, door_name: str) -> bool:
        """True while pressed and driving door_name."""
        return self._pressed and self._drives == door_name

    def reset(self) -> None:
        self._last_contact = -math.inf
        self._forced = False
        self._pressed = False

    def set_value(self, field: str, value: str) -> None:
        if field not in self.WRITABLE:
            raise ValueError('field not writable')
        self._forced = bool(_coerce(self, field, value))
        self._pressed = self._pressed or self._forced

    def _live_pressed(self) -> bool:
        """Forced and press_on evaluate live, the contact/latch component stays step-cached."""
        if self._forced:
            return True
        if self._press_on is not None and self._mech._semantics.regime(self._press_on):
            return True
        return self._pressed

    def asserts_regime(self, name: str) -> bool:
        return self._regime == name and self._live_pressed()

    def snapshot(self) -> SemanticEntitySnapshot:
        predicates = {'pressed': self._live_pressed()} if 'pressed' in self._predicates else {}
        return SemanticEntitySnapshot(self._entity, self.KIND, {}, {}, predicates)


@semantic_kind('sound')
class SoundSemantics(SemanticKind):
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac', 'mujoco', 'dummy', 'flatsim'})

    CONTINUOUS = ('volume_db',)
    PREDICATES = ('sounding',)
    QUANTA = {'volume_db': 1.0}
    WRITABLE = frozenset({'sounding', 'volume_db'})

    def __init__(
        self,
        mech: MechanismITF,
        entity: str,
        sound_on: str | None,
        regime: str | None,
        initial_sounding: bool,
        initial_volume: float | None,
        discrete: tuple[str, ...],
        continuous: tuple[str, ...],
        predicates: tuple[str, ...],
    ) -> None:
        super().__init__(entity, discrete, continuous, predicates)
        self._mech = mech
        self._sound_on = sound_on
        self._regime = regime
        self._initial_sounding = initial_sounding
        self._initial_volume = initial_volume
        self._volume = initial_volume
        self._override_sounding: bool | None = None
        self._last_natural_sounding = self._natural()

    @classmethod
    def attach(cls, mech: MechanismITF, entity: str, cfgs: Sequence[SemanticCfg], *, polygon: Sequence[tuple[float, float]] | None = None) -> SemanticKind:
        discrete, continuous, predicates = cls._classify(cfgs)
        params = _merge_params(cfgs)
        sound_on = params.get('sound_on')
        regime = params.get('regime')
        if sound_on is not None and sound_on.removeprefix('!') == regime:
            raise ValueError(f"sound {entity!r}: sound_on cannot equal regime (self-latch)")
        initial_sounding = False
        initial_volume: float | None = None
        for cfg in cfgs:
            if cfg.value is None:
                continue
            if cfg.name == 'volume_db':
                initial_volume = float(cfg.value)
            elif cfg.name == 'sounding':
                if sound_on is not None:
                    raise ValueError(f"sound {entity!r}: sounding value and sound_on are exclusive")
                initial_sounding = bool(cfg.value)
        return cls(mech, entity, sound_on, regime, initial_sounding, initial_volume, discrete, continuous, predicates)

    def _natural(self) -> bool:
        return self._mech._semantics.regime(self._sound_on) if self._sound_on is not None else self._initial_sounding

    def _live_sounding(self) -> bool:
        return self._override_sounding if self._override_sounding is not None else self._natural()

    def settle(self) -> None:
        natural = self._natural()
        if natural != self._last_natural_sounding:
            self._override_sounding = None
        self._last_natural_sounding = natural

    def reset(self) -> None:
        self._override_sounding = None
        self._last_natural_sounding = self._natural()
        self._volume = self._initial_volume

    def set_value(self, field: str, value: str) -> None:
        if field not in self.WRITABLE:
            raise ValueError('field not writable')
        if field == 'volume_db' and field not in self._continuous:
            raise ValueError('volume_db not declared on this sound')
        coerced = _coerce(self, field, value)
        if field == 'sounding':
            self._override_sounding = bool(coerced)
        else:
            self._volume = float(coerced)

    def asserts_regime(self, name: str) -> bool:
        return self._regime == name and self._live_sounding()

    def snapshot(self) -> SemanticEntitySnapshot:
        continuous: dict[str, float] = {}
        predicates: dict[str, bool] = {}
        if 'sounding' in self._predicates:
            predicates['sounding'] = self._live_sounding()
        if 'volume_db' in self._continuous and self._volume is not None:
            continuous['volume_db'] = self._volume
        return SemanticEntitySnapshot(self._entity, self.KIND, {}, continuous, predicates)


@semantic_kind('light')
class LightSemantics(SemanticKind):
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac', 'mujoco', 'dummy'})

    CONTINUOUS = ('dim', 'dead_fraction')
    PREDICATES = ('lit',)
    QUANTA = {'dim': 0.01, 'dead_fraction': 0.01}
    WRITABLE = frozenset({'lit', 'dim', 'dead_fraction'})
    RANGES = {'dim': (0.0, 1.0), 'dead_fraction': (0.0, 1.0)}

    def __init__(
        self,
        mech: MechanismITF,
        entity: str,
        light_on: str | None,
        initial_lit: bool,
        initial_dim: float,
        initial_dead_fraction: float,
        discrete: tuple[str, ...],
        continuous: tuple[str, ...],
        predicates: tuple[str, ...],
    ) -> None:
        super().__init__(entity, discrete, continuous, predicates)
        self._mech = mech
        self._light_on = light_on
        self._initial_lit = initial_lit
        self._initial_dim = initial_dim
        self._initial_dead_fraction = initial_dead_fraction
        self._dim = initial_dim
        self._dead_fraction = initial_dead_fraction
        self._override_lit: bool | None = None
        self._last_natural_lit = self._natural()

    @classmethod
    def attach(cls, mech: MechanismITF, entity: str, cfgs: Sequence[SemanticCfg], *, polygon: Sequence[tuple[float, float]] | None = None) -> SemanticKind:
        discrete, continuous, predicates = cls._classify(cfgs)
        params = _merge_params(cfgs)
        light_on = params.get('light_on')
        initial_lit = True
        initial_dim = 1.0
        initial_dead_fraction = 0.0
        for cfg in cfgs:
            if cfg.value is None:
                continue
            if cfg.name == 'dim':
                initial_dim = float(cfg.value)
            elif cfg.name == 'dead_fraction':
                initial_dead_fraction = float(cfg.value)
            elif cfg.name == 'lit':
                if light_on is not None:
                    raise ValueError(f"light {entity!r}: lit value and light_on are exclusive")
                initial_lit = bool(cfg.value)
        return cls(mech, entity, light_on, initial_lit, initial_dim, initial_dead_fraction, discrete, continuous, predicates)

    def _natural(self) -> bool:
        return self._mech._semantics.regime(self._light_on) if self._light_on is not None else self._initial_lit

    def _live_lit(self) -> bool:
        return self._override_lit if self._override_lit is not None else self._natural()

    def settle(self) -> None:
        natural = self._natural()
        if natural != self._last_natural_lit:
            self._override_lit = None
        self._last_natural_lit = natural

    def reset(self) -> None:
        self._override_lit = None
        self._last_natural_lit = self._natural()
        self._dim = self._initial_dim
        self._dead_fraction = self._initial_dead_fraction

    def set_value(self, field: str, value: str) -> None:
        if field not in self.WRITABLE:
            raise ValueError('field not writable')
        if field == 'dead_fraction' and field not in self._continuous:
            raise ValueError('dead_fraction not declared on this light')
        coerced = _coerce(self, field, value)
        if field == 'lit':
            self._override_lit = bool(coerced)
            return
        fraction = float(coerced)
        low, high = self.RANGES[field]
        if not low <= fraction <= high:
            raise ValueError('malformed value')
        if field == 'dim':
            self._dim = fraction
        else:
            self._dead_fraction = fraction

    def output(self) -> tuple[float, float]:
        """Rendered (dim, dead fraction), dim 0.0 while unlit."""
        dim = self._dim if self._live_lit() else 0.0
        dead_fraction = self._dead_fraction if 'dead_fraction' in self._continuous else 0.0
        return dim, dead_fraction

    def snapshot(self) -> SemanticEntitySnapshot:
        continuous: dict[str, float] = {}
        predicates: dict[str, bool] = {}
        if 'lit' in self._predicates:
            predicates['lit'] = self._live_lit()
        if 'dim' in self._continuous:
            continuous['dim'] = self._dim
        if 'dead_fraction' in self._continuous:
            continuous['dead_fraction'] = self._dead_fraction
        return SemanticEntitySnapshot(self._entity, self.KIND, {}, continuous, predicates)


@semantic_kind('occupancy_cap')
class OccupancyCapSemantics(SemanticKind):
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac', 'mujoco', 'flatsim'})

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
        cap = next((cfg.value for cfg in cfgs if cfg.name == 'cap' and cfg.value is not None), 0)
        ring = [(float(x), float(y)) for x, y in (polygon or [])]
        return cls(mech, entity, ring, float(cap), discrete, continuous, predicates)

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


@semantic_kind('articulated')
class ArticulatedSemantics(SemanticKind):
    SUPPORTED_SIMS = frozenset({'gazebo', 'isaac', 'mujoco', 'flatsim', 'dummy'})

    DISCRETE = ('state', 'target')
    PREDICATES = ('moving', 'blocked')
    FAMILY_QUANTA: ClassVar[dict[str, float]] = {'openness': 0.01, 'pos': 0.01, 'angle': 0.01}
    Q_QUANTA: ClassVar[dict[str, float]] = {'revolute': 0.01, 'continuous': 0.01, 'prismatic': 0.001}

    def __init__(self, entity: str, runtime: ArticulatedObject) -> None:
        articulation = runtime.articulation
        continuous, discrete = articulation.fields(articulation.initial())
        super().__init__(entity, (*self.DISCRETE, *discrete), tuple(continuous), (*articulation.states, *self.PREDICATES))
        self._runtime = runtime
        self._writable = articulation.writable_fields()

    @classmethod
    def attach(cls, mech: MechanismITF, entity: str, cfgs: Sequence[SemanticCfg], *, polygon: Sequence[tuple[float, float]] | None = None) -> SemanticKind:
        runtime = next((runtime for runtime in mech._articulated.values() if runtime.entity == entity), None)
        if runtime is None:
            raise SemanticAttachError(f"no articulated object {entity!r}")
        return cls(entity, runtime)

    def has_field(self, field: str) -> bool:
        return field in self._discrete or field in self._continuous or field in self._predicates or field in self._writable

    @classmethod
    def is_continuous(cls, field: str) -> bool:
        """Every state field but `state`, `target` and the `setting.` detents is a number."""
        return field not in cls.DISCRETE and not field.startswith('setting.')

    def quantum(self, field: str) -> float:
        family, _, name = field.partition('.')
        if family in self.FAMILY_QUANTA:
            return self.FAMILY_QUANTA[family]
        joint = self._runtime.articulation.joint_map.get(name) if family == 'q' else None
        return self.Q_QUANTA.get(str(joint.type), 1.0) if joint is not None else 1.0

    def reset(self) -> None:
        self._runtime.reset()

    def set_value(self, field: str, value: str) -> None:
        """Write `target` (a state name), `openness` or a writable family field, the joints it moves get new targets."""
        if field not in self._writable:
            raise ValueError('field not writable')
        runtime = self._runtime
        try:
            runtime.command(runtime.articulation.target_for(field, value, runtime.target))
        except ValueError:
            raise ValueError('malformed value') from None

    def snapshot(self) -> SemanticEntitySnapshot:
        runtime = self._runtime
        articulation = runtime.articulation
        continuous, settings = articulation.fields(runtime.current)
        discrete = {'state': articulation.state_of(runtime.current), 'target': articulation.state_of(runtime.target), **settings}
        predicates = {state: _at_state(runtime.current, values) for state, values in articulation.states.items()}
        predicates['moving'] = runtime.moving
        predicates['blocked'] = runtime.blocked
        return SemanticEntitySnapshot(self._entity, self.KIND, discrete, continuous, predicates)


def _at_state(current: dict[str, float], values: dict[str, float]) -> bool:
    return all(abs(current.get(joint, 0.0) - value) <= STATE_TOLERANCE for joint, value in values.items())


def _diff(prev: SemanticEntitySnapshot | None, snap: SemanticEntitySnapshot, quantum: Callable[[str], float]) -> list[SemanticChange]:
    """Emit one SemanticChange per changed field. Continuous fields use the per-field quantum (F2)."""
    prev_discrete = prev.discrete if prev is not None else {}
    prev_continuous = prev.continuous if prev is not None else {}
    prev_predicates = prev.predicates if prev is not None else {}
    changes: list[SemanticChange] = []
    for name, value in snap.discrete.items():
        old = prev_discrete.get(name)
        if old != value and _stringify(old) != _stringify(value):
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
        self._regime_stack: set[str] = set()

    def set_sim(self, name: str) -> None:
        self._sim = name

    def attach(self, kind: str, entity: str, cfgs: Sequence[SemanticCfg] = (), *, polygon: Sequence[tuple[float, float]] | None = None) -> bool:
        """Instantiate the kind for entity, skipping unsupported sims. True when an instance was created."""
        kind_cls = _SEMANTIC_KINDS.get(kind)
        if kind_cls is None:
            self._mech.node.get_logger().info(f"semantics: unknown kind {kind!r} for {entity!r}, skipping")
            return False
        if any(instance.KIND == kind for instance in self._instances.get(entity, [])):
            self._mech.node.get_logger().warning(f"semantics: {kind!r} already attached to {entity!r}, skipping duplicate")
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
        """Step, settle, then diff and batch changes: separate passes, so cross-instance
        regime consults see this tick's state regardless of attach order."""
        self._now = now
        changes: list[SemanticChange] = []
        for instances in self._instances.values():
            for instance in instances:
                instance.step(now)
        for instances in self._instances.values():
            for instance in instances:
                instance.settle()
        for entity, instances in self._instances.items():
            for instance in instances:
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
            if instance.has_field(field):
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
        """OR of every scripted instance currently asserting the named regime, negated by a leading '!'. A consult cycle resolves to False."""
        negated = name.startswith('!')
        bare = name.removeprefix('!')
        if bare in self._regime_stack:
            return False
        self._regime_stack.add(bare)
        try:
            asserted = any(instance.asserts_regime(bare) for instances in self._instances.values() for instance in instances)
        finally:
            self._regime_stack.discard(bare)
        return asserted != negated

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

    def door_driven(self, door_name: str) -> bool:
        """True while any pressed plate drives door_name."""
        return any(instance.drives_door(door_name) for instances in self._instances.values() for instance in instances if isinstance(instance, PressurePlateSemantics))

    def elevator_recalled(self, elev_name: str, now: float) -> bool:
        """HOOK-C: True when a recall regime registered for this elevator currently asserts."""
        recall = self._recall.get(elev_name)
        return recall is not None and self.regime(recall)

    def light_outputs(self) -> dict[str, tuple[float, float]]:
        """Current (output level, dead fraction) of every attached light, keyed by entity."""
        return {entity: instance.output() for entity, instances in self._instances.items() for instance in instances if isinstance(instance, LightSemantics)}
