"""Pure predicate evaluation shared by the online task judge and offline replay."""

from __future__ import annotations

import math
import typing
from collections.abc import Callable, Mapping

import attrs
import numpy as np

from arena_simulation_setup.shared.conditions import (
    Atom,
    EntityAtom,
    EpisodeCondition,
    MembershipAtom,
    NotAtom,
    WithinAtom,
    parse_atom,
)

if typing.TYPE_CHECKING:
    import shapely

    from arena_simulation_setup.shared.task import GoToPhase, TaskPhase
    from arena_simulation_setup.tree.World import LevelDescription

FLOAT_TOLERANCE = 1e-6
PARK_DRIFT_M = 0.05
PARK_TURN_RAD = math.radians(5.0)

Zones = Mapping[str, 'shapely.Polygon']
FieldLookup = Callable[[str, str], str | None]


@attrs.frozen
class Sample:
    """World state at one instant: robot poses (x, y, yaw) and ped positions (x, y) by name, a missing subject is nowhere if in `known`, else unresolvable."""

    t: float
    robots: Mapping[str, tuple[float, float, float]] = attrs.field(factory=dict)
    peds: Mapping[str, tuple[float, float]] = attrs.field(factory=dict)
    field: FieldLookup = attrs.field(default=lambda entity, name: None)
    known: frozenset[str] = attrs.field(factory=frozenset)

    def point(self, subject: str, robot: str) -> tuple[float, float] | None | typing.Literal[False]:
        """Planar position of a subject, False when it is known but absent, None when it is unknown."""
        name = robot if subject == 'robot' else subject
        pose = self.robots.get(name)
        if pose is not None:
            return pose[0], pose[1]
        ped = self.peds.get(name)
        if ped is not None:
            return ped
        return False if name in self.known or name in self.peds or name in self.robots else None


def zone_polygons(level: LevelDescription) -> dict[str, shapely.Polygon]:
    """Zone, door and elevator polygons of a compacted level, keyed by name."""
    import shapely

    polygons: dict[str, shapely.Polygon] = {}
    for name in level.zone_ref_names():
        corners = level.lookup_zone_polygon(name)
        if corners is None or len(corners) < 3:
            continue
        polygons[name] = shapely.Polygon([(c.x, c.y) for c in corners])
    return polygons


def values_equal(recorded: str, expected: str) -> bool:
    """Float compare within tolerance when both parse as float, else exact string."""
    try:
        return abs(float(recorded) - float(expected)) <= FLOAT_TOLERANCE
    except (TypeError, ValueError):
        return recorded == expected


def atom_holds(atom: Atom, sample: Sample, zones: Zones, robot: str = 'robot') -> bool | None:
    """Truth of one atom on one sample, None when an input it needs is missing."""
    import shapely

    if isinstance(atom, NotAtom):
        inner = atom_holds(atom.atom, sample, zones, robot)
        return None if inner is None else not inner
    if isinstance(atom, EntityAtom):
        current = sample.field(atom.entity, atom.field)
        return None if current is None else values_equal(current, atom.value)
    if isinstance(atom, MembershipAtom):
        polygon = zones.get(atom.zone)
        point = sample.point(atom.subject, robot)
        if polygon is None or point is None:
            return None
        if point is False:
            return False
        return bool(polygon.covers(shapely.Point(point)))
    a = sample.point(atom.subject, robot)
    b = sample.point(atom.other, robot)
    if a is None or b is None:
        return None
    if a is False or b is False:
        return False
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= atom.radius


def operator_verdict(
    op: str,
    p_series: np.ndarray | None,
    p_ok: bool,
    q_series: np.ndarray | None,
    q_ok: bool,
) -> bool | None:
    """One clause verdict from its atom series, None when any used atom is unresolvable."""
    if op in ('always', 'never', 'eventually'):
        if not p_ok:
            return None
        if op == 'always':
            return bool(np.all(p_series))
        if op == 'never':
            return not bool(np.any(p_series))
        return bool(np.any(p_series))

    if not (p_ok and q_ok):
        return None
    if op == 'before':
        first_p = _first_true(p_series)
        first_q = _first_true(q_series)
        return first_p is not None and (first_q is None or first_p < first_q)
    return not bool(np.any(p_series & q_series))


def _first_true(series: np.ndarray) -> int | None:
    idxs = np.flatnonzero(series)
    return int(idxs[0]) if len(idxs) else None


@attrs.define
class ConditionMonitor:
    """Incremental verdict of one clause over the samples of its scope, same truth table as operator_verdict."""

    condition: EpisodeCondition
    robot: str = 'robot'
    _p: Atom = attrs.field(init=False)
    _q: Atom | None = attrs.field(init=False)
    _unknown: bool = attrs.field(default=False, init=False)
    _any_p: bool = attrs.field(default=False, init=False)
    _all_p: bool = attrs.field(default=True, init=False)
    _first_p: int | None = attrs.field(default=None, init=False)
    _first_q: int | None = attrs.field(default=None, init=False)
    _cooccur: bool = attrs.field(default=False, init=False)
    _n: int = attrs.field(default=0, init=False)
    verdict: bool | None = attrs.field(default=None, init=False)
    decided: bool = attrs.field(default=False, init=False)

    def __attrs_post_init__(self) -> None:
        self._p = parse_atom(self.condition.p)
        self._q = parse_atom(self.condition.q) if self.condition.q is not None else None

    def update(self, sample: Sample, zones: Zones) -> bool | None:
        """Feed one sample, returning the verdict once it is decided, else None."""
        if self.decided:
            return self.verdict
        p = atom_holds(self._p, sample, zones, self.robot)
        q = atom_holds(self._q, sample, zones, self.robot) if self._q is not None else False
        if p is None or q is None:
            self._unknown = True
            self._n += 1
            return None
        op = self.condition.op
        if p:
            self._any_p = True
            if self._first_p is None:
                self._first_p = self._n
        else:
            self._all_p = False
        if q and self._first_q is None:
            self._first_q = self._n
        if p and q:
            self._cooccur = True
        self._n += 1

        if op == 'always' and not p:
            return self._decide(False)
        if op == 'never' and p:
            return self._decide(False)
        if op == 'eventually' and p:
            return self._decide(True)
        if op == 'never_during' and self._cooccur:
            return self._decide(False)
        if op == 'before' and self._first_q is not None and self._first_p is None:
            return self._decide(False)
        if op == 'before' and self._first_p is not None and (self._first_q is None or self._first_p < self._first_q):
            return self._decide(True)
        return None

    def finish(self) -> bool | None:
        """Verdict at the end of the scope, None when an atom was ever unresolvable."""
        if self.decided:
            return self.verdict
        if self._unknown or self._n == 0:
            return self._decide(None)
        op = self.condition.op
        if op == 'always':
            return self._decide(self._all_p)
        if op == 'never':
            return self._decide(not self._any_p)
        if op == 'eventually':
            return self._decide(self._any_p)
        if op == 'never_during':
            return self._decide(not self._cooccur)
        return self._decide(self._first_p is not None and (self._first_q is None or self._first_p < self._first_q))

    def _decide(self, verdict: bool | None) -> bool | None:
        self.verdict = verdict
        self.decided = True
        return verdict


@attrs.define
class ParkTimer:
    """Sim seconds a robot has stayed within PARK_DRIFT_M and PARK_TURN_RAD of one pose."""

    _anchor: tuple[float, float, float] | None = None
    _since: float = 0.0

    def reset(self) -> None:
        self._anchor = None

    def held_for(self, x: float, y: float, yaw: float, t: float) -> float:
        anchor = self._anchor
        if anchor is None or math.hypot(x - anchor[0], y - anchor[1]) > PARK_DRIFT_M or abs((yaw - anchor[2] + math.pi) % (2 * math.pi) - math.pi) > PARK_TURN_RAD:
            self._anchor = (x, y, yaw)
            self._since = t
        return t - self._since


def arrived(phase: GoToPhase, sample: Sample, zones: Zones, robot: str) -> bool | None:
    """Whether the robot has reached the goto target on this sample, None without a pose."""
    pose = sample.robots.get(robot)
    if pose is None:
        return None
    if phase.target is not None:
        if phase.target in zones:
            return atom_holds(MembershipAtom(subject='robot', zone=phase.target), sample, zones, robot)
        radius = phase.tolerance_radius if phase.tolerance_radius is not None else 0.0
        return atom_holds(WithinAtom(subject='robot', radius=radius, other=phase.target), sample, zones, robot)
    if phase.pose is None:
        return None
    radius = phase.tolerance_radius if phase.tolerance_radius is not None else 0.0
    if math.hypot(pose[0] - phase.pose.position.x, pose[1] - phase.pose.position.y) > radius:
        return False
    tol_ang = phase.tolerance_angle if phase.tolerance_angle is not None else 0.0
    if tol_ang > 0:
        dyaw = (pose[2] - phase.pose.orientation.to_yaw() + math.pi) % (2 * math.pi) - math.pi
        if abs(dyaw) > tol_ang:
            return False
    return True


@attrs.define
class PhaseMonitor:
    """Completion of one phase over samples: arrival, park, action result, then `until`, plus its scoped conditions."""

    phase: TaskPhase
    robot: str = 'robot'
    _until: Atom | None = attrs.field(init=False)
    _park: ParkTimer = attrs.field(factory=ParkTimer, init=False)
    conditions: list[ConditionMonitor] = attrs.field(init=False)
    arrived_at: float | None = attrs.field(default=None, init=False)
    held: bool = attrs.field(default=False, init=False)
    met_at: float | None = attrs.field(default=None, init=False)
    misfire: str | None = attrs.field(default=None, init=False)

    def __attrs_post_init__(self) -> None:
        self._until = parse_atom(self.phase.until) if self.phase.until is not None else None
        self.conditions = [ConditionMonitor(c, self.robot) for c in self.phase.conditions]

    @property
    def met(self) -> bool:
        return self.met_at is not None

    @property
    def waiting(self) -> bool:
        """Arrived or finished acting, and holding for park or `until`."""
        return self.arrived_at is not None and self.met_at is None

    def step(self, sample: Sample, zones: Zones, action_done: bool | None = None, signal: str | None = None) -> bool:
        """Feed one sample and return whether the phase is met, a `signal` sent outside the tolerance sets `misfire` instead."""
        from arena_simulation_setup.shared.task import GoToPhase

        for monitor in self.conditions:
            monitor.update(sample, zones)
        if self.met_at is not None:
            return True
        if self.misfire is not None:
            return False

        if isinstance(self.phase, GoToPhase) and self.phase.signal:
            if signal != self.phase.signal:
                return False
            pose = sample.robots.get(self.robot)
            at_goal = arrived(self.phase, sample, zones, self.robot)
            if not at_goal or pose is None:
                self.misfire = f"signaled {self.phase.signal} {self._goal_distance(sample):.2f} m from goal"
                return False
            self.arrived_at = sample.t
            self.held = True
        elif isinstance(self.phase, GoToPhase):
            at_goal = arrived(self.phase, sample, zones, self.robot)
            if at_goal:
                if self.arrived_at is None:
                    self.arrived_at = sample.t
                if not self.held:
                    hold = self.phase.hold_time or 0.0
                    pose = sample.robots[self.robot]
                    if hold <= 0 or self._park.held_for(pose[0], pose[1], pose[2], sample.t) >= hold:
                        self.held = True
            elif not self.held:
                self._park.reset()
                if self.phase.until is None:
                    self.arrived_at = None
        elif action_done:
            if self.arrived_at is None:
                self.arrived_at = sample.t
            self.held = True

        if not self.held:
            return False
        if self._until is not None and atom_holds(self._until, sample, zones, self.robot) is not True:
            return False
        self.met_at = sample.t
        return True

    def _goal_distance(self, sample: Sample) -> float:
        from arena_simulation_setup.shared.task import GoToPhase

        pose = sample.robots.get(self.robot)
        target = self.phase.target if isinstance(self.phase, GoToPhase) else None
        goal = sample.point(target, self.robot) if target is not None else None
        if goal is None or goal is False:
            goal = (self.phase.pose.position.x, self.phase.pose.position.y) if isinstance(self.phase, GoToPhase) and self.phase.pose is not None else None
        if pose is None or goal is None:
            return math.inf
        return math.hypot(pose[0] - goal[0], pose[1] - goal[1])

    def finish(self) -> list[bool | None]:
        """Close the scope and return the verdict of every scoped condition."""
        return [monitor.finish() for monitor in self.conditions]


__all__ = [
    'FLOAT_TOLERANCE',
    'PARK_DRIFT_M',
    'PARK_TURN_RAD',
    'ConditionMonitor',
    'FieldLookup',
    'ParkTimer',
    'PhaseMonitor',
    'Sample',
    'Zones',
    'arrived',
    'atom_holds',
    'operator_verdict',
    'values_equal',
    'zone_polygons',
]
