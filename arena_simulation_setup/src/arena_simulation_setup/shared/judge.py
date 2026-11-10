"""Pure predicate evaluation shared by the online task judge and offline replay."""

from __future__ import annotations

import math
import typing
from collections.abc import Callable, Mapping

import attrs
import numpy as np

from arena_simulation_setup.shared.conditions import (
    AllNode,
    AnyNode,
    Atom,
    CompareValue,
    EntityAtom,
    EpisodeCondition,
    FamilyTest,
    FieldTest,
    HoldNode,
    MembershipAtom,
    Node,
    NotAtom,
    NotNode,
    PlaysAtom,
    SeesAtom,
    WithinAtom,
    node_children,
    parse_point,
)

if typing.TYPE_CHECKING:
    import shapely

    from arena_simulation_setup.shared.task import GoToPhase, TaskPhase
    from arena_simulation_setup.tree.World import Level

FLOAT_TOLERANCE = 1e-6
PARK_DRIFT_M = 0.05
PARK_TURN_RAD = math.radians(5.0)
WALL_GRID_RESOLUTION_M = 0.05

Zones = Mapping[str, 'shapely.Polygon | shapely.MultiPolygon']
FieldLookup = Callable[[str, str], str | None]
FamilyLookup = Callable[[str, str], Mapping[str, str] | None]


@attrs.frozen
class WallGrid:
    """Wall cells of the flattened world as a north-up boolean grid (row 0 is the top edge), `origin` its lower-left corner."""

    cells: np.ndarray
    origin: tuple[float, float]
    resolution: float

    @classmethod
    def of_level(cls, level: Level, resolution: float = WALL_GRID_RESOLUTION_M) -> WallGrid:
        """The level's walls rasterized at `resolution` and thickened by one cell, doors and elevator cabins left open."""
        grid, origin = level.render_grid(resolution=resolution)
        walls = grid < 128
        cells = walls.copy()
        cells[1:, :] |= walls[:-1, :]
        cells[:-1, :] |= walls[1:, :]
        cells[:, 1:] |= walls[:, :-1]
        cells[:, :-1] |= walls[:, 1:]
        return cls(cells=cells, origin=(float(origin[0]), float(origin[1])), resolution=resolution)

    def cell(self, x: float, y: float) -> tuple[int, int]:
        rows, cols = self.cells.shape
        row = min(max(int(rows - (y - self.origin[1]) / self.resolution), 0), rows - 1)
        col = min(max(int((x - self.origin[0]) / self.resolution), 0), cols - 1)
        return row, col

    def clear(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """Whether no wall cell lies on the segment from `a` to `b`."""
        steps = int(math.hypot(b[0] - a[0], b[1] - a[1]) / self.resolution) + 2
        for i in range(steps):
            f = i / (steps - 1)
            row, col = self.cell(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            if self.cells[row, col]:
                return False
        return True


@attrs.frozen
class Sample:
    """World state at one instant: robot poses (x, y, yaw), ped positions (x, y), ped gesture clips and semantic fields by name, a missing subject is nowhere if in `known`, else unresolvable."""

    t: float
    robots: Mapping[str, tuple[float, float, float]] = attrs.field(factory=dict)
    peds: Mapping[str, tuple[float, float]] = attrs.field(factory=dict)
    field: FieldLookup = attrs.field(default=lambda entity, name: None)
    known: frozenset[str] = attrs.field(factory=frozenset)
    clips: Mapping[str, frozenset[str]] = attrs.field(factory=dict)
    walls: WallGrid | None = None
    family: FamilyLookup = attrs.field(default=lambda entity, name: None)

    def point(self, subject: str, robot: str) -> tuple[float, float] | None | typing.Literal[False]:
        """Planar position of a subject or an `x,y` literal, False when it is known but absent, None when it is unknown."""
        literal = parse_point(subject)
        if literal is not None:
            return literal
        name = robot if subject == 'robot' else subject
        pose = self.robots.get(name)
        if pose is not None:
            return pose[0], pose[1]
        ped = self.peds.get(name)
        if ped is not None:
            return ped
        return False if name in self.known or name in self.peds or name in self.robots else None


def zone_polygons(level: Level) -> dict[str, shapely.Polygon | shapely.MultiPolygon]:
    """Zone, door, elevator and stair polygons of a compacted level, keyed by name."""
    import shapely

    polygons: dict[str, shapely.Polygon | shapely.MultiPolygon] = {}
    for name in level.zone_ref_names():
        corners = level.lookup_zone_polygon(name)
        if corners is None or len(corners) < 3:
            continue
        polygons[name] = shapely.Polygon([(c.x, c.y) for c in corners])
    groups: dict[str, list[shapely.Polygon]] = {}
    for stair in level.stairs:
        half = polygons[stair.name]
        if stair.group and isinstance(half, shapely.Polygon):
            groups.setdefault(stair.group, []).append(half)
    polygons.update((name, shapely.MultiPolygon(halves)) for name, halves in groups.items())
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
    if isinstance(atom, PlaysAtom):
        clips = sample.clips.get(atom.subject)
        if clips is not None:
            return atom.clip in clips
        return False if atom.subject in sample.known and atom.subject not in sample.peds else None
    a = sample.point(atom.subject, robot)
    b = sample.point(atom.other, robot)
    if a is None or b is None:
        return None
    if a is False or b is False:
        return False
    if isinstance(atom, SeesAtom):
        if math.hypot(a[0] - b[0], a[1] - b[1]) > atom.range:
            return False
        return None if sample.walls is None else sample.walls.clear(a, b)
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= atom.radius


def compare(recorded: str, op: str, value: CompareValue) -> bool | None:
    """One recorded value against a comparison, None when an ordering op meets a non-numeric value."""
    if op in ('eq', 'ne'):
        equal = values_equal(recorded, value if isinstance(value, str) else str(value))
        return equal if op == 'eq' else not equal
    try:
        number = float(recorded)
    except (TypeError, ValueError):
        return None
    if isinstance(value, tuple):
        return value[0] - FLOAT_TOLERANCE <= number <= value[1] + FLOAT_TOLERANCE
    bound = float(value)
    if op == 'le':
        return number <= bound + FLOAT_TOLERANCE
    if op == 'ge':
        return number >= bound - FLOAT_TOLERANCE
    if op == 'lt':
        return number < bound - FLOAT_TOLERANCE
    return number > bound + FLOAT_TOLERANCE


def _all(verdicts: list[bool | None]) -> bool | None:
    if any(v is False for v in verdicts):
        return False
    return None if any(v is None for v in verdicts) else True


def _any(verdicts: list[bool | None]) -> bool | None:
    if any(v is True for v in verdicts):
        return True
    return None if any(v is None for v in verdicts) else False


def _family_holds(test: FamilyTest, sample: Sample) -> bool | None:
    members = sample.family(test.entity, test.family)
    if members is None:
        return None
    if not members:
        return False
    verdicts = [compare(recorded, test.op, test.value) for recorded in members.values()]
    return _any(verdicts) if test.quantifier == 'some' else _all(verdicts)


@attrs.define
class NodeMonitor:
    """Truth of a condition node over successive samples, every child evaluated each step, holds timed per node path."""

    node: Node
    robot: str = 'robot'
    _since: dict[tuple[int, ...], float] = attrs.field(factory=dict, init=False)

    def reset(self) -> None:
        self._since.clear()

    def step(self, sample: Sample, zones: Zones) -> bool | None:
        return self._eval(self.node, (), sample, zones)

    def _eval(self, node: Node, path: tuple[int, ...], sample: Sample, zones: Zones) -> bool | None:
        if isinstance(node, (AllNode, AnyNode)):
            verdicts = [self._eval(child, (*path, i), sample, zones) for i, child in enumerate(node.nodes)]
            return _all(verdicts) if isinstance(node, AllNode) else _any(verdicts)
        if isinstance(node, NotNode):
            inner = self._eval(node.node, (*path, 0), sample, zones)
            return None if inner is None else not inner
        if isinstance(node, HoldNode):
            inner = self._eval(node.node, (*path, 0), sample, zones)
            if inner is not True:
                self._since.pop(path, None)
                return inner
            since = self._since.setdefault(path, sample.t)
            return sample.t - since >= node.seconds
        if isinstance(node, FieldTest):
            recorded = sample.field(node.entity, node.field)
            return None if recorded is None else compare(recorded, node.op, node.value)
        if isinstance(node, FamilyTest):
            return _family_holds(node, sample)
        return atom_holds(node, sample, zones, self.robot)


def _has_hold(node: Node) -> bool:
    return isinstance(node, HoldNode) or any(_has_hold(child) for child in node_children(node))


def node_holds(node: Node, sample: Sample, zones: Zones, robot: str = 'robot') -> bool | None:
    """Truth of a node without holds on one sample, None when an input it needs is missing."""
    if _has_hold(node):
        raise ValueError('node_holds takes a node without holds, step a NodeMonitor for one with holds')
    return NodeMonitor(node, robot).step(sample, zones)


@attrs.define
class UntilMonitor:
    """Exit test of a phase or loop over samples, met on a sample its node holds on."""

    until: Node
    robot: str = 'robot'
    _monitor: NodeMonitor = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        self._monitor = NodeMonitor(self.until, self.robot)

    def step(self, sample: Sample, zones: Zones) -> bool:
        return self._monitor.step(sample, zones) is True


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
    assert p_series is not None and q_series is not None
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
    _p: NodeMonitor = attrs.field(init=False)
    _q: NodeMonitor | None = attrs.field(init=False)
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
        self._p = NodeMonitor(self.condition.p_node, self.robot)
        q = self.condition.q_node
        self._q = NodeMonitor(q, self.robot) if q is not None else None

    def update(self, sample: Sample, zones: Zones) -> bool | None:
        """Feed one sample, returning the verdict once it is decided, else None."""
        if self.decided:
            return self.verdict
        p = self._p.step(sample, zones)
        q = self._q.step(sample, zones) if self._q is not None else False
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
    _until: UntilMonitor | None = attrs.field(init=False)
    _stop: UntilMonitor | None = attrs.field(init=False)
    _park: ParkTimer = attrs.field(factory=ParkTimer, init=False)
    conditions: list[ConditionMonitor] = attrs.field(init=False)
    arrived_at: float | None = attrs.field(default=None, init=False)
    held: bool = attrs.field(default=False, init=False)
    met_at: float | None = attrs.field(default=None, init=False)
    misfire: str | None = attrs.field(default=None, init=False)

    def __attrs_post_init__(self) -> None:
        from arena_simulation_setup.shared.task import ReachPhase

        until = self.phase.until_node
        self._until = UntilMonitor(until, self.robot) if until is not None else None
        stop = self.phase.stop_node if isinstance(self.phase, ReachPhase) else None
        self._stop = UntilMonitor(stop, self.robot) if stop is not None else None
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
        elif action_done or (self._stop is not None and self._stop.step(sample, zones)):
            if self.arrived_at is None:
                self.arrived_at = sample.t
            self.held = True

        if not self.held:
            return False
        if self._until is not None and not self._until.step(sample, zones):
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
    'WALL_GRID_RESOLUTION_M',
    'ConditionMonitor',
    'FamilyLookup',
    'FieldLookup',
    'NodeMonitor',
    'ParkTimer',
    'PhaseMonitor',
    'Sample',
    'UntilMonitor',
    'WallGrid',
    'Zones',
    'arrived',
    'atom_holds',
    'compare',
    'node_holds',
    'operator_verdict',
    'values_equal',
    'zone_polygons',
]
