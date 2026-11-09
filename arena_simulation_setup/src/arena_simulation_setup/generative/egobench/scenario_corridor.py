"""Encounters in a corridor world: along its longest straight runs, and crossings out of the side branches of a run."""

from __future__ import annotations

import math

import attrs

from arena_simulation_setup.generative.world.layout import Layout, Segment
from arena_simulation_setup.shared.pathing import Point

from .encounter import CLEARANCE, CROSSING, LEAD_IN_M, LEAD_OUT_M, PED_RADIUS, ROBOT_RADIUS, Generated, Leg, Placement, Rejected, build, crossed, crossing_direction, fits, parallel, summary
from .sampler import Sample

_EPS = 1e-6
_PERPENDICULAR = 0.01  # |cos| below which a branch counts as a right angle


@attrs.frozen
class Arm:
    """A side branch leaving a run at `arc`, `direction` pointing away from the run."""

    arc: float
    direction: Point
    length: float
    opposite: float  # length of the branch straight across the run, 0 = none


@attrs.frozen
class Run:
    leg: Leg
    width: float
    arms: tuple[Arm, ...]

    def trimmed(self, margin: float) -> Run:
        """The run with `margin` cut off both ends, so a body at either end stays off the end wall."""
        leg = self.leg.sub(margin, self.leg.length - margin)
        arms = tuple(attrs.evolve(arm, arc=arm.arc - margin) for arm in self.arms if margin < arm.arc < self.leg.length - margin)
        return Run(leg=leg, width=self.width, arms=arms)

    def reversed(self) -> Run:
        """The same run driven the other way."""
        arms = tuple(attrs.evolve(arm, arc=self.leg.length - arm.arc) for arm in reversed(self.arms))
        return Run(leg=Leg(start=self.leg.end, end=self.leg.start), width=self.width, arms=arms)


def _key(p: Point) -> tuple[int, int]:
    return (round(p[0] / _EPS), round(p[1] / _EPS))


def _unit(a: Point, b: Point) -> Point:
    d = math.dist(a, b)
    return ((b[0] - a[0]) / d, (b[1] - a[1]) / d)


def _next(at: Point, heading: Point, used: set[int], segments: list[Segment], ends: dict[tuple[int, int], list[int]]) -> int | None:
    """The unused segment leaving `at` straight on along `heading`."""
    for i in ends.get(_key(at), []):
        if i in used:
            continue
        s = segments[i]
        far = s.b if _key(s.a) == _key(at) else s.a
        u = _unit(at, far)
        if u[0] * heading[0] + u[1] * heading[1] > 1 - _EPS:
            return i
    return None


def _far(s: Segment, near: Point) -> Point:
    return s.b if _key(s.a) == _key(near) else s.a


def _arms(leg: Leg, run: set[int], segments: list[Segment], width: float) -> tuple[Arm, ...]:
    """Every square segment that ends on or passes through the run, as the branches it leaves on either side."""
    u, n = leg.direction, leg.normal
    arms: list[Arm] = []
    for i, s in enumerate(segments):
        if i in run or math.dist(s.a, s.b) < _EPS:
            continue
        d = _unit(s.a, s.b)
        if abs(d[0] * u[0] + d[1] * u[1]) > _PERPENDICULAR:
            continue
        side_a = (s.a[0] - leg.start[0]) * n[0] + (s.a[1] - leg.start[1]) * n[1]
        side_b = (s.b[0] - leg.start[0]) * n[0] + (s.b[1] - leg.start[1]) * n[1]
        if side_a * side_b > 0 and min(abs(side_a), abs(side_b)) > _EPS:
            continue
        t = side_a / (side_a - side_b)
        cross = (s.a[0] + (s.b[0] - s.a[0]) * t, s.a[1] + (s.b[1] - s.a[1]) * t)
        arc = (cross[0] - leg.start[0]) * u[0] + (cross[1] - leg.start[1]) * u[1]
        if not _EPS < arc < leg.length - _EPS:
            continue
        for side, other in ((side_a, side_b), (side_b, side_a)):
            if abs(side) > width / 2:
                opposite = abs(other) if side * other < 0 and abs(other) > width / 2 else 0.0
                arms.append(Arm(arc=arc, direction=n if side > 0 else (-n[0], -n[1]), length=abs(side), opposite=opposite))
    return tuple(sorted(arms, key=lambda a: a.arc))


def runs(layout: Layout) -> list[Run]:
    """Maximal straight chains of the layout's segments, longest first."""
    segments = layout.segments
    ends: dict[tuple[int, int], list[int]] = {}
    for i, s in enumerate(segments):
        ends.setdefault(_key(s.a), []).append(i)
        ends.setdefault(_key(s.b), []).append(i)
    seen: set[frozenset[int]] = set()
    found: list[Run] = []
    for i, s in enumerate(segments):
        if math.dist(s.a, s.b) < _EPS:
            continue
        chain = {i}
        start, end = s.a, s.b
        forward = _unit(start, end)
        while (j := _next(end, forward, chain, segments, ends)) is not None:
            chain.add(j)
            end = _far(segments[j], end)
        backward = (-forward[0], -forward[1])
        while (j := _next(start, backward, chain, segments, ends)) is not None:
            chain.add(j)
            start = _far(segments[j], start)
        if frozenset(chain) in seen:
            continue
        seen.add(frozenset(chain))
        leg = Leg(start=start, end=end)
        width = min(segments[k].width for k in chain)
        found.append(Run(leg=leg, width=width, arms=_arms(leg, chain, segments, width)))
    return sorted(found, key=lambda r: r.leg.length, reverse=True)


def widest_offset(width: float) -> float:
    """The largest lateral offset a corridor of `width` holds with both bodies clear of its walls."""
    return width - 2 * (max(ROBOT_RADIUS, PED_RADIUS) + CLEARANCE)


def _crossing(sample: Sample, run: Run, arm: Arm) -> Placement | Rejected:
    robot = run.leg
    v = crossing_direction(sample, robot)
    if arm.direction[0] * v[0] + arm.direction[1] * v[1] > -1 + _EPS:
        return Rejected("the branch lies on the other side")
    meet = robot.at(arm.arc)
    if arm.opposite > 0:
        path = [(meet[0] + v[0] * min(LEAD_OUT_M, arm.opposite), meet[1] + v[1] * min(LEAD_OUT_M, arm.opposite))]
    else:
        side = run.width / 2 - PED_RADIUS - CLEARANCE
        far = (meet[0] + v[0] * side, meet[1] + v[1] * side)
        u = robot.direction
        path = [far, (far[0] - u[0] * LEAD_OUT_M, far[1] - u[1] * LEAD_OUT_M)]
    return crossed(sample, robot, arm.arc, path, lead_in=min(LEAD_IN_M, arm.length))


def generate(layout: Layout, sample: Sample) -> Generated | Rejected:
    """The first placement of `sample` that fits the corridor world, else why none did."""
    free = layout.geometry()
    reasons: list[str] = []
    margin = ROBOT_RADIUS + CLEARANCE
    usable = [found.trimmed(margin) for found in runs(layout) if found.leg.length > 2 * margin]
    for run in (driven for found in usable for driven in (found, found.reversed())):
        if sample.encounter in CROSSING:
            candidates = [_crossing(sample, run, arm) for arm in run.arms]
            if not candidates:
                reasons.append(f"the {run.leg.length:.1f} m run has no side branch to cross from")
        elif abs(sample.lateral_offset) > widest_offset(run.width):
            reasons.append(f"a lateral offset of {abs(sample.lateral_offset):.2f} m does not fit a {run.width:.2f} m corridor, which holds at most {widest_offset(run.width):.2f} m")
            continue
        else:
            candidates = [parallel(sample, run.leg)]
        for placement in candidates:
            if isinstance(placement, Rejected):
                reasons.append(placement.reason)
                continue
            why = fits(placement, free)
            if why is None:
                return build(sample, placement, notes=[f"corridor run of {run.leg.length:.2f} m, {run.width:.2f} m wide"])
            reasons.append(why)
    return Rejected(summary(reasons) or "the world has no corridor run")
