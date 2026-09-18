"""Level C - ego-relative spawn geometry for injected edge-case pedestrians.

A per-agent tail makes failure *possible*; the geometry and timing of the encounter make
it *happen*. Given the robot's route, this module solves where and when a pedestrian must
start walking so that the two arrive at the same place at the same time.

The robot's route is a **polyline** (see :mod:`pathing`), because a real route through a
multi-room world bends through doorways; the encounter is placed along it by arc length and
the approach angle is measured against the local heading there. A two-point route is the
straight line, which is what an open-area case reduces to.

The pedestrian's own walk is a straight constant-speed line: that is what makes the
co-arrival exact, and it is validated against free space before the case is accepted.
Everything here is pure geometry - no simulation, no RNG, no ROS - so the solved case is
reproducible from the numbers written into ``cases.jsonl``.

The solver runs at scenario-build time: ``Task._reset_episode`` awaits ``tm_robots.reset()``
before ``tm_obstacles.reset()``, so the robot's start and goal are known - under a robot
mode that assigns its goal *during* reset (``tm_robots:=edge_case`` and ``:=scenario`` do,
``:=explore`` does not).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence

import attrs

from task_generator.shared import Position

from .pathing import iter_points, point_at_arc, route_length

Point = tuple[float, float]

IsValid = Callable[[Position], bool]

#: Spacing for the walk-segment free-space scan, metres. Finer than the robot's footprint,
#: so a doorway-width gap cannot be stepped over.
_SEGMENT_STEP_M = 0.25

#: Lead-out relaxations tried, in order, when no approach angle yields free space. Shrinking
#: the walk-through distance is less damaging to the experiment than moving the encounter.
_LEAD_OUT_SCALES = (1.0, 0.5, 0.25)

#: Step and bounds for the encounter-point sweep along the robot's route: not every point
#: suits an encounter, and the fraction also sets the approach length.
_ENCOUNTER_STEP = 0.05
_ENCOUNTER_MIN = 0.1
_ENCOUNTER_MAX = 0.9


def encounter_fractions(requested: float) -> Iterator[float]:
    """Encounter fractions to try, nearest the requested one first.

    The requested value always comes first, so an unobstructed case gets exactly what it
    asked for and ``deviated`` stays false.
    """
    yield requested
    seen = {round(requested, 6)}
    for k in range(1, int((_ENCOUNTER_MAX - _ENCOUNTER_MIN) / _ENCOUNTER_STEP) + 1):
        for candidate in (requested - k * _ENCOUNTER_STEP, requested + k * _ENCOUNTER_STEP):
            value = round(candidate, 6)
            if value in seen or not _ENCOUNTER_MIN <= value <= _ENCOUNTER_MAX:
                continue
            seen.add(value)
            yield value

#: A pedestrian must not materialise on top of the robot's start pose. Comfortably
#: clear of the robot footprint plus SAFE_DIST without ruling out short approaches, which
#: are the ones that fit in a furnished room.
_MIN_ROBOT_CLEARANCE_M = 1.0


class GeometryError(RuntimeError):
    """No valid interception could be constructed. Always carries the reason - a case that
    cannot be built must be recorded and aborted, never quietly relocated."""


@attrs.frozen
class Encounter:
    """A solved interception, and enough provenance to reconstruct it exactly."""

    spawn: Point
    goal: Point
    encounter: Point
    t_encounter: float
    angle_requested: float
    angle_achieved: float
    encounter_at_requested: float
    encounter_at_achieved: float
    lead_out_achieved: float
    ped_speed: float
    co_arrival_offset: float
    retries: int
    #: Seconds the pedestrian waits at its spawn before setting off. Non-zero when the full
    #: lead-in (`ped_speed * t_encounter`) did not fit in free space - a fast overtaker behind a
    #: start that sits against a wall - so it starts closer and later, arriving on time.
    depart_at: float = 0.0

    @property
    def designed_ttc(self) -> float:
        """Time-to-collision at spawn, by construction. The *achieved* min-TTC is a measured
        quantity and will differ - a Level D tail that delays the pedestrian changes it, which
        is the entire point of the tail."""
        return self.t_encounter

    @property
    def designed_pet(self) -> float:
        """Post-encroachment time: the designed gap between the two arrivals at ``encounter``."""
        return abs(self.co_arrival_offset)

    @property
    def deviated(self) -> bool:
        """Whether the solver had to move off the requested factors to find free space."""
        return (
            abs(self.angle_achieved - self.angle_requested) > 1e-9
            or abs(self.encounter_at_achieved - self.encounter_at_requested) > 1e-9
        )


# Vector helpers
# --------------


def _unit(a: Point, b: Point) -> tuple[Point, float]:
    """Direction from ``a`` to ``b``, and the distance."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        raise GeometryError("robot start and goal coincide; there is no route to intercept")
    return (dx / length, dy / length), length


def rotate(v: Point, degrees: float) -> Point:
    """Rotate counter-clockwise. 180 deg negates, which is what makes it head-on."""
    rad = math.radians(degrees)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    return (v[0] * cos_r - v[1] * sin_r, v[0] * sin_r + v[1] * cos_r)


def _along(origin: Point, direction: Point, distance: float) -> Point:
    return (origin[0] + direction[0] * distance, origin[1] + direction[1] * distance)


def _sub_route(route: Sequence[Point], start_s: float, end_s: float) -> list[Point]:
    """The stretch of ``route`` between two arc lengths, endpoints interpolated.

    Skips a margin at each end before the route is probed for blockage: both endpoints are
    occupied by construction (the robot stands at the start, and the robots scenario mode
    ``forbid``s start and goal to the crowd).
    """
    if end_s <= start_s:
        return [point_at_arc(route, start_s)[0], point_at_arc(route, end_s)[0]]

    out = [point_at_arc(route, start_s)[0]]
    travelled = 0.0
    for a, b in zip(route, route[1:], strict=False):
        seg = math.dist(a, b)
        if seg > 1e-12 and start_s < travelled + seg < end_s:
            out.append(b)
        travelled += seg
    out.append(point_at_arc(route, end_s)[0])
    return out


# Free-space checks
# -----------------


def _shortest_clear_lead_in(encounter: Point, direction: Point, goal: Point, travel: float, is_valid: IsValid | None, *, min_lead_in: float, step: float = 0.5, spawn_valid: IsValid | None = None) -> float | None:
    """The longest lead-in distance (<= `travel`, >= `min_lead_in`) from which the walk to
    `goal` is clear and the spawn itself passes `spawn_valid`, stepping the spawn towards the
    encounter point; None if none is."""
    d = travel - step
    while d >= min_lead_in:
        spawn = _along(encounter, direction, -d)
        if (spawn_valid is None or spawn_valid(Position(x=spawn[0], y=spawn[1]))) and first_blocked(spawn, goal, is_valid) is None:
            return d
        d -= step
    return None


def first_blocked(a: Point, b: Point, is_valid: IsValid | None, *, step: float = _SEGMENT_STEP_M) -> Point | None:
    """First sampled point on ``a``->``b`` that is not navigable, or ``None`` if all are.

    Returning the offending point rather than a bool is what lets an abort say *where* the
    walk failed - "no navigable interception" alone sends the reader hunting.

    ``is_valid`` of ``None`` means the world declared no safe distance, so there is nothing
    to check against - the same convention the zone converter uses.
    """
    if is_valid is None:
        return None

    length = math.dist(a, b)
    steps = max(1, int(math.ceil(length / max(step, 1e-6))))
    for i in range(steps + 1):
        f = i / steps
        pt = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        if not is_valid(Position(x=pt[0], y=pt[1])):
            return pt
    return None


def segment_is_free(a: Point, b: Point, is_valid: IsValid | None, *, step: float = _SEGMENT_STEP_M) -> bool:
    """Whether every sampled point on ``a``->``b``, endpoints included, is navigable."""
    return first_blocked(a, b, is_valid, step=step) is None


# Candidate ordering
# ------------------


def angle_offsets(search: float, step: float) -> Iterator[float]:
    """Deviations from the requested angle, nearest first: 0, +s, -s, +2s, -2s, ...

    The requested angle is always tried first, so an unobstructed case gets exactly what it
    asked for and ``deviated`` stays false.
    """
    yield 0.0
    if search <= 0 or step <= 0:
        return
    k = 1
    while k * step <= search + 1e-9:
        yield k * step
        yield -k * step
        k += 1


def _candidates(
    angle: float,
    encounter_at: float,
    lead_out: float,
    *,
    angle_search: float,
    angle_step: float,
) -> Iterator[tuple[float, float, float]]:
    """(angle, encounter_at, lead_out) triples in decreasing order of fidelity to the request.

    Angle is the innermost loop because it is recorded explicitly per case, so a deviation
    there is visible in the results; the encounter point moving is comparatively silent and
    is therefore tried last.
    """
    seen: set[tuple[float, float, float]] = set()
    for enc in encounter_fractions(encounter_at):
        for scale in _LEAD_OUT_SCALES:
            for offset in angle_offsets(angle_search, angle_step):
                candidate = (angle + offset, enc, lead_out * scale)
                if candidate in seen:
                    continue
                seen.add(candidate)
                yield candidate


# The solver
# ----------


def solve_encounter(
    route: Sequence[Point],
    *,
    robot_speed: float,
    ped_speed: float,
    approach_angle: float = 180.0,
    encounter_at: float = 0.5,
    co_arrival_offset: float = 0.0,
    lead_out: float = 3.0,
    angle_search: float = 45.0,
    angle_step: float = 15.0,
    is_valid: IsValid | None = None,
    min_robot_clearance: float = _MIN_ROBOT_CLEARANCE_M,
    min_lead_in: float = 3.0,
    spawn_valid: IsValid | None = None,
) -> Encounter:
    """Place a pedestrian so it meets the robot at a designed point and time.

    ``route`` is the robot's path as a polyline - normally the planned route from
    :func:`pathing.plan_route`, which bends through doorways the way the robot actually will.
    A two-point route is the straight line, which is what an open-area case reduces to.

    ``approach_angle`` is the pedestrian's direction of travel relative to the robot's heading
    **at the encounter point**: 180 deg head-on, 90 deg crossing from the robot's right, 0 deg
    same-direction (the robot overtakes). Measuring against the local heading rather than a
    global one is what keeps the angle meaningful once the route turns a corner.
    ``encounter_at`` is a fraction along the route by arc length.

    ``co_arrival_offset`` shifts the pedestrian's arrival by that many seconds. It is applied
    as extra *distance* (``ped_speed * offset``) rather than as a spawn delay, because the
    Arena human adapter spawns every agent at episode start and has no ``spawn_tick``. For a
    constant-speed walker the two are equivalent; for a walker that dawdles en route - which
    is exactly what a Level D tail causes - they are not, so the achieved timing must be
    measured rather than assumed.

    When the full lead-in (``ped_speed * t_encounter``) does not fit in free space - a fast
    overtaker behind a start that sits against a wall - the pedestrian starts at the longest
    clear stretch behind the encounter (at least ``min_lead_in`` metres, or it is no approach)
    and that much *later* (``Encounter.depart_at``), still arriving on time.

    Raises :class:`GeometryError` when no candidate is navigable. Never relocates the case to
    somewhere that happens to be free: a moved spawn is a different experiment under the same
    case id.
    """
    if robot_speed <= 0:
        raise GeometryError(f"robot_speed must be positive, got {robot_speed}")
    if ped_speed <= 0:
        raise GeometryError(f"ped_speed must be positive, got {ped_speed}")
    if not 0.0 < encounter_at < 1.0:
        raise GeometryError(f"encounter_at must be strictly between 0 and 1, got {encounter_at}")
    if lead_out <= 0:
        raise GeometryError(f"lead_out must be positive, got {lead_out}")
    if len(route) < 2:
        raise GeometryError(f"route needs at least two points, got {len(route)}")

    robot_start, robot_goal = route[0], route[-1]
    leg_length = route_length(route)
    if leg_length < 1e-9:
        raise GeometryError("robot start and goal coincide; there is no route to intercept")

    attempts = 0
    reasons: list[str] = []
    for angle, enc_at, lead in _candidates(
        approach_angle, encounter_at, lead_out, angle_search=angle_search, angle_step=angle_step
    ):
        attempts += 1

        t_enc = enc_at * leg_length / robot_speed
        encounter, heading = point_at_arc(route, enc_at * leg_length)

        # Upstream distance: far enough that walking it takes until the robot arrives.
        travel = ped_speed * (t_enc + co_arrival_offset)
        if travel <= 1e-6:
            raise GeometryError(
                f"co_arrival_offset={co_arrival_offset:g}s cancels the pedestrian's approach "
                f"(it would have to start at or past the encounter point, {travel:.2f} m away); "
                f"the offset must exceed -{t_enc:.2f}s at encounter_at={enc_at:g}"
            )

        direction = rotate(heading, angle)
        spawn = _along(encounter, direction, -travel)
        goal = _along(encounter, direction, lead)

        walk = f"({spawn[0]:.2f},{spawn[1]:.2f})->({goal[0]:.2f},{goal[1]:.2f})"

        if math.dist(spawn, robot_start) < min_robot_clearance:
            reasons.append(f"angle={angle:g}: spawn {walk[: walk.index(')') + 1]} is {math.dist(spawn, robot_start):.2f} m from the robot's start")
            continue

        depart_at = 0.0
        blocked = first_blocked(spawn, goal, is_valid)
        # `spawn_valid` is the full free-space test (walls, furniture and the discs around the
        # robot's start and goal); `is_valid` the physical one the walk may cross.
        reserved = spawn_valid is not None and not spawn_valid(Position(x=spawn[0], y=spawn[1]))
        if blocked is not None or reserved:
            # The full lead-in does not fit. Start closer - at the longest clear stretch behind
            # the encounter, but not so close that the walk-in is trivial - and that much later.
            shorter = _shortest_clear_lead_in(encounter, direction, goal, travel, is_valid, min_lead_in=min_lead_in, spawn_valid=spawn_valid)
            if shorter is None or math.dist(_along(encounter, direction, -shorter), robot_start) < min_robot_clearance:
                why = f"hits occupied space at ({blocked[0]:.2f},{blocked[1]:.2f})" if blocked is not None else "starts in space reserved for the robot"
                reasons.append(f"angle={angle:g}, encounter_at={enc_at:g}, lead_out={lead:g}: walk {walk} {why}")
                continue
            spawn = _along(encounter, direction, -shorter)
            depart_at = (travel - shorter) / ped_speed

        return Encounter(
            spawn=spawn,
            goal=goal,
            encounter=encounter,
            t_encounter=t_enc,
            angle_requested=approach_angle,
            angle_achieved=angle,
            encounter_at_requested=encounter_at,
            encounter_at_achieved=enc_at,
            lead_out_achieved=lead,
            ped_speed=ped_speed,
            co_arrival_offset=co_arrival_offset,
            retries=attempts - 1,
            depart_at=depart_at,
        )

    detail = "; ".join(reasons[:3])
    leg = (
        f"({robot_start[0]:.2f},{robot_start[1]:.2f})->({robot_goal[0]:.2f},{robot_goal[1]:.2f}) "
        f"[{leg_length:.2f} m]"
    )

    # Diagnose the structural cause first: a leg modelled as a straight line can cut through a
    # wall, and no encounter on it is reachable. Only the interior of the leg is sampled; both
    # endpoints are forbidden by design.
    margin = min(min_robot_clearance, 0.4 * leg_length)
    blocked_leg = None
    if is_valid is not None:
        for probe in iter_points(_sub_route(route, margin, leg_length - margin), _SEGMENT_STEP_M):
            if not is_valid(Position(x=probe[0], y=probe[1])):
                blocked_leg = probe
                break
    if blocked_leg is not None:
        planned = "planned route" if len(route) > 2 else "straight-line leg"
        raise GeometryError(
            f"the robot's {planned} {leg} is itself blocked at "
            f"({blocked_leg[0]:.2f},{blocked_leg[1]:.2f}), so there is no valid space on the route "
            "to design an encounter against. On a planned (bent) route this is not a wall - the "
            "planner already routed around those - so look for something occupying the route "
            "itself: base-population agents, their forbidden discs, or furniture inflated since "
            "the route was planned. On a two-point route it usually is a wall between two rooms. "
            "Widen angle_search, move encounter_at, or set place=scenario to drop the tail "
            f"without a designed encounter. ({attempts} candidate(s) tried at "
            f"ped_speed={ped_speed:g} m/s.)"
        )

    raise GeometryError(
        f"no navigable interception after {attempts} candidate(s) on robot leg {leg} "
        f"at ped_speed={ped_speed:g} m/s "
        f"(angle={approach_angle:g} +/-{angle_search:g} deg, encounter_at={encounter_at:g}, "
        f"lead_out={lead_out:g} m): {detail or 'no candidates generated'}"
    )
