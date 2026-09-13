"""Criticality metrics — what actually happened in an episode.

Everything else in this package *designs* an encounter. This measures one. The distinction is
the whole point: ``designed_ttc`` is a construction, fixed the moment the pedestrian is placed,
and it is **not** an outcome. A Level A tail that makes an agent slow to react, or a Level D
tail that makes it dawdle mid-sequence, changes what actually happens without changing a single
designed number. Reporting the designed value as a result would report the experiment's input
as its output.

So these functions take a *trajectory* — a time series of robot and pedestrian poses sampled
from ``arena_peds`` — and reduce it to the panel from ``edge_case_changes.md`` section 9:
min-TTC, min-clearance, PET, freeze duration, intrusion time.

Pure functions over plain tuples: no ROS, no simulator, no I/O. The live sampling that feeds
them lives in :mod:`scoring`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import attrs

Point = tuple[float, float]

#: Robot speed below which it counts as not moving, m/s. Matches `FREEZE_SPEED_THRESHOLD`
#: used by the existing robot metrics, so "frozen" means the same thing across the codebase.
FREEZE_SPEED = 0.05

#: Personal space, metres. Surface-to-surface clearance below this counts as an intrusion.
PERSONAL_SPACE = 0.5

#: Settling window at episode start, seconds. The robot is spawned, nav2 is still bringing its
#: costmaps up, and nothing has been commanded yet - counting that as a freeze would score
#: every episode as frozen regardless of what the tail did.
SETTLE_S = 2.0


@attrs.frozen
class Sample:
    """One instant: where the robot is, and where everyone else is.

    ``radius`` values are surface radii, so clearance can be reported surface-to-surface
    rather than centre-to-centre - a 0.3 m gap between centres is a collision, not a near miss.
    """

    t: float
    robot: Point
    robot_vel: Point = (0.0, 0.0)
    robot_radius: float = 0.3
    #: name -> (position, velocity, radius)
    peds: dict[str, tuple[Point, Point, float]] = attrs.Factory(dict)


@attrs.frozen
class Criticality:
    """One episode's outcome. ``None`` means "not observed", never "zero"."""

    min_ttc_s: float | None = None
    min_ttc_with: str = ""
    min_ttc_at: float | None = None

    min_clearance_m: float | None = None
    min_clearance_with: str = ""
    min_clearance_at: float | None = None

    pet_s: float | None = None
    freeze_duration_s: float = 0.0
    intrusion_time_s: float = 0.0

    collided: bool = False
    #: Who closed the gap at the FIRST contact: "robot" (drove into a slow/still pedestrian),
    #: "pedestrian" (walked into a slow/still robot), "mutual" (both closing), "static" (overlap
    #: with neither closing - e.g. a spawn on top of the robot). "" = no contact.
    collision_fault: str = ""
    collision_with: str = ""
    collision_at: float | None = None
    collision_robot_closing: float | None = None
    collision_ped_closing: float | None = None
    samples: int = 0
    duration_s: float = 0.0

    @property
    def observed(self) -> bool:
        """Whether anything was measurable. A run with no pedestrian in sight yields a record
        full of `None`, which must not be mistaken for a safe episode."""
        return self.samples > 0 and self.min_clearance_m is not None


def time_to_collision(
    p_a: Point, v_a: Point, r_a: float,
    p_b: Point, v_b: Point, r_b: float,
) -> float | None:
    """Constant-velocity time until two discs touch, or ``None`` if they never do.

    Solves ``|Δp + Δv·t| = r_a + r_b`` for the smallest positive root. Constant velocity is an
    approximation over a single sample, which is why this is evaluated every frame and
    minimised rather than computed once.
    """
    dpx, dpy = p_b[0] - p_a[0], p_b[1] - p_a[1]
    dvx, dvy = v_b[0] - v_a[0], v_b[1] - v_a[1]
    r = r_a + r_b

    a = dvx * dvx + dvy * dvy
    if a < 1e-12:
        return None  # no relative motion: never closes
    b = 2.0 * (dpx * dvx + dpy * dvy)
    c = dpx * dpx + dpy * dpy - r * r

    if c <= 0.0:
        return 0.0  # already overlapping

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(disc)
    for root in ((-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)):
        if root >= 0.0:
            return root
    return None


def _speed(v: Point) -> float:
    return math.hypot(v[0], v[1])


def post_encroachment_time(samples: Sequence[Sample], encounter: Point) -> float | None:
    """|t_ped(E) − t_rob(E)| at the designed encounter point.

    How near a miss was in *time* rather than distance: both parties passed through the same
    place, this is the gap between them doing so. Needs the designed encounter point, so it is
    only defined for a `place: intercept` case; returns ``None`` otherwise.

    Uses the closest approach to ``E`` for each party, and only counts a party that actually
    came near it - a pedestrian that never went there has no crossing time, and pretending it
    did would invent a PET out of two unrelated moments.
    """
    if not samples:
        return None

    near = 1.5  # metres; "passed through E" rather than "was vaguely in the room"

    def crossing_time(track: Iterable[tuple[float, Point]]) -> float | None:
        best_t, best_d = None, math.inf
        for t, p in track:
            d = math.dist(p, encounter)
            if d < best_d:
                best_t, best_d = t, d
        return best_t if best_d <= near else None

    t_rob = crossing_time((s.t, s.robot) for s in samples)
    if t_rob is None:
        return None

    names = {n for s in samples for n in s.peds}
    best: float | None = None
    for name in names:
        t_ped = crossing_time((s.t, s.peds[name][0]) for s in samples if name in s.peds)
        if t_ped is None:
            continue
        pet = abs(t_ped - t_rob)
        if best is None or pet < best:
            best = pet
    return best


def score(
    samples: Sequence[Sample],
    *,
    encounter: Point | None = None,
    only: Sequence[str] | None = None,
    settle_s: float = SETTLE_S,
) -> Criticality:
    """Reduce a trajectory to the criticality panel.

    ``only`` restricts the pairwise metrics to named pedestrians — pass ``["edge_0"]`` to score
    the injected agent alone rather than whichever bystander happened to be closest. Without it
    a busy scenario reports the crowd's ambient near-misses and the tail's effect disappears
    into them.
    """
    if not samples:
        return Criticality()

    t0 = samples[0].t
    wanted = set(only) if only else None

    min_ttc: float | None = None
    ttc_with, ttc_at = "", None
    min_clear: float | None = None
    clear_with, clear_at = "", None
    freeze = 0.0
    intrusion = 0.0
    collided = False
    fault, fault_with, fault_at = "", "", None
    fault_rc: float | None = None
    fault_pc: float | None = None

    # Forward hold: the state observed at a sample is taken to persist until the next one.
    # Crediting an interval to the sample that *ends* it is subtly wrong - it would count the
    # window straddling the settle boundary as post-settle, and attribute a pedestrian's
    # proximity to time before it arrived.
    for i, s in enumerate(samples):
        dt = (samples[i + 1].t - s.t) if i + 1 < len(samples) else 0.0
        dt = max(0.0, dt)

        if s.t - t0 >= settle_s and _speed(s.robot_vel) < FREEZE_SPEED:
            freeze += dt

        intruding = False
        for name, (p, v, r) in s.peds.items():
            if wanted is not None and name not in wanted:
                continue

            clearance = math.dist(s.robot, p) - (s.robot_radius + r)
            if min_clear is None or clearance < min_clear:
                min_clear, clear_with, clear_at = clearance, name, s.t - t0
            if clearance <= 0.0:
                if not collided:
                    # attribute the FIRST contact by who was closing the gap along the
                    # centre line; below FREEZE_SPEED a party counts as standing
                    dx, dy = p[0] - s.robot[0], p[1] - s.robot[1]
                    norm = math.hypot(dx, dy) or 1.0
                    ux, uy = dx / norm, dy / norm
                    rc = s.robot_vel[0] * ux + s.robot_vel[1] * uy
                    pc = -(v[0] * ux + v[1] * uy)
                    r_moving, p_moving = rc >= FREEZE_SPEED, pc >= FREEZE_SPEED
                    fault = "mutual" if r_moving and p_moving else "robot" if r_moving else "pedestrian" if p_moving else "static"
                    fault_with, fault_at, fault_rc, fault_pc = name, s.t - t0, rc, pc
                collided = True
            if clearance < PERSONAL_SPACE:
                intruding = True

            ttc = time_to_collision(s.robot, s.robot_vel, s.robot_radius, p, v, r)
            if ttc is not None and (min_ttc is None or ttc < min_ttc):
                min_ttc, ttc_with, ttc_at = ttc, name, s.t - t0

        if intruding:
            intrusion += dt

    return Criticality(
        min_ttc_s=min_ttc,
        min_ttc_with=ttc_with,
        min_ttc_at=ttc_at,
        min_clearance_m=min_clear,
        min_clearance_with=clear_with,
        min_clearance_at=clear_at,
        pet_s=None if encounter is None else post_encroachment_time(samples, encounter),
        freeze_duration_s=freeze,
        intrusion_time_s=intrusion,
        collided=collided,
        collision_fault=fault,
        collision_with=fault_with,
        collision_at=fault_at,
        collision_robot_closing=fault_rc,
        collision_ped_closing=fault_pc,
        samples=len(samples),
        duration_s=samples[-1].t - t0,
    )


# Formations
# ----------


def _bare(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def formation_hold(samples: Sequence[Sample], members: Sequence[str], *, radius: float = 1.5, centre: Point | None = None) -> dict[str, Any]:
    """Whether a placed group formed, how long it stood together, and when it broke.

    The Type D success criterion (`new_plan_2.md` §10) is positional. With a `centre` (where
    the generator put the formation) the group counts as *formed* from the first sample at
    which every member is within `radius` m of it; `held_s` is how long that lasted and
    `broke_at` the first departure afterwards. Without a centre the members' first positions
    define it. `seen` is False when no member ever appeared on the stream, which is a build
    failure, not a quiet success; `formed_at` is None when they never all arrived.
    """
    wanted = {str(m) for m in members}
    out: dict[str, Any] = {"members": sorted(wanted), "seen": False, "formed_at": None, "held_s": None, "broke_at": None, "arrived": 0}
    if not wanted or not samples:
        return out
    t0 = samples[0].t
    ref: Point | None = centre
    formed_at: float | None = None
    held_until: float | None = None
    for sample in samples:
        present = {_bare(n): v[0] for n, v in sample.peds.items() if _bare(n) in wanted}
        if not present:
            continue
        out["seen"] = True
        if ref is not None:
            # How many were ever there at once: a line five of six reached is not "never formed".
            out["arrived"] = max(out["arrived"], sum(1 for p in present.values() if math.dist(p, ref) <= radius))
        if len(present) < len(wanted):
            if formed_at is None:
                continue
        if ref is None:
            ref = (sum(p[0] for p in present.values()) / len(present), sum(p[1] for p in present.values()) / len(present))
        together = all(math.dist(p, ref) <= radius for p in present.values())
        if formed_at is None:
            if together:
                formed_at = sample.t - t0
                held_until = formed_at
            continue
        if not together:
            out["broke_at"] = sample.t - t0
            break
        held_until = sample.t - t0
    if formed_at is not None:
        out["formed_at"] = formed_at
        out["held_s"] = (held_until or formed_at) - formed_at
    return out
