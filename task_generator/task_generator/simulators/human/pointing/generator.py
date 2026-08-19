"""Parametric PointAt animation generator.

Takes one of the recorded pointing clips as a *style template* (timing envelope,
idle body, gesture arc, elbow bend and swivel) and re-targets its pointing arm
to an arbitrary direction or 3-D point in the pedestrian-local frame, emitting a
clip in exactly the same on-wire format as the template
(``[{angles, root_xy_yaw, animation_state, t}, ...]`` as an object ``.npy``).

Design
------
1. **Envelope.**  The template's gesture is projected onto its own peak
   deviation, giving a scalar ``w(t) in [0, 1]``: 0 = idle, 1 = fully pointing.
   Everything the generator changes is blended by ``w(t)``, so the ramp-in,
   hold and release timing are the template's, unchanged.
2. **Residual.**  The projected gesture component is subtracted from the
   template's pointing arm, leaving an idle arm *plus* its natural micro-motion,
   which becomes the ``w = 0`` end of the blend.
3. **Aim IK** (closed form, per frame).  The pointing ray is the forearm axis
   (elbow -> wrist).  Given the desired ray direction ``f``, an elbow flexion
   ``e`` and a swivel angle ``phi``, the upper-arm direction is a point on the
   cone of half-angle ``e`` about ``f``, and the wire triple ``(y, p, r)`` then has
   an exact closed-form inverse (section 9b of convert_v2 plus a one-line twist
   solve).  ``phi`` is the arm's redundancy: it is searched over a full turn and
   scored against joint limits, self-collision clearance and the template's own
   swivel, which is how the generator stays inside the body's reachable set
   instead of pushing the hand through the chest.
4. **Body assist.**  Torso yaw/pitch and clavicle protraction/elevation are
   added as *deltas relative to the template's own aim*, so generating for the
   template's own target reproduces the template.  The head is never touched:
   the three head DOFs pass through from the template verbatim so a gaze
   controller can own them.

All angles are in the JOINTS.md **wire** convention (anatomical shoulder
triples).  Arena's RViz adapter and the Isaac bone map consume them directly.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import contract as C
from . import skeleton as S

TEMPLATES = ("point_to_right", "point_straight", "point_to_left")


def _animations_dir() -> str:
    """The recorded clips: package sibling in the source tree, ament share when installed."""
    local = Path(__file__).resolve().parent.parent / "animations"
    if local.is_dir():
        return str(local)
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        return str(local)
    return os.path.join(get_package_share_directory("task_generator"), "simulators", "human", "animations")


TEMPLATE_DIR = _animations_dir()
# under-relaxed wrist-to-aim fixed point: close targets do not contract otherwise
FIXED_POINT_DAMPING = 0.55

# torso-assist distribution across the spine stack (waist, spine, chest)
YAW_SPLIT = (0.45, 0.275, 0.275)
PITCH_SPLIT = (0.5, 0.25, 0.25)


# ---------------------------------------------------------------------------
# template
# ---------------------------------------------------------------------------


@dataclass
class Template:
    """A recorded clip decomposed into (idle body + residual arm) x envelope."""

    name: str
    frames: list[dict]
    dt: float
    side: str                     # which arm gestures
    w: np.ndarray                 # (T,) envelope in [0, 1]
    base: list[dict]              # per-frame angles with the gesture removed
    peak: int                     # frame index of maximum deviation
    aim_dir: np.ndarray           # unit forearm direction at the peak, ped-local
    elbow_bend: float             # elbow flexion at the peak (rad)
    swivel: float                 # elbow swivel at the peak (rad, 0 = elbow down)
    collar: tuple[float, float]   # (y, p) of the pointing clavicle at the peak
    hold: tuple[int, int]         # [start, end) frame range with w >= 0.95
    settled: tuple[int, int] = (0, 0)   # [start, end) core of the hold, aim steady

    @property
    def n(self) -> int:
        return len(self.frames)


def _load_frames(path: str) -> list[dict]:
    raw = np.load(path, allow_pickle=True)
    return [dict(f) for f in raw]


def _elbow_swivel(sh: np.ndarray, el: np.ndarray, aim_dir: np.ndarray) -> float:
    """Signed angle of the elbow off the 'straight down from the ray' reference."""
    f = S.unit(aim_dir)
    n0 = S.perp_component(f, S.DOWN)
    if np.linalg.norm(n0) < 1e-6:
        n0 = S.perp_component(f, -S.FWD)
    m0 = np.cross(f, n0)
    v = S.perp_component(f, np.asarray(el) - np.asarray(sh))
    return float(np.arctan2(np.dot(v, m0), np.dot(v, n0)))


def load_template(name: str = "point_to_right", height: float = 1.65) -> Template:
    if name in TEMPLATES:
        path = os.path.join(TEMPLATE_DIR, f"{name}.npy")
    else:
        path = name
        name = os.path.splitext(os.path.basename(path))[0]
    frames = _load_frames(path)
    body = S.Body(height)
    t = np.array([f["t"] for f in frames], dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.05

    # which arm gestures: larger total joint travel over the six arm DOFs
    travel = {}
    for side in ("l", "r"):
        m = np.array([[f["angles"][d] for d in C.arm_dofs(side)] for f in frames])
        travel[side] = float(np.abs(np.diff(m, axis=0)).sum())
    side = max(travel, key=travel.get)

    dofs = C.arm_dofs(side)
    A = np.array([[f["angles"][d] for d in dofs] for f in frames])
    d = A - A[0]                                   # deviation from the rest frame
    peak = int(np.argmax(np.linalg.norm(d, axis=1)))
    dp = d[peak]
    w = np.clip(d @ dp / float(dp @ dp), 0.0, 1.0)
    w[0] = 0.0
    # Snap the hold plateau to exactly 1 so the emitted clip holds the aim dead
    # steady.  The template's own micro-motion is not lost: it survives in the
    # residual `base` track below, which is added underneath the blend.
    w[w >= 0.95] = 1.0

    base = []
    for i, f in enumerate(frames):
        ang = dict(f["angles"])
        for k, name_dof in enumerate(dofs):
            ang[name_dof] = float(A[i, k] - w[i] * dp[k])
        base.append(ang)

    pos, _ = S.fk(frames[peak]["angles"], body)
    sh, el, wr = (pos[f"{side}_{k}"] for k in ("shoulder", "elbow", "wrist"))
    aim = S.unit(wr - el)
    hold_idx = np.flatnonzero(w >= 0.95)
    hold = (int(hold_idx[0]), int(hold_idx[-1]) + 1) if hold_idx.size else (peak, peak + 1)
    # The plateau opens while the arm is still settling out of its overshoot
    # (this clip swings ~13 deg past the target and eases back over 0.2 s) and
    # closes as the release begins.  The settled core is what "is it aimed?"
    # should be judged on -- the overshoot is style, not error.
    settled = []
    for i in range(*hold):
        p_i, _ = S.fk(frames[i]["angles"], body)
        d_i = S.unit(p_i[f"{side}_wrist"] - p_i[f"{side}_elbow"])
        if np.degrees(np.arccos(np.clip(float(d_i @ aim), -1.0, 1.0))) < 3.0:
            settled.append(i)
    settled = settled or [peak]

    return Template(
        name=name,
        frames=frames,
        dt=dt,
        side=side,
        w=w,
        base=base,
        peak=peak,
        aim_dir=aim,
        elbow_bend=float(frames[peak]["angles"][f"{side}_elbow"]),
        swivel=_elbow_swivel(sh, el, aim),
        collar=(
            float(frames[peak]["angles"][f"{side}_y_collar"]),
            float(frames[peak]["angles"][f"{side}_p_collar"]),
        ),
        hold=hold,
        settled=(int(settled[0]), int(settled[-1]) + 1),
    )


# ---------------------------------------------------------------------------
# target specification
# ---------------------------------------------------------------------------


def direction_from_angles(azimuth: float, elevation: float, deg: bool = True) -> np.ndarray:
    """Unit direction in the ped-local frame.

    ``azimuth``  +left / -right about the vertical axis (0 = straight ahead).
    ``elevation`` +up / -down from the horizontal.
    """
    a = np.radians(azimuth) if deg else float(azimuth)
    e = np.radians(elevation) if deg else float(elevation)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def angles_from_direction(d: np.ndarray | Sequence[float], deg: bool = True) -> tuple[float, float]:
    d = S.unit(d)
    az = float(np.arctan2(d[1], d[0]))
    el = float(np.arcsin(np.clip(d[2], -1.0, 1.0)))
    return (np.degrees(az), np.degrees(el)) if deg else (az, el)


@dataclass
class PointAtOptions:
    hand: str = "auto"            # 'auto' | 'l' | 'r'
    dominant: str = "r"           # hand used when the target is near the midline
    cross_deg: float = 20.0       # azimuth band around the midline kept for `dominant`
    elbow_bend: float | None = None   # rad, None -> the template's own bend
    torso_yaw_gain: float = 0.25   # 15 deg of torso turn for a level target at 60 deg azimuth
    torso_pitch_gain: float = 0.55
    collar_gain: float = 0.45
    swivel_samples: int = 72
    swivel_prior: float = 0.35    # weight pulling the elbow to the template's swivel
    limit_weight: float = 12.0    # weight on advisory-limit violation
    collide_weight: float = 35.0  # weight on capsule interpenetration
    clamp_limits: bool = True
    steady_hold: bool = True      # re-solve the aim across the settled hold
    relax_on_collision: bool = True   # retry with a bent elbow if the arm crowds the body
    unwrap_shoulder: bool = False     # trade the advisory shoulder window for no 2*pi steps
    root: str = "zero"            # 'zero' | 'template'
    hold_s: float | None = None   # stretch/shrink the hold plateau to this many seconds
    collar_scale: float = 1.0     # scale of the template's own clavicle excursion (shrug), 1 = as recorded
    upright: bool = False         # emit the shoulder against an upright spine (a walking ped does not blend the torso DOFs)


@dataclass
class PointAtClip:
    frames: list[dict]
    side: str
    target_dir: np.ndarray
    target_point: np.ndarray | None
    template: str
    envelope: np.ndarray = field(default_factory=lambda: np.zeros(0))
    report: dict = field(default_factory=dict)

    @property
    def peak(self) -> int:
        """Index of a fully-pointing frame (mid-hold)."""
        if self.envelope.size == 0:
            return len(self.frames) // 2
        hold = np.flatnonzero(self.envelope >= 0.999)
        return int(hold[len(hold) // 2]) if hold.size else int(np.argmax(self.envelope))

    @property
    def hold_range(self) -> tuple[int, int]:
        """First and last frame index of the settled hold, for looping it."""
        if self.envelope.size:
            hold = np.flatnonzero(self.envelope >= 0.999)
            if hold.size:
                return int(hold[0]), int(hold[-1])
        return self.peak, self.peak

    def to_array(self) -> np.ndarray:
        arr = np.empty(len(self.frames), dtype=object)
        for i, f in enumerate(self.frames):
            arr[i] = f
        return arr

    def save(self, path: str) -> str:
        np.save(path, self.to_array(), allow_pickle=True)
        return path

    @property
    def duration(self) -> float:
        return float(self.frames[-1]["t"]) if self.frames else 0.0


@dataclass
class HoldPose:
    """A held point: the DOFs the generator wrote at the peak, plus how it got there."""

    side: str
    angles: dict[str, float]      # spine assist + arm DOFs (PointAtGenerator._blended_dofs)
    swivel: float                 # elbow swivel the hold was solved with (rad)
    target_dir: np.ndarray
    target_point: np.ndarray | None


# ---------------------------------------------------------------------------
# arm IK
# ---------------------------------------------------------------------------


def _solve_wire_triple(collar_r: np.ndarray, u: np.ndarray, f: np.ndarray,
                       bend: float, warm: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    """Exact inverse of the wire shoulder chain for a given upper-arm direction.

    ``u`` upper-arm direction (world), ``f`` forearm direction (world), both
    unit and ``angle(u, f) == bend``.  Returns ``(y, p, r)``.
    """
    v = collar_r.T @ u
    p = float(np.arccos(np.clip(-v[2], -1.0, 1.0)))
    sp = np.sin(p)
    y = float(np.arctan2(v[1], v[0])) if sp > 1e-9 else float(warm[0])

    def finish(yy: float, pp: float) -> tuple[float, float, float]:
        a = S.arm_rot(yy, pp, 0.0)
        vloc = a.T @ (collar_r.T @ f)
        rr = float(np.arctan2(-vloc[1], vloc[0])) if np.sin(bend) > 1e-6 else float(warm[2])
        return float(yy), float(pp), rr

    def score(sol: tuple[float, float, float]) -> float:
        yy, pp, rr = sol
        pen = (
            C.saturation("l_y_shoulder", yy)
            + C.saturation("l_p_shoulder", pp)
            + C.saturation("l_r_shoulder", rr)
        )
        drift = (
            abs(C.wrap_pi(yy - warm[0])) + abs(pp - warm[1]) + abs(C.wrap_pi(rr - warm[2]))
        )
        # a value outside the emit window WILL be clamped, and a clamped
        # shoulder angle is a wrong pose -- so validity outranks continuity
        # absolutely, and drift only breaks ties among representations that fit
        return 1000.0 * pen + drift

    # spherical double cover: (y, p) and (y+pi, -p) name the same direction.
    # Prefer the branch that fits the advisory limits, warm start only breaks ties.
    cands = [finish(y, p), finish(C.wrap_pi(y + np.pi), -p)]
    best = min(cands, key=score)

    # Gimbal degeneracy: at p = 0 (arm down) or p = pi (arm up) the azimuth and
    # the twist are the same DOF -- only y-r resp. y+r is observable -- so a
    # saturated twist can be traded into the azimuth for free.
    yy, pp, rr = best
    if abs(np.sin(pp)) < 0.25 and C.saturation("l_r_shoulder", rr) > 0.0:
        rc = C.clamp("l_r_shoulder", rr)
        sign = 1.0 if np.cos(pp) > 0.0 else -1.0
        alt = (float(C.wrap_pi(yy - sign * (rr - rc))), pp, rc)
        if score(alt) < score(best):
            best = alt
    return best


def _arm_positions(shoulder: np.ndarray, collar_r: np.ndarray, y: float, p: float, r: float, bend: float, body: S.Body) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_r = collar_r @ S.arm_rot(y, p, r)
    el = shoulder + body.upperarm * (s_r @ S.DOWN)
    f_r = s_r @ S.rot_axis(S.ELBOW_AXIS, bend)
    wr = el + body.forearm * (f_r @ S.DOWN)
    return el, wr, f_r @ S.DOWN


def _arm_clearance(pos: dict, side: str, sh: np.ndarray, el: np.ndarray, wr: np.ndarray, body: S.Body) -> float:
    """Worst clearance of a candidate arm against the rest of the body."""
    tmp = dict(pos)
    tmp[f"{side}_shoulder"], tmp[f"{side}_elbow"], tmp[f"{side}_wrist"] = sh, el, wr
    return min(S.collision_clearances(tmp, body, side).values())


def _swivel_axes(f: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Reference frame the elbow swivel is measured in: phi = 0 drops the elbow
    straight down off the aim ray.  Falls back when the ray is vertical."""
    n0 = S.perp_component(f, S.DOWN)
    if np.linalg.norm(n0) < 1e-6:
        n0 = S.perp_component(f, -S.FWD)
    if np.linalg.norm(n0) < 1e-6:
        n0 = S.perp_component(f, np.array([0.0, S.REFLECT[side], 0.0]))
    return n0, np.cross(f, n0)


def solve_arm_ik(
    pos: dict,
    frames: dict,
    side: str,
    aim_dir: np.ndarray,
    body: S.Body,
    bend: float,
    swivel_pref: float,
    opts: PointAtOptions,
    warm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fixed_phi: float | None = None,
    all_candidates: bool = False,
) -> dict | list[dict]:
    """Closed-form aim IK with a scored search over the elbow swivel.

    ``pos``/``frames`` come from an FK pass on the *assisted* body pose (torso
    and collars already set), so the shoulder position and collar frame are
    final.  Returns the chosen solution and its diagnostics.

    ``fixed_phi`` skips the search and evaluates that one swivel: the caller
    resolves the redundancy once, at the fully-pointing frame, and reuses it for
    the whole clip.  That is both ~40x cheaper and strictly smoother -- a
    per-frame argmin can hop between equal-cost swivel branches and snap the
    elbow mid-hold.

    ``all_candidates`` returns every scored swivel, cheapest first, so the
    caller can re-rank them on something this function cannot see (the swept
    ramp path -- see ``PointAtGenerator._path_penalty``).
    """
    collar_r = frames[f"{side}_collar"]
    sh = pos[f"{side}_shoulder"]
    f = S.unit(aim_dir)

    n0, m0 = _swivel_axes(f, side)

    cb, sb = np.cos(bend), np.sin(bend)
    best = None
    cands = []
    if fixed_phi is not None:
        phis = np.array([float(fixed_phi)])
    else:
        phis = np.linspace(-np.pi, np.pi, int(opts.swivel_samples), endpoint=False)
        phis = np.concatenate([[swivel_pref], phis])
    for phi in phis:
        u = cb * f + sb * (np.cos(phi) * n0 + np.sin(phi) * m0)
        y, p, r = _solve_wire_triple(collar_r, S.unit(u), f, bend, warm)
        raw = {
            f"{side}_y_shoulder": y,
            f"{side}_p_shoulder": p,
            f"{side}_r_shoulder": r,
            f"{side}_elbow": bend,
        }
        sat = sum(C.saturation(k, v) for k, v in raw.items())
        yc, pc, rc = (
            C.clamp(f"{side}_y_shoulder", y),
            C.clamp(f"{side}_p_shoulder", p),
            C.clamp(f"{side}_r_shoulder", r),
        )
        bc = C.clamp(f"{side}_elbow", bend)
        el, wr, achieved = _arm_positions(sh, collar_r, yc, pc, rc, bc, body)
        clear = _arm_clearance(pos, side, sh, el, wr, body)
        aim_err = float(np.arccos(np.clip(np.dot(achieved, f), -1.0, 1.0)))
        cost = (
            aim_err
            + opts.limit_weight * sat
            + opts.collide_weight * max(0.0, -clear)
            + opts.swivel_prior * (1.0 - np.cos(phi - swivel_pref))
        )
        cand = {
            "cost": float(cost), "phi": float(phi), "raw": raw,
            "angles": {
                f"{side}_y_shoulder": yc, f"{side}_p_shoulder": pc,
                f"{side}_r_shoulder": rc, f"{side}_elbow": bc,
            },
            "saturation": float(sat), "clearance": float(clear),
            "aim_error": aim_err, "elbow": el, "wrist": wr, "achieved": achieved,
            "aim": f,
        }
        if all_candidates:
            cands.append(cand)
        if best is None or cost < best["cost"]:
            best = cand
    if all_candidates:
        cands.sort(key=lambda c: c["cost"])
        return cands
    return best


# ---------------------------------------------------------------------------
# body assist
# ---------------------------------------------------------------------------


def _triple_in_limits(side: str, triple: tuple[float, float, float], prev: tuple[float, float, float] | None = None, limits: dict[str, tuple[float, float]] | None = None) -> tuple[float, float, float]:
    """Pick the representation of a shoulder rotation that fits its limits.

    ``(y, p, r)`` and ``(y+pi, -p, r+pi)`` are the same rotation, and each has
    2*pi-equivalents on top.  Two things have to come out right:

    * the matrix decomposition always returns ``p >= 0``, which can strand the
      twist a half-turn outside its advisory range -- clamping *that* branch
      would twist the forearm somewhere else entirely rather than leave the
      pose alone, and
    * the choice has to be continuous in time.  Every representation renders
      identically, but the arm crosses ``p = 0`` and ``p = pi`` during a raise,
      and flipping branch there puts a 180-degree step into a joint that
      robot_state_publisher and every velocity check downstream will believe.
      ``prev`` (the previous frame's emitted triple) breaks the tie.
    """
    y, p, r = (float(v) for v in triple)
    lim = limits or C.EMIT_LIMITS
    names = [f"{side}_y_shoulder", f"{side}_p_shoulder", f"{side}_r_shoulder"]
    turns = (-2 * np.pi, 0.0, 2 * np.pi)
    cands = [
        (base[0] + ty, base[1], base[2] + tr)
        for base in ((y, p, r), (C.wrap_pi(y + np.pi), -p, C.wrap_pi(r + np.pi)))
        for ty in turns
        for tr in turns          # the product, not one or the other: crossing the
    ]                            # wrap usually moves the azimuth AND the twist

    def cost(c: tuple[float, float, float]) -> float:
        pen = sum(C.saturation(n, v, limits=lim) for n, v in zip(names, c, strict=True))
        drift = 0.0 if prev is None else sum(abs(a - b) for a, b in zip(c, prev, strict=True))
        # a value outside the emit window WILL be clamped, and a clamped
        # shoulder angle is a wrong pose -- so validity outranks continuity
        # absolutely, and drift only breaks ties among representations that fit
        return 1000.0 * pen + drift

    # deliberately NOT folded through to_limits first: that maps every
    # 2*pi-equivalent onto the same representative and takes away exactly the
    # freedom being searched here, which is what turns a smooth swing into a
    # full-turn wrap every time the twist crosses +-pi
    return min(cands, key=cost)


def _torso_assist(az: float, el: float, opts: PointAtOptions) -> tuple[float, float]:
    """(total yaw, total pitch) the spine stack should add for this aim.

    The turn is driven by the *horizontal* part of the aim (``cos(el)``): near
    the vertical poles the azimuth names no real direction, and turning the
    torso for it would twist the body under an arm that is simply going up --
    which is what jammed straight-up-and-hard-left into the head.
    """
    yaw = opts.torso_yaw_gain * az * float(np.cos(el))
    if el < -0.45:
        pitch = opts.torso_pitch_gain * (-el - 0.45)
    elif el > 0.95:
        pitch = -0.28 * (el - 0.95)
    else:
        pitch = 0.0
    return float(yaw), float(pitch)


def _collar_assist(side: str, az_t: float, el_t: float, opts: PointAtOptions) -> tuple[float, float]:
    """(y_collar, p_collar) for an aim expressed in the torso frame."""
    across = az_t * (1.0 if side == "r" else -1.0)   # >0 => target across the body
    y = opts.collar_gain * (max(0.0, across) + 0.35 * min(0.0, across))
    p = 0.42 * max(0.0, el_t - 0.30) + 0.18 * max(0.0, across)
    return float(y), float(p)


def _torso_frame_angles(torso_r: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    """(azimuth, elevation) of world direction ``d`` seen from the torso frame."""
    h = torso_r.T @ S.unit(d)
    return float(np.arctan2(h[1], h[0])), float(np.arcsin(np.clip(h[2], -1.0, 1.0)))


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------


class PointAtGenerator:
    """Generates PointAt clips from a style template.

    >>> gen = PointAtGenerator()
    >>> clip = gen.point_at(target=(2.5, 1.0, 1.6))       # ped-local metres
    >>> clip = gen.point_at(azimuth=-40, elevation=25)    # or a pure direction
    >>> clip.save("point_at.npy")
    """

    def __init__(self, template: str = "point_to_right", height: float = 1.65,
                 options: PointAtOptions | None = None):
        self.body = S.Body(height)
        self.template = load_template(template, height)
        self.opts = options or PointAtOptions()
        # the template's own aim, used to anchor every assist term so that
        # re-generating the template's target reproduces the template
        self.tpl_az, self.tpl_el = angles_from_direction(self.template.aim_dir, deg=False)
        pos, fr = S.fk(self.template.frames[self.template.peak]["angles"], self.body)
        self.tpl_az_torso, self.tpl_el_torso = _torso_frame_angles(fr["torso"], self.template.aim_dir)

    # -- hand choice ------------------------------------------------------
    def choose_hand(self, aim: np.ndarray, opts: PointAtOptions) -> str:
        """Point with the arm on the target's side.

        Keyed on the *lateral* component of the aim, not on the azimuth: near
        the vertical poles the azimuth is undefined (every azimuth names the
        same direction) while the lateral component correctly collapses to 0,
        so straight-up and straight-down fall back to the dominant hand instead
        of flipping arms on a numerically meaningless angle.
        """
        if opts.hand in ("l", "r"):
            return opts.hand
        aim = S.unit(aim)
        lateral = float(aim[1])
        # The band shrinks with the aim's elevation -- "20 deg off the midline"
        # is a much bigger lateral offset when pointing level than when pointing
        # nearly straight up -- but floors at 0.35 so the exact pole, where the
        # azimuth means nothing at all, still lands on the dominant hand.
        horizon = max(float(np.hypot(aim[0], aim[1])), 0.35)
        thresh = np.sin(np.radians(opts.cross_deg)) * horizon
        if lateral > thresh:
            return "l"
        if lateral < -thresh:
            return "r"
        return opts.dominant

    # -- main entry point -------------------------------------------------
    def point_at(
        self,
        target: Sequence[float] | np.ndarray | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
        deg: bool = True,
        options: PointAtOptions | None = None,
        **kw: object,
    ) -> PointAtClip:
        """Generate a clip pointing at a 3-D point or a direction.

        ``target``: (x, y, z) in the ped-local frame -- x forward, y left,
        z up, origin on the **floor** under the pelvis.  Alternatively give
        ``azimuth``/``elevation`` for a direction-only point (treated as a
        target at infinity).
        """
        opts = options or self.opts
        if kw:
            opts = PointAtOptions(**{**opts.__dict__, **kw})

        aim0, target_point = self._resolve_aim(target, azimuth, elevation, deg)
        az, el = angles_from_direction(aim0, deg=False)
        side = self.choose_hand(aim0, opts)
        near_target = False
        if target_point is not None and self._near_target(target_point, side):
            near_target, target_point = True, None
        bend = self.template.elbow_bend if opts.elbow_bend is None else float(opts.elbow_bend)

        tpl = self.template
        w_track, idx = self._timeline(opts)
        frames_out: list[dict] = []
        diag = []
        warm = (0.0, 0.0, 0.0)

        d_yaw, d_pitch = self._assist_deltas(az, el, opts)

        # The retarget is ONE rotation, solved on the fully-pointing frame and
        # then applied to the whole recorded arm track.  A per-frame re-solve
        # would pin the aim dead-on for every hold frame, but it would also
        # erase the template's own settle (this clip overshoots ~13 deg and
        # eases back over 0.4 s) and its ramp arc, replacing them with a
        # straight interpolation toward the final pose.  One rotation keeps the
        # recorded dynamics and swings them onto the new aim.
        tri = [f"{side}_y_shoulder", f"{side}_p_shoulder", f"{side}_r_shoulder"]
        # the template's peak frame -- the one its aim_dir was measured on, so
        # a zero retarget really is the identity (argmax(w) would land on the
        # first plateau frame, which is still 13 deg into the overshoot)
        anchor = tpl.peak
        anchor_posed, anchor_sol, _ = self._solve_frame(
            anchor, aim0, side, d_yaw, d_pitch, bend, target_point, opts,
            warm=warm, fixed_phi=None,
        )
        # ...expressed in the ped-local frame, not the collar frame: the collar
        # moves (torso assist, shrug) and a delta fixed to it would let the aim
        # ride along with the torso.  World-frame keeps the emitted arm path an
        # exact rigid rotation of the recorded one.
        _, fr_a_src = S.fk(self._source_pose(anchor, side), self.body)
        r_src_anchor = (
            fr_a_src[f"{side}_collar"]
            @ S.arm_rot(*(self._arm_track(anchor, side)[n] for n in tri))
        )

        def delta_for(posed: dict) -> np.ndarray:
            _, fr_a_out = S.fk(posed, self.body)
            return (fr_a_out[f"{side}_collar"]
                    @ S.arm_rot(*(posed[n] for n in tri))) @ r_src_anchor.T

        # The swivel fixes more than the hold pose: the ramp and release are the
        # recorded arc swung along r_delta's geodesic, and r_delta's axis is
        # set by whichever swivel the anchor picked.  Two swivels that tie at
        # the hold can sweep the arm through very different space on the way
        # there (down-and-across with the left arm is the classic: one branch
        # rolls the elbow behind the back and the forearm through the ribs).
        # So re-rank the anchor's candidates on anchor cost + swept-path
        # collision before committing.
        path = self._path_samples(idx, w_track, side, aim0, d_yaw, d_pitch, bend, opts)
        if path:
            pos_a, fr_a = S.fk(anchor_posed, self.body)
            cands = solve_arm_ik(pos_a, fr_a, side, anchor_sol["aim"], self.body, bend,
                                 self._swivel_pref(side), opts, warm=warm,
                                 all_candidates=True)
            floor = cands[0]["cost"]
            best_phi, best_total = anchor_sol["phi"], None
            for cand in cands:
                if cand["cost"] > floor + 3.0:
                    break          # sorted: nothing further can win
                posed_c = dict(anchor_posed)
                posed_c.update(cand["angles"])
                pen = self._path_penalty(path, side, delta_for(posed_c), bend)
                total = cand["cost"] + opts.collide_weight * pen
                if best_total is None or total < best_total - 1e-9:
                    best_total, best_phi = total, cand["phi"]
            if best_phi != anchor_sol["phi"]:
                anchor_posed, anchor_sol, _ = self._solve_frame(
                    anchor, aim0, side, d_yaw, d_pitch, bend, target_point, opts,
                    warm=warm, fixed_phi=best_phi,
                )
        phi = anchor_sol["phi"]          # elbow swivel, resolved once (see solve_arm_ik)
        warm = tuple(anchor_posed[n] for n in tri)
        r_delta = delta_for(anchor_posed)

        # Weight of the "hold the aim exactly" correction: 1 across the settled
        # core, cosine-ramped to 0 over three frames at each edge so the
        # recorded overshoot and release stay intact and nothing snaps.
        steady = self._steady_weights(idx, opts)
        emit = C.UNWRAP_LIMITS if opts.unwrap_shoulder else C.EMIT_LIMITS

        for k, (i, w) in enumerate(zip(idx, w_track, strict=True)):
            src_ang = self._source_pose(i, side)
            prev_tri = None if not frames_out else tuple(
                frames_out[-1]["angles"][n] for n in tri
            )
            deltas = self._frame_deltas(i, aim0, side, d_yaw, d_pitch, bend, opts)

            # --- retarget as a DELTA on the template's own trajectory, scaled
            # by the envelope.  Zero delta reproduces the template frame for
            # frame, a non-zero one rotates/leans the same gesture elsewhere.
            out = dict(src_ang)
            for name, dval in deltas.items():
                v = C.to_limits(name, float(src_ang[name] + w * dval))
                # clamp before the FK below: the collar frame the shoulder is
                # decomposed in has to be the one that will actually be emitted,
                # or the triple is solved against a pose nobody renders
                out[name] = C.clamp(name, v) if opts.clamp_limits else v
            # The shoulder is a rotation, not three independent angles: apply
            # the retarget as a rotation in the collar frame, so the recorded
            # arc swings rigidly onto the new aim instead of being interpolated
            # through the p = 0 pole (where the azimuth is undefined and the
            # hand would loop off on a wild path).
            _, fr_src = S.fk(src_ang, self.body)
            pos_out, fr_out = S.fk(out, self.body)
            r_world = S.slerp_rot(np.eye(3), r_delta, float(w)) @ (
                fr_src[f"{side}_collar"] @ S.arm_rot(*(src_ang[n] for n in tri))
            )
            if steady[k] > 0.0:
                r_solved, warm = self._steady_solve(pos_out, fr_out, side, aim0, target_point,
                                                    out[f"{side}_elbow"], warm, opts, phi)
                r_world = S.slerp_rot(r_world, r_solved, float(steady[k]))
            r_out = fr_out[f"{side}_collar"].T @ r_world
            for n, v in zip(tri, _triple_in_limits(side, S.arm_rot_inverse(r_out),
                                                   prev=prev_tri, limits=emit), strict=True):
                out[n] = v
            if opts.clamp_limits:
                # only the DOFs this generator writes: the rest are the
                # template's own captured values and the advisory limits are,
                # per JOINTS.md, advisory -- clamping them would quietly
                # rewrite the source pose (legs especially).
                for name in self._blended_dofs(side):
                    out[name] = C.clamp(name, out[name], limits=emit)

            src = tpl.frames[i]
            root = (0.0, 0.0, 0.0) if opts.root == "zero" else tuple(src["root_xy_yaw"])
            frames_out.append(
                {
                    "angles": {n: float(out[n]) for n in C.ROS_JOINT_ORDER},
                    "root_xy_yaw": root,
                    "animation_state": src.get("animation_state", 0),
                    "t": float(k * tpl.dt),
                }
            )
            diag.append({"w": float(w), "sol": anchor_sol, "src": i,
                         "steady": float(steady[k]), "anchor": bool(i == anchor)})

        clip = PointAtClip(
            frames=frames_out, side=side, target_dir=aim0,
            target_point=target_point, template=tpl.name,
            envelope=np.asarray(w_track, dtype=float),
        )
        clip.report = self.analyze(clip, diag)
        clip.report["near_target_fallback"] = near_target
        clip.report["elbow_bend_deg"] = float(np.degrees(bend))
        clip.report["swivel_rad"] = float(phi)

        # A straight-arm point into the lower-side band lays the upper arm along
        # the ribs.  People solve that by bending the elbow and holding it off
        # the body, so try that before declaring the target unreachable, only
        # keep it if it actually buys clearance.
        if (
            opts.relax_on_collision
            and opts.elbow_bend is None
            and clip.report["clearance_m"] < 0.0
        ):
            best = clip
            for extra in (0.22, 0.45, 0.70):
                alt = self.point_at(
                    target=target if target_point is not None or target is not None else None,
                    azimuth=azimuth, elevation=elevation, deg=deg,
                    options=opts, elbow_bend=bend + extra, relax_on_collision=False,
                )
                # clearance is worth buying, but not with the aim
                if (
                    alt.report["clearance_m"] > best.report["clearance_m"]
                    and alt.report["aim_error_deg"] <= max(1.0, clip.report["aim_error_deg"])
                ):
                    best = alt
                if best.report["clearance_m"] >= 0.005:
                    break
            best.report["relaxed"] = best is not clip
            return best
        clip.report["relaxed"] = False
        return clip

    # -- gesture chaining -------------------------------------------------
    def aim_of(self, target: Sequence[float] | np.ndarray) -> np.ndarray:
        """Unit aim direction :meth:`point_at` would use for ``target``."""
        return self._resolve_aim(target, None, None, True)[0]

    def hold_pose(self, clip: PointAtClip) -> HoldPose:
        """The pose ``clip`` holds at its peak, as the start of a :meth:`retarget`."""
        ang = clip.frames[clip.peak]["angles"]
        return HoldPose(
            side=clip.side,
            angles={n: float(ang[n]) for n in self._blended_dofs(clip.side)},
            swivel=float(clip.report["swivel_rad"]),
            target_dir=np.asarray(clip.target_dir, dtype=float),
            target_point=None if clip.target_point is None else np.asarray(clip.target_point, dtype=float),
        )

    # -- helpers ----------------------------------------------------------
    def _steady_solve(self, pos_out: dict, fr_out: dict, side: str, aim0: np.ndarray, target_point: np.ndarray | None,
                      bend_i: float, warm: tuple[float, float, float], opts: PointAtOptions, phi: float | None,
                      swivel_pref: float | None = None) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Exact-aim arm rotation (world) for one emitted body pose, and the new warm start.

        Solves against the elbow angle actually being emitted this frame (the
        template's own elbow micro-motion rides on top of the target bend),
        otherwise the forearm -- and so the ray -- lands a degree or two off
        what the shoulder was solved for.
        """
        tri = [f"{side}_y_shoulder", f"{side}_p_shoulder", f"{side}_r_shoulder"]
        pref = self._swivel_pref(side) if swivel_pref is None else swivel_pref
        aim_i = aim0
        if target_point is not None:
            aim_i = self._aim_for_point(
                pos_out, fr_out, side, target_point, bend_i, warm, opts, phi, swivel_pref=pref
            )
        sol_i = solve_arm_ik(pos_out, fr_out, side, aim_i, self.body, bend_i,
                             pref, opts, warm=warm, fixed_phi=phi)
        warm = tuple(sol_i["angles"][n] for n in tri)
        return fr_out[f"{side}_collar"] @ S.arm_rot(*(sol_i["angles"][n] for n in tri)), warm

    def _resolve_aim(self, target: Sequence[float] | np.ndarray | None, azimuth: float | None,
                     elevation: float | None, deg: bool) -> tuple[np.ndarray, np.ndarray | None]:
        """(unit aim direction, target point or None) from the point_at arguments."""
        target_point = None
        if target is not None:
            target_point = np.asarray(target, dtype=float).reshape(3)
            if not np.all(np.isfinite(target_point)):
                raise ValueError(f"target must be finite, got {target}")
        elif azimuth is None and elevation is None:
            raise ValueError("give either target=(x, y, z) or azimuth=/elevation=")

        if target_point is not None:
            # reference the direction from the chest, the ray's rough origin
            ref = np.array([0.0, 0.0, 0.78 * self.body.hip_height + self.body.torso_height])
            delta = target_point - ref
            if np.linalg.norm(delta) < 1e-6:
                raise ValueError("target coincides with the pedestrian's chest")
            aim0 = S.unit(delta)
        else:
            aim0 = direction_from_angles(azimuth or 0.0, elevation or 0.0, deg=deg)
        return aim0, target_point

    def _near_target(self, target_point: np.ndarray, side: str) -> bool:
        """A target closer to the shoulder than the hand is cannot be aimed at
        along the forearm: the ray from the wrist points away from it, and
        the fixed point below would converge onto the reversed ray (a hand
        pointing at its own shoulder).  Inside that sphere fall back to the
        direction taken from the chest -- which is what a person does when
        indicating something on their own body."""
        pos_pk, _ = S.fk(self.template.frames[self.template.peak]["angles"], self.body)
        # Well outside the arm, not merely outside it: the forearm ray is
        # anchored at the wrist, so a target a hand's length past the
        # fingertip is a near-singular thing to aim at (and not something a
        # person points at either -- they gesture toward it).
        return float(np.linalg.norm(target_point - pos_pk[f"{side}_shoulder"])) <= (
            self.body.arm_reach + 0.45
        )

    def _path_samples(self, idx: list[int], w_track: np.ndarray, side: str, aim0: np.ndarray, d_yaw: float, d_pitch: float, bend: float, opts: PointAtOptions) -> list[dict]:
        """Everything about the ramp/release frames that does not depend on the
        swivel: source arm rotation, assisted body FK.  Computed once so the
        candidate ranking in ``point_at`` is just matrix products."""
        tri = [f"{side}_y_shoulder", f"{side}_p_shoulder", f"{side}_r_shoulder"]
        out = []
        for i, w in zip(idx, w_track, strict=True):
            if not (0.05 < w < 0.97):
                continue
            src_ang = self._source_pose(i, side)
            deltas = self._frame_deltas(i, aim0, side, d_yaw, d_pitch, bend, opts)
            posed = dict(src_ang)
            for name, dval in deltas.items():
                v = C.to_limits(name, float(src_ang[name] + w * dval))
                posed[name] = C.clamp(name, v) if opts.clamp_limits else v
            _, fr_src = S.fk(src_ang, self.body)
            pos_out, fr_out = S.fk(posed, self.body)
            out.append({
                "w": float(w),
                "r_src": fr_src[f"{side}_collar"] @ S.arm_rot(*(src_ang[n] for n in tri)),
                "pos": pos_out,
                "collar": fr_out[f"{side}_collar"],
                "bend": float(posed[f"{side}_elbow"]),
            })
        return out

    def _path_penalty(self, path: list[dict], side: str, r_delta: np.ndarray, bend: float) -> float:
        """Summed capsule interpenetration (m) of the arm along the swept
        ramp/release for one candidate retarget rotation."""
        pen = 0.0
        for smp in path:
            r_world = S.slerp_rot(np.eye(3), r_delta, smp["w"]) @ smp["r_src"]
            y, p, r = S.arm_rot_inverse(smp["collar"].T @ r_world)
            sh = smp["pos"][f"{side}_shoulder"]
            el, wr, _ = _arm_positions(sh, smp["collar"], y, p, r, smp["bend"], self.body)
            pen += max(0.0, -_arm_clearance(smp["pos"], side, sh, el, wr, self.body))
        return pen

    def _steady_weights(self, idx: list[int], opts: PointAtOptions) -> np.ndarray:
        """Per-frame weight of the exact-aim correction over the settled hold."""
        n = len(idx)
        w = np.zeros(n)
        if not opts.steady_hold:
            return w
        s0, s1 = self.template.settled
        inside = np.array([s0 <= i < s1 for i in idx], dtype=float)
        if not inside.any():
            return w
        run = np.flatnonzero(inside)
        a, b = int(run[0]), int(run[-1])
        ramp = min(3, max(0, (b - a) // 2))
        for k in range(a, b + 1):
            edge = min(k - a, b - k, ramp)
            w[k] = 0.5 - 0.5 * np.cos(np.pi * (edge + 1) / (ramp + 1))
        return w

    def _arm_track(self, i: int, side: str) -> dict:
        """The template's *gesture* arm at frame ``i``, expressed on ``side``.

        Mirroring a pose across the sagittal plane negates the azimuth and the
        twist and leaves flexion alone (JOINTS.md 1a: the wire uses the same
        sign convention on both sides, so the mirror is explicit here rather
        than implied by the convention).
        """
        tpl = self.template
        a = tpl.frames[i]["angles"]
        b = tpl.base[i]
        s = tpl.side
        mirror = side != s
        k = self.opts.collar_scale
        return {
            f"{side}_y_collar": b[f"{s}_y_collar"] + k * (a[f"{s}_y_collar"] - b[f"{s}_y_collar"]),
            f"{side}_p_collar": b[f"{s}_p_collar"] + k * (a[f"{s}_p_collar"] - b[f"{s}_p_collar"]),
            f"{side}_y_shoulder": -a[f"{s}_y_shoulder"] if mirror else a[f"{s}_y_shoulder"],
            f"{side}_p_shoulder": a[f"{s}_p_shoulder"],
            f"{side}_r_shoulder": -a[f"{s}_r_shoulder"] if mirror else a[f"{s}_r_shoulder"],
            f"{side}_elbow": a[f"{s}_elbow"],
        }

    def _source_pose(self, i: int, side: str) -> dict:
        """Template frame with the gesture moved onto the pointing arm.

        Starts from the residual track (the gesture projected out of the arm
        that recorded it, so it hangs idle when the other hand is doing the
        pointing) and puts the recorded gesture back on ``side``.  For the
        template's own side this is exactly the template frame again.
        """
        pose = dict(self.template.base[i])
        pose.update(self._arm_track(i, side))
        return pose

    def _frame_deltas(self, i: int, aim0: np.ndarray, side: str, d_yaw: float, d_pitch: float, bend: float,
                      opts: PointAtOptions) -> dict[str, float]:
        """Every non-shoulder delta for one frame: torso lean/turn, clavicle,
        elbow bend.  The head is deliberately left alone.  All are differences
        against what the template does for its *own* target, so an unchanged
        target gives zero."""
        tpl = self.template
        src = self._source_pose(i, side)
        deltas: dict[str, float] = {}
        for seg, fy, fp in zip(C.SPINE_SEGMENTS, YAW_SPLIT, PITCH_SPLIT, strict=True):
            deltas[f"y_{seg}"] = fy * d_yaw
            deltas[seg] = fp * d_pitch
        leaned = dict(src)
        for name, dv in deltas.items():
            leaned[name] = C.clamp(name, src[name] + dv) if opts.clamp_limits else src[name] + dv

        _, fr = S.fk(leaned, self.body)
        az_t, el_t = _torso_frame_angles(fr["torso"], aim0)
        cy, cp = _collar_assist(side, az_t, el_t, opts)
        cy0, cp0 = _collar_assist(tpl.side, self.tpl_az_torso, self.tpl_el_torso, opts)
        deltas[f"{side}_y_collar"] = cy - cy0
        deltas[f"{side}_p_collar"] = cp - cp0
        deltas[f"{side}_elbow"] = bend - tpl.elbow_bend
        return deltas

    def _solve_frame(self, i: int, aim0: np.ndarray, side: str, d_yaw: float, d_pitch: float, bend: float, target_point: np.ndarray | None,
                     opts: PointAtOptions, warm: tuple[float, float, float], fixed_phi: float | None,
                     swivel_pref: float | None = None) -> tuple[dict, dict, dict[str, float]]:
        """Fully-pointing pose for one source frame, plus its deltas vs source."""
        src = self._source_pose(i, side)
        deltas = self._frame_deltas(i, aim0, side, d_yaw, d_pitch, bend, opts)
        posed = dict(src)
        for name, dv in deltas.items():
            v = src[name] + dv
            posed[name] = C.clamp(name, v) if opts.clamp_limits else v

        # --- aim IK
        pos, fr = S.fk(posed, self.body)
        pref = self._swivel_pref(side) if swivel_pref is None else swivel_pref
        aim = aim0
        if target_point is not None:
            aim = self._aim_for_point(pos, fr, side, target_point, bend, warm, opts, fixed_phi, swivel_pref=pref)
        sol = solve_arm_ik(pos, fr, side, aim, self.body, bend, pref,
                           opts, warm=warm, fixed_phi=fixed_phi)
        posed.update(sol["angles"])
        return posed, sol, deltas

    def _swivel_pref(self, side: str) -> float:
        """The elbow swivel is chiral: mirroring the body across the sagittal
        plane sends phi -> -phi (it is measured about a cross product), so the
        template's right-arm swivel becomes its negative on the left arm."""
        return self.template.swivel * (1.0 if side == self.template.side else -1.0)

    def _blended_dofs(self, side: str) -> list[str]:
        return (
            [f"y_{s}" for s in C.SPINE_SEGMENTS]
            + list(C.SPINE_SEGMENTS)
            + [f"r_{s}" for s in C.SPINE_SEGMENTS]
            + C.arm_dofs(side)
        )

    def _assist_deltas(self, az: float, el: float, opts: PointAtOptions) -> tuple[float, float]:
        y_new, p_new = _torso_assist(az, el, opts)
        y_tpl, p_tpl = _torso_assist(self.tpl_az, self.tpl_el, opts)
        return y_new - y_tpl, p_new - p_tpl

    def _timeline(self, opts: PointAtOptions) -> tuple[np.ndarray, list[int]]:
        """Frame index track and envelope, optionally with a stretched hold."""
        tpl = self.template
        idx = list(range(tpl.n))
        if opts.hold_s is None:
            return tpl.w.copy(), idx
        # stretch the *settled* core, not the whole plateau: the frames before
        # it are the overshoot and belong to the ramp, and squeezing the hold
        # below the settle time should drop hold frames, not settled ones
        h0, h1 = tpl.settled
        plateau = self._plateau(max(1, int(round(opts.hold_s / tpl.dt))))
        idx = idx[:h0] + plateau + idx[h1:]
        w = np.concatenate([tpl.w[:h0], np.full(len(plateau), 1.0), tpl.w[h1:]])
        return w, idx

    def _plateau(self, want: int) -> list[int]:
        """``want`` source indices ping-ponging through the settled core so the micro-motion never jumps."""
        h0, h1 = self.template.settled
        span = max(1, h1 - h0)
        period = max(1, 2 * span - 2)
        return [h0 + (j % period if (j % period) < span else period - (j % period))
                for j in range(want)]

    def _aim_for_point(self, pos: dict, fr: dict, side: str, target_point: np.ndarray, bend: float, warm: tuple[float, float, float],
                       opts: PointAtOptions, fixed_phi: float | None = None, swivel_pref: float | None = None) -> np.ndarray:
        """Fixed point: the forearm axis must pass through the target, and the
        wrist (the ray's origin) moves as the arm turns."""
        pref = self._swivel_pref(side) if swivel_pref is None else swivel_pref
        sh = pos[f"{side}_shoulder"]
        aim = S.unit(target_point - sh)
        best, best_err = aim, np.inf
        for _ in range(12):
            sol = solve_arm_ik(pos, fr, side, aim, self.body, bend,
                               pref, opts, warm=warm,
                               fixed_phi=fixed_phi)
            delta = target_point - sol["wrist"]
            if np.linalg.norm(delta) < 1e-4:
                return aim
            want = S.unit(delta)
            err = float(np.arccos(np.clip(float(want @ sol["achieved"]), -1.0, 1.0)))
            if err < best_err:
                best, best_err = aim, err
            if err < 1e-4:
                return aim
            if np.dot(want, aim) < -0.5:     # target behind the wrist: keep the outward ray
                break
            aim = S.unit(aim + FIXED_POINT_DAMPING * (want - aim))
        return best

    # -- diagnostics ------------------------------------------------------
    def analyze(self, clip: PointAtClip, diag: list | None = None) -> dict:
        """Aim error, clearance and limit saturation over the hold plateau."""
        side = clip.side
        s0, s1 = self.template.settled
        settled = [i for i, f in enumerate(clip.frames)] if diag is None else [
            i for i, d in enumerate(diag) if s0 <= d["src"] < s1
        ]
        # judge the aim on the locked part of the hold, the ramped edges are
        # the recorded settle-in/release and are reported separately
        hold = [i for i in settled if (diag[i]["steady"] if diag else 1.0) >= 0.999]
        hold = hold or settled or (
            [int(np.argmax([d["w"] for d in diag]))] if diag else [0]
        )
        anchor_i = next(
            (i for i, d in enumerate(diag or []) if d.get("anchor")), hold[len(hold) // 2]
        )
        rows = []
        for i in hold:
            ang = clip.frames[i]["angles"]
            pos, _ = S.fk(ang, self.body)
            el, wr = pos[f"{side}_elbow"], pos[f"{side}_wrist"]
            achieved = S.unit(wr - el)
            if clip.target_point is not None:
                want = S.unit(clip.target_point - wr)
            else:
                want = clip.target_dir
            err = float(np.degrees(np.arccos(np.clip(np.dot(achieved, want), -1.0, 1.0))))
            worst, clear = S.worst_clearance(pos, self.body, side)
            sat = 0.0
            sat_joints = {}
            if diag is not None:
                sat = diag[i]["sol"]["saturation"]
                sat_joints = {
                    k: round(float(v), 4)
                    for k, v in (
                        (k, C.saturation(k, v)) for k, v in diag[i]["sol"]["raw"].items()
                    )
                    if v > 1e-4
                }
            rows.append((err, clear, worst, sat, wr, sat_joints))
        errs = np.array([r[0] for r in rows])
        # a collision anywhere in the clip is a collision, so scan every frame
        # the idle head and tail are the template's own pose verbatim, so their
        # clearance is the source's, not this target's -- judge the gesture
        whole = []
        active = [i for i, d in enumerate(diag or []) if d["w"] >= 0.05]
        for i in (active or range(len(clip.frames))):
            pos_i, _ = S.fk(clip.frames[i]["angles"], self.body)
            whole.append((*S.worst_clearance(pos_i, self.body, side)[::-1], i))
        clear_v, clear_pair, clear_i = min(whole)
        clears = np.array([r[1] for r in rows])
        k = int(np.argmin(clears))
        az, el_ = angles_from_direction(clip.target_dir)
        anchor_err = float(errs[hold.index(anchor_i)]) if anchor_i in hold else float(errs.min())
        settle_err = 0.0
        for i in settled:
            pos_i, _ = S.fk(clip.frames[i]["angles"], self.body)
            d_i = S.unit(pos_i[f"{side}_wrist"] - pos_i[f"{side}_elbow"])
            want_i = (
                S.unit(clip.target_point - pos_i[f"{side}_wrist"])
                if clip.target_point is not None else clip.target_dir
            )
            settle_err = max(
                settle_err,
                float(np.degrees(np.arccos(np.clip(float(d_i @ want_i), -1.0, 1.0)))),
            )
        m = np.array([[fr["angles"][n] for n in C.ROS_JOINT_ORDER] for fr in clip.frames])
        steps = np.abs(np.diff(m, axis=0)) if len(m) > 1 else np.zeros((1, len(m[0])))
        j = int(np.unravel_index(int(np.argmax(steps)), steps.shape)[1])
        return {
            "template": clip.template,
            "max_joint_step_rad": float(steps.max()),
            "max_joint_step_joint": C.ROS_JOINT_ORDER[j],
            "side": side,
            "azimuth_deg": float(az),
            "elevation_deg": float(el_),
            "aim_error_deg": float(errs.max()),
            "aim_error_anchor_deg": anchor_err,
            "aim_settle_deg": settle_err,
            "aim_error_mean_deg": float(errs.mean()),
            "clearance_m": float(clear_v),
            "clearance_pair": clear_pair,
            "clearance_frame": int(clear_i),
            "saturation_rad": float(max(r[3] for r in rows)),
            "hold_frames": len(hold),
            "n_frames": len(clip.frames),
            "duration_s": clip.duration,
            "wrist_xyz": [float(v) for v in rows[k][4]],
            "saturated_joints": {
                k2: v2 for r in rows for k2, v2 in r[5].items()
            },
            "ok": bool(errs.max() < 5.0 and clear_v > -0.005),
        }


def point_at(target: Sequence[float] | np.ndarray | None = None, azimuth: float | None = None, elevation: float | None = None,
             template: str = "point_to_right", height: float = 1.65, **kw: object) -> PointAtClip:
    """One-shot convenience wrapper around :class:`PointAtGenerator`."""
    return PointAtGenerator(template=template, height=height).point_at(
        target=target, azimuth=azimuth, elevation=elevation, **kw
    )
