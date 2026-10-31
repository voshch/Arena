"""SMPL-X motion -> Arena wire angles: a rotation retarget as the start, then whole-body IK on the GPU.

1. Rotation retarget (closed form, vectorized over frames). Each Arena segment takes the SMPL-X segment's rotation
   away from ITS OWN rest, so the two rest shapes never meet and twist is kept. Fitting positions instead would turn
   every rest-shape difference into a pose: SMPL-X's clavicle rises 25 deg at rest (a constant shrug) and its rest limbs
   are kinked sideways (left arm 19 deg, legs 9-14 deg), which the contract's one-plane hinges can only fake by
   twisting the whole limb (knees turned out ~60 deg).
     pelvis/spine1/spine2 -> waist/spine/chest; neck+head -> head triple; collar -> 2-DOF clavicle (twist dropped)
     arms and legs: T-pose map Rx(+-90 deg) for the arms, then each limb's rest map keeps the in-plane part of SMPL-X's
     rest kink as the rest hinge angle, drops the sideways part, and lays the limb on SMPL-X's rest line
     (shoulder -> wrist, hip -> ankle); wrist -> pronation + extension; ankle -> yaw + sagittal; pelvis -> root.
2. Whole-body IK (``ik.solve``): all frames of all clips at once, matching what the eye reads (hand and foot reach
   scaled to Arena's limb lengths, hands relative to each other and to the neck, stance width, hand orientation, foot
   direction, bone directions) while staying near the rotation retarget, inside the contract's hard limits, and smooth
   in time. The rotation retarget alone puts a bent arm's hand 10-27 deg off, because Arena's forearm is 0.6x its upper
   arm and SMPL-X's are equal.

The pelvis tilt (pitch/roll) has no wire DOF: ``fold_tilt=True`` puts it into the waist and hips (exact body shape,
upright root); ``fold_tilt=False`` leaves the body pelvis-relative for a renderer that applies ``root_quat``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .contract import SK, WIRE
from .smplx import J, Motion

if TYPE_CHECKING:
    from .ik import IKConfig

IDX = {k: i for i, k in enumerate(WIRE)}

# Contract arm-down rest -> SMPL-X T-pose rest, per side (palms in <-> palms down).
ARM_REST = {"l": SK.rot_axis([1, 0, 0], np.pi / 2), "r": SK.rot_axis([1, 0, 0], -np.pi / 2)}

# Per limb: (proximal joint, hinge joint, end joint, T-pose map). Collars, spine, head and feet keep the plain rest
# map: their SMPL-X landmarks are not the contract's (collar joint at the sternum, foot joint at the ball of the foot).
LIMBS = {f"{s}_{n}": (f"{s}_{a}", f"{s}_{b}", f"{s}_{c}", m) for s in "lr" for n, a, b, c, m in (("arm", "shoulder", "elbow", "wrist", ARM_REST[s]), ("leg", "hip", "knee", "ankle", np.eye(3)))}
# IK bone-direction targets, in this order: (source joint, its rest-mapped bone)
BONE_JOINTS = [f"{s}_{j}" for s in "lr" for j in ("shoulder", "elbow", "hip", "knee")]


@dataclass
class Clip:
    """A retargeted clip at a fixed rate: what Arena's Animation npz stores."""

    angles: dict[str, np.ndarray]  # wire DOF -> (N,) rad
    t: np.ndarray  # (N,) s
    root_xy_yaw: np.ndarray  # (N, 3) pelvis x, y (m) and heading (rad); Arena plays clips in place and ignores it
    root_z: np.ndarray  # (N,) pelvis height above the lowest joint (m)
    root_quat: np.ndarray  # (N, 4) pelvis orientation in its heading frame, xyzw
    fold_tilt: bool = True
    meta: dict = field(default_factory=dict)

    @property
    def fps(self) -> float:
        return float(1.0 / np.median(np.diff(self.t))) if len(self.t) > 1 else 0.0

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0


@dataclass
class Prepared:
    """One motion ready for the IK: rotation-retarget start (T, 40) and per-frame targets in the body base frame."""

    init: np.ndarray
    targets: dict[str, np.ndarray]
    src: np.ndarray  # (T, 22, 3) source joints in the base frame, pelvis at the origin
    root_xy_yaw: np.ndarray
    root_z: np.ndarray
    root_quat: np.ndarray
    fps: float
    fold_tilt: bool
    scale: dict[str, float]  # source -> contract length ratios used by the reach targets

    def __len__(self) -> int:
        return len(self.init)


# ---- small vectorized rotation helpers ---------------------------------------------------------------------------


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking direction a onto direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    ax = np.cross(a, b)
    s, c = np.linalg.norm(ax), float(np.dot(a, b))
    return np.eye(3) if s < 1e-9 else SK.rot_axis(ax / s, float(np.arctan2(s, c)))


def _rot(axis: Sequence[float] | np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Batched SK.rot_axis: (T,) angles about one axis -> (T, 3, 3)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    s, c = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def _T(m: np.ndarray) -> np.ndarray:
    return np.swapaxes(m, -1, -2)


def _euler(m: np.ndarray, seq: str) -> np.ndarray:
    return Rotation.from_matrix(m).as_euler(seq)  # upper case = intrinsic, matches the contract's right-multiplied chains


def _hinge(m: np.ndarray) -> np.ndarray:
    """Angle of the contract hinge rot_axis((0,-1,0), a) = Ry(-a) closest to m, batched."""
    return -np.arctan2(m[..., 0, 2], m[..., 2, 2])


def _wrap_near(v: np.ndarray, ref: np.ndarray | float) -> np.ndarray:
    return ref + np.angle(np.exp(1j * (v - ref)))


def heading(pelvis_rot: np.ndarray) -> np.ndarray:
    """Heading from the pelvis' left axis: stays defined when the body lies down. (3, 3) or (T, 3, 3)."""
    left = pelvis_rot[..., :, 1]
    return np.arctan2(-left[..., 0], left[..., 1])


def _arm_inverse(m: np.ndarray) -> np.ndarray:
    """Inverse of SK.arm_rot = Rz(y) R(-Y, p) R(-Z, r) for (T, 3, 3), each frame on the branch nearest the previous
    one (the sphere's double cover (y, p) ~ (y + pi, -p); a straight-down/up arm holds its azimuth)."""
    T = len(m)
    d = m @ SK.DOWN
    p = np.arccos(np.clip(-d[:, 2], -1, 1))
    y = np.arctan2(d[:, 1], d[:, 0])
    cands = []
    for yy, pp in ((y, p), (y + np.pi, -p)):
        n = _T(_rot(SK.ARM_AXES[0], yy) @ _rot(SK.ARM_AXES[1], pp)) @ m
        cands.append((yy, pp, np.arctan2(-n[:, 1, 0], n[:, 0, 0])))
    out = np.zeros((T, 3))
    prev = np.zeros(3)
    pole = np.sin(p) < 1e-4
    for t in range(T):
        best, cost = None, np.inf
        for yy, pp, rr in cands:
            yv = prev[0] if pole[t] else _wrap_near(yy[t], prev[0])
            c = (yv, pp[t], _wrap_near(rr[t], prev[2]))
            k = abs(c[0] - prev[0]) + abs(c[1] - prev[1]) + abs(c[2] - prev[2])
            if k < cost:
                best, cost = c, k
        assert best is not None  # cost starts at inf, so the first candidate is always taken
        out[t] = prev = np.asarray(best)
    return out


# ---- rest maps -----------------------------------------------------------------------------------------------------


def _limb_rest(upper: np.ndarray, lower: np.ndarray, m0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rest maps of a limb's two bones from SMPL-X's rest bones (ROS axes). The rest kink splits into a bend in the
    contract hinge plane, kept as the rest hinge angle e0, and a sideways part the hinge cannot make, dropped. The bent
    contract limb is then laid on the rest line."""
    qb = _align(m0 @ SK.DOWN, upper) @ m0
    v = qb.T @ lower
    e0 = float(np.arctan2(v[0], -v[2]))  # SK.rot_axis(ELBOW_AXIS, e) @ DOWN = (sin e, 0, -cos e)
    hinge = SK.rot_axis(SK.ELBOW_AXIS, e0)
    line = np.linalg.norm(upper) * SK.DOWN + np.linalg.norm(lower) * (hinge @ SK.DOWN)
    qa = _align(m0 @ line, upper + lower) @ m0
    return qa, qa @ hinge


def rest_maps(motion: Motion) -> dict[str, np.ndarray]:
    """Per-joint rest map: contract rest frame -> SMPL-X rest frame (both in ROS axes)."""
    Q = {j: np.eye(3) for j in J}
    for s in "lr":
        Q[f"{s}_wrist"] = ARM_REST[s]
    for a, b, c, m0 in LIMBS.values():
        Q[a], Q[b] = _limb_rest(motion.rest_vector(a, b), motion.rest_vector(b, c), m0)
    return Q


def source_lengths(motion: Motion) -> dict[str, float]:
    """Source limb reach at rest: the straight-line rest distance (shoulder -> wrist, hip -> ankle), not the bone sum.
    SMPL-X's rest limbs are kinked, so a rest limb, fully extended in its own geometry, then maps to a straight
    Arena limb instead of one bent to shorten it."""
    n = lambda a, b: float(np.linalg.norm(motion.rest_vector(a, b)))  # noqa: E731
    out = {f"{s}_arm": n(f"{s}_shoulder", f"{s}_wrist") for s in "lr"}  # per side: SMPL-X's kinks are asymmetric
    out.update({f"{s}_leg": n(f"{s}_hip", f"{s}_ankle") for s in "lr"})
    out.update(arm=0.5 * (out["l_arm"] + out["r_arm"]), leg=0.5 * (out["l_leg"] + out["r_leg"]), clavicle=0.5 * sum(n("spine3", f"{s}_shoulder") for s in "lr"))
    return out


def contract_lengths(height: float = 1.75) -> dict[str, float]:
    b = SK.Body(height)
    arm, leg = b.upperarm + b.forearm, b.thigh + b.tibia
    return {"arm": arm, "leg": leg, "l_arm": arm, "r_arm": arm, "l_leg": leg, "r_leg": leg, "clavicle": b.neck_shoulder}


CONTACT_DIST = 0.25  # m


def _contact(d: np.ndarray) -> np.ndarray:
    """Per-frame weight 0 .. 1, rising as a source distance (m) falls below ~CONTACT_DIST. Absolute hand relations
    only matter near contact: far apart, the two skeletons' different spans would fight the reach."""
    return np.exp(-((d / CONTACT_DIST) ** 2))


# ---- stage 1: rotation retarget + IK targets -----------------------------------------------------------------------


def prepare(motion: Motion, fold_tilt: bool = True, height: float = 1.75) -> Prepared:
    """Rotation retarget of every frame (vectorized) and the IK targets, all in the body base frame: the heading frame
    (fold_tilt) or the pelvis frame."""
    G, P, T = motion.global_rots, motion.joints, len(motion)
    g = lambda j: G[:, J[j]]  # noqa: E731
    Q = rest_maps(motion)
    th = np.zeros((T, len(WIRE)))
    put = lambda k, v: th.__setitem__((slice(None), IDX[k]), v)  # noqa: E731

    yaw = heading(g("pelvis"))
    root_r = _rot(SK.UP, yaw)
    base = root_r if fold_tilt else g("pelvis")
    BT = _T(base)
    root_xy_yaw = np.stack([P[:, 0, 0], P[:, 0, 1], np.unwrap(yaw)], 1)
    root_z = P[:, 0, 2] - P[:, :, 2].min(1)  # pelvis above the lowest joint (the floor)
    root_quat = Rotation.from_matrix(_T(root_r) @ g("pelvis")).as_quat()

    # spine: each segment takes its SMPL-X counterpart's frame (the Euler split is exact, so the chain lands on it)
    cur = base
    for seg, j in (("waist", "pelvis"), ("spine", "spine1"), ("chest", "spine2")):
        r, y, p = _euler(_T(cur) @ g(j), "XZY").T  # segment_rot = Rx(r) Rz(y) Ry(p)
        put(f"r_{seg}", r)
        put(f"y_{seg}", y)
        put(seg, p)
        cur = g(j)
    chest = cur
    r, y, q = _euler(_T(chest) @ g("head"), "XZY").T  # head_rot = Rx(r) Rz(y) R(-Y, p)
    put("r_head", r)
    put("y_head", y)
    put("p_head", -q)

    for s in "lr":
        sign = 1.0 if s == "l" else -1.0
        d = (_T(chest) @ g(f"{s}_collar")) @ np.array([0.0, sign, 0.0])  # where the rest clavicle swings to
        cp, cy = np.arcsin(np.clip(d[:, 2], -1, 1)), np.arctan2(d[:, 0], sign * d[:, 1])
        put(f"{s}_y_collar", cy)
        put(f"{s}_p_collar", cp)
        ay, ap = SK.COLLAR_AXES[s]
        c_r = chest @ _rot(ay, cy) @ _rot(ap, cp)
        U, Fo = g(f"{s}_shoulder") @ Q[f"{s}_shoulder"], g(f"{s}_elbow") @ Q[f"{s}_elbow"]
        arm = _arm_inverse(_T(c_r) @ U)
        for k, v in zip(("y_shoulder", "p_shoulder", "r_shoulder"), arm.T, strict=True):
            put(f"{s}_{k}", v)
        S = c_r @ _rot(SK.ARM_AXES[0], arm[:, 0]) @ _rot(SK.ARM_AXES[1], arm[:, 1]) @ _rot(SK.ARM_AXES[2], arm[:, 2])
        e = _hinge(_T(S) @ Fo)
        put(f"{s}_elbow", e)
        f_r = S @ _rot(SK.ELBOW_AXIS, e)
        a1, b1, _ = _euler(_T(f_r @ Q[f"{s}_elbow"].T @ ARM_REST[s]) @ (g(f"{s}_wrist") @ ARM_REST[s]), "ZXY").T
        put(f"{s}_r_wrist", -a1 if s == "l" else a1)
        put(f"{s}_wrist", b1 if s == "l" else -b1)

        Th, Sh = g(f"{s}_hip") @ Q[f"{s}_hip"], g(f"{s}_knee") @ Q[f"{s}_knee"]
        hy, hp, hr = _euler(BT @ Th, "ZXY").T  # = Rz(-y) Rx(+-p) Ry(-r) of the contract, exactly
        put(f"{s}_y_hip", -hy)
        put(f"{s}_p_hip", sign * hp)
        put(f"{s}_r_hip", -hr)
        k = _hinge(_T(Th) @ Sh)
        put(f"{s}_knee", k)
        k_r = Th @ _rot(SK.ELBOW_AXIS, k)
        fy, fb, _ = _euler(_T(k_r @ Q[f"{s}_knee"].T) @ g(f"{s}_ankle"), "ZYX").T  # Rz(-y_ankle) Ry(-ankle)
        put(f"{s}_y_ankle", -fy)
        put(f"{s}_ankle", -fb)
    th = np.unwrap(th, axis=0)  # never a 2 pi jump between frames

    # IK targets (base frame); reach and relations scaled from the source's to the contract's lengths
    src_len, con_len = source_lengths(motion), contract_lengths(height)
    scale = {k: con_len[k] / src_len[k] for k in src_len}
    scale["upper"] = (con_len["clavicle"] + con_len["arm"]) / (src_len["clavicle"] + src_len["arm"])
    q = np.einsum("tij,tkj->tki", BT, P - P[:, :1])  # source joints in the base frame
    v = lambda a, b: q[:, J[b]] - q[:, J[a]]  # noqa: E731
    hip_gap = np.array([0.0, SK.Body(height).waist_length, 0.0])  # contract l_hip - r_hip in the base frame
    tg = {
        "dirs": np.stack([BT @ g(j) @ Q[j] @ SK.DOWN for j in BONE_JOINTS], 1),  # (T, 8, 3)
        "fwd": np.stack([(BT @ g(j) @ Q[j])[:, :, 0] for j in BONE_JOINTS], 1),  # twist anchor, upper bones used
        "reach": np.stack([v(f"{s}_shoulder", f"{s}_wrist") * scale[f"{s}_arm"] for s in "lr"] + [v(f"{s}_hip", f"{s}_ankle") * scale[f"{s}_leg"] for s in "lr"], 1),  # (T, 4, 3)
        "hands": v("r_wrist", "l_wrist") * scale["upper"],
        "hand_neck": np.stack([v("neck", f"{s}_wrist") * scale["upper"] for s in "lr"], 1),
        # stance relative to the hips: Arena's hip joints sit 2.3x further apart than SMPL-X's, so an absolute foot gap
        # would pull Arena's feet in (knock-knees). Hands stay absolute: a clap must close.
        "feet": hip_gap + (v("r_ankle", "l_ankle") - v("r_hip", "l_hip")) * scale["leg"],
        # contact weights: on while the source hands touch each other or come to the head
        "hands_w": _contact(np.linalg.norm(v("r_wrist", "l_wrist"), axis=-1)),
        "hand_neck_w": np.stack([_contact(np.linalg.norm(v(f"{s}_wrist", "head"), axis=-1)) for s in "lr"], 1),
        "hand_rot": np.stack([BT @ g(f"{s}_wrist") @ ARM_REST[s] for s in "lr"], 1),  # (T, 2, 3, 3)
        "foot_fwd": np.stack([BT @ g(f"{s}_ankle") @ Q[f"{s}_knee"] @ SK.FWD for s in "lr"], 1),
        # absolute wrist/elbow targets (ik.ANCHOR_JOINTS order), off unless a caller sets weights (pair contacts)
        "anchor": np.zeros((T, 4, 3)),
        "anchor_w": np.zeros((T, 4)),
    }
    return Prepared(init=th, targets=tg, src=q, root_xy_yaw=root_xy_yaw, root_z=root_z, root_quat=root_quat, fps=motion.fps, fold_tilt=fold_tilt, scale=scale)


# ---- API -----------------------------------------------------------------------------------------------------------


def resample(prep: Prepared, angles: np.ndarray, fps: float = 20.0) -> Clip:
    t_in = np.arange(len(prep)) / prep.fps
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / fps) if len(t_in) > 1 else np.zeros(1)
    t_out = np.minimum(t_out, t_in[-1])  # arange can overshoot the end by rounding

    def lerp(v: np.ndarray) -> np.ndarray:
        return np.interp(t_out, t_in, v)

    quat = Slerp(t_in, Rotation.from_quat(prep.root_quat))(t_out).as_quat() if len(t_in) > 1 else prep.root_quat
    return Clip(angles={k: lerp(angles[:, i]) for i, k in enumerate(WIRE)}, t=t_out, root_xy_yaw=np.stack([lerp(prep.root_xy_yaw[:, i]) for i in range(3)], 1), root_z=lerp(prep.root_z), root_quat=quat, fold_tilt=prep.fold_tilt)


def retarget_batch(motions: list[Motion], fold_tilt: bool = True, fps: float = 20.0, method: str = "ik", device: str = "auto", config: IKConfig | None = None, preps: list[Prepared] | None = None, chunk_frames: int = 400_000, verbose: bool = False) -> list[Clip]:
    """Retarget many motions at once: rotation retarget per motion, then one batched IK over all their frames.

    ``method``: "ik" (default, whole body), "rotation" (bone directions only, no reach), or "none" (the closed-form
    rotation retarget as is). ``config``: an ``ik.IKConfig`` overriding the method's weights.
    """
    from . import ik

    preps = preps or [prepare(m, fold_tilt) for m in motions]
    if method == "none":
        sols = [p.init for p in preps]
    else:
        sols = ik.solve(preps, config or ik.IKConfig.preset(method), device=device, chunk_frames=chunk_frames, verbose=verbose)
    return [resample(p, s, fps) for p, s in zip(preps, sols, strict=True)]


def retarget(motion: Motion, fold_tilt: bool = True, fps: float = 20.0, method: str = "ik", device: str = "auto", config: IKConfig | None = None) -> Clip:
    """Retarget one SMPL-X motion onto Arena's wire and resample it to ``fps`` (Arena clips run at 20)."""
    return retarget_batch([motion], fold_tilt, fps, method, device, config)[0]
