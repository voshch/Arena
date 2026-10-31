"""Two-person interactions: retarget both people of a pair so that their contacts survive on Arena's bodies.

Arena plays every clip in place: a simulator poses each human at its agent's pose and ignores the clip's root motion.
A pair therefore becomes two clips plus a fixed placement of each person, and the hands have to meet the partner
where the partner's Arena body actually is. Steps:

1. **Window** (``window="auto"``): keep the part of the source where the two touch (padded), so the clips hold the
   interaction itself; Arena's interaction manager walks the agents into formation before it plays the clips.
   ``"all"`` keeps everything (conversations, waving: no contact).
2. **Placement**: each person's pelvis position and heading over the window, weighted by contact strength (how the
   source stands while touching), i.e. the pose its agent stands at.
3. **Contacts**: per frame, each wrist and elbow of one person is matched to the nearest of the partner's body
   points; within ~35 cm (wrist joint to the partner's joint) it is a contact, weighted by distance, and the offset is kept in the partner's heading frame.
4. **IK with anchors**: both people are solved in one batch (:func:`ik.solve`), then each contact becomes an anchor
   target on the Arena body, the partner's point plus the source offset (hand to hand contacts meet halfway), and
   the IK runs again (``passes`` times). Everything else (reach, style, limits, smoothing) is the single-clip IK.

The fidelity check compares each contact's vector (effector minus partner point) on the two Arena bodies with the
source's, on frames in contact.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from .retarget import Clip, Prepared, prepare, resample
from .smplx import J, Motion, load

if TYPE_CHECKING:
    from .ik import IKConfig

EFFECTORS = ("l_wrist", "r_wrist", "l_elbow", "r_elbow")  # = ik.ANCHOR_JOINTS
# partner body points a hand or elbow can touch: source joint -> Arena FK point(s) (averaged)
PARTNER = {
    "pelvis": ("pelvis",),
    "spine1": ("spine",),
    "spine2": ("chest",),
    "spine3": ("chest", "torso"),
    "neck": ("torso",),
    "head": ("head",),
    "l_shoulder": ("l_shoulder",),
    "r_shoulder": ("r_shoulder",),
    "l_elbow": ("l_elbow",),
    "r_elbow": ("r_elbow",),
    "l_wrist": ("l_wrist",),
    "r_wrist": ("r_wrist",),
    "l_hip": ("l_hip",),
    "r_hip": ("r_hip",),
    "l_knee": ("l_knee",),
    "r_knee": ("r_knee",),
}
PARTNER_NAMES = tuple(PARTNER)
CONTACT_REACH = 0.35  # m: beyond this a source effector is not touching (wrist joints: the palm is ~8 cm further)
CONTACT_SCALE = 0.20  # m: weight exp(-(d / scale)^2)
WINDOW_ON = 0.5  # contact weight that counts as touching for the auto window
WINDOW_PAD = 0.75  # s around the touching frames


@dataclass
class Placement:
    """Where a person's agent stands: pelvis x, y (m) and heading (rad), in the source world (ROS axes)."""

    x: float
    y: float
    yaw: float

    def to_local(self, other: Placement) -> tuple[float, float, float]:
        """``other`` seen from this placement: x forward, y left, relative heading."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        dx, dy = other.x - self.x, other.y - self.y
        return c * dx + s * dy, -s * dx + c * dy, float(np.angle(np.exp(1j * (other.yaw - self.yaw))))


@dataclass
class PairResult:
    clips: tuple[Clip, Clip]
    placements: tuple[Placement, Placement]
    window: tuple[int, int]  # source frames kept
    contact: dict = field(default_factory=dict)  # fidelity of the contacts, see contact_error


# ---- loading -------------------------------------------------------------------------------------------------------


def load_pair(path: str | Path, up: str = "auto", fps: float | None = None, model: str | Path | None = None) -> tuple[Motion, Motion]:
    """A two-person source: an ``.npz`` with ``p0_`` / ``p1_`` key prefixes (InterMask export), or a directory holding
    ``P1.npz`` and ``P2.npz`` (the Inter-X dataset's layout, SMPL-X parameters, Y up)."""
    p = Path(path)
    if p.is_dir():
        files = [p / "P1.npz", p / "P2.npz"]
        if not all(f.is_file() for f in files):
            raise ValueError(f"{p}: a pair directory needs P1.npz and P2.npz")
        up = "y" if up == "auto" else up
        return load(files[0], up=up, fps=fps, model=model), load(files[1], up=up, fps=fps, model=model)
    return load(p, up=up, fps=fps, model=model, person=0), load(p, up=up, fps=fps, model=model, person=1)


def is_pair_file(path: str | Path) -> bool:
    p = Path(path)
    if p.is_dir():
        return (p / "P1.npz").is_file() and (p / "P2.npz").is_file()
    with np.load(p) as z:
        return any(k.startswith("p0_") for k in z.files) and any(k.startswith("p1_") for k in z.files)


def _slice(m: Motion, f0: int, f1: int) -> Motion:
    return replace(m, global_rots=m.global_rots[f0:f1], joints=m.joints[f0:f1])


# ---- source contacts -----------------------------------------------------------------------------------------------


def _heading_rot(yaw: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.zeros(yaw.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = c, -s, s, c, 1.0
    return R


def source_contacts(m: Motion, partner: Motion, partner_yaw: np.ndarray) -> dict[str, np.ndarray]:
    """Per frame and effector (EFFECTORS order): the nearest partner point (index into PARTNER_NAMES), the contact
    weight and the offset effector - point in the partner's heading frame."""
    E = m.joints[:, [J[e] for e in EFFECTORS]]  # (T, 4, 3)
    Pp = partner.joints[:, [J[n] for n in PARTNER_NAMES]]  # (T, K, 3)
    d = np.linalg.norm(E[:, :, None] - Pp[:, None], axis=-1)  # (T, 4, K)
    j = d.argmin(-1)
    dmin = np.take_along_axis(d, j[..., None], -1)[..., 0]
    w = np.exp(-((dmin / CONTACT_SCALE) ** 2)) * (dmin < CONTACT_REACH)
    off = E - Pp[np.arange(len(j))[:, None], j]
    off = np.einsum("tji,tkj->tki", _heading_rot(partner_yaw), off)  # into the partner's heading frame
    return {"point": j, "w": w, "offset": off}


def contact_strength(m0: Motion, m1: Motion) -> np.ndarray:
    """(T,) strongest contact weight between the two, either way round."""
    yaw0, yaw1 = (_yaw(m) for m in (m0, m1))
    return np.maximum(source_contacts(m0, m1, yaw1)["w"].max(1), source_contacts(m1, m0, yaw0)["w"].max(1))


def _yaw(m: Motion) -> np.ndarray:
    from .retarget import heading

    return np.unwrap(heading(m.global_rots[:, J["pelvis"]]))


def window(m0: Motion, m1: Motion, mode: str | tuple[int, int] = "auto") -> tuple[int, int]:
    """Source frames to keep: "all", "auto" (touching frames, padded; all if they never touch) or explicit."""
    T = min(len(m0), len(m1))
    if isinstance(mode, tuple):
        f0, f1 = max(0, mode[0]), min(T, mode[1])
        if f0 >= f1:
            raise ValueError(f"window {mode} holds no frames of {T}")
        return f0, f1
    if mode == "all":
        return 0, T
    if mode != "auto":
        raise ValueError(f"window must be 'auto', 'all' or (f0, f1), got {mode!r}")
    on = np.flatnonzero(contact_strength(_slice(m0, 0, T), _slice(m1, 0, T)) > WINDOW_ON)
    if not len(on):
        return 0, T
    pad = int(round(WINDOW_PAD * m0.fps))
    return max(0, int(on[0]) - pad), min(T, int(on[-1]) + pad + 1)


def placement(prep: Prepared, weight: np.ndarray | None = None) -> Placement:
    """Mean pelvis position and heading, weighted (e.g. by contact strength: stand as the source does while touching;
    a small floor keeps frames without contact in when there is none)."""
    r = prep.root_xy_yaw
    w = np.ones(len(r)) if weight is None else np.asarray(weight, dtype=float) + 0.02
    w = w / w.sum()
    yaw = float(np.angle((w * np.exp(1j * r[:, 2])).sum()))
    return Placement(float(w @ r[:, 0]), float(w @ r[:, 1]), yaw)


# ---- Arena side ----------------------------------------------------------------------------------------------------


class _ArenaWorld:
    """Contract FK of a solved clip, placed at its agent's pose and grounded (lowest point on the floor)."""

    def __init__(self, height: float = 1.75):
        from .fk_torch import ContractFK

        self.fk = ContractFK(height, device="cpu", dtype=torch.float64)

    def __call__(self, angles: np.ndarray, place: Placement) -> tuple[dict[str, np.ndarray], np.ndarray]:
        with torch.no_grad():
            pos, _ = self.fk(torch.as_tensor(angles, dtype=torch.float64))
        pos = {k: v.numpy() for k, v in pos.items()}
        floor = np.min(np.stack([v[:, 2] for v in pos.values()], 1), 1)  # (T,) lowest point, base frame
        R = _heading_rot(np.array(place.yaw))
        t = np.array([place.x, place.y, 0.0])
        world = {}
        for k, v in pos.items():
            w = v @ R.T + t
            w[:, 2] -= floor
            world[k] = w
        return world, floor

    @staticmethod
    def point(world: dict[str, np.ndarray], name: str) -> np.ndarray:
        return np.mean([world[k] for k in PARTNER[name]], 0)


def _anchors(c: dict, world: dict, partner_world: dict, place: Placement, partner_place: Placement, floor: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Anchor targets (T, 4, 3) in this person's base frame and their weights (T, 4)."""
    T = len(floor)
    tgt = np.zeros((T, 4, 3))
    Rp = _heading_rot(np.array(partner_place.yaw))
    for e, name in enumerate(EFFECTORS):
        for k, pname in enumerate(PARTNER_NAMES):
            sel = c["point"][:, e] == k
            if not sel.any():
                continue
            Y = _ArenaWorld.point(partner_world, pname)[sel]
            off = c["offset"][sel, e] @ Rp.T * scale
            if pname in EFFECTORS:  # hand to hand / elbow to elbow: both move, meeting halfway
                tgt[sel, e] = (world[name][sel] + Y) / 2 + off / 2
            else:
                tgt[sel, e] = Y + off
    R = _heading_rot(np.array(place.yaw))
    base = (tgt - np.array([place.x, place.y, 0.0])) @ R  # world -> heading frame (R^T applied on the right)
    base[..., 2] += floor[:, None]
    return base, c["w"].copy()


def contact_error(cs: list[dict], worlds: list[dict], places: list[Placement], scale: float, on: float = 0.5) -> dict[str, float]:
    """Contact vectors on the Arena bodies vs the source's (scaled), on frames in contact: mean / p95 error (cm),
    plus the share of source contact frames."""
    errs = []
    for i in (0, 1):
        c, w, wp = cs[i], worlds[i], worlds[1 - i]
        Rp = _heading_rot(np.array(places[1 - i].yaw))
        for e, name in enumerate(EFFECTORS):
            for k, pname in enumerate(PARTNER_NAMES):
                sel = (c["point"][:, e] == k) & (c["w"][:, e] > on)
                if sel.any():
                    v = w[name][sel] - _ArenaWorld.point(wp, pname)[sel]
                    errs.append(np.linalg.norm(v - c["offset"][sel, e] @ Rp.T * scale, axis=-1))
    if not errs:
        return {"contact_frames": 0.0}
    e = np.concatenate(errs) * 100
    n = len(cs[0]["w"])
    share = float(max((cs[i]["w"] > on).any(1).mean() for i in (0, 1)))
    return {"contact_mean": float(e.mean()), "contact_p95": float(np.percentile(e, 95)), "contact_frames": share, "n": float(n)}


# ---- API -----------------------------------------------------------------------------------------------------------


def retarget_pairs(pairs: list[tuple[Motion, Motion]], window_mode: str | tuple[int, int] = "auto", passes: int = 2, fps: float = 20.0, method: str = "ik", device: str = "auto", config: IKConfig | None = None, height: float = 1.75, verbose: bool = False, chunk_frames: int = 400_000) -> list[PairResult]:
    """Retarget two-person motions; every person of every pair goes through one batched IK per pass."""
    from . import ik

    cfg = config or ik.IKConfig.preset("ik" if method == "none" else method)
    wins = [window(a, b, window_mode) for a, b in pairs]
    ms = [_slice(m, *w) for (a, b), w in zip(pairs, wins, strict=True) for m in (a, b)]
    preps = [prepare(m, True, height) for m in ms]
    yaws = [_yaw(m) for m in ms]
    cs = [source_contacts(ms[i], ms[i ^ 1], yaws[i ^ 1]) for i in range(len(ms))]
    strength = [np.maximum(cs[i]["w"].max(1), cs[i ^ 1]["w"].max(1)) for i in range(len(ms))]
    places = [placement(p, w) for p, w in zip(preps, strength, strict=True)]
    scale = float(np.mean([p.scale["upper"] for p in preps]))  # source -> Arena upper-body size, for the offsets
    arena = _ArenaWorld(height)
    sols = [p.init for p in preps] if method == "none" else ik.solve(preps, cfg, device=device, chunk_frames=chunk_frames, verbose=verbose)
    before = None
    for _ in range(passes if method != "none" else 0):
        worlds = [arena(s, pl) for s, pl in zip(sols, places, strict=True)]
        if before is None:
            before = [contact_error(cs[i : i + 2], [w for w, _ in worlds[i : i + 2]], places[i : i + 2], scale) for i in range(0, len(ms), 2)]
        for i in range(len(ms)):
            (w, floor), (wp, _) = worlds[i], worlds[i ^ 1]
            tgt, wt = _anchors(cs[i], w, wp, places[i], places[i ^ 1], floor, scale)
            preps[i].targets["anchor"], preps[i].targets["anchor_w"] = tgt, wt
        sols = ik.solve(preps, cfg, device=device, chunk_frames=chunk_frames, verbose=verbose)
    worlds = [arena(s, pl)[0] for s, pl in zip(sols, places, strict=True)]
    out = []
    for n, i in enumerate(range(0, len(ms), 2)):
        ce = contact_error(cs[i : i + 2], worlds[i : i + 2], places[i : i + 2], scale)
        if before is not None and "contact_mean" in before[n]:
            ce["contact_mean_noanchor"] = before[n]["contact_mean"]
        out.append(PairResult(clips=(resample(preps[i], sols[i], fps), resample(preps[i + 1], sols[i + 1], fps)), placements=(places[i], places[i + 1]), window=wins[n], contact=ce))
    return out


def retarget_pair(m0: Motion, m1: Motion, window_mode: str | tuple[int, int] = "auto", passes: int = 2, fps: float = 20.0, method: str = "ik", device: str = "auto", config: IKConfig | None = None, height: float = 1.75, verbose: bool = False, chunk_frames: int = 400_000) -> PairResult:
    return retarget_pairs([(m0, m1)], window_mode, passes, fps, method, device, config, height, verbose, chunk_frames)[0]


def pair_annotation(res: PairResult, names: tuple[str, str], role: int, fps_src: float) -> dict:
    """The ``pair`` block of each clip's annotation: partner clip, where the partner stands seen from this person
    (x forward, y left, m; relative heading, rad) and the source window."""
    me, other = res.placements[role], res.placements[1 - role]
    dx, dy, dyaw = me.to_local(other)
    return {"pair": {"role": "ab"[role], "partner": names[1 - role], "partner_offset": [round(float(dx), 4), round(float(dy), 4), round(float(dyaw), 4)], "separation": round(float(np.hypot(dx, dy)), 4), "source_window_s": [round(res.window[0] / fps_src, 3), round(res.window[1] / fps_src, 3)]}}


__all__ = ["EFFECTORS", "PairResult", "Placement", "contact_error", "is_pair_file", "load_pair", "pair_annotation", "retarget_pair", "retarget_pairs", "source_contacts", "window"]
