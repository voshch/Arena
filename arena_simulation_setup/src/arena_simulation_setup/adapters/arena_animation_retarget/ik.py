"""Whole-body IK for the 40-DOF wire, batched over every frame of every clip, on the GPU when there is one.

One least-squares objective per frame, solved jointly with L-BFGS from the rotation retarget:

  reach      hand - shoulder and ankle - hip, the source vectors scaled to Arena's limb lengths (normalized by them).
             Puts the hands and feet where the source's are, relative to the limb roots, despite Arena's
             proportions (forearm 0.6x the upper arm; SMPL-X 1:1).
  relations  left - right hand and hands - neck base, absolute, weighted in only while the source hands are within
             ~25 cm of each other / the head (claps close, hands reach the face; far apart the two skeletons'
             different spans would fight the reach); left - right ankle relative to the hips (stance; Arena's hips
             are 2.3x wider than SMPL-X's)
  dirs/twist rest-corrected bone directions and upper-bone forward axes (the rotation retarget's own targets): keep
             the elbow/knee swivel and limb twist, i.e. the style of the source, where reach leaves them free
  hand_rot   full hand orientation (chordal); foot_fwd: toe direction
  prior      stay near the rotation retarget, per DOF group (the spine, head and collars are already exact)
  limits     hard limits from JOINTS.md (a stiff penalty, then a clamp) where anatomy fixes them (elbows/knees bend one way, wrists, ankles,
             collars, head); the advisory gait limits are not applied (Arena's own clips exceed them)
  anchor     absolute targets for the wrists and elbows (base frame), per-frame weights; zero unless a caller sets
             them, e.g. :mod:`.pair`, which puts each person's hands where they meet the partner's body
  smooth     second differences in time, within each clip

All terms are means over frames, so the weights do not depend on how many clips are batched.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from .contract import WIRE
from .fk_torch import IDX, ContractFK

if TYPE_CHECKING:
    from .retarget import Prepared

# Hard limits (rad) from JOINTS.md's table, only where anatomy fixes them.
HARD_LIMITS: dict[str, tuple[float, float]] = {}
for _s in "lr":
    HARD_LIMITS.update({f"{_s}_elbow": (0.0, 2.6), f"{_s}_knee": (-2.6, 0.0), f"{_s}_r_wrist": (-1.4, 1.4), f"{_s}_wrist": (-1.4, 1.2), f"{_s}_ankle": (-0.9, 0.6), f"{_s}_y_ankle": (-0.6, 0.6), f"{_s}_y_collar": (-0.5, 0.5), f"{_s}_p_collar": (-0.2, 0.6)})
HARD_LIMITS.update({"r_head": (-1.0, 1.0), "y_head": (-1.4, 1.4), "p_head": (-1.5, 1.5)})

ANCHOR_JOINTS = ("l_wrist", "r_wrist", "l_elbow", "r_elbow")  # order of the anchor targets

GROUPS = {
    "core": [k for k in WIRE if k.endswith(("waist", "spine", "chest", "head", "collar"))],
    "limb": [k for k in WIRE if k.endswith(("shoulder", "elbow", "hip", "knee"))],
    "distal": [k for k in WIRE if k.endswith(("wrist", "ankle"))],
}


@dataclass
class IKConfig:
    reach: float = 10.0
    hands: float = 3.0  # peak, at contact (weights fade out beyond ~25 cm)
    hand_neck: float = 3.0
    feet: float = 0.3
    dirs: float = 0.1
    twist: float = 0.01
    hand_rot: float = 0.05
    foot_fwd: float = 0.1
    anchor: float = 10.0  # times the per-frame anchor weights (all zero unless set, e.g. by pair contacts)
    prior: dict = field(default_factory=lambda: {"core": 1.0, "limb": 0.003, "distal": 0.01})
    limits: float = 100.0
    smooth: float = 3.0  # on second differences (rad/frame^2) at the source rate
    iters: int = 300

    @classmethod
    def preset(cls, method: str) -> IKConfig:
        if method == "ik":
            return cls()
        if method == "rotation":  # the rotation retarget's limb fit: bone directions only, no reach or relations
            return cls(reach=0.0, hands=0.0, hand_neck=0.0, feet=0.0, dirs=1.0, twist=0.01, hand_rot=0.05, foot_fwd=0.1, prior={"core": 1e3, "limb": 1e-4, "distal": 0.01}, limits=0.0, smooth=0.0)
        raise ValueError(f"unknown method {method!r} (ik, rotation)")

    def with_(self, **kw: object) -> IKConfig:
        return replace(self, **kw)


def pick_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _chunks(lengths: list[int], budget: int) -> list[list[int]]:
    out, cur, n = [], [], 0
    for i, L in enumerate(lengths):
        if cur and n + L > budget:
            out.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += L
    return out + ([cur] if cur else [])


def solve(preps: list[Prepared], config: IKConfig | None = None, device: str = "auto", chunk_frames: int = 400_000, height: float = 1.75, verbose: bool = False) -> list[np.ndarray]:
    """IK for every prepared motion; returns the (T, 40) wire angles per motion, in WIRE order."""
    cfg = config or IKConfig()
    dev = pick_device(device)
    fk = ContractFK(height, device=dev, dtype=torch.float32)
    out: list[np.ndarray | None] = [None] * len(preps)
    for chunk in _chunks([len(p) for p in preps], chunk_frames):
        sol = _solve_chunk([preps[i] for i in chunk], cfg, fk, dev, verbose)
        for i, s in zip(chunk, sol, strict=True):
            out[i] = s
    done = [s for s in out if s is not None]
    assert len(done) == len(preps)  # the chunks cover every motion
    return done


def _solve_chunk(preps: list[Prepared], cfg: IKConfig, fk: ContractFK, dev: torch.device, verbose: bool) -> list[np.ndarray]:
    f32 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    cat = lambda k: f32(np.concatenate([p.targets[k] for p in preps]))  # noqa: E731
    theta0 = f32(np.concatenate([p.init for p in preps]))
    tg = {k: cat(k) for k in preps[0].targets}
    n = theta0.shape[0]
    # second differences only inside a clip
    seg = np.concatenate([np.full(len(p), i) for i, p in enumerate(preps)])
    inner = torch.as_tensor((seg[:-2] == seg[1:-1]) & (seg[1:-1] == seg[2:]), device=dev) if n > 2 else None

    w_prior = torch.zeros(len(WIRE), device=dev)
    for g, names in GROUPS.items():
        w_prior[[IDX[k] for k in names]] = cfg.prior[g]
    lim_i = [IDX[k] for k in HARD_LIMITS]
    lim_lo = torch.tensor([HARD_LIMITS[k][0] for k in HARD_LIMITS], device=dev)
    lim_hi = torch.tensor([HARD_LIMITS[k][1] for k in HARD_LIMITS], device=dev)
    L = fk.lengths
    DOWN = fk.DOWN
    has_anchor = "anchor_w" in tg and bool((tg["anchor_w"] > 0).any())

    def loss_fn(theta: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pos, fr = fk(theta)
        terms: dict[str, torch.Tensor] = {}
        sq = lambda x: (x * x).reshape(x.shape[0], -1).sum(1).mean()  # noqa: E731  (sum per frame, mean over frames)
        if cfg.reach:
            vec = torch.stack([pos[f"{s}_wrist"] - pos[f"{s}_shoulder"] for s in "lr"] + [pos[f"{s}_ankle"] - pos[f"{s}_hip"] for s in "lr"], 1)
            norm = torch.tensor([L["arm"], L["arm"], L["leg"], L["leg"]], device=dev)[None, :, None]
            terms["reach"] = cfg.reach * sq((vec - tg["reach"]) / norm)
        if cfg.hands:
            e = (pos["l_wrist"] - pos["r_wrist"] - tg["hands"]) / L["arm"]
            terms["hands"] = cfg.hands * ((e * e).sum(-1) * tg["hands_w"]).mean()
        if cfg.hand_neck:
            hn = torch.stack([pos[f"{s}_wrist"] - pos["torso"] for s in "lr"], 1)
            e = (hn - tg["hand_neck"]) / L["arm"]
            terms["hand_neck"] = cfg.hand_neck * ((e * e).sum(-1) * tg["hand_neck_w"]).sum(-1).mean()
        if cfg.feet:
            terms["feet"] = cfg.feet * sq((pos["l_ankle"] - pos["r_ankle"] - tg["feet"]) / L["leg"])
        if cfg.dirs or cfg.twist:
            frames = torch.stack([fr[f"{s}_{b}"] for s in "lr" for b in ("upperarm", "forearm", "thigh", "shin")], 1)
            if cfg.dirs:
                terms["dirs"] = cfg.dirs * sq(frames @ DOWN - tg["dirs"])
            if cfg.twist:
                up = [0, 2, 4, 6]  # upper bones: upper arms and thighs
                terms["twist"] = cfg.twist * sq(frames[:, up, :, 0] - tg["fwd"][:, up])
        if cfg.hand_rot:
            hands = torch.stack([fr["l_hand"], fr["r_hand"]], 1)
            terms["hand_rot"] = cfg.hand_rot * sq(hands - tg["hand_rot"])
        if cfg.foot_fwd:
            feet = torch.stack([fr["l_foot"][..., 0], fr["r_foot"][..., 0]], 1)
            terms["foot_fwd"] = cfg.foot_fwd * sq(feet - tg["foot_fwd"])
        if cfg.anchor and has_anchor:
            e = (torch.stack([pos[k] for k in ANCHOR_JOINTS], 1) - tg["anchor"]) / L["arm"]
            terms["anchor"] = cfg.anchor * ((e * e).sum(-1) * tg["anchor_w"]).sum(-1).mean()
        d = theta - theta0
        terms["prior"] = (w_prior * d * d).sum(-1).mean()
        if cfg.limits:
            x = theta[:, lim_i]
            terms["limits"] = cfg.limits * ((torch.relu(lim_lo - x) ** 2 + torch.relu(x - lim_hi) ** 2).sum(-1)).mean()
        if cfg.smooth and inner is not None:
            acc = theta[2:] - 2 * theta[1:-1] + theta[:-2]
            terms["smooth"] = cfg.smooth * ((acc * acc).sum(-1) * inner).sum() / max(int(inner.sum()), 1)
        return sum(terms.values(), theta.new_zeros(())), terms

    theta = theta0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=cfg.iters, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss, _ = loss_fn(theta)
        loss.backward()
        return loss

    start = None
    if verbose:
        with torch.no_grad():
            start = loss_fn(theta0)
    opt.step(closure)
    if start is not None:
        l0, t0 = start
        with torch.no_grad():
            l1, t1 = loss_fn(theta)
        fmt = lambda t: " ".join(f"{k}={float(v):.2e}" for k, v in t.items())  # noqa: E731
        print(f"  ik: {len(preps)} clips, {n} frames on {dev}: loss {float(l0):.3e} -> {float(l1):.3e}\n    start {fmt(t0)}\n    end   {fmt(t1)}")
    if cfg.limits:
        with torch.no_grad():  # the penalty leaves a small overshoot where other terms push: clamp it
            theta[:, lim_i] = torch.maximum(torch.minimum(theta[:, lim_i], lim_hi), lim_lo)
    sol = theta.detach().cpu().numpy().astype(float)
    if not np.all(np.isfinite(sol)):
        raise FloatingPointError("IK diverged (non-finite angles)")
    out, i = [], 0
    for p in preps:
        out.append(sol[i : i + len(p)])
        i += len(p)
    return out


__all__ = ["ANCHOR_JOINTS", "HARD_LIMITS", "IKConfig", "pick_device", "solve"]
