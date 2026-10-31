"""How faithfully a clip reproduces its SMPL-X source on the contract skeleton (what RViz draws), vectorized.

Each clip frame is compared with the nearest source frame, in the body base frame. Reported as (mean, p95):

* ``*_line`` (deg): whole-limb direction, shoulder -> wrist / hip -> ankle, against the raw source joints.
* ``*_reach`` (% of limb length): hand rel. shoulder / ankle rel. hip against the source vector scaled to Arena's
  limb length, i.e. where the hand or foot ends up. Includes direction and extension.
* ``hands_gap`` / ``feet_gap`` (% of limb length): left - right hand / ankle, scaled likewise (feet relative to the
  hips). ``hands_contact``: hands_gap over the frames where the source hands are within 20 cm (0 if none).
* ``*_upper``/``*_lower`` (deg): each bone against SMPL-X's rest-corrected bone (the rotation retarget's target).
  The IK trades a few degrees here for reach, because the two skeletons' proportions differ.
* ``trunk``, ``head`` (deg; different landmarks on the two skeletons, a few deg are rest shape), ``*_hand_rot`` (deg,
  full rotation incl. twist), ``step`` (deg, largest frame-to-frame change of any DOF: flips and jitter).
"""

from __future__ import annotations

import numpy as np
import torch

from .contract import WIRE
from .fk_torch import ContractFK
from .retarget import LIMBS, Clip, Prepared, prepare
from .smplx import J, Motion

LINES = [f"{k}_line" for k in LIMBS]
REACH = [f"{k}_reach" for k in LIMBS]
BENDS = [f"{k}_{b}" for k in LIMBS for b in ("upper", "lower")]
_POINTS = {"arm": ("shoulder", "elbow", "wrist"), "leg": ("hip", "knee", "ankle")}
_FRAMES = {"arm": ("upperarm", "forearm"), "leg": ("thigh", "shin")}
_BONE_ORDER = [f"{s}_{j}" for s in "lr" for j in ("shoulder", "elbow", "hip", "knee")]  # = retarget.BONE_JOINTS


def _ang(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def _rot_ang(m: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((np.trace(m, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def score(motion: Motion | None, clip: Clip, height: float = 1.75, prep: Prepared | None = None, device: str = "cpu") -> dict[str, tuple[float, float]]:
    if prep is None:
        if motion is None:
            raise ValueError("score needs the source motion or its prep")
        prep = prepare(motion, clip.fold_tilt, height)
    idx = np.clip(np.round(clip.t * prep.fps).astype(int), 0, len(prep) - 1)  # nearest source frame
    tg = {k: v[idx] for k, v in prep.targets.items()}
    q = prep.src[idx]
    theta = np.stack([clip.angles[k] for k in WIRE], 1)
    with torch.no_grad():
        fk = ContractFK(height, device=device, dtype=torch.float64)
        pos, fr = fk(torch.as_tensor(theta, dtype=torch.float64, device=device))
        pos = {k: v.cpu().numpy() for k, v in pos.items()}
        fr = {k: v.cpu().numpy() for k, v in fr.items()}
    L = fk.lengths
    err: dict[str, np.ndarray] = {}
    err["trunk"] = _ang(q[:, J["spine3"]] - q[:, J["pelvis"]], pos["torso"] - pos["pelvis"])
    err["head"] = _ang(q[:, J["head"]] - q[:, J["neck"]], fr["head"][:, :, 2])
    reach_i = {"l_arm": 0, "r_arm": 1, "l_leg": 2, "r_leg": 3}
    for k, (ja, _, jc, _) in LIMBS.items():
        s, kind = k[0], k[2:]
        ca, _, cc = (f"{s}_{n}" for n in _POINTS[kind])
        v = pos[cc] - pos[ca]
        err[f"{k}_line"] = _ang(q[:, J[jc]] - q[:, J[ja]], v)
        err[f"{k}_reach"] = 100 * np.linalg.norm(v - tg["reach"][:, reach_i[k]], axis=-1) / L[kind]
        for b, f, j in (("upper", _FRAMES[kind][0], ja), ("lower", _FRAMES[kind][1], f"{s}_{_POINTS[kind][1]}")):
            err[f"{k}_{b}"] = _ang(fr[f"{s}_{f}"][:, :, 2] * -1, tg["dirs"][:, _BONE_ORDER.index(j)])
    err["hands_gap"] = 100 * np.linalg.norm(pos["l_wrist"] - pos["r_wrist"] - tg["hands"], axis=-1) / L["arm"]
    close = np.linalg.norm(q[:, J["l_wrist"]] - q[:, J["r_wrist"]], axis=-1) < 0.2  # source hands touching
    err["hands_contact"] = err["hands_gap"][close] if close.any() else np.zeros(1)
    err["feet_gap"] = 100 * np.linalg.norm(pos["l_ankle"] - pos["r_ankle"] - tg["feet"], axis=-1) / L["leg"]
    for i, s in enumerate("lr"):
        err[f"{s}_hand_rot"] = _rot_ang(np.swapaxes(tg["hand_rot"][:, i], -1, -2) @ fr[f"{s}_hand"])
    step = np.degrees(np.abs(np.diff(theta, axis=0)).max(1)) if len(theta) > 1 else np.zeros(1)
    err["step"] = step
    out = {k: (float(np.mean(v)), float(np.percentile(v, 95))) for k, v in err.items()}
    out["step"] = (float(np.percentile(step, 95)), float(step.max()))  # (p95, max) for steps
    return out


def _worst(sc: dict[str, tuple[float, float]], keys: list[str], unit: str = "") -> str:
    k = max(keys, key=lambda k: sc[k][0])
    return f"{k} {sc[k][0]:.1f}/{sc[k][1]:.1f}{unit}"


def summary(sc: dict[str, tuple[float, float]]) -> str:
    return f"reach {_worst(sc, REACH, '%')}, line {_worst(sc, LINES)}, hands gap {sc['hands_gap'][0]:.1f}% (contact {sc['hands_contact'][0]:.1f}%), bone {_worst(sc, BENDS)}, hand rot {sc['l_hand_rot'][0]:.0f}/{sc['r_hand_rot'][0]:.0f}, step max {sc['step'][1]:.0f} deg  (mean/p95)"


def table(sc: dict[str, tuple[float, float]]) -> str:
    return "\n".join(f"  {k:12s} {m:7.1f} {p:7.1f}" for k, (m, p) in sc.items())
