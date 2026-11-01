"""Offline layer-quality measurements on the shipped clips and the aim solver.

Prompt-free, every number read from joint states through the rig's own FK (``pointing.skeleton.fk``):

- plausibility: foot skating, floor penetration, lowest-foot height and jerk, per clip, on the rig as
  Arena plays it (root motion dropped, pelvis at hip height) and optionally on the generator's joint
  positions before retargeting (``--source clip=path.npy``, HumanML3D (T, 22, 3), y up)
- loop seams: the pose jump where a looping clip wraps, against the clip's own frame-to-frame motion
  and the playback gate's 0.25 m per 50 ms link step
- targeting error: point and halt invocations on a grid of targets inside the arm aiming envelope,
  through the real GestureLayer, median per invocation over the parked hold

    python3 -m arena_runtime.humans.layer_quality --out /tmp/layer_quality [--source hug=/path/hug.npy ...]
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from arena_runtime.humans.animation_manager import AnimationManager
from arena_runtime.humans.gestures import BODY_HEIGHT, Channel, GestureLayer, GestureRequest, world_to_local
from arena_runtime.humans.gestures.qa import MAX_HOLD_AIM_DEG, MAX_LINK_STEP_M
from arena_runtime.humans.pointing import skeleton as S
from arena_runtime.humans.pointing.contract import ROS_JOINT_ORDER

FPS = 20.0
DT = 1.0 / FPS
IDLE, WALKING = 0, 1
WALK_SPEED = 1.2  # m/s

CONTACT_HEIGHT_M = 0.05
SKID_M_PER_FRAME = 0.025
LINKS = ("l_wrist", "r_wrist", "l_elbow", "r_elbow", "head")
FEET = ("l_ankle", "r_ankle", "l_foot", "r_foot")

H3D_FEET = (7, 8, 10, 11)  # HumanML3D l/r ankle, l/r foot
H3D_LINKS = (20, 21, 18, 19, 15)  # HumanML3D l/r wrist, l/r elbow, head


# -- positions ---------------------------------------------------------------------------------


def rig_positions(frames: Sequence[dict], body: S.Body, joints: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """(T, 3) track per rig link of a clip as Arena plays it: root fixed, joints outside the clip's mask at rest."""
    out: dict[str, list] = {}
    for f in frames:
        angles = dict.fromkeys(ROS_JOINT_ORDER, 0.0)
        for name, v in f["angles"].items():
            if joints is None or name in joints:
                angles[name] = float(v)
        pos, _ = S.fk(angles, body)
        for k, v in pos.items():
            out.setdefault(k, []).append(v)
    return {k: np.asarray(v) for k, v in out.items()}


def h3d_z_up(positions: np.ndarray) -> np.ndarray:
    """HumanML3D (x left, y up, z forward) -> (x forward, y left, z up)."""
    p = np.asarray(positions, dtype=float)
    return np.stack([p[..., 2], p[..., 0], p[..., 1]], axis=-1)


# -- plausibility ------------------------------------------------------------------------------


def plausibility(feet: np.ndarray, links: np.ndarray, fps: float = FPS) -> dict[str, float]:
    """Skating, floor penetration, lowest-foot height and link jerk of feet (T, F, 3) and links (T, L, 3), z up, floor at z = 0."""
    feet = np.asarray(feet, dtype=float)
    low = feet[..., 2].min(axis=1)
    planted = feet[:-1, :, 2] < CONTACT_HEIGHT_M
    skid = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=-1)
    with_contact = planted.any(axis=1)
    skating = (planted & (skid > SKID_M_PER_FRAME)).any(axis=1)
    jerk = np.diff(np.asarray(links, dtype=float), n=3, axis=0) * fps**3
    return {
        "skate_ratio": float(skating.sum() / with_contact.sum()) if with_contact.any() else float("nan"),
        "penetration_cm": 100.0 * float(np.clip(-low, 0.0, None).mean()),
        "penetration_max_cm": 100.0 * float(max(0.0, -low.min())),
        "lowest_foot_cm": 100.0 * float(low.mean()),
        "jerk_m_s3": float(np.linalg.norm(jerk, axis=-1).mean()) if len(jerk) else float("nan"),
    }


def rig_plausibility(frames: Sequence[dict], body: S.Body, joints: Sequence[str] | None) -> dict[str, float]:
    pos = rig_positions(frames, body, joints)
    return plausibility(np.stack([pos[f] for f in FEET], axis=1), np.stack([pos[k] for k in LINKS], axis=1))


def source_plausibility(positions: np.ndarray) -> dict[str, float]:
    p = h3d_z_up(positions)
    return plausibility(p[:, H3D_FEET], p[:, H3D_LINKS])


# -- loop seams --------------------------------------------------------------------------------


def loop_seam(frames: Sequence[dict], body: S.Body, joints: Sequence[str] | None, loop_from: int = 0, reverse: bool = False) -> dict[str, float | str | bool]:
    """Link jump from the last played frame to the loop-in frame, next to the clip's own per-frame link motion."""
    pos = rig_positions(frames, body, joints)
    tracks = np.stack([pos[k] for k in LINKS], axis=1)  # (T, L, 3)
    steps = np.linalg.norm(np.diff(tracks, axis=0), axis=-1).max(axis=1)  # worst link per frame
    last = 1 if reverse and len(tracks) > 1 else len(tracks) - 1
    seam = np.linalg.norm(tracks[last] - tracks[loop_from], axis=-1)
    worst = int(seam.argmax())
    names = [n for n in ROS_JOINT_ORDER if joints is None or n in joints]
    last = np.array([frames[last]["angles"].get(n, 0.0) for n in names])
    first = np.array([frames[loop_from]["angles"].get(n, 0.0) for n in names])
    ang = np.abs((last - first + np.pi) % (2 * np.pi) - np.pi)
    return {
        "seam_m": float(seam[worst]),
        "seam_link": LINKS[worst],
        "step_median_m": float(np.median(steps)),
        "step_p95_m": float(np.percentile(steps, 95)),
        "seam_over_p95": float(seam[worst] / np.percentile(steps, 95)) if np.percentile(steps, 95) > 0 else float("inf"),
        "seam_joint_rad": float(ang.max()) if len(ang) else 0.0,
        "passes_gate": bool(seam[worst] <= MAX_LINK_STEP_M),
    }


# -- targeting ---------------------------------------------------------------------------------


def targeting_grid(azimuths_deg: Sequence[float] = tuple(range(-90, 91, 15)), distances_m: Sequence[float] = (1.5, 3.0, 5.0), heights_m: Sequence[float] = (0.4, 1.2, 1.7)) -> list[tuple[float, float, float]]:
    """Target points in the ped-local frame (x forward, y left, z up): the robot body, a chest, a head."""
    return [(d * math.cos(math.radians(a)), d * math.sin(math.radians(a)), h) for a in azimuths_deg for d in distances_m for h in heights_m]


def aim_errors(slot: str, target_local: Sequence[float], *, moving: bool, hold_s: float = 2.5, origin: str = "wrist") -> dict[str, float] | None:
    """One invocation through the real GestureLayer: forearm-to-target angle per tick over the parked hold (horizontal only for halt), measured from the wrist or the pelvis (``origin``)."""
    log = _Log()
    mgr = AnimationManager(logger=log, fps=FPS)
    layer = GestureLayer(mgr, log)
    mgr.gesture_hook = layer
    body = S.Body(BODY_HEIGHT)
    pose = (0.0, 0.0, 0.0)
    target = np.asarray(target_local, dtype=float)
    channel = Channel(slot, tuple(target), hand="")
    state = WALKING if moving else IDLE
    errs: list[float] = []
    for _ in range(int(round(hold_s / DT))):
        req = GestureRequest(channels=(channel,), pose=pose, moving=moving)
        angles = mgr.compute(1, state, WALK_SPEED if moving else 0.0, DT, gesture=req)
        ag = layer._agents.get(1)
        st = None if ag is None else ag.slots.get("arm")
        parked = st is not None and st.phase == "hold" and st.clip is not None and st.t >= st.clip.hold_end / st.clip.fps and not st.release_pending
        if not parked or st is None or st.clip is None:
            continue
        pos, _ = S.fk(angles, body)
        side = st.clip.side
        fore = pos[f"{side}_wrist"] - pos[f"{side}_elbow"]
        want = world_to_local(tuple(target), pose) - pos["pelvis" if origin == "body" else f"{side}_wrist"]
        if slot.startswith("halt"):
            fore, want = fore * [1, 1, 0], want * [1, 1, 0]
        errs.append(math.degrees(math.acos(float(np.clip(S.unit(fore) @ S.unit(want), -1.0, 1.0)))))
    if not errs:
        return None
    return {"median_deg": float(np.median(errs)), "max_deg": float(np.max(errs)), "ticks": len(errs)}


def targeting(slot: str, moving: bool, grid: Sequence[tuple[float, float, float]] | None = None, origin: str = "wrist") -> dict:
    rows = []
    for tgt in grid or targeting_grid():
        res = aim_errors(slot, tgt, moving=moving, origin=origin)
        az = math.degrees(math.atan2(tgt[1], tgt[0]))
        rows.append({"azimuth_deg": round(az, 1), "distance_m": round(math.hypot(tgt[0], tgt[1]), 2), "height_m": tgt[2], **(res or {"median_deg": None})})
    med = np.array([r["median_deg"] for r in rows if r["median_deg"] is not None])
    return {
        "slot": slot,
        "moving": moving,
        "origin": origin,
        "invocations": len(rows),
        "parked": int(len(med)),
        "median_deg": float(np.median(med)) if len(med) else None,
        "p90_deg": float(np.percentile(med, 90)) if len(med) else None,
        "max_deg": float(med.max()) if len(med) else None,
        "over_gate": int((med > MAX_HOLD_AIM_DEG).sum()),
        "rows": rows,
    }


class _Log:
    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass


# -- driver ------------------------------------------------------------------------------------


def clip_report(sources: dict[str, Path]) -> list[dict]:
    mgr = AnimationManager(logger=_Log(), fps=FPS)
    mgr.cache_animations()
    body = S.Body(BODY_HEIGHT)
    rows = []
    for name, anim in sorted(mgr.animations.items()):
        if not anim.frames:  # synthesized
            continue
        joints = anim.joints or None
        frames = anim.frames
        row: dict = {"clip": name, "frames": len(frames), "loop": anim.loop, "mask": "upper-body" if joints else "whole-body"}
        after = rig_plausibility(frames, body, joints)
        row.update({f"rig_{k}": v for k, v in after.items()})
        if joints:
            for k in ("skate_ratio", "penetration_cm", "penetration_max_cm", "lowest_foot_cm"):
                row[f"rig_{k}"] = None
        if name in sources:
            before = source_plausibility(np.load(sources[name]))
            row.update({f"src_{k}": v for k, v in before.items()})
        if anim.loop:
            row["reverse"] = anim.reverse
            row.update({f"loop_{k}": v for k, v in loop_seam(frames, body, joints, anim.loop_from, anim.reverse).items()})
        rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/tmp/layer_quality"))
    ap.add_argument("--source", action="append", default=[], help="clip=path to the generator's (T, 22, 3) joint positions")
    ap.add_argument("--skip-targeting", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {k: Path(v) for k, v in (s.split("=", 1) for s in args.source)}
    report: dict = {"clips": clip_report(sources)}
    if not args.skip_targeting:
        report["targeting"] = [targeting(slot, moving) for slot in ("arm", "halt") for moving in (False, True)]
        report["targeting"] += [targeting("halt", moving, origin="body") for moving in (False, True)]
    (args.out / "layer_quality.json").write_text(json.dumps(report, indent=1, default=str))
    for row in report["clips"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    for t in report.get("targeting", []):
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in t.items() if k != "rows"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
