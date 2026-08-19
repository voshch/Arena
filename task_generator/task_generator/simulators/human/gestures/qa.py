"""Offline gesture QA: run scripted intents through the real GestureLayer and inspect the result.

Usage (container):
    python3 -m task_generator.simulators.human.gestures.qa --out /tmp/gesture_qa [--video]
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import attrs
import numpy as np

from task_generator.simulators.human.animation_mananager import AnimationManager
from task_generator.simulators.human.gestures import BODY_HEIGHT, Channel, GestureLayer, GestureRequest, world_to_local
from task_generator.simulators.human.pointing import skeleton as S
from task_generator.simulators.human.pointing.contract import ROS_JOINT_ORDER

DT = 0.05
ANIMATIONS = Path(__file__).resolve().parents[1] / "animations"
MAX_JOINT_STEP_RAD = 0.6  # per 50 ms tick, informational (shoulder triples may wrap)
MAX_LINK_STEP_M = 0.25  # per 50 ms tick, wrist/elbow/head jump that reads as a snap
MAX_COLLAR_RAD = 0.45  # clavicle elevation anywhere in a clip, the recorded template shrugs to its 0.6 limit
MAX_HOLD_AIM_DEG = 10.0  # forearm off the target while the arm slot is parked (a tracking arm on a walker included)
REST_EPS_RAD = 0.03  # every arm DOF within this of zero = the arm hangs
LINKS = ("l_wrist", "r_wrist", "l_elbow", "r_elbow", "head")
ARM_DOFS = tuple(f"{s}_{d}" for s in ("l", "r") for d in ("y_collar", "p_collar", "y_shoulder", "p_shoulder", "r_shoulder", "elbow"))
IDLE = 0


@attrs.frozen
class Cue:
    """The channels held from ``t`` on: an empty list clears."""

    t: float
    channels: Sequence[Channel] = ()
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    moving: bool = False


@attrs.frozen
class Case:
    name: str
    cues: Sequence[Cue]
    duration: float
    min_clips: int = 0  # arm clips the layer must install (a tracking arm retargets many times)
    rest_windows: Sequence[tuple[float, float]] = ()  # the arm must hang at some tick inside each (t0, t1)


@attrs.frozen
class Result:
    name: str
    frames: list[dict]
    reports: list[dict]
    warnings: list[str]
    max_step_rad: float
    max_step_joint: str
    max_link_m: float
    max_link: str
    max_link_t: float
    max_collar_rad: float
    max_hold_aim_deg: float
    rest_times: list[float]
    arm_clips: int
    missing_rest: list[tuple[float, float]]

    @property
    def ok(self) -> bool:
        return (
            not self.warnings
            and self.max_link_m <= MAX_LINK_STEP_M
            and self.max_collar_rad <= MAX_COLLAR_RAD
            and self.max_hold_aim_deg <= MAX_HOLD_AIM_DEG
            and not self.missing_rest
            and all(r.get("ok", True) and not r.get("relaxed") and not r.get("near_target_fallback") for r in self.reports)
        )


class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        self.lines.append(msg)


def run(case: Case, *, agent_id: int = 1) -> Result:
    log = _Log()
    mgr = AnimationManager(ANIMATIONS, logger=log, fps=20.0)
    layer = GestureLayer(mgr, log)
    mgr.gesture_hook = layer
    reports: list[dict] = []
    seen: list[object] = []  # clip objects kept alive so identity is stable
    cues = sorted(case.cues, key=lambda c: c.t)
    frames: list[dict] = []
    prev: dict[str, float] | None = None
    prev_pos: dict | None = None
    max_step, max_joint = 0.0, ""
    max_link, max_link_name, max_link_t = 0.0, "", 0.0
    body = S.Body(BODY_HEIGHT)
    n = int(round(case.duration / DT))
    max_collar, max_hold_aim = 0.0, 0.0
    rest_times: list[float] = []
    arm_clips = 0
    for i in range(n):
        t = i * DT
        cue = None
        for c in cues:
            if c.t <= t + 1e-9:
                cue = c
        req = None if cue is None else GestureRequest(channels=tuple(cue.channels), pose=cue.pose, moving=cue.moving)
        angles = mgr.compute(agent_id, IDLE, 0.0, DT, gesture=req)
        ag = layer._agents.get(agent_id)
        arm_st = None
        if ag is not None:
            for st in ag.slots.values():
                if st.slot == "arm":
                    arm_st = st
                if st.clip is not None and not any(st.clip is c for c in seen):
                    seen.append(st.clip)
                    arm_clips += st.slot == "arm"
                    reports.append({"slot": st.slot, "kind": st.kind, "t": t, **{k: v for k, v in st.clip.report.items() if not isinstance(v, (list, dict, np.ndarray))}})
        if prev is not None:
            for j in ROS_JOINT_ORDER:
                d = abs(angles[j] - prev[j])
                if d > max_step:
                    max_step, max_joint = d, j
        pos, _ = S.fk(angles, body)
        if prev_pos is not None:
            for link in LINKS:
                d = float(np.linalg.norm(pos[link] - prev_pos[link]))
                if d > max_link:
                    max_link, max_link_name, max_link_t = d, link, t
        max_collar = max(max_collar, abs(angles["l_p_collar"]), abs(angles["r_p_collar"]))
        if all(abs(angles[d]) < REST_EPS_RAD for d in ARM_DOFS):
            rest_times.append(t)
        parked = arm_st is not None and arm_st.phase == "hold" and arm_st.clip is not None and arm_st.t >= arm_st.clip.hold_end / arm_st.clip.fps and not arm_st.release_pending
        if parked and cue is not None:
            side = arm_st.clip.side
            ch = next((c for c in cue.channels if c.slot in ("arm", "arm_l", "arm_r")), None)
            if ch is not None:
                target = world_to_local(ch.at, cue.pose)
                forearm = S.unit(pos[f"{side}_wrist"] - pos[f"{side}_elbow"])
                want = S.unit(target - pos[f"{side}_wrist"])
                max_hold_aim = max(max_hold_aim, math.degrees(math.acos(float(np.clip(forearm @ want, -1.0, 1.0)))))
        prev, prev_pos = angles, pos
        frames.append({"angles": dict(angles), "t": t})
    missing = [w for w in case.rest_windows if not any(w[0] <= rt <= w[1] for rt in rest_times)]
    return Result(case.name, frames, reports, list(log.lines), max_step, max_joint, max_link, max_link_name, max_link_t, max_collar, max_hold_aim, rest_times, arm_clips, missing)


def _pose(x: float, y: float, yaw_deg: float) -> tuple[float, float, float]:
    return (x, y, math.radians(yaw_deg))


def arm(at: tuple[float, float, float], **opts: str) -> Channel:
    return Channel("arm", at, hand=opts.get("dominant", ""))


def head(at: tuple[float, float, float]) -> Channel:
    return Channel("head", at)


def default_cases() -> list[Case]:
    p = _pose(10.0, 2.5, 180.0)  # facing west
    fan = [(5.0, 0.5, 1.0), (5.0, 2.5, 1.5), (6.0, 1.0, 2.2), (4.0, 3.5, 0.3), (7.0, 2.0, 1.8)]
    burst = [Cue(0.9 * i, [arm(at)], p) for i, at in enumerate(fan)] + [Cue(0.9 * len(fan), [], p)]
    chain = [(5.0, 0.5, 1.2), (5.0, 2.5, 1.2), (5.0, 4.5, 1.2)]  # right of the ped, midline, left of the ped
    chained = [Cue(1.2 * i, [arm(at)], p) for i, at in enumerate(chain)] + [Cue(1.2 * len(chain), [], p)]
    cells = [(7.0, 3.5, 1.8), (7.0, 1.5, 1.8), (7.0, 3.5, 0.6)]  # grid cells 2 s each with a 1 s rest between, arm must hang in every rest
    grid = [c for i, at in enumerate(cells) for c in (Cue(3.0 * i, [arm(at), head(at)], p), Cue(3.0 * i + 2.0, [], p))]
    walk = [Cue(0.25 * i, [arm((10.0, 8.0, 1.5)), head((10.0, 8.0, 1.5))], (5.0 + 0.25 * i, 3.0, 0.0), moving=True) for i in range(40)]  # walker heading east past a target on its left
    return [
        Case("single_level", [Cue(0.0, [arm((5.0, 2.5, 1.3))], p), Cue(2.5, [], p)], 5.0),
        Case("single_high", [Cue(0.0, [arm((8.0, 2.5, 3.5))], p), Cue(2.5, [], p)], 5.0),
        Case("single_low_cross", [Cue(0.0, [arm((7.0, 4.5, 0.2))], p), Cue(2.5, [], p)], 5.0),
        Case("burst_5", burst, 0.9 * len(fan) + 2.5),
        Case("hand_switch", [Cue(0.0, [arm((5.0, 0.5, 1.2))], p), Cue(2.0, [Channel("arm_r", (5.0, 4.5, 1.2))], p), Cue(4.0, [], p)], 6.5),
        Case("forced_left", [Cue(0.0, [Channel("arm_l", (5.0, 0.5, 1.2))], p), Cue(2.5, [], p)], 5.0),
        Case("look_only", [Cue(0.0, [head((6.0, 4.0, 1.6))], p), Cue(2.0, [], p)], 3.5),
        Case("walking", [Cue(0.0, [arm((6.0, 3.0, 1.2))], p, moving=True), Cue(2.5, [], p)], 5.0),
        Case("point_and_gaze", [Cue(0.0, [arm((5.0, 0.5, 1.2)), head((6.0, 4.0, 1.6))], p), Cue(2.5, [arm((5.0, 0.5, 1.2))], p), Cue(3.5, [], p)], 6.0),
        Case("chained_across_midline", chained, 1.2 * len(chain) + 2.5),
        Case("dominant_left", [Cue(0.0, [arm((5.0, 2.5, 1.2), dominant="l")], p), Cue(2.5, [], p)], 5.0),
        Case("grid_rest_between", grid, 3.0 * len(cells) + 2.0, min_clips=len(cells), rest_windows=[(3.0 * i + 2.0, 3.0 * i + 4.5) for i in range(len(cells))]),
        Case("walker_tracks", walk, 10.0 + 2.0, min_clips=8),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/gesture_qa"))
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    body = S.Body(BODY_HEIGHT)
    from task_generator.simulators.human.pointing import render

    failed = 0
    summary = []
    for case in default_cases():
        if args.only and case.name not in args.only:
            continue
        res = run(case)
        first = next((c for c in case.cues if c.channels), None)
        local_target = None if first is None else world_to_local(first.channels[0].at, first.pose)
        render.filmstrip(res.frames, body, args.out / f"{case.name}.png", n=14, views=("front", "side", "top"), target=local_target, title=f"{case.name}  link step {res.max_link_m:.2f} m ({res.max_link} @ {res.max_link_t:.2f}s)  joint step {res.max_step_rad:.2f} rad ({res.max_step_joint})")
        if args.video:
            render.video(res.frames, body, args.out / f"{case.name}.mp4", target=local_target)
        row = {"case": case.name, "ok": res.ok, "max_link_m": round(res.max_link_m, 3), "max_link": res.max_link, "max_link_t": res.max_link_t, "max_step_rad": round(res.max_step_rad, 3), "max_step_joint": res.max_step_joint, "warnings": res.warnings, "reports": res.reports}
        summary.append(row)
        failed += not res.ok
        flags = [f"{r['slot']}:{r['kind']} aim={r.get('aim_error_deg', 0):.2f} clr={r.get('clearance_m', 0):.3f}{' RELAXED' if r.get('relaxed') else ''}{' NEAR' if r.get('near_target_fallback') else ''}" for r in res.reports]
        print(f"{'OK ' if res.ok else 'BAD'} {case.name:18s} link {res.max_link_m:.2f} m ({res.max_link} @ {res.max_link_t:.2f}s) joint {res.max_step_rad:.2f} rad ({res.max_step_joint}) {'; '.join(flags)} {' '.join(res.warnings)}")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
