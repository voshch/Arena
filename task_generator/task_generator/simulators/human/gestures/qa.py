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
from task_generator.simulators.human.gestures import BODY_HEIGHT, GestureLayer, GestureRequest
from task_generator.simulators.human.pointing import skeleton as S
from task_generator.simulators.human.pointing.contract import ROS_JOINT_ORDER

DT = 0.05
ANIMATIONS = Path(__file__).resolve().parents[1] / "animations"
MAX_JOINT_STEP_RAD = 0.6  # per 50 ms tick, informational (shoulder triples may wrap)
MAX_LINK_STEP_M = 0.25  # per 50 ms tick, wrist/elbow/head jump that reads as a snap
LINKS = ("l_wrist", "r_wrist", "l_elbow", "r_elbow", "head")
IDLE = 0


@attrs.frozen
class Cue:
    """One intent held from ``t`` on: kind '' clears."""

    t: float
    kind: str
    at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    moving: bool = False
    opts: dict = attrs.Factory(dict)


@attrs.frozen
class Case:
    name: str
    cues: Sequence[Cue]
    duration: float


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

    @property
    def ok(self) -> bool:
        return not self.warnings and self.max_link_m <= MAX_LINK_STEP_M and all(r.get("ok", True) and not r.get("relaxed") and not r.get("near_target_fallback") for r in self.reports)


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
    layer = GestureLayer(mgr, log, sync=True)
    mgr.gesture_hook = layer
    reports: list[dict] = []
    seen: set[int] = set()
    cues = sorted(case.cues, key=lambda c: c.t)
    frames: list[dict] = []
    prev: dict[str, float] | None = None
    prev_pos: dict | None = None
    max_step, max_joint = 0.0, ""
    max_link, max_link_name, max_link_t = 0.0, "", 0.0
    body = S.Body(BODY_HEIGHT)
    n = int(round(case.duration / DT))
    for i in range(n):
        t = i * DT
        cue = None
        for c in cues:
            if c.t <= t + 1e-9:
                cue = c
        req = None if cue is None or not cue.kind else GestureRequest(kind=cue.kind, at=cue.at, pose=cue.pose, moving=cue.moving, opts=dict(cue.opts))
        angles = mgr.compute(agent_id, IDLE, 0.0, DT, gesture=req)
        ag = layer._agents.get(agent_id)
        if ag is not None:
            for st in ag.slots.values():
                if st.clip is not None and id(st.clip) not in seen:
                    seen.add(id(st.clip))
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
        prev, prev_pos = angles, pos
        frames.append({"angles": dict(angles), "t": t})
    return Result(case.name, frames, reports, list(log.lines), max_step, max_joint, max_link, max_link_name, max_link_t)


def _pose(x: float, y: float, yaw_deg: float) -> tuple[float, float, float]:
    return (x, y, math.radians(yaw_deg))


def default_cases() -> list[Case]:
    p = _pose(10.0, 2.5, 180.0)  # facing west
    fan = [(5.0, 0.5, 1.0), (5.0, 4.5, 1.5), (6.0, 2.5, 2.2), (4.0, 1.0, 0.3), (7.0, 4.0, 1.8)]
    burst = [Cue(0.0 + 0.9 * i, "point", at, p) for i, at in enumerate(fan)] + [Cue(0.9 * len(fan), "")]
    return [
        Case("single_level", [Cue(0.0, "point", (5.0, 2.5, 1.3), p), Cue(2.5, "")], 5.0),
        Case("single_high", [Cue(0.0, "point", (8.0, 2.5, 3.5), p), Cue(2.5, "")], 5.0),
        Case("single_low_cross", [Cue(0.0, "point", (7.0, 4.5, 0.2), p), Cue(2.5, "")], 5.0),
        Case("burst_5", burst, 0.9 * len(fan) + 2.5),
        Case("hand_switch", [Cue(0.0, "point", (5.0, 0.5, 1.2), p), Cue(2.0, "point", (5.0, 4.5, 1.2), p), Cue(4.0, "")], 6.5),
        Case("forced_left", [Cue(0.0, "point", (5.0, 0.5, 1.2), p, opts={"hand": "left"}), Cue(2.5, "")], 5.0),
        Case("look_only", [Cue(0.0, "look", (6.0, 4.0, 1.6), p), Cue(2.0, "")], 3.5),
        Case("walking", [Cue(0.0, "point", (6.0, 3.0, 1.2), p, moving=True), Cue(2.5, "")], 5.0),
        Case("behind_then_front", [Cue(0.0, "point", (14.0, 2.5, 1.2), p), Cue(1.5, "point", (6.0, 2.5, 1.2), p), Cue(4.0, "")], 6.0),
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
        cue_target = next((c.at for c in case.cues if c.kind), None)
        local_target = None
        if cue_target is not None:
            from task_generator.simulators.human.gestures import world_to_local

            local_target = world_to_local(cue_target, next(c.pose for c in case.cues if c.kind))
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
