"""Hold: re-solve a retargeted clip's arm(s) so the hand closes on a prop (long cane, IV pole, walking frame, rollator).

Text-to-motion pins (e.g. Kimodo's hand constraints) bring a hand only near a handle: they are soft, the retarget changes
the body's proportions and the character a simulator draws differs again from Arena's template. Holding a prop needs
millimetres and a few degrees, so this stage solves the arm last, per clip frame, on the body that will be drawn: a
character bundle (shared.actor_pose.ActorPose, the simulators' own pose of its .rig and skin) or Arena's template.

Per frame, the arm DOFs (collar, shoulder, elbow, forearm twist, wrist flexion and tilt) are solved by bounded least
squares (scipy, warm-started from the previous frame, the first frame from random starts within the limits) so that the
fist closed on the prop (arena_runtime hands.palm_frame(...).grip / .axis: the cylinder the finger curls close on) has its
centre on the handle and its axis along it. Legs, trunk and head stay as retargeted. Targets are in the body frame, x
forward, y left, z up from the floor (the soles), all lengths in m:

  cane      right hand (side=l mirrors), long cane, two-point touch: the fist at (gx, gy, gz), the cane (length, held grip_at
            below its top) running from the fist to a tip that sweeps +-arc across the midline in step with the clip's own
            gait (on the left at each right heel strike, from the ankles), lifted `tap` between the touches.
  rolling   the same, the tip always on the floor (tap 0).
  iv        left hand (side=r mirrors) round a vertical pole at (gx, gy, gz), thumb up: the pole runs diagonally across the
            palm (`angle`), the elbow at the side.
  frame     both hands on a pick-up walking frame's side bars, `width` apart: the clip's own two-hand path fore-aft and in
            height (the frame's lift and placing), averaged over both hands (rigid frame) and held at the bar height
            `handle` while the hands move back relative to the body (the frame stands, the person steps in).
  rollator  both hands on a rollator's grips at (gx, +-width/2, gz).

Keys (all tasks): radius, angle (the grip cylinder, hands.Grip), comfort (1: soft costs keeping the elbow within
elbow..elbow_hi rad, wrist flexion within wflex, tilt within wdev, forearm twist within wtwist), elbowdir (weight of the
elbow pointing back and `elbowout` outward from the shoulder-wrist line), fa (canes: most degrees between the cane and
the forearm line), wrap (1: the thumb closes over the fingers), starts (random starts of the first frame).
The defaults fit the props they were made for (IVStand, WalkerFitted, RollatorFitted, a 1.27 m cane); `Hold.parse`
takes overrides as TASK:key=value,... .

The finger curls that close this body's hand on the handle (hands.grasp_on) are written into the clip and reported, for
an agent type's `pose.hold.<side>.curls`. A held clip carries no `body/` track: that track holds the source's own torso,
which the arm was not solved against, so players pose the clip from its angles.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from arena_assets.utils.geom import unit
from arena_simulation_setup.shared.actor_pose import ActorPose, TemplatePose
from arena_simulation_setup.shared.body import DEFAULT_STANCE

from .contract import load_module
from .ik import HARD_LIMITS
from .retarget import Clip

HANDS = load_module(Path("humans/hands.py"), "arena_wire_hands", "ARENA_HANDS")

ARM = ("y_collar", "p_collar", "y_shoulder", "p_shoulder", "r_shoulder", "elbow", "r_wrist", "wrist", "d_wrist")
WIDE = {"y_shoulder": (-4.7, 4.7), "p_shoulder": (-1.6, 3.3), "r_shoulder": (-3.1, 3.1)}  # the shoulder triple's full wrap
CANE = dict(gx=0.30, gy=-0.10, gz=0.90, arc=0.30, tap=0.04, length=1.27, grip_at=0.1, radius=0.0125, angle=-60.0, fa=12.0)
TASKS: dict[str, dict[str, float | str]] = {
    "cane": dict(CANE, side="r"),
    "rolling": dict(CANE, tap=0.0, side="r"),
    "iv": dict(side="l", gx=0.32, gy=0.24, gz=1.02, radius=0.008, angle=42.0, comfort=1, elbow=1.0, elbow_hi=1.9, wflex=0.35, wdev=0.25, wtwist=0.6, elbowdir=1.0, elbowout=0.35, wrap=1),
    "frame": dict(side="lr", width=0.42, radius=0.009, angle=0.0, handle=0.659),
    "rollator": dict(side="lr", gx=0.36, width=0.444, gz=0.86, radius=0.018, angle=-35.0, comfort=1, elbow=0.3, elbow_hi=0.6, wflex=0.5, wdev=0.25),
}


OPTIONAL = {"comfort", "elbow", "elbow_hi", "wflex", "wdev", "wtwist", "elbowdir", "elbowout", "fa", "wrap", "starts"}  # keys any task takes


@dataclass(frozen=True)
class Hold:
    """A prop held through a clip: the task and its keys (TASKS defaults, overridden)."""

    task: str
    params: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str) -> Hold:
        """TASK or TASK:key=value,key=value (e.g. iv:gx=0.3,angle=40)."""
        task, _, rest = text.partition(":")
        if task not in TASKS:
            raise ValueError(f"unknown hold task {task!r} ({', '.join(TASKS)})")
        p = dict(TASKS[task])
        for item in filter(None, rest.split(",")):
            key, sep, value = item.partition("=")
            key = key.strip()
            if not sep:
                raise ValueError(f"hold key {item!r}: use key=value")
            if key not in TASKS[task] and key not in OPTIONAL:
                raise ValueError(f"hold {task}: unknown key {key!r} ({', '.join(sorted(set(TASKS[task]) | OPTIONAL))})")
            p[key] = value.strip() if key == "side" else float(value)
        side = str(p["side"])
        if task in ("frame", "rollator") and side != "lr" or task not in ("frame", "rollator") and side not in ("l", "r"):
            raise ValueError(f"hold {task}: side {side!r} ({'lr' if task in ('frame', 'rollator') else 'l or r'})")
        return cls(task, p)

    @property
    def sides(self) -> tuple[str, ...]:
        return ("l", "r") if self.params["side"] == "lr" else (str(self.params["side"]),)


@functools.lru_cache(maxsize=8)
def load_body(spec: str | Path | None) -> tuple[Callable[[Mapping[str, float]], dict[str, np.ndarray]], Mapping[str, np.ndarray], str]:
    """(positions of semantic angles, stance, label) of a character bundle directory or model URI (Cat/name), the template without one;
    loaded once per spec."""
    if spec is None:
        return TemplatePose().positions, DEFAULT_STANCE, "template"
    path = Path(spec)
    if not path.is_dir():  # a model URI: the Human asset resolvers (local trees, then the configured buckets)
        import asyncio

        from arena_simulation_setup.tree.assets.Human import HumanIdentifier

        try:
            found = asyncio.run(HumanIdentifier.parse(str(spec)).resolve_path())
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"hold body {spec}: neither a bundle directory nor a Human model Arena can resolve") from exc
        path = Path(found)
    body = ActorPose(path)
    return body.positions, body.stance, path.name


def _gait_phase(lead: np.ndarray) -> np.ndarray:
    """Phase (rad) per frame, 0 at each right heel strike: the frames where the right ankle leads the left the most."""
    n = len(lead)
    x = lead - lead.mean()
    peaks = [k for k in range(1, n - 1) if x[k] >= x[k - 1] and x[k] > x[k + 1] and x[k] > 0.5 * x.max()]
    keep: list[int] = []
    for k in peaks:
        if keep and k - keep[-1] < 6:
            if x[k] > x[keep[-1]]:
                keep[-1] = k
            continue
        keep.append(k)
    if len(keep) < 2:
        raise ValueError("hold cane/rolling: the clip has fewer than two strides (right heel strikes) to time the sweep")
    period = float(np.median(np.diff(keep)))
    anchors = [keep[0] - period * m for m in range(int(keep[0] / period) + 2, 0, -1)] + keep + [keep[-1] + period * m for m in range(1, int((n - keep[-1]) / period) + 2)]
    return np.interp(np.arange(n), anchors, 2 * math.pi * np.arange(len(anchors)))


def apply(clip: Clip, hold: Hold, body: str | Path | None = None, verbose: bool = False) -> tuple[Clip, dict]:
    """The clip with its arm(s) re-solved to hold the prop and the hand(s) closed on it, and a report (errors, curls)."""
    positions, _stance, label = load_body(body)
    p = hold.params
    sides = hold.sides
    T = len(clip.t)
    base = [{k: float(v[i]) for k, v in clip.angles.items()} for i in range(T)]
    radius, angle = float(p["radius"]), float(p["angle"])
    mirror = {s: (1.0 if s == "l" else -1.0) for s in "lr"}  # y sign of a left-side target for this hand
    start = [positions(b) for b in base] if hold.task in ("cane", "rolling", "frame") else []
    phase, path = np.zeros(T), np.zeros((T, 3))  # set below for the tasks that use them

    if hold.task in ("cane", "rolling"):
        phase = _gait_phase(np.array([q["right_ankle"][0] - q["left_ankle"][0] for q in start]))  # the tip lands opposite the striking foot, either hand
    if hold.task == "frame":
        path = np.array([np.mean([HANDS.palm_frame(q, s).grip(radius, 0.0) for s in sides], axis=0) for q in start])
        dx = np.gradient(path[:, 0]) if T > 1 else np.zeros(T)
        w = np.clip(dx / (0.5 * dx.max()), 0.0, 1.0) if dx.max() > 0 else np.zeros(T)
        path[:, 2] = p["handle"] + w * (path[:, 2] - path[:, 2].min())

    def targets(i: int) -> dict[str, tuple[np.ndarray, np.ndarray, bool, np.ndarray]]:
        """{side: (grip centre, axis, axis sign free, soft palm direction)}"""
        if hold.task in ("cane", "rolling"):
            s = sides[0]
            g = np.array([p["gx"], p["gy"] if s == "r" else -p["gy"], p["gz"]])  # gy is given for the right hand
            c = math.cos(phase[i])
            ty, tz = -p["arc"] * c * mirror[s], p["tap"] * (1.0 - c * c)
            lg = p["length"] - p["grip_at"]
            dz, dy = g[2] - tz, ty - g[1]
            tip = np.array([g[0] + math.sqrt(max(1e-6, lg * lg - dz * dz - dy * dy)), ty, tz])
            return {s: (g, unit(tip - g), False, np.array([0.0, -0.7 * mirror[s], -0.7]))}  # palm toward the midline and down
        if hold.task == "iv":
            s = sides[0]
            down = np.array([0.0, 0.0, -1.0])  # thumb up: the fist axis (index -> little finger) points down the pole, either hand
            return {s: (np.array([p["gx"], mirror[s] * p["gy"], p["gz"]]), down, False, np.array([0.0, -mirror[s], 0.0]))}
        half = p["width"] / 2
        if hold.task == "rollator":
            return {s: (np.array([p["gx"], mirror[s] * half, p["gz"]]), np.array([1.0, 0.0, 0.0]), True, np.array([0.0, 0.0, -1.0])) for s in sides}
        return {s: (np.array([path[i, 0], mirror[s] * half, path[i, 2]]), np.array([1.0, 0.0, 0.0]), True, np.array([0.0, 0.0, -1.0])) for s in sides}

    names = [f"{s}_{d}" for s in sides for d in ARM]
    bound = [WIDE[n[2:]] if n[2:] in WIDE else HARD_LIMITS.get(n, (-math.pi, math.pi)) for n in names]
    lo, hi = np.array([b_[0] for b_ in bound]), np.array([b_[1] for b_ in bound])
    cane = hold.task in ("cane", "rolling")

    def forearm_deg(q: Mapping[str, np.ndarray], s: str, ax: np.ndarray) -> float:
        w = "left" if s == "l" else "right"
        return math.degrees(math.acos(max(-1.0, min(1.0, float(unit(q[f"{w}_wrist"] - q[f"{w}_elbow"]) @ ax)))))

    def axis_err(f: object, ax: np.ndarray, free: bool) -> float:
        a = f.axis(angle)  # type: ignore[attr-defined]
        c = abs(float(a @ ax)) if free else math.copysign(1.0, angle) * float(a @ ax)
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    solved = np.zeros((T, len(names)))
    prev: np.ndarray | None = None
    errors = []
    for i in range(T):
        b, tg = base[i], targets(i)
        kim = np.array([b.get(n, 0.0) for n in names])

        def res(x: np.ndarray, b: dict = b, tg: dict = tg, kim: np.ndarray = kim, prev: np.ndarray | None = prev) -> np.ndarray:
            a = b | dict(zip(names, x, strict=True))
            q = positions(a)
            r: list[float] = []
            for s, (g, ax, free, palm_dir) in tg.items():
                f = HANDS.palm_frame(q, s)
                r.extend((f.grip(radius, angle) - g) / 0.004)
                ax_f = f.axis(angle)
                r.extend((np.cross(ax_f, ax) if free else (math.copysign(1.0, angle) * ax_f - ax)) / 0.02)
                r.append((1.0 - float(f.palm @ unit(palm_dir))) / 0.5)
                if cane:
                    r.append(max(0.0, forearm_deg(q, s, ax) - p["fa"]) / 3.0)
                if p.get("elbowdir"):
                    w_ = "left" if s == "l" else "right"
                    sh, el, wr = q[f"{w_}_shoulder"], q[f"{w_}_elbow"], q[f"{w_}_wrist"]
                    line = unit(wr - sh)
                    off = (el - sh) - float((el - sh) @ line) * line
                    want = unit(np.array([-1.0, p.get("elbowout", 0.35) * mirror[s], 0.0]))
                    want = unit(want - float(want @ line) * line)
                    r.append((1.0 - float(unit(off) @ want)) / 0.15 * p["elbowdir"])
            if p.get("comfort"):
                for s in tg:
                    r.append(max(0.0, p.get("elbow", 0.35) - a[f"{s}_elbow"]) / 0.1)
                    r.append(max(0.0, a[f"{s}_elbow"] - p.get("elbow_hi", 9.0)) / 0.1)
                    r.append(max(0.0, abs(a[f"{s}_wrist"]) - p.get("wflex", 0.5)) / 0.15)
                    r.append(max(0.0, abs(a[f"{s}_d_wrist"]) - p.get("wdev", 0.25)) / 0.15)
                    r.append(max(0.0, abs(a[f"{s}_r_wrist"]) - p.get("wtwist", 9.0)) / 0.15)
            r.extend((x - kim) * 0.05)
            if prev is not None:
                r.extend((x - prev) * 0.6)
            return np.array(r)

        x0 = np.clip(kim if prev is None else prev, lo + 1e-6, hi - 1e-6)
        if prev is None:
            rng = np.random.default_rng(0)
            starts = [x0] + [rng.uniform(lo, hi) for _ in range(int(p.get("starts", 25)))]
            sol = min((least_squares(res, s0, bounds=(lo, hi), x_scale=0.3, max_nfev=200) for s0 in starts), key=lambda r: r.cost)
        else:
            sol = least_squares(res, x0, bounds=(lo, hi), x_scale=0.3, max_nfev=400)
        prev = solved[i] = sol.x
        q = positions(b | dict(zip(names, sol.x, strict=True)))
        for s, (g, ax, free, _) in tg.items():
            f = HANDS.palm_frame(q, s)
            errors.append((float(np.linalg.norm(f.grip(radius, angle) - g)), axis_err(f, ax, free)))
        if verbose and i % 20 == 0:
            print(f"  hold {hold.task} frame {i}/{T}: grip {errors[-1][0] * 1000:.1f} mm, axis {errors[-1][1]:.1f} deg")

    # finger curls of this body's hand on the handle, held through the clip
    curls = {s: HANDS.grasp_on(positions, s, radius, angle, wrap=bool(p.get("wrap", 0))) for s in sides}
    angles = {k: v.copy() for k, v in clip.angles.items()}
    for j, n in enumerate(names):
        angles[n] = solved[:, j]
    for s in sides:
        for n, v in HANDS.shape(s, curls[s]).items():
            angles[n] = np.full(T, float(v))
    # the body track would play the source's torso under the solved arm: drop it, so players pose the held clip from its
    # angles over the character's stance, exactly what was solved
    e = np.array(errors)
    report = {
        "task": hold.task,
        "body": label,
        "sides": "".join(sides),
        "params": {k: v for k, v in p.items() if k != "side"},
        "curls": {s: [float(c) for c in curls[s]] for s in sides},
        "grip_mm": [round(float(e[:, 0].mean() * 1000), 2), round(float(np.percentile(e[:, 0], 95) * 1000), 2)],
        "axis_deg": [round(float(e[:, 1].mean()), 2), round(float(np.percentile(e[:, 1], 95)), 2)],
    }
    return replace(clip, angles=angles, body_rotations=None, body_pelvis=None), report


def annotation(report: dict) -> dict:
    """The clip annotation's `hold` block: what the hand holds, on which body, the curls for pose.hold and the residual errors."""
    return {"hold": {k: report[k] for k in ("task", "body", "sides", "curls", "grip_mm", "axis_deg")} | {"radius": report["params"]["radius"], "angle": report["params"]["angle"]}}


__all__ = ["TASKS", "Hold", "annotation", "apply", "load_body"]
