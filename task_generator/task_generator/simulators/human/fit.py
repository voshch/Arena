"""Fit a walking clip of semantic joint angles to a library gait profile yaml."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs
import numpy as np

from .gait import LIMITS
from .profile import DEFAULT_WALK, IDLE, JOINT_NAMES, CadenceLaw, GainLaw, GaitProfile, JointSignal, PhaseWarp, Signal, dump_gait_profile, load_library

LEFT_HIP = "l_r_hip"
RIGHT_HIP = "r_r_hip"
DEFAULT_FPS = 20.0
DEFAULT_HARMONICS = 3
MAX_HARMONICS = 8
MIN_CYCLES = 2
SPLIT_DEADBAND = 0.02
SPLIT_SEARCH = (0.25, 0.75)
SYMMETRY_TOLERANCE = 0.05
SYMMETRY_FLOOR = 0.01
ZERO_TOLERANCE = 1e-6
FLAT_RMS = 1e-6

_TAU = 2.0 * math.pi
_PAIR_KEYS = {"r_hip": "hip"}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CLIP_LAYOUT = "a .npy object array of frames, each {'angles': {<joint from JOINT_NAMES>: <rad>}}, the per-frame layout AnimationClip.frames() gives (arena_simulation_setup.tree.assets.Animation)"
_EXAMPLE = "python3 -m task_generator.simulators.human.fit walk_slow.npy -o walk_slow.yaml --speed 0.8 --report walk_slow.json"


class FitError(ValueError):
    """A clip or setting the fitter cannot turn into a gait profile, the message says what to change."""


@attrs.frozen
class FitReport:
    """What the fit measured on the clip and how well the emitted profile reproduces it."""

    name: str
    fps: float
    frames: int
    frames_used: int
    cycles: int
    period_s: float
    cadence_hz: float
    cadence_law: str
    speed: float | None
    harmonics: int
    symmetric: bool
    split: float
    split_fitted: float
    heel_strike_fraction: float
    rms: Mapping[str, float]
    worst_joint: str
    worst_rms: float
    limits_widened: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)

    def text(self) -> str:
        law = f"through {self.speed} m/s on the default base" if self.cadence_law == "through_speed" else "fixed at the measured cadence (pass --speed to fit per_speed)"
        lines = [
            f"profile {self.name}: {self.cycles} cycles over {self.frames_used} of {self.frames} frames at {self.fps} fps, {self.harmonics} harmonics",
            f"cadence: {self.cadence_hz:.6f} Hz (period {self.period_s:.6f} s), law {law}",
            f"symmetric: {'true' if self.symmetric else 'false'}",
            f"phase_warp.split: {self.split} (fitted {self.split_fitted:.6f}, heel-strike fraction {self.heel_strike_fraction:.6f})",
            f"limits widened: {', '.join(self.limits_widened) if self.limits_widened else 'none'}",
            f"worst joint: {self.worst_joint}, rms {self.worst_rms:.3e} rad",
            "rms per joint (rad):",
        ]
        lines += [f"  {joint:<14}{rms:.3e}" for joint, rms in sorted(self.rms.items(), key=lambda item: -item[1])]
        return "\n".join(lines)


def load_clip(path: str | Path) -> list[dict[str, float]]:
    """Per-frame joint angles of a clip in the AnimationManager .npy layout."""
    path = Path(path)
    if not path.is_file():
        raise FitError(f"clip {str(path)!r} does not exist, pass {_CLIP_LAYOUT}")
    try:
        frames = np.load(path, allow_pickle=True)
        return [{str(joint): float(angle) for joint, angle in frame["angles"].items()} for frame in frames]
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError, EOFError, pickle.UnpicklingError) as e:
        raise FitError(f"clip {str(path)!r} is not readable as a clip ({type(e).__name__}: {e}), pass {_CLIP_LAYOUT}") from e


def _design(psi: np.ndarray, harmonics: int) -> np.ndarray:
    columns = [np.ones_like(psi)]
    for k in range(1, harmonics + 1):
        columns += [np.sin(k * psi), np.cos(k * psi)]
    return np.column_stack(columns)


def _design_slope(psi: np.ndarray, harmonics: int) -> np.ndarray:
    columns = [np.zeros_like(psi)]
    for k in range(1, harmonics + 1):
        columns += [k * np.cos(k * psi), -k * np.sin(k * psi)]
    return np.column_stack(columns)


def _solve(design: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _warp(phi: np.ndarray, split: float) -> np.ndarray:
    if split == 0.5:
        return phi
    u = np.mod(phi, _TAU)
    base = phi - u
    knee = _TAU * split
    return np.where(u < knee, base + u / (2.0 * split), base + math.pi + (u - knee) / (2.0 * (1.0 - split)))


def _short(detail: str) -> FitError:
    return FitError(f"{detail}: the fit needs at least {MIN_CYCLES} whole walking cycles, record or cut a longer stretch of steady walking")


def _period_frames(hip: np.ndarray) -> float:
    """Dominant period of the left hip swing in frames, from the first autocorrelation peak."""
    n = len(hip)
    last = min(n - 2, n // MIN_CYCLES + 1)
    if last < 2:
        raise _short(f"the clip has only {n} frames")
    frame = np.arange(n, dtype=float)
    x = hip - np.polyval(np.polyfit(frame, hip, 1), frame)
    power = float(x @ x) / n
    if math.sqrt(power) < FLAT_RMS:
        raise FitError(f"{LEFT_HIP} is constant over the clip, so it holds no walking cycle: pass a clip of steady walking, a held pose belongs in a profile of means like seated.yaml")
    acf = np.array([float(x[: n - lag] @ x[lag:]) / (n - lag) for lag in range(last + 1)]) / power
    negative = np.flatnonzero(acf < 0.0)
    peaks = [lag for lag in range(int(negative[0]) if negative.size else last, last) if acf[lag] > 0.0 and acf[lag] >= acf[lag - 1] and acf[lag] > acf[lag + 1] and lag <= n / MIN_CYCLES]
    if not peaks:
        raise _short(f"the {LEFT_HIP} swing does not repeat within half of the {n} frame clip")
    best = max(acf[lag] for lag in peaks)
    lag = next(lag for lag in peaks if acf[lag] >= 0.8 * best)
    curvature = acf[lag - 1] - 2.0 * acf[lag] + acf[lag + 1]
    return float(lag + (0.5 * (acf[lag - 1] - acf[lag + 1]) / curvature if curvature < 0.0 else 0.0))


def _refine_rate(hip: np.ndarray, rate: float, harmonics: int) -> float:
    """Least-squares phase rate (rad per frame) of the harmonic model on the hip, started from the autocorrelation estimate."""
    frame = np.arange(len(hip), dtype=float)
    start = rate
    for _ in range(50):
        design = _design(rate * frame, harmonics)
        coef = _solve(design, hip)
        slope = frame * (_design_slope(rate * frame, harmonics) @ coef)
        step = float(_solve(np.column_stack([design, slope]), hip - design @ coef)[-1])
        rate += step
        if abs(rate - start) > 0.2 * start:
            return start
        if abs(step) < 1e-15 * rate:
            break
    return rate


def _cycle_phase(hips: np.ndarray, rate: float, cycles: int, harmonics: int) -> np.ndarray:
    """Phase per frame from frame 0: a uniform ramp between cycle anchors, each anchored where the hip pair matches the mean waveform."""
    frame = np.arange(len(hips), dtype=float)
    theta = rate * frame
    whole = theta < _TAU * cycles
    coef = _solve(_design(theta[whole], harmonics), hips[whole])
    at: list[float] = []
    for cycle in range(cycles):
        inside = (theta >= _TAU * cycle) & (theta < _TAU * (cycle + 1))
        delta = 0.0
        for _ in range(10):
            psi = theta[inside] + delta
            slope = (_design_slope(psi, harmonics) @ coef).ravel()
            step = float((hips[inside] - _design(psi, harmonics) @ coef).ravel() @ slope / (slope @ slope))
            delta += step
            if abs(step) < 1e-15:
                break
        at.append((_TAU * (cycle + 0.5) - max(-math.pi / 4.0, min(math.pi / 4.0, delta))) / rate)
    anchors = np.array(at)
    centers = _TAU * (np.arange(cycles) + 0.5)
    phase = np.interp(frame, anchors, centers)
    phase = np.where(frame < anchors[0], centers[0] + rate * (frame - anchors[0]), phase)
    return np.where(frame > anchors[-1], centers[-1] + rate * (frame - anchors[-1]), phase)


def _align(theta: np.ndarray, hip: np.ndarray, split: float, harmonics: int, target: float, offset: float) -> float:
    """Phase offset at which the fitted left hip fundamental sits at the library walk's phase."""
    relax = 1.0
    previous = math.inf
    for _ in range(200):
        coef = _solve(_design(_warp(theta + offset, split), harmonics), hip)
        error = math.remainder(math.atan2(coef[2], coef[1]) - target, _TAU)
        if abs(error) < 1e-14:
            break
        if abs(error) > previous:
            relax *= 0.5
        previous = abs(error)
        offset += relax * error
    return offset


def _minimize(cost: Callable[[float], float], lo: float, hi: float) -> float:
    grid = np.linspace(lo, hi, 51)
    costs = [cost(float(s)) for s in grid]
    best = int(np.argmin(costs))
    a, b = float(grid[max(best - 1, 0)]), float(grid[min(best + 1, len(grid) - 1)])
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    cost_c, cost_d = cost(c), cost(d)
    while b - a > 1e-7:
        if cost_c < cost_d:
            b, d, cost_d = d, c, cost_c
            c = b - ratio * (b - a)
            cost_c = cost(c)
        else:
            a, c, cost_c = c, d, cost_d
            d = a + ratio * (b - a)
            cost_d = cost(d)
    return 0.5 * (a + b)


def _peak_phase(coef: np.ndarray, harmonics: int) -> float:
    psi = np.linspace(0.0, _TAU, 3600, endpoint=False)
    values = _design(psi, harmonics) @ coef
    i = int(np.argmax(values))
    before, at, after = values[i - 1], values[i], values[(i + 1) % len(values)]
    curvature = before - 2.0 * at + after
    return float(psi[i] + (0.5 * (before - after) / curvature * (psi[1] - psi[0]) if curvature < 0.0 else 0.0))


def _signal(coef: np.ndarray) -> Signal:
    harmonics: list[tuple[float, float]] = []
    for a, b in zip(coef[1::2], coef[2::2], strict=True):
        amp = math.hypot(a, b)
        harmonics.append((amp, math.atan2(b, a)) if amp >= ZERO_TOLERANCE else (0.0, 0.0))
    while harmonics and harmonics[-1] == (0.0, 0.0):
        harmonics.pop()
    return Signal(float(coef[0]) if abs(coef[0]) >= ZERO_TOLERANCE else 0.0, tuple(harmonics))


def _joint_signals(joints: Sequence[str], coef: np.ndarray, harmonics: int, tolerance: float) -> tuple[dict[str, JointSignal], bool]:
    """Bindings per joint and whether every left/right pair is one signal half a cycle apart."""
    column = {joint: coef[:, i] for i, joint in enumerate(joints)}
    pairs = [(joint, f"r_{joint[2:]}") for joint in joints if joint.startswith("l_") and f"r_{joint[2:]}" in column]
    flip = np.array([1.0] + [(-1.0) ** k for k in range(1, harmonics + 1) for _ in range(2)])
    symmetric = True
    for left, right in pairs:
        gap = column[right] - flip * column[left]
        amplitude = math.sqrt(max(float(column[left][1:] @ column[left][1:]), float(column[right][1:] @ column[right][1:])) / 2.0)
        if math.sqrt(gap[0] ** 2 + float(gap[1:] @ gap[1:]) / 2.0) > max(tolerance * amplitude, SYMMETRY_FLOOR):
            symmetric = False
    bound: dict[str, JointSignal] = {}
    for left, right in pairs:
        if symmetric:
            key = _PAIR_KEYS.get(left[2:], left[2:])
            shared = _signal(0.5 * (column[left] + flip * column[right]))
            bound[left], bound[right] = JointSignal(key, 0.0, shared), JointSignal(key, math.pi, shared)
        else:
            bound[left], bound[right] = JointSignal(left, 0.0, _signal(column[left])), JointSignal(right, 0.0, _signal(column[right]))
    paired = {joint for pair in pairs for joint in pair}
    bound.update({joint: JointSignal(joint, 0.0, _signal(column[joint])) for joint in joints if joint not in paired})
    return {joint: js for joint, js in bound.items() if js.signal != Signal(0.0)}, symmetric


def fit_clip(
    frames: Sequence[Mapping[str, float]],
    fps: float = DEFAULT_FPS,
    *,
    name: str,
    harmonics: int = DEFAULT_HARMONICS,
    speed: float | None = None,
    symmetry_tolerance: float = SYMMETRY_TOLERANCE,
) -> tuple[GaitProfile, FitReport]:
    """Gait profile reproducing a clip of per-frame joint angles sampled at fps, with the report of what was measured."""
    if not 1 <= harmonics <= MAX_HARMONICS:
        raise FitError(f"harmonics takes 1 to {MAX_HARMONICS}, got {harmonics}")
    if not fps > 0.0:
        raise FitError(f"fps takes a positive frame rate in Hz, got {fps}")
    if speed is not None and not speed > 0.0:
        raise FitError(f"speed takes the clip's walking speed in m/s, above 0, got {speed}")
    if not frames:
        raise FitError(f"the clip has no frames, pass {_CLIP_LAYOUT}")
    joints = [joint for joint in JOINT_NAMES if joint in frames[0]]
    missing = [hip for hip in (LEFT_HIP, RIGHT_HIP) if hip not in joints]
    if missing:
        raise FitError(f"the clip lacks {missing}, the sagittal hips set the cycle: joint names must be the bare JOINT_NAMES (no id suffix), got {sorted(frames[0])[:6]}...")
    try:
        angles = np.array([[frame[joint] for joint in joints] for frame in frames], dtype=float)
    except KeyError as e:
        raise FitError(f"a frame lacks joint {e.args[0]!r} that frame 0 carries, every frame must list the same joints") from e
    n = len(angles)
    left, right = joints.index(LEFT_HIP), joints.index(RIGHT_HIP)

    period = _period_frames(angles[:, left])
    if period <= 2.0 * harmonics:
        raise FitError(f"one cycle spans {period:.1f} frames, harmonic {harmonics} needs more than {2 * harmonics}: lower --harmonics or pass the clip's real --fps")
    rate = _refine_rate(angles[:, left], _TAU / period, harmonics)
    cycles = int(math.floor(n * rate / _TAU + 1e-6))
    if cycles < MIN_CYCLES:
        raise _short(f"the clip holds {n * rate / _TAU:.2f} cycles (period {_TAU / rate / fps:.2f} s, clip {n / fps:.2f} s)")
    theta = _cycle_phase(angles[:, [left, right]], rate, cycles, harmonics)
    used = theta - theta[0] < _TAU * cycles
    if int(used.sum()) <= 2 * (2 * harmonics + 1):
        raise FitError(f"{int(used.sum())} frames over {cycles} cycles are too few for {harmonics} harmonics: lower --harmonics or pass a longer clip")

    target = load_library()[DEFAULT_WALK].joints[LEFT_HIP].signal.harmonics[0][1]
    offsets = {0.5: _align(theta[used], angles[used, left], 0.5, harmonics, target, 0.0)}

    def coefficients(split: float) -> tuple[np.ndarray, np.ndarray]:
        offsets[split] = _align(theta[used], angles[used, left], split, harmonics, target, offsets.get(split, offsets[0.5]))
        design = _design(_warp(theta[used] + offsets[split], split), harmonics)
        coef = _solve(design, angles[used])
        return coef, angles[used] - design @ coef

    def cost(split: float) -> float:
        return float((coefficients(split)[1] ** 2).sum())

    uniform = coefficients(0.5)[0]
    heel_strike = ((_peak_phase(uniform[:, right], harmonics) - _peak_phase(uniform[:, left], harmonics)) % _TAU) / _TAU
    split_fitted = _minimize(cost, *SPLIT_SEARCH)
    split = 0.5 if abs(split_fitted - 0.5) <= SPLIT_DEADBAND else round(split_fitted, 4)
    coef = coefficients(split)[0]
    bound, symmetric = _joint_signals(joints, coef, harmonics, symmetry_tolerance)

    cadence = float(fps * rate / _TAU)
    default = CadenceLaw()
    if speed is None:
        law = CadenceLaw(base=cadence, per_speed=0.0, lo=cadence, hi=cadence)
    else:
        law = CadenceLaw(base=default.base, per_speed=(cadence - default.base) / speed, lo=min(default.lo, cadence), hi=max(default.hi, cadence))
    profile = GaitProfile(name=name, joints=bound, symmetric=symmetric, phase_warp=PhaseWarp(split), cadence=law, gain=GainLaw(lo=1.0, hi=1.0, factor=1.0))
    cycle = [profile.evaluate(_TAU * step / 720.0, 1.0) for step in range(720)]
    limits: dict[str, tuple[float, float]] = {}
    for i, joint in enumerate(joints):
        lo, hi = LIMITS[JOINT_NAMES.index(joint)]
        reach = [pose.get(joint, 0.0) for pose in cycle]
        low, high = min(float(angles[:, i].min()), *reach), max(float(angles[:, i].max()), *reach)
        if low < lo or high > hi:
            limits[joint] = (min(lo, low), max(hi, high))
    profile = attrs.evolve(profile, limits=limits)

    fitted = np.array([[pose.get(joint, 0.0) for joint in joints] for pose in (profile.evaluate(float(phi), 1.0) for phi in theta + offsets[split])])
    rms = {joint: float(math.sqrt(float(((fitted[:, i] - angles[:, i]) ** 2).mean()))) for i, joint in enumerate(joints)}
    worst = max(rms, key=lambda joint: rms[joint])
    report = FitReport(
        name=name,
        fps=float(fps),
        frames=n,
        frames_used=int(used.sum()),
        cycles=cycles,
        period_s=1.0 / cadence,
        cadence_hz=cadence,
        cadence_law="fixed" if speed is None else "through_speed",
        speed=speed,
        harmonics=harmonics,
        symmetric=symmetric,
        split=split,
        split_fitted=split_fitted,
        heel_strike_fraction=float(heel_strike),
        rms=rms,
        worst_joint=worst,
        worst_rms=rms[worst],
        limits_widened=tuple(limits),
    )
    return profile, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m task_generator.simulators.human.fit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Fit a walking clip to a gait profile for the pose profile library.\n\n"
            "Finds the walking cycle from the sagittal hips, fits a mean plus harmonics per joint over whole cycles,\n"
            "decides whether left and right are one signal half a cycle apart, and measures the cadence and the\n"
            "stance split. Writes the profile yaml and prints a fit report (rms error per joint, cycles, cadence,\n"
            "split, symmetry, worst joint)."
        ),
        epilog=(f"example:\n  {_EXAMPLE}\n\nnext: move PROFILE.yaml into task_generator/simulators/human/profiles/ (the library looks profiles up by\nfile stem) and name it from an agent type yaml as pose: {{walk: {{base: <stem>}}}}."),
    )
    parser.add_argument("clip", type=Path, metavar="CLIP.npy", help=f"walking clip of semantic joint angles: {_CLIP_LAYOUT}. Needs at least {MIN_CYCLES} whole cycles of steady walking")
    parser.add_argument("-o", "--output", type=Path, required=True, metavar="PROFILE.yaml", help="where the library profile yaml is written, in the form of profiles/walk_cmu_12_01.yaml")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, metavar="F", help=f"frame rate of the clip in Hz (default {DEFAULT_FPS}, the rate AnimationManager plays the shipped clips at). It sets the measured cadence")
    parser.add_argument("--harmonics", type=int, default=DEFAULT_HARMONICS, metavar="N", help=f"harmonics fitted per joint, 1 to {MAX_HARMONICS} (default {DEFAULT_HARMONICS}). More follow the clip closer and need more frames per cycle")
    parser.add_argument("--speed", type=float, metavar="V", help="walking speed of the clip in m/s. Given: the cadence law keeps the default base and takes per_speed through (V, measured cadence). Omitted: the law is fixed at the measured cadence")
    parser.add_argument("--report", type=Path, metavar="REPORT.json", help="also write the fit report as json")
    parser.add_argument("--name", metavar="NAME", help="name: field of the profile, letters, digits and underscores (default: the stem of PROFILE.yaml, which is what the library goes by)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if not arguments:
        print(f"{parser.format_usage()}example: {_EXAMPLE}\n--help explains every option and what the output is for", file=sys.stderr)
        return 2
    args = parser.parse_args(arguments)
    name = args.output.stem if args.name is None else args.name
    if not args.clip.is_file():
        parser.error(f"clip {str(args.clip)!r} does not exist, pass {_CLIP_LAYOUT}")
    if not 1 <= args.harmonics <= MAX_HARMONICS:
        parser.error(f"--harmonics takes 1 to {MAX_HARMONICS}, got {args.harmonics}")
    if not args.fps > 0.0:
        parser.error(f"--fps takes a positive frame rate in Hz, got {args.fps}")
    if args.speed is not None and not args.speed > 0.0:
        parser.error(f"--speed takes the clip's walking speed in m/s, above 0, got {args.speed} (omit it for a fixed cadence)")
    if _NAME.fullmatch(name) is None or name == IDLE:
        parser.error(f"profile name {name!r} must be letters, digits and underscores, not starting with a digit and not {IDLE!r} (the builtin idle): pass --name NAME")
    for target in (args.output, args.report):
        if target is not None and not target.resolve().parent.is_dir():
            parser.error(f"directory {str(target.resolve().parent)!r} for {target.name} does not exist, create it or pass another path")
    try:
        profile, report = fit_clip(load_clip(args.clip), args.fps, name=name, harmonics=args.harmonics, speed=args.speed)
    except FitError as e:
        print(f"fit: {e}", file=sys.stderr)
        return 1
    args.output.write_text(dump_gait_profile(profile))
    if args.report is not None:
        args.report.write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    print(report.text())
    print(f"wrote {args.output}" + (f" and {args.report}" if args.report is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
