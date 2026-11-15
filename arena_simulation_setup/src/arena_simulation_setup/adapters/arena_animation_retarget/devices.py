"""Device tracks: where the assistive device an agent type holds sits in its body frame over one gait cycle, for humansim.

humansim detects a robot touching a held cane, rollator or walking frame (arena_humansim agent types, `device:`). It knows
each agent's position, heading and gait phase, not its arm, so the device's body-frame geometry is fitted here once per
agent type: the type's gait profile is posed over one cycle on the character it was solved for (`ActorPose`, the pose the
simulators draw), the grips (arena_runtime `hands.palm_frame(...).grip`, the cylinder the fingers close on) place the
device, and every coordinate is fitted as harmonics of the warped gait phase, the convention of the profile's own signals:
value = mean + sum_k amp_k * sin(k * w + phase_k), w the phase after the profile's phase warp.

  cane           a segment from `grip` (the fist's grip centre) along the grip axis to `tip`, `length - grip_at` beyond it
  rollator       a box of the object's footprint (`size`), `origin` on the floor under the middle of the two grips
  walking_frame  the same, `origin.z` its lift while it is carried forward

each with `model`, the held object's model renderers draw (WhiteCane, RollingCane, RollatorFitted, WalkerFitted).

Body frame: x forward, y left, z up from the soles, origin under the agent's position (humansim's pose), metres.

  python3 -m arena_simulation_setup.adapters.arena_animation_retarget.devices [AGENT_TYPE ...]

prints the `device:` block of each agent type, ready for humansim's config/agent_types/<type>.yaml, and the largest fit
residual to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import attrs
import numpy as np
import yaml

from .contract import load_module
from .hold import CANE, HANDS, TASKS, load_body

PROFILE = load_module(Path("humans/profile.py"), "arena_wire_profile", "ARENA_PROFILE")

SAMPLES = 360
MAX_HARMONICS = 8
TOLERANCE = 0.002  # m: the fewest harmonics that fit a coordinate this closely, or as closely as any count does plus GAIN
GAIN = 0.001  # m: a harmonic that improves the fit by less is noise of the profile's own fit, not motion
DECIMALS = 4


@attrs.frozen
class DeviceSource:
    """One agent type's device: its gait profile, the character the profile's arm was solved on and how the hands hold it."""

    kind: str
    profile: str
    body: str
    grip: str  # hold.TASKS entry whose radius and angle the hands close on
    obj: str  # the held object's model: drawn by renderers, its footprint is a rollator's or walking frame's box


SOURCES: dict[str, DeviceSource] = {
    "blind_white_cane": DeviceSource("cane", "walk_whitecane", "Common/Human/casual_female_caucasian_middleage_whitecane", "cane", "Common/Object/WhiteCane"),
    "blind_rolling_cane": DeviceSource("cane", "walk_rollingcane", "Common/Human/casual_male_asian_young_rollingcane", "rolling", "Common/Object/RollingCane"),
    "elder_rollator": DeviceSource("rollator", "walk_rollator", "Common/Human/casual_male_caucasian_old_rollator", "rollator", "Common/Object/RollatorFitted"),
    "patient_walking_frame": DeviceSource("walking_frame", "walk_walkingframe", "Hospital/Human/patient_female_asian_old", "frame", "Common/Object/WalkerFitted"),
}


def fit_track(w: np.ndarray, v: np.ndarray, tolerance: float = TOLERANCE, max_harmonics: int = MAX_HARMONICS) -> tuple[float, list[list[float]], float]:
    """(mean, [[amp, phase], ...], largest residual) of v(w) with the fewest harmonics within `tolerance`, or within GAIN of
    the best any count up to `max_harmonics` reaches when none does."""
    fits: list[tuple[float, list[list[float]], float]] = []
    for k in range(max_harmonics + 1):
        cols = [np.ones_like(w)] + [f(n * w) for n in range(1, k + 1) for f in (np.sin, np.cos)]
        coef, *_ = np.linalg.lstsq(np.stack(cols, axis=1), v, rcond=None)
        harmonics = [[round(math.hypot(s, c), DECIMALS), round(math.atan2(c, s), DECIMALS)] for s, c in zip(coef[1::2], coef[2::2], strict=True)]
        mean = round(float(coef[0]), DECIMALS) + 0.0  # + 0.0: no -0.0 in the yaml
        fits.append((mean, harmonics, float(np.max(np.abs(evaluate(w, mean, harmonics) - v)))))
    floor = min(err for *_, err in fits)
    return next(fit for fit in fits if fit[2] <= max(tolerance, floor + GAIN))


def evaluate(w: np.ndarray, mean: float, harmonics: Sequence[Sequence[float]]) -> np.ndarray:
    """mean + sum_k amp_k * sin(k * w + phase_k), the signal convention of the gait profiles and of humansim's tracks."""
    out = np.full_like(np.asarray(w, dtype=float), mean)
    for k, (amp, ph) in enumerate(harmonics, start=1):
        out += amp * np.sin(k * w + ph)
    return out


def cycle(profile: object, positions: Callable[[Mapping[str, float]], Mapping[str, np.ndarray]], samples: int = SAMPLES) -> tuple[np.ndarray, list[Mapping[str, np.ndarray]]]:
    """(warped phase, body-frame joint positions) at `samples` raw phases over one cycle, posed as the simulators pose them."""
    phi = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    warp = profile.phase_warp  # type: ignore[attr-defined]
    w = np.array([warp(p) for p in phi])
    return w, [positions(profile.evaluate(p, 1.0)) for p in phi]  # type: ignore[attr-defined]


def cane_points(poses: Sequence[Mapping[str, np.ndarray]], grip: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """(grip centres, tips) of a cane in the right fist: the tip `length - grip_at` from the grip centre, against the grip axis."""
    reach = grip["length"] - grip["grip_at"]
    frames = [HANDS.palm_frame(q, "r") for q in poses]
    centres = np.array([f.grip(grip["radius"], grip["angle"]) for f in frames])
    tips = np.array([c - reach * f.axis(grip["angle"]) for c, f in zip(centres, frames, strict=True)])  # the axis points up the cane for a negative angle
    return centres, tips


def box_origin(poses: Sequence[Mapping[str, np.ndarray]], grip: Mapping[str, float]) -> np.ndarray:
    """The device origin per pose: under the middle of the two grips, at the height the grips stand above their mean."""
    mids = np.array([np.mean([HANDS.palm_frame(q, s).grip(grip["radius"], grip["angle"]) for s in "lr"], axis=0) for q in poses])
    mids[:, 2] -= mids[:, 2].min()  # the lift of a carried frame; a pushed rollator stays on the floor
    return mids


def footprint(obj: str) -> dict[str, float]:
    """Footprint of an object model from its annotation's bounding box, the model facing -y (Arena's mobility aids): length along
    the heading, width across it, height, and the box centre's offset ahead of / left of the object origin."""
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier

    path = Path(asyncio.run(ObjectIdentifier.parse(obj).resolve_path()))
    (x0, x1), (y0, y1), (_z0, z1) = yaml.safe_load((path / "annotation.yaml").read_text())["bounding_box"]
    return {"length": round(y1 - y0, DECIMALS), "width": round(x1 - x0, DECIMALS), "height": round(z1, DECIMALS), "ahead": round(-(y0 + y1) / 2, DECIMALS) + 0.0, "left": round((x0 + x1) / 2, DECIMALS) + 0.0}


def track_block(w: np.ndarray, points: np.ndarray) -> tuple[dict[str, object], float]:
    """{x, y, z} of fitted tracks (a plain number for a constant one) and the largest residual."""
    out: dict[str, object] = {}
    worst = 0.0
    for axis, v in zip("xyz", points.T, strict=True):
        mean, harmonics, err = fit_track(w, v)
        out[axis] = mean if not harmonics else {"mean": mean, "harmonics": harmonics}
        worst = max(worst, err)
    return out, worst


def device_block(name: str) -> tuple[dict[str, object], float]:
    """The `device:` block of a humansim agent type and its largest fit residual (m)."""
    src = SOURCES[name]
    profile = PROFILE.load_library()[src.profile]
    positions, _stance, _label = load_body(src.body)
    w, poses = cycle(profile, positions)
    grip = {**CANE, **TASKS[src.grip]}
    if src.kind == "cane":
        centres, tips = cane_points(poses, grip)
        start, e0 = track_block(w, centres)
        end, e1 = track_block(w, tips)
        return {"kind": src.kind, "model": src.obj, "grip": start, "tip": end, "radius": grip["radius"]}, max(e0, e1)
    origin, err = track_block(w, box_origin(poses, grip))
    return {"kind": src.kind, "model": src.obj, "origin": origin, "size": footprint(src.obj)}, err


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m arena_simulation_setup.adapters.arena_animation_retarget.devices", description=__doc__.split("\n\n")[0])
    parser.add_argument("agent_types", nargs="*", metavar="AGENT_TYPE", help=f"agent types to fit (default: all of {', '.join(SOURCES)})")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    unknown = sorted(set(args.agent_types) - set(SOURCES))
    if unknown:
        parser.error(f"no device source for {', '.join(unknown)}, known: {', '.join(SOURCES)}")
    for name in args.agent_types or list(SOURCES):
        block, err = device_block(name)
        src = SOURCES[name]
        print(f"# {name}: profile {src.profile} posed on {src.body}, largest fit residual {1000 * err:.1f} mm", file=sys.stderr)
        print(f"# {name}\n" + yaml.safe_dump({"device": block}, default_flow_style=None, sort_keys=False, width=200))
    return 0


if __name__ == "__main__":
    sys.exit(main())
