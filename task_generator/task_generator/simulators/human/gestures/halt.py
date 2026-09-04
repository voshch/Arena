"""``halt``: the same baked arm swing as ``point``, held at a fixed
 elevation facing the target's
bearing instead of tracking its exact height/distance -- "stop, hol
d position" rather than "point
at exactly this". Mirrors ``point.py``'s structure; see that file f
or the baked-table mechanics.

The skeleton (JOINTS.md v3) has no wrist or finger DOF, so there is
 no literal open-palm hand
shape to render -- "halt" is distinguished from "point" only by arm
 posture (fixed hold elevation,
azimuth-only tracking), not by hand shape. A visually distinct bent
-elbow "stop" silhouette would
need a dedicated reference template baked the same way as ``point_t
o_right``
(``python3 -m task_generator.simulators.human.pointing.table --temp
late <name>``); this
implementation reuses the existing point table as a first cut.
"""

from __future__ import annotations

import math

import numpy as np

from task_generator.simulators.human.pointing import HoldPose, PointAtClip, PointAtOptions
from task_generator.simulators.human.pointing import skeleton as S
from task_generator.simulators.human.pointing.contract import SPINE_SEGMENTS, arm_dofs
from task_generator.simulators.human.pointing.generator import direction_from_angles
from task_generator.simulators.human.pointing.table import BakedPointAt

from . import BODY_HEIGHT, BREATH_AMP_RAD, RELEASE_STRETCH, GestureClip, ease_to_rest, move_time, resample

FPS = 20.0
HANDS = {"auto": "auto", "left": "l", "right": "r", "l": "l", "r": "r"}
DOMINANT = {"left": "l", "right": "r", "l": "l", "r": "r"}
SLOT_HAND = {"halt_l": "l", "halt_r": "r"}
HOLD_ELEVATION_DEG = 15.0  # forward and slightly up, roughly chest/shoulder height


class HaltGesture:
    def __init__(self) -> None:
        self._gen = BakedPointAt(height=BODY_HEIGHT)

    def joints(self, side: str, moving: bool) -> set[str]:
        joints = set(arm_dofs(side))
        if not moving:
            joints |= {f"{axis}_{s}" for s in SPINE_SEGMENTS for axis in ("y", "r")} | set(SPINE_SEGMENTS)
        return joints

    def breathing(self, side: str) -> dict[str, float]:
        return {f"{side}_p_shoulder": BREATH_AMP_RAD, f"{side}_elbow": BREATH_AMP_RAD}

    def _options(self, opts: dict) -> PointAtOptions:
        hand = HANDS.get(str(opts.get("hand", "auto")), "auto")
        dominant = DOMINANT.get(str(opts.get("dominant", "r")), "r")
        return PointAtOptions(hand=hand, dominant=dominant, upright=bool(opts.get("moving", False)))

    @staticmethod
    def _azimuth_deg(local: np.ndarray) -> float:
        d = S.unit(np.asarray(local, dtype=float))
        return math.degrees(math.atan2(d[1], d[0]))

    def resolve_hand(self, local: np.ndarray, opts: dict) -> str:
        """Same rule as ``point``: dominant hand near the midline, else the target-side arm."""
        return self._gen.choose_hand(self._gen.aim_of(local), self._options({**opts, "hand": "auto"}))

    def bind(self, slot: str, local: np.ndarray, opts: dict) -> dict:
        hand = SLOT_HAND.get(slot) or self.resolve_hand(local, opts)
        return {**opts, "hand": hand}

    def start(self, local: np.ndarray, opts: dict) -> GestureClip:
        clip = self._gen.point_at(azimuth=self._azimuth_deg(local), elevation=HOLD_ELEVATION_DEG, options=self._options(opts))
        h0, peak = clip.hold_range[0], clip.peak
        swept = math.acos(float(np.clip(np.dot(S.DOWN, S.unit(clip.target_dir)), -1.0, 1.0)))
        ramp = resample(clip.frames[:h0], max(2, int(round(move_time(swept) * FPS))))
        release = resample(clip.frames[peak:], int(round(RELEASE_STRETCH * (len(clip.frames) - peak))))
        frames = ramp + list(clip.frames[h0 : peak + 1])
        hold_start = len(ramp)
        return self._wrap(clip, frames, hold_start, hold_start + (peak - h0), self._gen.hold_pose(clip), release)

    def retarget(self, hold: object, local: np.ndarray, opts: dict) -> GestureClip:
        assert isinstance(hold, HoldPose)
        options = self._options(opts)
        az = self._azimuth_deg(local)
        aim = direction_from_angles(az, HOLD_ELEVATION_DEG, deg=True)
        swept = math.acos(float(np.clip(np.dot(S.unit(hold.target_dir), aim), -1.0, 1.0)))
        clip = self._gen.retarget(hold, azimuth=az, elevation=HOLD_ELEVATION_DEG, options=options, transition_s=move_time(swept))
        new_hold = self._gen.hold_pose(clip)
        lowered = math.acos(float(np.clip(np.dot(S.DOWN, S.unit(new_hold.target_dir)), -1.0, 1.0)))
        release = ease_to_rest(new_hold.angles, int(round(move_time(lowered) * RELEASE_STRETCH * FPS)))
        return self._wrap(clip, clip.frames, clip.hold_range[0], clip.peak, new_hold, release)

    def _wrap(self, clip: PointAtClip, frames: list[dict], hold_start: int, hold_end: int, hold: HoldPose, release: list[dict]) -> GestureClip:
        return GestureClip(frames=frames, fps=FPS, hold_start=hold_start, hold_end=hold_end, side=clip.side, hold=hold, report=clip.report, release=release)
