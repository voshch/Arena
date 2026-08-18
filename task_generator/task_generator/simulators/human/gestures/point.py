"""``point``: PointAt clips retargeted to a ped-local target, gaze companion on the head slot."""

from __future__ import annotations

import math

import numpy as np

from task_generator.simulators.human.pointing import HoldPose, PointAtClip, PointAtGenerator, PointAtOptions
from task_generator.simulators.human.pointing import skeleton as S
from task_generator.simulators.human.pointing.contract import SPINE_SEGMENTS, arm_dofs

from . import BODY_HEIGHT, BREATH_AMP_RAD, RELEASE_STRETCH, GestureClip, ease_to_rest, move_time, resample

SWIVEL_SAMPLES = 24
FPS = 20.0
LIGHT_RETARGET_RAD = math.radians(20.0)  # below this sweep the swivel is kept and only the aim is re-solved
HANDS = {"auto": "auto", "left": "l", "right": "r", "l": "l", "r": "r"}


class PointGesture:
    slot = "arm"
    REACH_IN = 1.57
    REACH_OUT = 1.92

    def __init__(self) -> None:
        self._gen = PointAtGenerator(height=BODY_HEIGHT, options=PointAtOptions(swivel_samples=SWIVEL_SAMPLES))

    def joints(self, side: str, moving: bool) -> set[str]:
        joints = set(arm_dofs(side))
        if not moving:
            joints |= {f"y_{s}" for s in SPINE_SEGMENTS} | set(SPINE_SEGMENTS)
        return joints

    def reach(self, local: np.ndarray) -> float:
        return abs(math.atan2(local[1], local[0]))

    def companions(self) -> list[tuple[str, str]]:
        return [("head", "look")]

    def breathing(self, side: str) -> dict[str, float]:
        return {f"{side}_p_shoulder": BREATH_AMP_RAD, f"{side}_elbow": BREATH_AMP_RAD}

    def _options(self, opts: dict) -> PointAtOptions:
        hand = HANDS.get(str(opts.get("hand", "auto")), "auto")
        return PointAtOptions(swivel_samples=SWIVEL_SAMPLES, hand=hand)

    def start(self, local: np.ndarray, opts: dict) -> GestureClip:
        clip = self._gen.point_at(target=local, options=self._options(opts))
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
        aim = self._gen.aim_of(local)
        swept = math.acos(float(np.clip(np.dot(S.unit(hold.target_dir), aim), -1.0, 1.0)))
        transition_s = move_time(swept)
        if swept < LIGHT_RETARGET_RAD:
            clip = self._gen.retarget_light(hold, target=local, options=options, transition_s=transition_s)
        else:
            clip = self._gen.retarget(hold, target=local, options=options, transition_s=transition_s)
        new_hold = self._gen.hold_pose(clip)
        lowered = math.acos(float(np.clip(np.dot(S.DOWN, S.unit(new_hold.target_dir)), -1.0, 1.0)))
        release = ease_to_rest(new_hold.angles, int(round(move_time(lowered) * RELEASE_STRETCH * FPS)))
        return self._wrap(clip, clip.frames, clip.hold_range[0], clip.peak, new_hold, release)

    def _wrap(self, clip: PointAtClip, frames: list[dict], hold_start: int, hold_end: int, hold: HoldPose, release: list[dict]) -> GestureClip:
        return GestureClip(frames=frames, fps=FPS, hold_start=hold_start, hold_end=hold_end, side=clip.side, hold=hold, report=clip.report, release=release)
