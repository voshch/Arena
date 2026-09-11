"""``halt``: the ``point`` swing held at a fixed elevation on the target's bearing, azimuth-only tracking, palm pushed out at the target."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from task_generator.simulators.human.pointing import skeleton as S
from task_generator.simulators.human.pointing.contract import ROS_JOINT_ORDER, SPINE_SEGMENTS, clamp, wrap_pi, wrist_dofs
from task_generator.simulators.human.pointing.generator import direction_from_angles

from . import BODY_HEIGHT, GestureClip
from .point import PointGesture

HOLD_ELEVATION_DEG = 15.0
EXTENSION_RAD = math.radians(70.0)  # wrist extension on the hold: fingers up, palm at the target


class HaltGesture(PointGesture):
    SLOT_HAND = {"halt_l": "l", "halt_r": "r"}

    def __init__(self) -> None:
        super().__init__()
        self._body = S.Body(BODY_HEIGHT)

    def _aim(self, local: np.ndarray) -> tuple[np.ndarray, dict]:
        d = S.unit(np.asarray(local, dtype=float))
        azimuth = math.degrees(math.atan2(d[1], d[0]))
        return direction_from_angles(azimuth, HOLD_ELEVATION_DEG, deg=True), {"azimuth": azimuth, "elevation": HOLD_ELEVATION_DEG}

    def joints(self, side: str, moving: bool) -> set[str]:
        return super().joints(side, moving) | set(wrist_dofs(side))

    def start(self, local: np.ndarray, opts: dict) -> GestureClip:
        clip = super().start(local, opts)
        # one twist for the whole swing: it is only defined once the forearm has left the vertical
        twist = self._palm_twist(self._forearm(clip.frames[clip.hold_end], clip.side, opts), clip.side)
        return self._palm_out(clip, opts, lambda _fore: twist, ramp=True)

    def retarget(self, hold: object, local: np.ndarray, opts: dict) -> GestureClip:
        clip = super().retarget(hold, local, opts)
        side = clip.side
        prev = [self._palm_twist(self._forearm(clip.frames[0], side, opts), side)]

        def track(fore: np.ndarray) -> float:
            # the arm stays raised through the transition, so the palm follows it frame by frame
            prev[0] += float(wrap_pi(self._palm_twist(fore, side) - prev[0]))
            return prev[0]

        return self._palm_out(clip, opts, track, ramp=False)

    def _forearm(self, frame: dict, side: str, opts: dict) -> np.ndarray:
        """World forearm frame of a (possibly partial) clip frame, against an upright torso when the torso is not blended."""
        pose = {**dict.fromkeys(ROS_JOINT_ORDER, 0.0), **frame["angles"]}
        if opts.get("moving", False):
            for seg in SPINE_SEGMENTS:
                pose[f"r_{seg}"] = pose[f"y_{seg}"] = pose[seg] = 0.0
        _, frames = S.fk(pose, self._body)
        return frames[f"{side}_forearm"]

    @staticmethod
    def _palm_twist(fore: np.ndarray, side: str) -> float:
        """Pronation that turns the back of the hand up, so the extension lifts the fingers and faces the palm down the aim."""
        axis = fore @ S.WRIST_AXES[side][0]
        dorsal = fore @ np.array([0.0, S.REFLECT[side], 0.0])  # back of the hand at zero pronation
        up = S.perp_component(fore @ S.DOWN, S.UP)
        return math.atan2(float(axis @ np.cross(dorsal, up)), float(dorsal @ up))

    def _palm_out(self, clip: GestureClip, opts: dict, twist: Callable[[np.ndarray], float], *, ramp: bool) -> GestureClip:
        """Wrist DOFs on the clip and its release arc. ``ramp``: the clip rises from the hanging arm, else it starts on a held palm."""
        side = clip.side
        tw, ext = wrist_dofs(side)
        park = self._forearm(clip.frames[clip.hold_end], side, opts)
        park_twist = self._palm_twist(park, side)

        def palm(frames: Sequence[dict], twist: Callable[[np.ndarray], float], rest: dict | None) -> list[dict]:
            """The palm opens with the forearm's rise from ``rest`` (the arc's hanging end) to the park pose, fully open without one."""
            lo, hi = (self._rise(self._forearm(rest, side, opts)), self._rise(park)) if rest is not None else (0.0, 0.0)
            out = []
            for frame in frames:
                fore = self._forearm(frame, side, opts)
                w = min(1.0, max(0.0, (self._rise(fore) - lo) / (hi - lo))) if hi - lo > 1e-6 else 1.0
                angles = dict(frame["angles"])
                angles[tw] = clamp(tw, w * twist(fore))
                angles[ext] = clamp(ext, w * EXTENSION_RAD)
                out.append({**frame, "angles": angles})
            return out

        frames = palm(clip.frames, twist, clip.frames[0] if ramp else None)
        release = None if clip.release is None else palm(clip.release, lambda _fore: park_twist, clip.release[-1])
        return GestureClip(frames=frames, fps=clip.fps, hold_start=clip.hold_start, hold_end=clip.hold_end, side=side, hold=clip.hold, report=clip.report, release=release, joints=clip.joints, loop=clip.loop)

    @staticmethod
    def _rise(fore: np.ndarray) -> float:
        """Angle of the forearm off hanging straight down."""
        return math.acos(float(np.clip(S.DOWN @ (fore @ S.DOWN), -1.0, 1.0)))
