"""Canned clips on the ``body`` slot: the manager's cached animation, blended over the joints it moves."""

from __future__ import annotations

import typing

import numpy as np

from ..gait import GaitGenerator
from . import GestureClip

if typing.TYPE_CHECKING:
    from task_generator.simulators.human.animation_mananager import AnimationManager


class ClipGesture:
    def __init__(self, manager: AnimationManager) -> None:
        self._manager = manager

    def joints(self, side: str, moving: bool) -> set[str]:
        """Unannotated clips take the whole body."""
        del side, moving
        return set(GaitGenerator.JOINT_NAMES)

    def breathing(self, side: str) -> dict[str, float]:
        del side
        return {}

    def bind(self, slot: str, local: np.ndarray, opts: dict) -> dict:
        del slot, local
        return dict(opts)

    def start(self, local: np.ndarray, opts: dict) -> GestureClip:
        del local
        name = str(opts.get("clip", ""))
        anim = self._manager.animations.get(name)
        if anim is None or anim.n_frames == 0:
            raise ValueError(f"unknown clip {name!r}")
        last = anim.n_frames - 1
        # A looping clip (wave, talk_with_arm_gesture, ...) has no one-shot wind-up to protect -
        # it's already mid-cycle from frame 0, so it's safe to release as soon as the driving
        # gesture disappears (modulo the regular HOLD_MIN_S flicker debounce in GestureLayer).
        # Gating a full loop length behind hold_start (as for genuine one-shot clips) made
        # GestureLayer._release()'s min_t = hold_start/fps + HOLD_MIN_S measure from the END of
        # the whole clip, which could outlast a short interaction duration and leave the clip
        # visibly playing well after the BT step that requested it had already moved on.
        hold_start = 0 if anim.loop else last
        joints = frozenset(anim.joints) if anim.joints else None
        return GestureClip(frames=list(anim.frames), fps=anim.fps, hold_start=hold_start, hold_end=last, side="", hold=name, report={}, joints=joints, loop=anim.loop)

    def retarget(self, hold: object, local: np.ndarray, opts: dict) -> GestureClip:
        del hold
        return self.start(local, opts)
