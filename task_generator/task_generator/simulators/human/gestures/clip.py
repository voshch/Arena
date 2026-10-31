"""Canned clips on the ``body`` slot: the manager's cached animation, blended over the joints it moves."""

from __future__ import annotations

import typing

import numpy as np

from ..gait import GaitGenerator
from . import GestureClip

if typing.TYPE_CHECKING:
    from task_generator.simulators.human.animation_manager import AnimationManager


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
        anim = self._manager.clip(name)
        if anim is None or anim.n_frames == 0:
            raise ValueError(f"unknown clip {name!r}")
        last = anim.n_frames - 1
        # a looping clip has no wind-up to protect beyond its intro, it may release once the loop runs
        hold_start = anim.loop_from if anim.loop else last
        joints = frozenset(anim.joints) if anim.joints else None
        release = list(anim.release) if anim.release else None
        return GestureClip(frames=list(anim.frames), fps=anim.fps, hold_start=hold_start, hold_end=last, side="", hold=name, report={}, release=release, joints=joints, loop=anim.loop, reverse=anim.reverse, loop_from=anim.loop_from)

    def retarget(self, hold: object, local: np.ndarray, opts: dict) -> GestureClip:
        del hold
        return self.start(local, opts)
