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
        joints = frozenset(anim.joints) if anim.joints else None
        return GestureClip(frames=list(anim.frames), fps=anim.fps, hold_start=last, hold_end=last, side="", hold=name, report={}, joints=joints, loop=anim.loop)

    def retarget(self, hold: object, local: np.ndarray, opts: dict) -> GestureClip:
        del hold
        return self.start(local, opts)
