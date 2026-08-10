from __future__ import annotations

from collections import defaultdict
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING
import attrs

import rclpy

import numpy as np

from .gait import GaitGenerator

JOINT_NAMES = GaitGenerator.JOINT_NAMES

if TYPE_CHECKING:
    from builtin_interfaces.msg import Time
    from sensor_msgs.msg import JointState
else:
    try:
        from sensor_msgs.msg import JointState
    except ImportError:
        JointState = None  # type: ignore[assignment,misc]


# SUPPORTED_ANIMATIONS = {"wave": "wave.npy", "head_nod": "head_nod.npy"}
SUPPORTED_ANIMATIONS = {"gorilla": "gorilla.npy", "t-pose": "t-pose.npy", "jump": "jump.npy"}


@attrs.define
class Animation:
    name: str
    frames: list[dict]
    n_frames: int
    duration: float

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict:
        return self.frames[index]


class AnimationManager:
    """
    Takes animation state, resolves joint angles
    """

    def __init__(
        self,
        animation_database_path: str | Path,
        fps: float = 20.0,
    ) -> None:
        """
        Args:
            animation_database_path: path to animation database.
            fps: playback rate of the source clip (MoMask/HumanML3D clips
                are commonly 20 fps).
        """
        self.database_path = Path(animation_database_path)
        assert self.database_path.is_dir(), f"Path does not exist {str(self.database_path.absolute())}"
        self.animations: dict[str, Animation] = {}

        self._fps = fps
        self._playhead: dict[int, float] = {}
        self._ped_anim: dict[int, str] = {}  # Which pedestrian is playing which animation

    def cache_animations(self, animation_name: list[str]):
        """
        Load animations
        """
        for name in animation_name:
            if name not in SUPPORTED_ANIMATIONS.keys():
                raise ValueError(f"Animation {name} is not supported.List of supported animations: {SUPPORTED_ANIMATIONS.keys()}")

            path = Path(os.path.join(self.database_path, SUPPORTED_ANIMATIONS[name]))
            assert path.is_file(), f"Animation {name} does not appear at {str(path)}"

            anim_frames = np.load(path, allow_pickle=True)
            n_frames = len(anim_frames)
            duration = n_frames / self._fps
            self.animations[name] = Animation(name=name, frames=anim_frames, n_frames=n_frames, duration=duration)
            logger = rclpy.logging.get_logger("task_generator")
            logger.info(f"Animation loaded: [{name}]: [{n_frames} frames - {duration}s at {self._fps}]")

    def get_current_ped_animation(self, agent_id: int) -> Animation:
        """
        Get the current playing animation of given pedestrian
        """
        anim_name = self._ped_anim[agent_id]
        return self.animations[anim_name]

    def _start_offset(self, agent_id: int) -> float:
        anim = self.get_current_ped_animation(agent_id)
        return (agent_id % 360) / 360.0 * anim.duration

    def _get_playhead(self, agent_id: int) -> float:
        if agent_id not in self._playhead:
            self._playhead[agent_id] = self._start_offset(agent_id)
        return self._playhead[agent_id]

    def forget(self, agent_id: int) -> None:
        """Drop a despawned agent's playhead state."""
        self._playhead.pop(agent_id, None)

    def phase(self, agent_id: int) -> float:
        """Fractional clip position as a 0..2pi angle, for ped.gait_phase."""
        anim = self.get_current_ped_animation(agent_id)
        t = self._get_playhead(agent_id)
        return ((t / anim.duration) % 1.0) * 2.0 * math.pi

    def check_animations_cached(self):
        if self.animations is None or len(self.animations.keys()) == 0:
            raise ValueError(f"Animations are not loaded. Hint: Try {self.__class__}.cache_animations(<animation names>) first.")

    def set_ped_anim(self, agent_id: int, anim: Animation):
        self._ped_anim[agent_id] = anim.name

    def compute(
        self,
        agent_id: int,
        animation_state: int,
        speed: float,  # noqa: ARG002 - dummy: canned playback ignores this
        dt: float,
    ) -> dict[str, float]:
        """Advance this agent's playhead by dt and sample the clip there."""
        self.check_animations_cached()
        # Sample for testing only
        anim_name = list(self.animations.keys())[agent_id % len(self.animations.keys())]
        self.set_ped_anim(agent_id, self.animations[anim_name])

        anim = self.get_current_ped_animation(agent_id)

        t = (self._get_playhead(agent_id) + dt) % anim.duration
        t = min(t, anim.duration - 1.0 / self._fps)
        self._playhead[agent_id] = t
        return self._sample(t, anim)

    def _sample(self, t: float, anim: Animation) -> dict[str, float]:
        """Linear-interpolated sample between the two nearest clip frames."""
        pos = t * self._fps
        i0 = int(math.floor(pos))
        frac = pos - i0

        # if i0 >= anim.n_frames - 1:
        #     return dict(anim[-1]["angles"])

        i0 %= anim.n_frames
        i1 = (i0 + 1) % anim.n_frames
        a0 = anim[i0]["angles"]
        a1 = anim[i1]["angles"]
        return {name: a0[name] * (1.0 - frac) + a1[name] * frac for name in JOINT_NAMES}

    def joint_state(self, angles: dict[str, float], stamp: Time | None = None) -> JointState:
        """Same shape as GaitGenerator.joint_state: bare unsuffixed names."""
        msg = JointState()
        if stamp is not None:
            msg.header.stamp = stamp
        msg.name = list(JOINT_NAMES)
        msg.position = [angles[name] for name in JOINT_NAMES]
        return msg
