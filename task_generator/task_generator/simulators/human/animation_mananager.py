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


# Default supported animations mapping to their file names
SUPPORTED_ANIMATIONS = {"gorilla": "gorilla.npy", "t-pose": "t-pose.npy", "jump": "jump.npy"}


@attrs.define
class Animation:
    name: str
    frames: list[dict]
    n_frames: int
    duration: float
    loop: bool = True

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict:
        return self.frames[index]


class AnimationManager:
    """
    Manages heterogeneous pedestrian animations with support for looping,
    clamped one-shot, and overlay/fusing animations on top of base walk cycles.
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

        # Core playhead tracking (per agent)
        self._playhead: dict[int, float] = {}
        self._overlay_playhead: dict[int, float] = {}

        # State tracking (per agent)
        self._ped_anim: dict[int, str] = {}  # Which pedestrian is playing which base animation
        self._ped_blend: dict[int, dict] = {}  # Overlay/fusing configuration per pedestrian

        # Procedural gait generator used as high-fidelity base walking/running fallback
        self.gait_generator = GaitGenerator()

        # Default one-shot animation list
        self.one_shot_animations = {"t-pose", "jump", "seated", "sit", "lie"}

        # Optional mapping from Pedestrian.msg animation_state to animation names
        self.state_to_animation_map: dict[int, str] = {0: "gorilla", 1: "jump"}

    def map_state_to_animation(self, state: int, anim_name: str) -> None:
        """
        Map a Pedestrian.msg animation state index to a registered animation.
        """
        self.state_to_animation_map[state] = anim_name

    def cache_animations(self, animation_name: list[str], loop_mapping: dict[str, bool] | None = None) -> None:
        """
        Load animations from database and cache them.

        Args:
            animation_name: List of animation names to load.
            loop_mapping: Optional map specifying whether each animation loops.
                          If omitted, defaults to loop=False for one_shot_animations, True otherwise.
        """
        if loop_mapping is None:
            loop_mapping = {}

        for name in animation_name:
            filename = SUPPORTED_ANIMATIONS.get(name, f"{name}.npy")
            path = Path(os.path.join(self.database_path, filename))
            if not path.is_file():
                path = Path(os.path.join(self.database_path, f"{name}.npy"))

            assert path.is_file(), f"Animation {name} does not appear at {str(path)}"

            anim_frames = np.load(path, allow_pickle=True)
            n_frames = len(anim_frames)
            duration = n_frames / self._fps

            # TODO: Use annotation from dataset istead
            # For now, use hardcoded annotation
            is_loop = loop_mapping.get(name, name not in self.one_shot_animations)

            self.animations[name] = Animation(name=name, frames=anim_frames, n_frames=n_frames, duration=duration, loop=is_loop)
            logger = rclpy.logging.get_logger("task_generator")
            logger.info(f"Animation loaded: [{name}]: [{n_frames} frames - {duration}s at {self._fps}, loop={is_loop}]")

    def get_current_ped_animation(self, agent_id: int) -> Animation | None:
        """
        Get the current playing base animation of given pedestrian.
        """
        anim_name = self._ped_anim.get(agent_id)
        if not anim_name:
            return None
        return self.animations.get(anim_name)

    def _start_offset(self, agent_id: int, anim: Animation) -> float:
        """
        Stagger starts for looping animations so agents do not walk in lockstep.
        """
        return (agent_id % 360) / 360.0 * anim.duration

    def _get_playhead(self, agent_id: int) -> float:
        if agent_id not in self._playhead:
            anim = self.get_current_ped_animation(agent_id)
            if anim:
                self._playhead[agent_id] = self._start_offset(agent_id, anim) if anim.loop else 0.0
            else:
                self._playhead[agent_id] = 0.0
        return self._playhead[agent_id]

    def _get_overlay_playhead(self, agent_id: int, overlay_name: str) -> float:
        if agent_id not in self._overlay_playhead:
            anim = self.animations.get(overlay_name)
            if anim:
                self._overlay_playhead[agent_id] = self._start_offset(agent_id, anim) if anim.loop else 0.0
            else:
                self._overlay_playhead[agent_id] = 0.0
        return self._overlay_playhead[agent_id]

    def forget(self, agent_id: int) -> None:
        """Drop a despawned agent's state completely."""
        self._playhead.pop(agent_id, None)
        self._overlay_playhead.pop(agent_id, None)
        self._ped_anim.pop(agent_id, None)
        self._ped_blend.pop(agent_id, None)
        self.gait_generator.forget(agent_id)

    def phase(self, agent_id: int) -> float:
        """Fractional clip position as a 0..2pi angle, for ped.gait_phase."""
        anim = self.get_current_ped_animation(agent_id)
        if anim:
            t = self._get_playhead(agent_id)
            return ((t / anim.duration) % 1.0) * 2.0 * math.pi
        return self.gait_generator.phase(agent_id)

    def check_animations_cached(self) -> None:
        if not self.animations:
            raise ValueError(f"Animations are not loaded. Hint: Try {self.__class__.__name__}.cache_animations(<animation names>) first.")

    def set_ped_anim(self, agent_id: int, anim_name: str | None) -> None:
        """
        Set or clear the active base animation for a pedestrian.
        If transition occurs, the playhead is safely reset.
        """
        if anim_name is None:
            self._ped_anim.pop(agent_id, None)
            self._playhead.pop(agent_id, None)
            return

        self.check_animations_cached()
        if anim_name not in self.animations:
            raise ValueError(f"Animation '{anim_name}' is not loaded.")

        old_anim = self._ped_anim.get(agent_id)
        if old_anim != anim_name:
            self._ped_anim[agent_id] = anim_name
            # Reset playhead for the new animation
            anim = self.animations[anim_name]
            self._playhead[agent_id] = self._start_offset(agent_id, anim) if anim.loop else 0.0

    def set_ped_blend(
        self,
        agent_id: int,
        overlay_anim_name: str | None,
        blend_joints: list[str] | set[str] | None = None,
        blend_weight: float = 1.0,
        loop: bool | None = None,
    ) -> None:
        """
        Register an overlay animation that blends over specific joints (e.g. waving arms).

        Args:
            agent_id: Target agent's ID.
            overlay_anim_name: Loaded animation to overlay, or None to clear the overlay.
            blend_joints: Set/List of JOINT_NAMES to blend. Defaults to both arms.
            blend_weight: Interpolation factor (0.0 = base gait only, 1.0 = overlay only).
            loop: Explicit loop override for overlay. Defaults to overlay_anim's loop flag.
        """
        if overlay_anim_name is None:
            self._ped_blend.pop(agent_id, None)
            self._overlay_playhead.pop(agent_id, None)
            return

        self.check_animations_cached()
        if overlay_anim_name not in self.animations:
            raise ValueError(f"Animation '{overlay_anim_name}' is not loaded.")

        if blend_joints is None:
            # Default to left and right arms (collar, shoulder, elbow joints)
            blend_joints = {"l_y_collar", "l_p_collar", "l_y_shoulder", "l_p_shoulder", "l_r_shoulder", "l_elbow", "r_y_collar", "r_p_collar", "r_y_shoulder", "r_p_shoulder", "r_r_shoulder", "r_elbow"}
        else:
            blend_joints = set(blend_joints)

        self._ped_blend[agent_id] = {"overlay_anim": overlay_anim_name, "joints": blend_joints, "weight": blend_weight, "loop": loop}

        # Safe reset/initialization for the overlay playhead
        anim = self.animations[overlay_anim_name]
        is_loop = loop if loop is not None else anim.loop
        self._overlay_playhead[agent_id] = self._start_offset(agent_id, anim) if is_loop else 0.0

    def compute(
        self,
        agent_id: int,
        animation_state: int,
        speed: float,
        dt: float,
    ) -> dict[str, float]:
        """
        Resolves joint angles by advancing playheads and performing dynamic blending.
        """
        self.check_animations_cached()

        # --- 1. Resolve base joint angles ---
        base_anim_name = self._ped_anim.get(agent_id)

        # Check map-based automatic override if no base animation is explicitly pinned
        if not base_anim_name and animation_state in self.state_to_animation_map:
            mapped_name = self.state_to_animation_map[animation_state]
            if mapped_name in self.animations:
                base_anim_name = mapped_name

        if base_anim_name and base_anim_name in self.animations:
            anim = self.animations[base_anim_name]
            t = self._get_playhead(agent_id) + dt
            if anim.loop:
                t %= anim.duration
            else:
                t = min(t, anim.duration - 1.0 / self._fps)
            self._playhead[agent_id] = t
            base_angles = self._sample(t, anim)
        else:
            # Procedural high-fidelity GaitGenerator
            base_angles = self.gait_generator.compute(agent_id, animation_state, speed, dt)

        # --- 2. apply overlay animation blending ---
        blend_info = self._ped_blend.get(agent_id)
        if blend_info:
            overlay_name = blend_info["overlay_anim"]
            if overlay_name in self.animations:
                overlay_anim = self.animations[overlay_name]
                overlay_t = self._get_overlay_playhead(agent_id, overlay_name) + dt

                is_loop = blend_info["loop"]
                if is_loop is None:
                    is_loop = overlay_anim.loop

                if is_loop:
                    overlay_t %= overlay_anim.duration
                else:
                    overlay_t = min(overlay_t, overlay_anim.duration - 1.0 / self._fps)

                self._overlay_playhead[agent_id] = overlay_t
                overlay_angles = self._sample(overlay_t, overlay_anim)

                # Perform joint angle fusion/interpolation
                weight = blend_info["weight"]
                joints_to_blend = blend_info["joints"]
                for joint in joints_to_blend:
                    if joint in base_angles and joint in overlay_angles:
                        base_angles[joint] = (1.0 - weight) * base_angles[joint] + weight * overlay_angles[joint]

        return base_angles

    def _sample(self, t: float, anim: Animation) -> dict[str, float]:
        """Linear-interpolated sample between the two nearest clip frames."""
        pos = t * self._fps
        i0 = int(math.floor(pos))
        frac = pos - i0

        if not anim.loop and i0 >= anim.n_frames - 1:
            return dict(anim[-1]["angles"])

        i0 %= anim.n_frames
        i1 = (i0 + 1) % anim.n_frames
        a0 = anim[i0]["angles"]
        a1 = anim[i1]["angles"]
        return {name: a0[name] * (1.0 - frac) + a1[name] * frac for name in JOINT_NAMES}

    def joint_state(self, angles: dict[str, float], stamp: Time | None = None) -> JointState | None:
        """Same shape as GaitGenerator.joint_state: bare unsuffixed names."""
        if JointState is None:
            return None
        msg = JointState()
        if stamp is not None:
            msg.header.stamp = stamp
        msg.name = list(JOINT_NAMES)
        msg.position = [angles.get(name, 0.0) for name in JOINT_NAMES]
        return msg
