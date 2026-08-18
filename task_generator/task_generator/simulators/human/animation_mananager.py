from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import numpy as np
import rclpy
import yaml

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
SUPPORTED_ANIMATIONS = {"hug": "hug.npy", "t-pose": "t-pose.npy", "jump": "jump.npy", "shake_hand": "shake_hand.npy", "sit": "sit.npy", "talk_with_arm_gesture": "talk_with_arm_gesture.npy", "wave": "wave.npy"}


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

    def __eq__(self, value: object, /) -> bool:
        # NOTE: Ensure animation name is unique
        if value is None:
            return False
        if not isinstance(value, Animation):
            raise NotImplementedError
        return self.name == value.name


class AnimationManager:
    """
    Manages heterogeneous pedestrian animations with support for looping,
    clamped one-shot, and overlay/fusing animations on top of base walk cycles.
    """

    # Use GaitGenerator to systhesis poses instead of replaying animation
    USE_SYNTHESIS = ["walk", "run", "idle"]

    # Automatically use GaitGenerator to synthesis poses for upper body for realistic movements
    AUTO_BLEND_SYNTHESIS = ["wave"]

    # Default state to animation map matching Pedestrian.msg constants
    STATE_TO_ANIMATION_MAP: dict[int, str] = {
        0: "idle",  # IDLE
        1: "walk",  # WALKING
        2: "run",  # RUNNING
        3: "idle",  # PANIC (treated as idle/synthesis fallback for now)
        4: "idle",  # SURPRISED (treated as idle/synthesis fallback for now)
        5: "idle",  # CURIOUS (treated as idle/synthesis fallback for now)
        6: "idle",  # THREATENING (treated as idle/synthesis fallback for now)
        7: "hug",
        8: "jump",
        9: "point_straight",
        10: "shake_hand",
        11: "sit",
        12: "talk_with_arm_gesture",
        13: "wave",
        14: "wave_high",
        15: "collapse_to_ground",
    }

    def __init__(
        self,
        animation_database_path: str | Path,
        logger: rclpy.impl.rcutils_logger.RcutilsLogger,
        fps: float = 20.0,
    ) -> None:
        """
        Args:
            animation_database_path: path to animation database.
            fps: playback rate of the source clip (MoMask/HumanML3D clips
                are commonly 20 fps).
        """
        self.logger = logger
        self.database_path = Path(animation_database_path)
        assert self.database_path.is_dir(), f"Path does not exist {str(self.database_path.absolute())}"
        self._fps = fps

        self.animations: dict[str, Animation] = {}
        # Default one-shot animation list
        self.one_shot_animations = {"t-pose", "jump", "seated", "sit", "lie"}

        self.cache_animations()

        # Core playhead tracking (per agent)
        self._playhead: dict[int, float] = {}
        self._overlay_playhead: dict[int, float] = {}

        # State tracking (per agent)
        self._ped_anim: dict[int, str] = {}  # Which pedestrian is playing which base animation
        self._ped_blend: dict[int, dict] = {}  # Overlay/fusing configuration per pedestrian

        # Procedural gait generator used as high-fidelity base walking/running fallback
        self.gait_generator = GaitGenerator()

        # Transition tracking per agent (for smooth interpolation between clips/synthesized states)
        self._transitions: dict[int, dict] = {}
        # Tracking the previous animation state integer per agent
        self._ped_anim_state: dict[int, int] = {}

    def map_state_to_animation(self, state: int, anim_name: str) -> None:
        """
        Map a Pedestrian.msg animation state index to a registered animation.
        """
        self.STATE_TO_ANIMATION_MAP[state] = anim_name

    def cache_animations(self, animation_name: list[str] | None = None, loop_mapping: dict[str, bool] | None = None) -> None:
        """
        Load animations from database and cache them.

        Args:
            animation_name: List of animation names to load.
            loop_mapping: Optional map specifying whether each animation loops.
                          If omitted, defaults to loop=False for one_shot_animations, True otherwise.
        """
        if loop_mapping is None:
            loop_mapping = {}

        # Attempt to load annotations from a database metadata/annotation file if present
        annotations = {}
        annotations_path = self.database_path / "annotations.yaml"
        if annotations_path.is_file():
            try:
                with open(annotations_path) as f:
                    annotations = yaml.safe_load(f) or {}
                    self.logger.info(annotations)
                self.logger.info(f"Loaded animation annotations from {annotations_path.name}")
            except Exception as e:
                self.logger.warning(f"Failed to load annotations from {annotations_path}: {e}")

        # Update one_shot_animations if defined in annotations
        if "one_shot_animations" in annotations:
            self.one_shot_animations.update(annotations["one_shot_animations"])

        if animation_name is None:
            animation_name = [f.name.split(".")[0] for f in self.database_path.iterdir() if f.suffix == ".npy"]

        for name in self.USE_SYNTHESIS:
            self.animations[name] = Animation(name=name, frames=[], n_frames=0, duration=0.0, loop=True)
            self.logger.info(f"Animation loaded (default, use systhesis): [{name}]")

        for name in animation_name:
            if name in self.animations.keys():
                self.logger.warning(f"Animation `{name}` is already cached, overiding...")

            filename = SUPPORTED_ANIMATIONS.get(name, f"{name}.npy")
            path = Path(os.path.join(self.database_path, filename))
            if not path.is_file():
                path = Path(os.path.join(self.database_path, f"{name}.npy"))

            assert path.is_file(), f"Animation {name} does not appear at {str(path)}"

            anim_frames = np.load(path, allow_pickle=True)
            n_frames = len(anim_frames)
            duration = n_frames / self._fps

            # Use annotation from dataset if available, falling back to loop_mapping arg,
            # and finally falling back to loop=False for one_shot_animations, True otherwise.
            is_loop = annotations.get("loop_mapping", {}).get(name)
            if is_loop is None:
                is_loop = loop_mapping.get(name, name not in self.one_shot_animations)

            assert n_frames > 0, "Animation does not contain any frames"
            assert duration > 0.0, f"Animation duration is invalid, got: {duration}"

            self.animations[name] = Animation(name=name, frames=anim_frames, n_frames=n_frames, duration=duration, loop=is_loop)
            self.logger.info(f"Animation loaded: [{name}]: [{n_frames} frames - {duration}s at {self._fps}, loop={is_loop}]")

    def get_current_ped_animation(self, agent_id: int) -> Animation | None:
        """
        Get the current playing base animation of given pedestrian.
        """
        anim_name = self._ped_anim.get(agent_id)
        if not anim_name:
            return None
        assert anim_name in self.animations.keys(), f"Animation `{anim_name}` was not cached"
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
        self._transitions.pop(agent_id, None)
        self._ped_anim_state.pop(agent_id, None)
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
        # Clear active base
        if anim_name is None:
            self._ped_anim.pop(agent_id, None)
            self._playhead.pop(agent_id, None)
            return

        # Check
        self.check_animations_cached()
        if anim_name not in self.animations:
            raise ValueError(f"Animation '{anim_name}' is not loaded.")

        # Set active base
        old_anim = self._ped_anim.get(agent_id)

        # Update active base if animation transition happens, else do nothing
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

        assert 0.0 <= blend_weight <= 1.0

        # Ignore if overlay animation does not change
        if agent_id in self._ped_blend.keys():
            if self._ped_blend[agent_id]["overlay_anim_name"] == overlay_anim_name:
                return
        else:
            self._ped_blend[agent_id] = {"overlay_anim_name": overlay_anim_name, "joints": blend_joints, "weight": blend_weight, "loop": loop}

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
        assert animation_state in self.STATE_TO_ANIMATION_MAP, f"Can not map `animation_state` {animation_state} to any animation"

        # Determine target/next animation name
        next_anim_name = self.STATE_TO_ANIMATION_MAP[animation_state]
        next_anim = self.animations[next_anim_name]

        # Check if animation is changing (transition)
        cur_anim_name = self._ped_anim.get(agent_id)

        # If transitioning to a new base animation
        if cur_anim_name is not None and cur_anim_name != next_anim_name:
            # Initialize or update transition tracking
            from_anim = self.animations[cur_anim_name]
            from_playhead = self._get_playhead(agent_id)

            # Record transition state
            self._transitions[agent_id] = {
                "from_anim": from_anim,
                "from_state": self._ped_anim_state.get(agent_id, 0),
                "from_playhead": from_playhead,
                "progress": 0.0,
                "duration": 0.5,  # Smooth 0.5s blend window
            }

            # Safely set the new base animation in state
            self.set_ped_anim(agent_id, next_anim_name)
        elif cur_anim_name is None:
            # No transition, just initialize
            self.set_ped_anim(agent_id, next_anim_name)

        # Track the current state integer for potential transition fallbacks
        self._ped_anim_state[agent_id] = animation_state

        # Now compute the base target angles for this step
        target_playhead = self._get_playhead(agent_id)
        if next_anim.name in self.USE_SYNTHESIS or next_anim.name in self.AUTO_BLEND_SYNTHESIS:
            target_angles = self.gait_generator.compute(agent_id, animation_state, speed, dt)
        else:
            # Advance base playhead
            next_playhead = target_playhead + dt
            self._playhead[agent_id] = next_playhead
            target_angles = self._sample_single_animation(next_playhead, next_anim)

        # Handle active transition blending
        transition = self._transitions.get(agent_id)
        if transition is not None:
            # Advance transition progress
            transition["progress"] += dt
            progress = transition["progress"]
            duration = transition["duration"]

            # Blend weight
            weight = min(progress / duration, 1.0)

            # Advance previous animation's playhead and compute its angles
            from_anim = transition["from_anim"]
            from_state = transition["from_state"]

            if from_anim.name in self.USE_SYNTHESIS or from_anim.name in self.AUTO_BLEND_SYNTHESIS:
                # Synthesized fallback
                from_angles = self.gait_generator.compute(agent_id, from_state, speed, dt)
            else:
                # Recorded animation playhead advancement
                next_from_playhead = transition["from_playhead"] + dt
                transition["from_playhead"] = next_from_playhead
                from_angles = self._sample_single_animation(next_from_playhead, from_anim)

            # Perform linear interpolation blending
            base_angles = {}
            for name in JOINT_NAMES:
                val_from = from_angles.get(name, 0.0)
                val_to = target_angles.get(name, 0.0)
                base_angles[name] = val_from * (1.0 - weight) + val_to * weight

            # If transition is finished, clear it
            if progress >= duration:
                self._transitions.pop(agent_id, None)
        else:
            base_angles = target_angles

        # Apply overlay animation blending
        if next_anim.name in self.AUTO_BLEND_SYNTHESIS:
            self.set_ped_blend(agent_id, next_anim.name)

        base_angles = self._sample_animation_overlay(
            agent_id,
            base_angles,
            dt,
        )

        return base_angles

    def _sample(self, current_playhead: float, next_playhead: float, cur_anim: Animation | None, next_anim: Animation) -> dict[str, float]:
        """Linear-interpolated sample between the two nearest clip frames."""
        if cur_anim is None:
            # assert current_playhead == next_playhead == 0, f"Current animation is None, implying this pedestrian was not assigned with any animation, hence playheads should be 0, got (current_playhead, next_playhead): ({current_playhead}, {next_playhead})."
            # pose_angles = next_anim[0]["angles"]
            # return {name: pose_angles[name] for name in JOINT_NAMES}
            return self._sample_single_animation(next_playhead, next_anim)

        cur_pos = current_playhead * self._fps
        next_pos = next_playhead * self._fps
        cur_frame_indice = int(math.floor(cur_pos))
        next_frame_indice = int(math.floor(next_pos))
        frac = next_pos - cur_frame_indice

        # Interpolate between the two nearest frames of the same animation
        if next_anim == cur_anim:
            return self._sample_single_animation(next_playhead, next_anim)

        # Transition between two animations
        # TODO: Enhace, use better blending mechanism
        # For now, use linear interpolation only
        else:
            cur_pos = current_playhead * self._fps
            next_pos = next_playhead * self._fps
            cur_frame_indice = int(math.floor(cur_pos))
            next_frame_indice = int(math.floor(next_pos))

            # Safe modulo wrapping to avoid IndexError on infinite playheads
            wrapped_cur_frame = cur_frame_indice % cur_anim.n_frames
            wrapped_next_frame = next_frame_indice % next_anim.n_frames

            if wrapped_cur_frame != len(cur_anim):
                self.logger.warning(f"Animation {next_anim.name} interupts animation {cur_anim.name}. Animation {cur_anim.name} progress: {wrapped_cur_frame}/{cur_anim.n_frames}.")
            if wrapped_next_frame != 0:
                self.logger.warning(f"Animation {next_anim.name} start playing at suspicious frame indice: {wrapped_next_frame}/{next_anim.n_frames}, instead of 0.")
            current_pose_angles = cur_anim[wrapped_cur_frame]["angles"]
            next_pose_angles = next_anim[wrapped_next_frame]["angles"]

            return {name: current_pose_angles[name] * 0.5 + next_pose_angles[name] * 0.5 for name in JOINT_NAMES}

    def _sample_animation_overlay(
        self,
        agent_id: int,
        base_angles: dict[str, float],
        dt: float,
    ) -> dict[str, float]:
        """
        Sample and apply the active overlay animation on top of the base pose if animation blending is required.

        Only joints registered in `blend_joints` are modified.

        Args:
            agent_id: Pedestrian ID.
            base_angles: Absolute joint angles produced by the base animation.
            dt: Simulation timestep.

        Returns:
            Blended absolute joint angles.
        """
        blend_info = self._ped_blend.get(agent_id)
        if blend_info is None:
            return base_angles

        overlay_name = blend_info["overlay_anim_name"]
        if overlay_name in self.USE_SYNTHESIS:
            raise ValueError(f"Animation overlay only support {self.USE_SYNTHESIS} animations as base animation, and the rest as overlay. Got as {overlay_name} as base animation.")

        assert overlay_name in self.animations.keys(), f"Animation {overlay_name} not found."
        overlay_anim = self.animations[overlay_name]

        # Advance overlay playhead
        overlay_t = self._get_overlay_playhead(agent_id, overlay_name) + dt

        is_loop = blend_info["loop"]
        if is_loop is None:
            is_loop = overlay_anim.loop

        if is_loop:
            overlay_t %= overlay_anim.duration
        else:
            overlay_t = min(
                overlay_t,
                overlay_anim.duration - 1.0 / self._fps,
            )

        self._overlay_playhead[agent_id] = overlay_t

        # Sample overlay pose
        overlay_angles = self._sample_single_animation(
            overlay_t,
            overlay_anim,
        )

        weight = blend_info["weight"]
        joints_to_blend = blend_info["joints"]

        # Apply additive overlay
        for joint in joints_to_blend:
            overlay = overlay_angles[joint]
            base = base_angles[joint]
            base_angles[joint] = base * (1 - weight) + overlay * weight

        return base_angles

    def _sample_single_animation(
        self,
        playhead: float,
        anim: Animation,
    ) -> dict[str, float]:
        """Linearly interpolate a pose inside a single animation."""
        pos = playhead * self._fps
        frame = int(math.floor(pos))
        frac = pos - frame

        if not anim.loop and frame >= anim.n_frames - 1:
            return dict(anim[-1]["angles"])

        frame %= anim.n_frames
        next_frame = (frame + 1) % anim.n_frames

        current_angles = anim[frame]["angles"]
        next_angles = anim[next_frame]["angles"]

        return {name: current_angles[name] * (1.0 - frac) + next_angles[name] * frac for name in JOINT_NAMES if name in current_angles and name in next_angles}

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
