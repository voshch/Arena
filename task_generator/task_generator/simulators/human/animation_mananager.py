from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import numpy as np

from .gait import GaitGenerator

JOINT_NAMES = GaitGenerator.JOINT_NAMES

if TYPE_CHECKING:
    import rclpy.impl.rcutils_logger
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
    fps: float = 20.0
    loop_from: int = 0  # looping clips wrap back to this frame, so an intro plays once

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


@attrs.define
class Overlay:
    """One agent's overlay: the clip object plus its blend and envelope state."""

    anim: Animation
    joints: set[str]
    weight: float
    loop: bool
    fade_in_s: float = 0.0
    fade_out_s: float = 0.0
    on_end: Callable[[int], None] | None = None
    elapsed: float = 0.0          # seconds since the overlay was set
    release_s: float = 0.0        # >0 once clear_ped_blend(fade_out_s) was called
    release_from: float = 1.0     # envelope value when the release started
    release_elapsed: float = 0.0
    ended: bool = False           # on_end already fired
    ramps: dict[str, list[float]] = attrs.field(factory=dict)  # joint -> [from, to, elapsed, duration], per-joint weight ramp

    def joint_weight(self, joint: str) -> float:
        ramp = self.ramps.get(joint)
        if ramp is None:
            return 1.0
        w0, w1, elapsed, dur = ramp
        return w1 if elapsed >= dur else w0 + (w1 - w0) * elapsed / dur

    def advance_ramps(self, dt: float) -> None:
        """Tick the per-joint ramps, joints that faded out leave the blend set."""
        for joint, ramp in list(self.ramps.items()):
            ramp[2] += dt
            if ramp[2] >= ramp[3]:
                del self.ramps[joint]
                if ramp[1] <= 0.0:
                    self.joints.discard(joint)

    @property
    def releasing(self) -> bool:
        return self.release_s > 0.0


class AnimationManager:
    """
    Manages heterogeneous pedestrian animations with support for looping,
    clamped one-shot, and overlay/fusing animations on top of base walk cycles.
    """

    # Use GaitGenerator to systhesis poses instead of replaying animation
    USE_SYNTHESIS = ["walk", "run", "idle"]

    # Automatically use GaitGenerator to synthesis poses for upper body for realistic movements
    AUTO_BLEND_SYNTHESIS = ["wave"]

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
        self._overlay_playhead: dict[tuple[int, str], float] = {}

        # State tracking (per agent)
        self._ped_anim: dict[int, str] = {}  # Which pedestrian is playing which base animation
        self._ped_blend: dict[int, dict[str, Overlay]] = {}  # per pedestrian: overlays by slot, applied in insertion order
        self._transients: dict[int, set[str]] = {}  # transient clip names per owning agent

        # Called by compute() after the base pose, before the overlay: (agent_id, gesture, dt)
        self.gesture_hook: Callable[[int, object, float], None] | None = None

        # Procedural gait generator used as high-fidelity base walking/running fallback
        self.gait_generator = GaitGenerator()

        # Pedestrian.msg IDLE/WALKING/RUNNING/PANIC/SURPRISED/CURIOUS/THREATENING
        self.state_to_animation_map: dict[int, str] = {0: "idle", 1: "walk", 2: "run", 3: "idle", 4: "idle", 5: "idle", 6: "idle"}

    def map_state_to_animation(self, state: int, anim_name: str) -> None:
        """
        Map a Pedestrian.msg animation state index to a registered animation.
        """
        self.state_to_animation_map[state] = anim_name

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

        if animation_name is None:
            animation_name = [f.name.split(".")[0] for f in self.database_path.iterdir() if f.suffix == ".npy"]

        for name in self.USE_SYNTHESIS:
            self.animations[name] = Animation(name=name, frames=[], n_frames=0, duration=0.0, loop=True, fps=self._fps)
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

            # TODO: Use annotation from dataset istead
            # For now, use hardcoded annotation
            is_loop = loop_mapping.get(name, name not in self.one_shot_animations)

            assert n_frames > 0, "Animation does not contain any frames"
            assert duration > 0.0, f"Animation duration is invalid, got: {duration}"

            self.animations[name] = Animation(name=name, frames=anim_frames, n_frames=n_frames, duration=duration, loop=is_loop, fps=self._fps)
            self.logger.info(f"Animation loaded: [{name}]: [{n_frames} frames - {duration}s at {self._fps}, loop={is_loop}]")

    def register_transient(self, name: str, frames: Sequence[dict], fps: float | None = None, loop: bool = False, owner: int | None = None, loop_from: int = 0) -> Animation:
        """Register a generated clip (list of {"angles": ...} frames) under ``name``, optionally owned by an agent."""
        fps = self._fps if fps is None else fps
        n_frames = len(frames)
        assert n_frames > 0, "Animation does not contain any frames"
        assert 0 <= loop_from < n_frames, f"loop_from {loop_from} outside the clip"
        anim = Animation(name=name, frames=list(frames), n_frames=n_frames, duration=n_frames / fps, loop=loop, fps=fps, loop_from=loop_from)
        self.animations[name] = anim
        if owner is not None:
            self._transients.setdefault(owner, set()).add(name)
        return anim

    def drop_animation(self, name: str) -> None:
        """Forget a cached or transient clip. Overlays already holding it keep playing."""
        self.animations.pop(name, None)
        for owned in self._transients.values():
            owned.discard(name)

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

    def _overlay_start(self, agent_id: int, overlay: Overlay) -> float:
        return self._start_offset(agent_id, overlay.anim) if overlay.loop and overlay.anim.loop_from == 0 else 0.0

    def _get_overlay_playhead(self, agent_id: int, slot: str, overlay: Overlay) -> float:
        key = (agent_id, slot)
        if key not in self._overlay_playhead:
            self._overlay_playhead[key] = self._overlay_start(agent_id, overlay)
        return self._overlay_playhead[key]

    def forget(self, agent_id: int) -> None:
        """Drop a despawned agent's state completely, transients it owns included."""
        self._playhead.pop(agent_id, None)
        for key in [k for k in self._overlay_playhead if k[0] == agent_id]:
            del self._overlay_playhead[key]
        self._ped_anim.pop(agent_id, None)
        self._ped_blend.pop(agent_id, None)
        for name in self._transients.pop(agent_id, set()):
            self.animations.pop(name, None)
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
        overlay: Animation | str | None,
        blend_joints: list[str] | set[str] | None = None,
        blend_weight: float = 1.0,
        loop: bool | None = None,
        fade_in_s: float = 0.0,
        fade_out_s: float = 0.0,
        on_end: Callable[[int], None] | None = None,
        slot: str = "arm",
        carry_ramps: bool = False,
    ) -> None:
        """
        Register an overlay animation that blends over specific joints (e.g. waving arms).

        Args:
            agent_id: Target agent's ID.
            overlay: Animation (or loaded animation name) to overlay, or None to clear it.
                     The same object is a no-op, a different object (even same name) resets
                     the playhead.
            blend_joints: Set/List of JOINT_NAMES to blend. Defaults to both arms.
            blend_weight: Interpolation factor (0.0 = base gait only, 1.0 = overlay only).
            loop: Explicit loop override for overlay. Defaults to overlay's loop flag.
            fade_in_s: Envelope ramps 0->1 over this long from the overlay start.
            fade_out_s: Non-looping overlays ramp 1->0 over their last fade_out_s.
            on_end: Called once with agent_id when a non-looping overlay reaches its end or a fade-out completes.
            slot: Overlay slot. Slots are independent (own playhead, envelope, on_end) and applied in insertion order.
            carry_ramps: When replacing a live overlay in this slot, keep its blend set and per-joint ramps instead of blend_joints.
        """
        if overlay is None:
            self.clear_ped_blend(agent_id, slot=slot)
            return

        if isinstance(overlay, str):
            self.check_animations_cached()
            if overlay not in self.animations:
                raise ValueError(f"Animation '{overlay}' is not loaded.")
            overlay = self.animations[overlay]

        if blend_joints is None:
            # Default to left and right arms (collar, shoulder, elbow joints)
            blend_joints = {"l_y_collar", "l_p_collar", "l_y_shoulder", "l_p_shoulder", "l_r_shoulder", "l_elbow", "r_y_collar", "r_p_collar", "r_y_shoulder", "r_p_shoulder", "r_r_shoulder", "r_elbow"}
        else:
            blend_joints = set(blend_joints)

        assert 0.0 <= blend_weight <= 1.0

        # Ignore if overlay animation does not change
        slots = self._ped_blend.setdefault(agent_id, {})
        current = slots.get(slot)
        if current is not None and current.anim is overlay:
            return

        is_loop = loop if loop is not None else overlay.loop
        new = Overlay(anim=overlay, joints=blend_joints, weight=blend_weight, loop=is_loop, fade_in_s=fade_in_s, fade_out_s=fade_out_s, on_end=on_end)
        if carry_ramps and current is not None:
            new.joints = set(current.joints)
            new.ramps = {joint: list(ramp) for joint, ramp in current.ramps.items()}
        slots[slot] = new
        # Safe reset/initialization for the overlay playhead
        self._overlay_playhead[(agent_id, slot)] = self._overlay_start(agent_id, new)

    def set_overlay_joints(self, agent_id: int, slot: str, joints: set[str], fade_s: float = 0.0) -> None:
        """Change a live overlay's blend set in place: leaving joints ramp to 0 and entering joints ramp to 1 over fade_s, playhead and envelope untouched."""
        overlay = self._ped_blend.get(agent_id, {}).get(slot)
        if overlay is None:
            return
        joints = set(joints)
        if fade_s <= 0.0:
            overlay.joints = joints
            overlay.ramps.clear()
            return
        for joint in overlay.joints - joints:
            overlay.ramps[joint] = [overlay.joint_weight(joint), 0.0, 0.0, fade_s]
        for joint in joints - overlay.joints:
            overlay.ramps[joint] = [0.0, 1.0, 0.0, fade_s]
        for joint in joints & overlay.joints:
            ramp = overlay.ramps.get(joint)
            if ramp is not None and ramp[1] <= 0.0:
                overlay.ramps[joint] = [overlay.joint_weight(joint), 1.0, 0.0, fade_s]
        overlay.joints |= joints

    def _drop_overlay(self, agent_id: int, slot: str) -> None:
        slots = self._ped_blend.get(agent_id)
        if slots is None:
            return
        slots.pop(slot, None)
        self._overlay_playhead.pop((agent_id, slot), None)
        if not slots:
            del self._ped_blend[agent_id]

    def clear_ped_blend(self, agent_id: int, fade_out_s: float = 0.0, slot: str | None = None) -> None:
        """Drop the overlay(s) now, or ramp their envelope to 0 over fade_out_s and then drop. ``slot=None`` clears every slot."""
        slots = self._ped_blend.get(agent_id)
        if slots is None:
            return
        for name in [slot] if slot is not None else list(slots):
            overlay = slots.get(name)
            if overlay is None:
                continue
            if fade_out_s <= 0.0:
                self._drop_overlay(agent_id, name)
                continue
            if overlay.releasing:
                continue
            overlay.release_from = self._envelope(overlay, self._get_overlay_playhead(agent_id, name, overlay))
            overlay.release_s = fade_out_s
            overlay.release_elapsed = 0.0

    def _envelope(self, overlay: Overlay, playhead: float) -> float:
        """Overlay weight multiplier in [0, 1] at the given (unclamped) playhead."""
        if overlay.releasing:
            return overlay.release_from * max(0.0, 1.0 - overlay.release_elapsed / overlay.release_s)
        env = 1.0
        if overlay.fade_in_s > 0.0:
            env = min(env, overlay.elapsed / overlay.fade_in_s)
        if not overlay.loop and overlay.fade_out_s > 0.0:
            env = min(env, (overlay.anim.duration - playhead) / overlay.fade_out_s)
        return max(0.0, min(1.0, env))

    def _end_overlay(self, agent_id: int, slot: str, overlay: Overlay, drop: bool) -> None:
        if not overlay.ended:
            overlay.ended = True
            if overlay.on_end is not None:
                overlay.on_end(agent_id)
        if drop and self._ped_blend.get(agent_id, {}).get(slot) is overlay:
            self._drop_overlay(agent_id, slot)

    @staticmethod
    def _loop_time(anim: Animation, t: float) -> float:
        """Playhead of a looping clip: the intro before ``loop_from`` plays once, the rest wraps."""
        head = anim.loop_from / anim.fps
        if t < head:
            return t
        return head + (t - head) % (anim.duration - head)

    def compute(
        self,
        agent_id: int,
        animation_state: int,
        speed: float,
        dt: float,
        gesture: object = None,
    ) -> dict[str, float]:
        """
        Resolves joint angles by advancing playheads and performing dynamic blending.
        ``gesture`` is handed per agent to ``gesture_hook``.
        """
        self.check_animations_cached()
        assert animation_state in self.state_to_animation_map, f"Can not map `animation_state` {animation_state} to any animation"

        cur_anim = self.get_current_ped_animation(agent_id)
        cur_playhead = self._get_playhead(agent_id)

        # Set next animation
        self.set_ped_anim(agent_id, self.state_to_animation_map[animation_state])
        next_anim = self.get_current_ped_animation(agent_id)
        assert isinstance(next_anim, Animation)

        # Resolve base joint angles
        if next_anim.name in self.USE_SYNTHESIS or next_anim.name in self.AUTO_BLEND_SYNTHESIS:
            # Procedural high-fidelity GaitGenerator
            base_angles = self.gait_generator.compute(agent_id, animation_state, speed, dt)
        else:
            next_playhead = self._get_playhead(agent_id) + dt
            self._playhead[agent_id] = next_playhead
            base_angles = self._sample(cur_playhead, next_playhead, cur_anim, next_anim)

        if self.gesture_hook is not None:
            self.gesture_hook(agent_id, gesture, dt)

        # Apply overlay animation blending
        if next_anim.name in self.AUTO_BLEND_SYNTHESIS:
            self.set_ped_blend(agent_id, next_anim)

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
        slots = self._ped_blend.get(agent_id)
        if not slots:
            return base_angles
        for slot, overlay in list(slots.items()):
            self._sample_overlay_slot(agent_id, slot, overlay, base_angles, dt)
        return base_angles

    def _sample_overlay_slot(self, agent_id: int, slot: str, overlay: Overlay, base_angles: dict[str, float], dt: float) -> None:
        overlay_anim = overlay.anim
        if overlay_anim.name in self.USE_SYNTHESIS:
            raise ValueError(f"Animation overlay only support {self.USE_SYNTHESIS} animations as base animation, and the rest as overlay. Got as {overlay_anim.name} as base animation.")

        # Advance overlay playhead and envelope clocks
        raw_t = self._get_overlay_playhead(agent_id, slot, overlay) + dt
        overlay.elapsed += dt
        overlay.advance_ramps(dt)
        if overlay.releasing:
            overlay.release_elapsed += dt

        at_end = False
        if overlay.loop:
            overlay_t = self._loop_time(overlay_anim, raw_t)
        else:
            at_end = raw_t >= overlay_anim.duration
            overlay_t = min(
                raw_t,
                overlay_anim.duration - 1.0 / overlay_anim.fps,
            )

        self._overlay_playhead[(agent_id, slot)] = overlay_t

        weight = overlay.weight * self._envelope(overlay, raw_t)

        if weight > 0.0:
            # Sample overlay pose
            overlay_angles = self._sample_single_animation(
                overlay_t,
                overlay_anim,
            )
            # Apply additive overlay
            for joint in overlay.joints:
                w = weight * overlay.joint_weight(joint)
                base_angles[joint] = base_angles[joint] * (1 - w) + overlay_angles[joint] * w

        if overlay.releasing and overlay.release_elapsed >= overlay.release_s:
            self._end_overlay(agent_id, slot, overlay, drop=True)
        elif at_end:
            self._end_overlay(agent_id, slot, overlay, drop=overlay.fade_out_s > 0.0)

    def _sample_single_animation(
        self,
        playhead: float,
        anim: Animation,
    ) -> dict[str, float]:
        """Linearly interpolate a pose inside a single animation."""
        pos = playhead * anim.fps
        frame = int(math.floor(pos))
        frac = pos - frame

        if not anim.loop and frame >= anim.n_frames - 1:
            return dict(anim[-1]["angles"])

        frame %= anim.n_frames
        next_frame = frame + 1
        if next_frame >= anim.n_frames:
            next_frame = anim.loop_from

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
