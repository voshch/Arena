"""Gait synthesis for pedestrian skeleton animation.

Emits semantic joint angles per the wire contract in JOINTS.md: values match the
ros4hri human_description URDF axes for every joint except the shoulder triples, which
are anatomical (sagittal flexion, antiphase baked into the emitted values).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .profile import JOINT_NAMES, GaitProfile, PoseProfile, default_profile

if TYPE_CHECKING:
    from builtin_interfaces.msg import Time
    from sensor_msgs.msg import JointState
else:
    try:
        from sensor_msgs.msg import JointState
    except ImportError:
        JointState = None  # type: ignore[assignment,misc]

# Animation state constants matching Pedestrian.msg
_IDLE = 0
_WALKING = 1
_RUNNING = 2
# PANIC=3, SURPRISED=4, CURIOUS=5, THREATENING=6 -> treated as idle

WRIST_JOINTS: tuple[str, ...] = ("l_r_wrist", "l_wrist", "r_r_wrist", "r_wrist")

# Advisory generator-side joint limits: (lo, hi) in radians, ordered to match GaitGenerator.JOINT_NAMES.
LIMITS: tuple[tuple[float, float], ...] = (
    (-0.6, 0.6),  # r_waist
    (-0.8, 0.8),  # y_waist
    (-0.2, 1.0),  # waist
    (-0.3, 0.3),  # r_spine
    (-0.4, 0.4),  # y_spine
    (-0.1, 0.5),  # spine
    (-0.3, 0.3),  # r_chest
    (-0.4, 0.4),  # y_chest
    (-0.1, 0.5),  # chest
    (-1.0, 1.0),  # r_head
    (-1.4, 1.4),  # y_head
    (-1.5, 1.5),  # p_head
    (-0.5, 0.5),  # l_y_collar
    (-0.2, 0.6),  # l_p_collar
    (-3.1, 3.1),  # l_y_shoulder
    (-1.0, 3.3),  # l_p_shoulder
    (-1.6, 1.6),  # l_r_shoulder
    (0.0, 2.5),  # l_elbow
    (-0.5, 0.5),  # r_y_collar
    (-0.2, 0.6),  # r_p_collar
    (-3.1, 3.1),  # r_y_shoulder
    (-1.0, 3.3),  # r_p_shoulder
    (-1.6, 1.6),  # r_r_shoulder
    (0.0, 2.5),  # r_elbow
    (-0.1, 0.6),  # l_y_hip
    (-0.4, 3.3),  # l_p_hip
    (-0.4, 0.7),  # l_r_hip
    (-2.5, 0.0),  # l_knee
    (-0.1, 0.6),  # r_y_hip
    (-0.4, 3.3),  # r_p_hip
    (-0.4, 0.7),  # r_r_hip
    (-2.5, 0.0),  # r_knee
    (-0.6, 0.6),  # l_y_ankle
    (-0.9, 0.6),  # l_ankle
    (-0.6, 0.6),  # r_y_ankle
    (-0.9, 0.6),  # r_ankle
    (-1.4, 1.4),  # l_r_wrist
    (-1.3, 1.3),  # l_wrist
    (-1.4, 1.4),  # r_r_wrist
    (-1.3, 1.3),  # r_wrist
)


class GaitGenerator:
    """Deterministic per-agent gait synthesis emitting semantic joint angles per the JOINTS.md wire contract."""

    # the profiled joints, then the wrists: wire DOFs the gait never moves (0.0, see JOINTS.md, Wrists)
    JOINT_NAMES: tuple[str, ...] = (*JOINT_NAMES, *WRIST_JOINTS)

    def __init__(self) -> None:
        self._phase: dict[int, float] = {}

    def _get_phase(self, agent_id: int) -> float:
        if agent_id not in self._phase:
            self._phase[agent_id] = (agent_id % 360) * math.pi / 180.0
        return self._phase[agent_id]

    def _set_phase(self, agent_id: int, phi: float) -> None:
        self._phase[agent_id] = phi

    def forget(self, agent_id: int) -> None:
        """Drop accumulated phase state for a despawned agent."""
        self._phase.pop(agent_id, None)

    def phase(self, agent_id: int) -> float:
        """Return the current walk-cycle phase in radians for an agent, initializing it if unseen."""
        return self._get_phase(agent_id)

    def compute(
        self,
        agent_id: int,
        animation_state: int,
        speed: float,
        dt: float,
        *,
        phase: float | None = None,
        profile: PoseProfile | None = None,
    ) -> dict[str, float]:
        """Return base-joint-name -> angle for all 40 joints, clamped to limits.

        Phase advances by dt each call and is keyed per agent_id, unless `phase` supplies it.
        animation_state: int matching Pedestrian.msg constants (IDLE=0, WALKING=1, RUNNING=2).
        """
        if profile is None:
            profile = default_profile()
        angles: dict[str, float] = {name: 0.0 for name in self.JOINT_NAMES}

        if animation_state == _WALKING:
            gait: GaitProfile | None = profile.walk
            angles = self._gait_cycle(agent_id, speed, dt, profile.walk, phase)
        elif animation_state == _RUNNING:
            gait = profile.run
            angles = self._gait_cycle(agent_id, speed, dt, profile.run, phase)
        elif profile.idle is not None:
            gait = profile.idle
            angles = self._gait_idle_profile(agent_id, dt, profile.idle, phase)
        else:
            gait = None
            angles = self._gait_idle(agent_id, dt, phase)

        limits = gait.limits if gait is not None else {}
        return {name: _clamp(angles.get(name, 0.0), *limits.get(name, LIMITS[i])) for i, name in enumerate(self.JOINT_NAMES)}

    def _advance(self, agent_id: int, step: float, phase: float | None) -> float:
        if phase is None:
            phi = self._get_phase(agent_id) + step
        else:
            phi = phase
        self._set_phase(agent_id, phi)
        return phi

    def _gait_cycle(self, agent_id: int, speed: float, dt: float, gait: GaitProfile, phase: float | None) -> dict[str, float]:
        speed_abs = abs(speed)
        cadence = gait.cadence(speed_abs)
        phi = self._advance(agent_id, math.copysign(2.0 * math.pi * cadence * dt, speed), phase)

        angles = {name: 0.0 for name in self.JOINT_NAMES}
        angles.update(gait.evaluate(phi, gait.gain(speed_abs)))
        return angles

    def _gait_idle_profile(self, agent_id: int, dt: float, gait: GaitProfile, phase: float | None) -> dict[str, float]:
        phi = self._advance(agent_id, 2.0 * math.pi * 0.25 * dt, phase)
        angles = {name: 0.0 for name in self.JOINT_NAMES}
        angles.update(gait.evaluate(phi, 1.0))
        return angles

    def _gait_idle(self, agent_id: int, dt: float, phase: float | None) -> dict[str, float]:
        phi = self._advance(agent_id, 2.0 * math.pi * 0.25 * dt, phase)

        # breathing sway plus a slow incommensurate gaze wander
        waist = 0.03 * math.sin(phi)
        y_head = 0.06 * math.sin(0.3 * phi)
        p_head = 0.02 * math.sin(0.5 * phi + 1.0)

        return {
            "r_waist": 0.0,
            "y_waist": 0.0,
            "waist": waist,
            "r_spine": 0.0,
            "y_spine": 0.0,
            "spine": 0.0,
            "r_chest": 0.0,
            "y_chest": 0.0,
            "chest": 0.0,
            "r_head": 0.0,
            "y_head": y_head,
            "p_head": p_head,
            "l_y_collar": 0.0,
            "l_p_collar": 0.0,
            "l_y_shoulder": 0.0,
            "l_p_shoulder": 0.0,
            "l_r_shoulder": 0.0,
            "l_elbow": 0.0,
            "r_y_collar": 0.0,
            "r_p_collar": 0.0,
            "r_y_shoulder": 0.0,
            "r_p_shoulder": 0.0,
            "r_r_shoulder": 0.0,
            "r_elbow": 0.0,
            "l_y_hip": 0.0,
            "l_p_hip": 0.0,
            "l_r_hip": 0.0,
            "l_knee": 0.0,
            "r_y_hip": 0.0,
            "r_p_hip": 0.0,
            "r_r_hip": 0.0,
            "r_knee": 0.0,
            "l_y_ankle": 0.0,
            "l_ankle": 0.0,
            "r_y_ankle": 0.0,
            "r_ankle": 0.0,
        }

    def joint_state(
        self,
        angles: dict[str, float],
        stamp: Time | None = None,
    ) -> JointState:
        """Build a sensor_msgs/JointState from a compute() result with bare semantic names."""
        msg = JointState()
        if stamp is not None:
            msg.header.stamp = stamp
        msg.name = list(self.JOINT_NAMES)
        msg.position = [angles[name] for name in self.JOINT_NAMES]
        return msg


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
