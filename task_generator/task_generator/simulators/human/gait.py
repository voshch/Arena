"""Gait synthesis for pedestrian skeleton animation.

Emits semantic joint angles per the wire contract in JOINTS.md: values match the
ros4hri human_description URDF axes for every joint except the shoulder triples, which
are anatomical (sagittal flexion, antiphase baked into the emitted values).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

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

# Advisory generator-side joint limits: (lo, hi) in radians, ordered to match JOINT_NAMES.
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
)

# Walk-cycle joint profiles baked from the polished CMU 12_01 clip (the
# arena_humans posture pipeline output): mean + 3 sine harmonics per signal,
# radians, value = gain * (mean + sum amp_k * sin(k*(phi + shift) + phase_k)).
# Limb pairs share one canonical profile evaluated half a cycle apart, so L/R
# antiphase is exact by construction; reserved joints stay 0 per JOINTS.md.
_WALK_PROFILE: dict[str, tuple[float, tuple[tuple[float, float], ...]]] = {
    "hip": (+0.1875, ((0.4535, +0.0041), (0.1231, +0.7792), (0.0525, +1.6039))),
    "knee": (-0.5526, ((0.3626, -2.2253), (0.3583, -1.7656), (0.1199, -1.6237))),
    "p_shoulder": (+0.0250, ((0.2977, +3.1387), (0.0146, -2.8025), (0.0016, +3.0495))),
    "elbow": (+0.3183, ((0.2007, +3.1396), (0.0085, -1.5596), (0.0069, +3.1002))),
    "waist": (-0.0003, ((0.0027, -0.3283), (0.0118, -2.6790), (0.0009, +0.7895))),
    "r_head": (+0.0024, ((0.0613, +0.5242), (0.0050, -1.3198), (0.0192, +2.4158))),
    "y_head": (+0.0018, ((0.0443, -1.1709), (0.0074, +0.3561), (0.0042, +2.6086))),
    "p_head": (-0.0007, ((0.0097, +1.5574), (0.0207, -1.9386), (0.0019, +0.0378))),
}

_PROFILE_SIDES: tuple[tuple[str, str, float], ...] = (
    ("l_r_hip", "hip", 0.0),
    ("r_r_hip", "hip", math.pi),
    ("l_knee", "knee", 0.0),
    ("r_knee", "knee", math.pi),
    ("l_p_shoulder", "p_shoulder", 0.0),
    ("r_p_shoulder", "p_shoulder", math.pi),
    ("l_elbow", "elbow", 0.0),
    ("r_elbow", "elbow", math.pi),
    ("waist", "waist", 0.0),
    ("r_head", "r_head", 0.0),
    ("y_head", "y_head", 0.0),
    ("p_head", "p_head", 0.0),
)


def _profile_angles(phi: float, gain: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for joint, key, shift in _PROFILE_SIDES:
        mean, harmonics = _WALK_PROFILE[key]
        v = mean
        for k, (amp, ph) in enumerate(harmonics, start=1):
            v += amp * math.sin(k * (phi + shift) + ph)
        out[joint] = gain * v
    return out


class GaitGenerator:
    """Deterministic per-agent gait synthesis emitting semantic joint angles per the JOINTS.md wire contract."""

    JOINT_NAMES: tuple[str, ...] = (
        "r_waist",
        "y_waist",
        "waist",
        "r_spine",
        "y_spine",
        "spine",
        "r_chest",
        "y_chest",
        "chest",
        "r_head",
        "y_head",
        "p_head",
        "l_y_collar",
        "l_p_collar",
        "l_y_shoulder",
        "l_p_shoulder",
        "l_r_shoulder",
        "l_elbow",
        "r_y_collar",
        "r_p_collar",
        "r_y_shoulder",
        "r_p_shoulder",
        "r_r_shoulder",
        "r_elbow",
        "l_y_hip",
        "l_p_hip",
        "l_r_hip",
        "l_knee",
        "r_y_hip",
        "r_p_hip",
        "r_r_hip",
        "r_knee",
        "l_y_ankle",
        "l_ankle",
        "r_y_ankle",
        "r_ankle",
    )

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
    ) -> dict[str, float]:
        """Return base-joint-name -> angle for all 36 joints, clamped to limits.

        Phase advances by dt each call and is keyed per agent_id.
        animation_state: int matching Pedestrian.msg constants (IDLE=0, WALKING=1, RUNNING=2).
        """
        angles: dict[str, float] = {name: 0.0 for name in self.JOINT_NAMES}

        if animation_state == _WALKING:
            angles = self._gait_walk(agent_id, speed, dt)
        elif animation_state == _RUNNING:
            angles = self._gait_run(agent_id, speed, dt)
        else:
            angles = self._gait_idle(agent_id, dt)

        return {name: _clamp(angles.get(name, 0.0), LIMITS[i][0], LIMITS[i][1]) for i, name in enumerate(self.JOINT_NAMES)}

    def _gait_walk(self, agent_id: int, speed: float, dt: float) -> dict[str, float]:
        speed_abs = abs(speed)
        cadence = _clamp(0.4 + 0.55 * speed_abs, 0.4, 2.2)
        phi = self._get_phase(agent_id)
        phi += math.copysign(2.0 * math.pi * cadence * dt, speed)
        self._set_phase(agent_id, phi)

        g = _clamp(speed_abs / 1.2, 0.2, 1.0)
        angles = {name: 0.0 for name in self.JOINT_NAMES}
        angles.update(_profile_angles(phi, g))
        return angles

    def _gait_run(self, agent_id: int, speed: float, dt: float) -> dict[str, float]:
        speed_abs = abs(speed)
        cadence = _clamp(0.4 + 0.55 * speed_abs, 0.4, 2.2)
        phi = self._get_phase(agent_id)
        phi += math.copysign(2.0 * math.pi * cadence * dt, speed)
        self._set_phase(agent_id, phi)

        g = _clamp(speed_abs / 1.2, 0.2, 1.0)
        angles = {name: 0.0 for name in self.JOINT_NAMES}
        angles.update(_profile_angles(phi, 1.6 * g))
        return angles

    def _gait_idle(self, agent_id: int, dt: float) -> dict[str, float]:
        phi = self._get_phase(agent_id)
        phi += 2.0 * math.pi * 0.25 * dt
        self._set_phase(agent_id, phi)

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
