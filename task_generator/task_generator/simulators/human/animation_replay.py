"""
AnimationReplayer: a GaitGenerator-compatible driver that, instead of
procedurally synthesizing idle/walk/run, plays back one fixed text2motion
clip -- loaded once from a .npy file -- and applies it to
every pedestrian agent BaseHumanSimulator drives.

DROP-IN CONTRACT with GaitGenerator (base.py only calls these four):
  - compute(agent_id, animation_state, speed, dt) -> dict[JOINT_NAME, float]
  - forget(agent_id) -> None
  - phase(agent_id) -> float                      (used for ped.gait_phase)
  - joint_state(angles, stamp=...) -> sensor_msgs/JointState
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

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


class AnimationReplayer:
    """Loads one joint record and loops it for every agent.

    Each agent gets its own playhead so despawn/respawn or per-agent dt
    jitter doesn't affect others, but they all sample the same clip data.
    """

    def __init__(
        self,
        npy_path: str | Path,
        fps: float,
        *,
        loop: bool = True,
        stagger_agents: bool = True,
    ) -> None:
        """
        Args:
            npy_path: path to a joint-angles array.
            fps: playback rate of the source clip (MoMask/HumanML3D clips
                are commonly 20 fps).
            loop: if True, clip repeats forever; if False, each agent
                freezes on the last frame once it reaches clip end.
            stagger_agents: if True, agents' playheads start at different
                offsets into the clip (spread by agent_id, mirroring how
                GaitGenerator seeds phase from id) so a crowd doesn't look
                like one synchronized flash mob. Set False to make every
                agent play the clip in lockstep instead.
        """
        raw = np.load(npy_path, allow_pickle=True)

        self._frames: list[dict] = raw
        self._fps = fps
        self._n = len(self._frames)
        self._duration = self._n / fps
        self._loop = loop
        self._stagger = stagger_agents
        self._playhead: dict[int, float] = {}

    def _start_offset(self, agent_id: int) -> float:
        if not self._stagger:
            return 0.0
        return (agent_id % 360) / 360.0 * self._duration

    def _get_playhead(self, agent_id: int) -> float:
        if agent_id not in self._playhead:
            self._playhead[agent_id] = self._start_offset(agent_id)
        return self._playhead[agent_id]

    def forget(self, agent_id: int) -> None:
        """Drop a despawned agent's playhead state."""
        self._playhead.pop(agent_id, None)

    def phase(self, agent_id: int) -> float:
        """Fractional clip position as a 0..2pi angle, for ped.gait_phase."""
        t = self._get_playhead(agent_id)
        return ((t / self._duration) % 1.0) * 2.0 * math.pi

    def compute(
        self,
        agent_id: int,
        animation_state: int,  # noqa: ARG002 - dummy: canned playback ignores this
        speed: float,  # noqa: ARG002 - dummy: canned playback ignores this
        dt: float,
    ) -> dict[str, float]:
        """Advance this agent's playhead by dt and sample the clip there.

        animation_state/speed are intentionally unused -- see module
        docstring. If you want the clip to only advance while
        WALKING/RUNNING and hold frame 0 on IDLE, gate the `t = ... + dt`
        line below on animation_state instead of always advancing.
        """
        t = self._get_playhead(agent_id) + dt
        if self._loop:
            t %= self._duration
        else:
            t = min(t, self._duration - 1.0 / self._fps)
        self._playhead[agent_id] = t
        return self._sample(t)

    def _sample(self, t: float) -> dict[str, float]:
        """Linear-interpolated sample between the two nearest clip frames."""
        pos = t * self._fps
        i0 = int(math.floor(pos))
        frac = pos - i0

        if not self._loop and i0 >= self._n - 1:
            return dict(self._frames[self._n - 1]["angles"])

        i0 %= self._n
        i1 = (i0 + 1) % self._n
        a0 = self._frames[i0]["angles"]
        a1 = self._frames[i1]["angles"]
        return {name: a0[name] * (1.0 - frac) + a1[name] * frac for name in JOINT_NAMES}

    def joint_state(self, angles: dict[str, float], stamp: Time | None = None) -> JointState:
        """Same shape as GaitGenerator.joint_state: bare unsuffixed names."""
        msg = JointState()
        if stamp is not None:
            msg.header.stamp = stamp
        msg.name = list(JOINT_NAMES)
        msg.position = [angles[name] for name in JOINT_NAMES]
        return msg
