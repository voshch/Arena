"""Wire-contract constants (JOINTS.md v3, 36 DOF) as the PointAt generator needs them."""

from __future__ import annotations

import numpy as np

from ..gait import LIMITS as _LIMITS
from ..gait import GaitGenerator

# publish-all order, matching GaitGenerator.JOINT_NAMES
ROS_JOINT_ORDER: list[str] = list(GaitGenerator.JOINT_NAMES)

# advisory per-joint limits (rad), JOINTS.md / GaitGenerator.LIMITS
LIMITS: dict[str, tuple[float, float]] = dict(zip(ROS_JOINT_ORDER, _LIMITS, strict=True))
assert len(ROS_JOINT_ORDER) == 36

SPINE_SEGMENTS = ("waist", "spine", "chest")

# Limits actually applied when emitting.  Identical to the advisory table except
# for the shoulder azimuth and twist: the (y, p, r) decomposition can land a
# half-turn outside the +-1.6 rad advisory window, and clamping that branch
# would put a 180-degree step into a joint every downstream velocity check
# trusts.  Widened by 0.35 rad past a half turn on both spherical DOFs, just
# enough for the continuity search to step around the wrap (convert_v2 widens
# the same twist bound for the same reason, section 7 note on SPEC_ARM bounds3).
# Reports still measure saturation against the advisory LIMITS above.
EMIT_LIMITS = dict(LIMITS)
_SPHERICAL_SLACK = np.pi + 0.35
for _s in ("l", "r"):
    EMIT_LIMITS[f"{_s}_r_shoulder"] = (-_SPHERICAL_SLACK, _SPHERICAL_SLACK)
    EMIT_LIMITS[f"{_s}_y_shoulder"] = (-_SPHERICAL_SLACK, _SPHERICAL_SLACK)


# Opt-in window for consumers that rate-limit or interpolate joint values: a
# full turn of room on the two spherical shoulder DOFs lets the representation
# follow the arm without ever wrapping, at the cost of emitting angles well
# outside the advisory table (see PointAtOptions.unwrap_shoulder).
UNWRAP_LIMITS = dict(EMIT_LIMITS)
for _s in ("l", "r"):
    UNWRAP_LIMITS[f"{_s}_r_shoulder"] = (-2 * np.pi, 2 * np.pi)
    UNWRAP_LIMITS[f"{_s}_y_shoulder"] = (-2 * np.pi, 2 * np.pi)


def arm_dofs(side: str) -> list[str]:
    """The six DOFs this generator re-solves for the pointing arm."""
    return [
        f"{side}_y_collar", f"{side}_p_collar",
        f"{side}_y_shoulder", f"{side}_p_shoulder", f"{side}_r_shoulder",
        f"{side}_elbow",
    ]


def clamp(name: str, value: float, limits: dict[str, tuple[float, float]] = EMIT_LIMITS) -> float:
    lo, hi = limits[name]
    return float(np.clip(value, lo, hi))


def saturation(name: str, value: float, eps: float = 1e-4, limits: dict[str, tuple[float, float]] = LIMITS) -> float:
    """How far past the limit a raw value sits (rad, 0.0 if inside)."""
    lo, hi = limits[name]
    if value < lo - eps:
        return float(lo - value)
    if value > hi + eps:
        return float(value - hi)
    return 0.0


def wrap_pi(a: float | np.ndarray) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def to_limits(name: str, value: float, limits: dict[str, tuple[float, float]] = EMIT_LIMITS) -> float:
    """Pick the 2*pi-equivalent representative of ``value`` that fits the joint.

    Wrap-aware blending can land a joint on an angle that is geometrically
    right but numerically a full turn outside its range (typical for the
    shoulder azimuth, which is meaningless while the arm hangs and can start
    the blend anywhere).  Clamping that directly would destroy the pose, so
    rotate it back into range first.
    """
    lo, hi = limits[name]
    best, best_pen = value, None
    for k in (-1, 0, 1):
        v = value + 2 * np.pi * k
        pen = max(0.0, lo - v) + max(0.0, v - hi)
        if best_pen is None or pen < best_pen - 1e-12:
            best, best_pen = v, pen
    return float(best)
