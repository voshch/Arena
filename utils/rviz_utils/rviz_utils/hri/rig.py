"""Semantic to ros4hri URDF joint translation for pedestrian animation.

Pedestrian.joint_state carries anatomical shoulder triples (y, p, r): p is sagittal
flexion forward, same sign convention both sides; y is arm yaw about body-up; r is
twist about the limb's own long axis. They compose intrinsically as
    R = Rz(y) . Raxis((0,-1,0), p) . Raxis(d, r)
with d the image of (0,0,-1) under the first two rotations (JOINTS.md Section 1a).
The forked ros4hri human_description URDF instead reads the shoulder chain as
    left:  R = Rz(-a) . Rx(b) . Rz(c)
    right: R = Rz(a) . Rx(-b) . Rz(-c)
so this adapter composes R from the wire triple and extracts (a, b, c) per side via
closed-form ZXZ Euler decomposition.

A given R has exactly two ZXZ decompositions, (a, b, c) and (a+pi, -b, c+pi) (mod 2pi).
Extraction picks whichever has a in [0, pi), by detecting the other branch (a negative)
and folding it back with that identity. This keeps a=c=pi/2 constant with b=p
passthrough across the whole y=r=0 family, on both sides of b=0.

Degenerate branch (|sin(b)| < 1e-6, limb pointing along the twist axis): a and c are
individually ill-conditioned, only their sum or difference survives. Hold a at pi/2
and fold the rest into c.

All other joints (torso triple, hips, knees, elbows, head, ankles) pass straight
through: the fork keeps every other frame body-aligned at rest, so the wire and raw
URDF axes already coincide there.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

_HALF_PI = math.pi / 2
_DEGENERATE_EPS = 1e-6

_SHOULDER_SIDES = ("l", "r")
_SHOULDER_SUFFIXES = ("y_shoulder", "p_shoulder", "r_shoulder")


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' rotation formula about an arbitrary (non-unit) axis."""
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(theta), math.sin(theta)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _wrap(theta: float) -> float:
    """Normalize an angle into (-pi, pi]."""
    return math.atan2(math.sin(theta), math.cos(theta))


def _compose_arm_rotation(y: float, p: float, r: float) -> np.ndarray:
    """R = Rz(y) . Raxis((0,-1,0), p) . Raxis(d, r), d = that partial rotation on (0,0,-1)."""
    partial = _rot_z(y) @ _rot_axis(np.array([0.0, -1.0, 0.0]), p)
    d = partial @ np.array([0.0, 0.0, -1.0])
    return partial @ _rot_axis(d, r)


def _extract_urdf_triple(rot: np.ndarray, side: str) -> tuple[float, float, float]:
    """Closed-form ZXZ extraction of (a, b, c) for one shoulder side.

    left:  Rz(-a) Rx(b) Rz(c) = rot
    right: Rz(a) Rx(-b) Rz(-c) = rot
    """
    r00, _, r02 = rot[0]
    r10, _, r12 = rot[1]
    r20, r21, r22 = rot[2]

    sin_b_mag = math.hypot(r02, r12)

    if sin_b_mag < _DEGENERATE_EPS:
        a = _HALF_PI
        b = math.atan2(sin_b_mag, r22)
        if r22 >= 0.0:
            c = _HALF_PI + math.atan2(r10, r00) if side == "l" else _HALF_PI - math.atan2(r10, r00)
        else:
            c = math.atan2(-r10, r00) - _HALF_PI if side == "l" else math.atan2(r10, r00) - _HALF_PI
        return a, b, _wrap(c)

    b0 = math.atan2(sin_b_mag, r22)
    if side == "l":
        a0 = math.atan2(-r02, -r12)
        c0 = math.atan2(r20, r21)
    else:
        a0 = math.atan2(-r02, r12)
        c0 = math.atan2(r20, -r21)

    if a0 < 0.0:
        a, b, c = a0 + math.pi, -b0, c0 + math.pi
    else:
        a, b, c = a0, b0, c0
    return _wrap(a), b, _wrap(c)


def semantic_to_rig(names: Sequence[str], positions: Sequence[float]) -> list[float]:
    """Translate semantic joint positions into raw ros4hri URDF positions."""
    values = dict(zip(names, positions, strict=True))

    overrides: dict[str, float] = {}
    for side in _SHOULDER_SIDES:
        keys = [f"{side}_{suffix}" for suffix in _SHOULDER_SUFFIXES]
        if not all(key in values for key in keys):
            continue
        y, p, r = (values[key] for key in keys)
        rot = _compose_arm_rotation(y, p, r)
        a, b, c = _extract_urdf_triple(rot, side)
        overrides[keys[0]], overrides[keys[1]], overrides[keys[2]] = a, b, c

    return [overrides.get(name, value) for name, value in zip(names, positions, strict=True)]
