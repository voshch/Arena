from __future__ import annotations

import math

import numpy as np
import pytest

from rviz_utils.hri.rig import semantic_to_rig

_HALF_PI = math.pi / 2


def _Rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _Rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _Raxis(axis: np.ndarray, a: float) -> np.ndarray:
    """Rodrigues' rotation formula, spec-independent of rig.py's own implementation."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(a), math.sin(a)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _semantic_rotation(y: float, p: float, r: float) -> np.ndarray:
    """JOINTS.md Section 1a: R = Rz(y) . Raxis((0,-1,0), p) . Raxis((0,0,-1), r),
    the twist intrinsic about the rest limb axis (equivalently the world-frame
    rotation about the current limb direction applied after yaw and flexion)."""
    partial = _Rz(y) @ _Raxis((0.0, -1.0, 0.0), p)
    return partial @ _Raxis((0.0, 0.0, -1.0), r)


def _left_chain(a: float, b: float, c: float) -> np.ndarray:
    return _Rz(-a) @ _Rx(b) @ _Rz(c)


def _right_chain(a: float, b: float, c: float) -> np.ndarray:
    return _Rz(a) @ _Rx(-b) @ _Rz(-c)


_SHOULDER_NAMES = (
    "l_y_shoulder",
    "l_p_shoulder",
    "l_r_shoulder",
    "r_y_shoulder",
    "r_p_shoulder",
    "r_r_shoulder",
)


def _translate_shoulders(y: float, p: float, r: float) -> dict[str, float]:
    """Feed the same wire triple through both sides."""
    positions = [y, p, r, y, p, r]
    out = semantic_to_rig(_SHOULDER_NAMES, positions)
    return dict(zip(_SHOULDER_NAMES, out, strict=True))


_P_SWEEP = np.linspace(-3.0, 3.0, 25)


@pytest.mark.parametrize("p", _P_SWEEP)
def test_sagittal_only_constants_left(p: float) -> None:
    """wire (0, p, 0) extracts to the constants (pi/2, p, pi/2)."""
    s = _translate_shoulders(0.0, p, 0.0)
    assert math.isclose(s["l_y_shoulder"], _HALF_PI, abs_tol=1e-9)
    assert math.isclose(s["l_p_shoulder"], p, abs_tol=1e-9)
    assert math.isclose(s["l_r_shoulder"], _HALF_PI, abs_tol=1e-9)


@pytest.mark.parametrize("p", _P_SWEEP)
def test_sagittal_only_constants_right(p: float) -> None:
    s = _translate_shoulders(0.0, p, 0.0)
    assert math.isclose(s["r_y_shoulder"], _HALF_PI, abs_tol=1e-9)
    assert math.isclose(s["r_p_shoulder"], p, abs_tol=1e-9)
    assert math.isclose(s["r_r_shoulder"], _HALF_PI, abs_tol=1e-9)


def _random_wire_triples(n: int, seed: int) -> list[tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    ys = rng.uniform(-3.1, 3.1, size=n)
    ps = rng.uniform(-1.0, 3.3, size=n)
    rs = rng.uniform(-1.6, 1.6, size=n)
    return list(zip(ys, ps, rs, strict=True))


@pytest.mark.parametrize("triple", _random_wire_triples(200, seed=0))
def test_round_trip_left(triple: tuple[float, float, float]) -> None:
    y, p, r = triple
    s = _translate_shoulders(y, p, r)
    target = _semantic_rotation(y, p, r)
    got = _left_chain(s["l_y_shoulder"], s["l_p_shoulder"], s["l_r_shoulder"])
    assert np.allclose(got, target, atol=1e-9)


@pytest.mark.parametrize("triple", _random_wire_triples(200, seed=0))
def test_round_trip_right(triple: tuple[float, float, float]) -> None:
    y, p, r = triple
    s = _translate_shoulders(y, p, r)
    target = _semantic_rotation(y, p, r)
    got = _right_chain(s["r_y_shoulder"], s["r_p_shoulder"], s["r_r_shoulder"])
    assert np.allclose(got, target, atol=1e-9)


def _random_yr_pairs(n: int, seed: int) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    ys = rng.uniform(-3.1, 3.1, size=n)
    rs = rng.uniform(-1.6, 1.6, size=n)
    return list(zip(ys, rs, strict=True))


@pytest.mark.parametrize("pair", _random_yr_pairs(50, seed=1))
def test_degenerate_p_zero_folds_correctly(pair: tuple[float, float]) -> None:
    """p=0 is the degenerate branch (sin(b)~0). a/c individually are ambiguous, only
    their composition through the URDF chain has to reproduce the semantic rotation."""
    y, r = pair
    s = _translate_shoulders(y, 0.0, r)
    target = _semantic_rotation(y, 0.0, r)

    assert math.isclose(s["l_y_shoulder"], _HALF_PI, abs_tol=1e-9)
    assert math.isclose(s["r_y_shoulder"], _HALF_PI, abs_tol=1e-9)

    got_l = _left_chain(s["l_y_shoulder"], s["l_p_shoulder"], s["l_r_shoulder"])
    got_r = _right_chain(s["r_y_shoulder"], s["r_p_shoulder"], s["r_r_shoulder"])
    assert np.allclose(got_l, target, atol=1e-9)
    assert np.allclose(got_r, target, atol=1e-9)


def test_non_shoulder_joints_passthrough() -> None:
    names = ["l_r_hip", "l_elbow", "waist", "r_knee", "p_head"]
    positions = [0.45, 0.8, 0.03, -0.9, 0.1]
    out = semantic_to_rig(names, positions)
    assert out == positions


def test_torso_and_ankle_passthrough() -> None:
    names = ["r_waist", "y_waist", "l_ankle", "r_ankle"]
    positions = [0.1, -0.2, 0.3, -0.15]
    out = semantic_to_rig(names, positions)
    assert out == positions


def test_mixed_sequence_preserves_order_and_length() -> None:
    names = ["waist", "l_y_shoulder", "l_p_shoulder", "l_r_shoulder", "r_waist", "l_ankle"]
    positions = [0.02, 0.0, 0.5, 0.0, 0.1, 0.3]
    out = semantic_to_rig(names, positions)
    assert len(out) == len(names)
    assert out[0] == 0.02
    assert math.isclose(out[1], _HALF_PI, abs_tol=1e-9)
    assert math.isclose(out[2], 0.5, abs_tol=1e-9)
    assert math.isclose(out[3], _HALF_PI, abs_tol=1e-9)
    assert out[4] == 0.1
    assert out[5] == 0.3
