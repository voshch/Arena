from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from arena_simulation_setup.utils.geometry import Orientation, Pose, Position, Vector3

_angle = st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False)
_coord = st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


@st.composite
def orientations(draw):
    yaw = draw(_angle)
    return Orientation.from_yaw(yaw)


@st.composite
def positions(draw):
    x = draw(_coord)
    y = draw(_coord)
    z = draw(_coord)
    return Position(x, y, z)


@st.composite
def poses(draw):
    pos = draw(positions())
    yaw = draw(_angle)
    return Pose(pos, Orientation.from_yaw(yaw))


@given(st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_from_yaw_to_yaw_roundtrip(yaw):
    o = Orientation.from_yaw(yaw)
    assert o.to_yaw() == pytest.approx(yaw, abs=1e-5)


@given(orientations(), orientations(), orientations())
@settings(max_examples=100)
def test_quaternion_mul_associativity(a, b, c):
    lhs = (a * b) * c
    rhs = a * (b * c)
    assert lhs.w == pytest.approx(rhs.w, abs=1e-5)
    assert lhs.x == pytest.approx(rhs.x, abs=1e-5)
    assert lhs.y == pytest.approx(rhs.y, abs=1e-5)
    assert lhs.z == pytest.approx(rhs.z, abs=1e-5)


@given(positions(), positions(), positions())
@settings(max_examples=100)
def test_vector3_translation_associativity(a, b, c):
    lhs = (a + b) + c
    rhs = a + (b + c)
    assert (lhs.x, lhs.y, lhs.z) == pytest.approx((rhs.x, rhs.y, rhs.z), abs=1e-5)


@given(poses())
@settings(max_examples=100)
def test_pose_parse_idempotent(p):
    from arena_simulation_setup.utils.cattrs import converter

    p2 = converter.structure(p, Pose)
    assert p2 is p
