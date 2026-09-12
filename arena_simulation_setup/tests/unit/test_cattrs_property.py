from __future__ import annotations

import math

import attrs
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from arena_simulation_setup.utils.cattrs import Parseable, Serializable, converter
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position


# Property: structure(unstructure(x)) == x for geometry types that are Parseable+Serializable


@given(
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_position_roundtrip(x, y, z):
    p = Position(x, y, z)
    raw = list(p)
    p2 = Position.parse(raw)
    assert (p2.x, p2.y, p2.z) == pytest.approx((p.x, p.y, p.z), abs=1e-9)


@given(
    st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_orientation_yaw_roundtrip(yaw):
    o = Orientation.from_yaw(yaw)
    yaw2 = o.to_yaw()
    assert yaw2 == pytest.approx(yaw, abs=1e-5)


@given(
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_pose_converter_idempotent(x, y, yaw):
    from arena_simulation_setup.utils.cattrs import converter

    p = Pose(Position(x, y, 0), Orientation.from_yaw(yaw))
    p2 = converter.structure(p, Pose)
    assert p2 is p
