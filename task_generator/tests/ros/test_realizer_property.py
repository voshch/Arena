from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


_floats = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)


def _make_realizer(x, y, prefix=""):
    from task_generator.manager.realizer import Realizer

    return Realizer(Realizer._Configuration(x=x, y=y, prefix=prefix))


@given(_floats, _floats, _floats, _floats)
@settings(max_examples=100)
def test_realize_position_inv_round_trip(ox, oy, px, py):
    from arena_simulation_setup.utils.geometry import Position

    r = _make_realizer(ox, oy)
    p = Position(x=px, y=py)
    fwd = r._realize_position(p)
    back = r._realize_position_inv(fwd)
    assert math.isclose(back.x, p.x, abs_tol=1e-6)
    assert math.isclose(back.y, p.y, abs_tol=1e-6)


@given(_floats, _floats, _floats, _floats)
@settings(max_examples=100)
def test_combined_offset_equivalence(ox1, oy1, ox2, oy2):
    from arena_simulation_setup.utils.geometry import Position

    r1 = _make_realizer(ox1, oy1)
    r2 = _make_realizer(ox2, oy2)
    r_combined = _make_realizer(ox1 + ox2, oy1 + oy2)
    p = Position(x=3.0, y=5.0)
    two_step = r2._realize_position(r1._realize_position(p))
    combined = r_combined._realize_position(p)
    assert math.isclose(two_step.x, combined.x, abs_tol=1e-6)
    assert math.isclose(two_step.y, combined.y, abs_tol=1e-6)


@given(_floats, _floats, _floats, _floats)
@settings(max_examples=50)
def test_pose_round_trip(ox, oy, px, py):
    from arena_simulation_setup.utils.geometry import Orientation, Pose, Position
    r = _make_realizer(ox, oy)
    p = Pose(Position(px, py), Orientation.from_yaw(0.0))
    fwd = r._realize_pose(p)
    back = r._realize_position_inv(fwd.position)
    assert math.isclose(back.x, p.position.x, abs_tol=1e-6)
    assert math.isclose(back.y, p.position.y, abs_tol=1e-6)
