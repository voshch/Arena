from __future__ import annotations

import math

import pytest

from arena_simulation_setup.utils.geometry import (
    Orientation,
    Pose,
    Position,
    PositionRadius,
    Scale,
    Vector3,
)


# ---------------------------------------------------------------------------
# Vector3 arithmetic
# ---------------------------------------------------------------------------


def test_vector3_add():
    a = Vector3(1.0, 2.0, 3.0)
    b = Vector3(4.0, 5.0, 6.0)
    r = a + b
    assert (r.x, r.y, r.z) == pytest.approx((5, 7, 9))


def test_vector3_sub():
    a = Vector3(3.0, 5.0, 7.0)
    b = Vector3(1.0, 2.0, 3.0)
    r = a - b
    assert (r.x, r.y, r.z) == pytest.approx((2, 3, 4))


def test_vector3_mul():
    v = Vector3(1.0, 2.0, 3.0)
    r = v * 2.0
    assert (r.x, r.y, r.z) == pytest.approx((2, 4, 6))


def test_vector3_rmul():
    v = Vector3(1.0, 2.0, 3.0)
    r = 3.0 * v
    assert (r.x, r.y, r.z) == pytest.approx((3, 6, 9))


def test_vector3_truediv():
    v = Vector3(2.0, 4.0, 6.0)
    r = v / 2.0
    assert (r.x, r.y, r.z) == pytest.approx((1, 2, 3))


def test_vector3_norm_euclidean():
    v = Vector3(3.0, 4.0, 0.0)
    assert v.norm() == pytest.approx(5.0)


def test_vector3_norm_zero():
    v = Vector3(0.0, 0.0, 0.0)
    assert v.norm() == pytest.approx(0.0)


def test_vector3_normalized():
    v = Vector3(3.0, 0.0, 4.0)
    n = v.normalized()
    assert math.sqrt(n.x**2 + n.y**2 + n.z**2) == pytest.approx(1.0)


def test_vector3_normalized_zero_vector():
    v = Vector3(0.0, 0.0, 0.0)
    n = v.normalized()
    assert (n.x, n.y, n.z) == pytest.approx((0, 0, 0))


def test_vector3_to_orientation_pos_x():
    v = Vector3(1.0, 0.0, 0.0)
    o = v.to_orientation()
    assert o.to_yaw() == pytest.approx(0.0, abs=1e-6)


def test_vector3_to_orientation_pos_y():
    v = Vector3(0.0, 1.0, 0.0)
    o = v.to_orientation()
    assert o.to_yaw() == pytest.approx(math.pi / 2, abs=1e-6)


def test_vector3_to_orientation_neg_x():
    v = Vector3(-1.0, 0.0, 0.0)
    o = v.to_orientation()
    assert abs(o.to_yaw()) == pytest.approx(math.pi, abs=1e-6)


def test_vector3_to_orientation_neg_y():
    v = Vector3(0.0, -1.0, 0.0)
    o = v.to_orientation()
    assert o.to_yaw() == pytest.approx(-math.pi / 2, abs=1e-6)


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


def test_position_parse_2d():
    p = Position.parse([1.0, 2.0])
    assert (p.x, p.y, p.z) == pytest.approx((1.0, 2.0, 0.0))


def test_position_parse_3d():
    p = Position.parse([1.0, 2.0, 3.0])
    assert (p.x, p.y, p.z) == pytest.approx((1.0, 2.0, 3.0))


def test_position_parse_idempotent():
    from arena_simulation_setup.utils.cattrs import converter

    orig = Position(5.0, 6.0, 7.0)
    copy = converter.structure(orig, Position)
    assert copy is orig


def test_position_parse_invalid_length():
    with pytest.raises((ValueError, TypeError)):
        Position.parse([1.0, 2.0, 3.0, 4.0])


def test_position_iter():
    p = Position(1.0, 2.0, 3.0)
    assert list(p) == pytest.approx([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


def test_orientation_parse_quaternion_list():
    o = Orientation.parse([1.0, 0.0, 0.0, 0.0])
    assert (o.w, o.x, o.y, o.z) == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_orientation_parse_euler_list():
    o = Orientation.parse([0.0, 0.0, 0.0])
    assert (o.w, o.x, o.y, o.z) == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_orientation_parse_yaw_float():
    o = Orientation.parse(math.pi / 2)
    assert o.to_yaw() == pytest.approx(math.pi / 2, abs=1e-6)


def test_orientation_parse_idempotent():
    from arena_simulation_setup.utils.cattrs import converter

    orig = Orientation.identity()
    copy = converter.structure(orig, Orientation)
    assert copy is orig


def test_orientation_parse_invalid_type():
    with pytest.raises((ValueError, TypeError)):
        Orientation.parse("invalid")


def test_orientation_identity():
    o = Orientation.identity()
    assert (o.w, o.x, o.y, o.z) == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_orientation_from_euler_to_euler_xyz():
    angles = (0.1, 0.2, 0.3)
    o = Orientation.from_euler(angles, order='xyz')
    back = o.to_euler(order='xyz')
    assert back == pytest.approx(angles, abs=1e-6)


def test_orientation_from_euler_to_euler_zyx():
    angles = (0.1, 0.2, 0.3)
    o = Orientation.from_euler(angles, order='zyx')
    back = o.to_euler(order='zyx')
    assert back == pytest.approx(angles, abs=1e-6)


def test_orientation_from_yaw_to_yaw():
    yaw = 1.23
    o = Orientation.from_yaw(yaw)
    assert o.to_yaw() == pytest.approx(yaw, abs=1e-6)


def test_orientation_mul_orientation():
    a = Orientation.from_yaw(math.pi / 4)
    b = Orientation.from_yaw(math.pi / 4)
    c = a * b
    assert c.to_yaw() == pytest.approx(math.pi / 2, abs=1e-6)


def test_orientation_mul_vector3_x_axis():
    identity = Orientation.identity()
    v = Vector3(1.0, 0.0, 0.0)
    result = identity * v
    assert (result.x, result.y, result.z) == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)


def test_orientation_mul_vector3_rotation():
    rot = Orientation.from_yaw(math.pi / 2)
    v = Vector3(1.0, 0.0, 0.0)
    result = rot * v
    assert (result.x, result.y) == pytest.approx((0.0, 1.0), abs=1e-6)


def test_orientation_mul_invalid():
    o = Orientation.identity()
    with pytest.raises((ValueError, TypeError)):
        o * "bad"


def test_orientation_iter():
    o = Orientation(1.0, 0.0, 0.0, 0.0)
    assert list(o) == pytest.approx([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------


def test_pose_parse_2d():
    p = Pose.parse([1.0, 2.0])
    assert p.position.x == pytest.approx(1.0)
    assert p.position.y == pytest.approx(2.0)
    assert p.orientation.to_yaw() == pytest.approx(0.0)


def test_pose_parse_3d_with_yaw():
    p = Pose.parse([1.0, 2.0, 1.5])
    assert p.position.x == pytest.approx(1.0)
    assert p.orientation.to_yaw() == pytest.approx(1.5, abs=1e-6)


def test_pose_parse_6tuple():
    p = Pose.parse([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    assert p.position.z == pytest.approx(3.0)


def test_pose_parse_7tuple():
    p = Pose.parse([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    assert p.position.x == pytest.approx(1.0)
    assert p.orientation.w == pytest.approx(1.0)


def test_pose_parse_nested():
    p = Pose.parse([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    assert p.position.x == pytest.approx(1.0)


def test_pose_parse_idempotent():
    from arena_simulation_setup.utils.cattrs import converter

    orig = Pose(Position(1.0, 2.0, 3.0), Orientation.identity())
    copy = converter.structure(orig, Pose)
    assert copy is orig


def test_pose_parse_invalid():
    with pytest.raises((ValueError, TypeError)):
        Pose.parse("bad")


def test_pose_to_2d():
    p = Pose.parse([3.0, 4.0, 0.5])
    x, y, yaw = p.to_2d()
    assert (x, y) == pytest.approx((3.0, 4.0))
    assert yaw == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# PositionRadius
# ---------------------------------------------------------------------------


def test_positionradius_parse_2d():
    pr = PositionRadius.parse([1.0, 2.0])
    assert pr.radius == pytest.approx(1.0)


def test_positionradius_parse_3d():
    pr = PositionRadius.parse([1.0, 2.0, 0.5])
    assert pr.x == pytest.approx(1.0)
    assert pr.y == pytest.approx(2.0)
    assert pr.z == pytest.approx(0.5)
    assert pr.radius == pytest.approx(1.0)


def test_positionradius_iter():
    pr = PositionRadius(1.0, 2.0, 0.0, 3.0)
    assert list(pr) == pytest.approx([1.0, 2.0, 0.0, 3.0])


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


def test_scale_parse_3d():
    s = Scale.parse([2.0, 3.0, 4.0])
    assert (s.x, s.y, s.z) == pytest.approx((2.0, 3.0, 4.0))


def test_scale_parse_invalid():
    with pytest.raises((ValueError, TypeError)):
        Scale.parse([1.0, 2.0])


def test_scale_parse_idempotent():
    from arena_simulation_setup.utils.cattrs import converter

    s = Scale(2.0, 3.0, 4.0)
    s2 = converter.structure(s, Scale)
    assert s2 is s
