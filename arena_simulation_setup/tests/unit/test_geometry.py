from __future__ import annotations

import math

import numpy as np
import pytest

from arena_simulation_setup.utils.geometry import (
    Facing,
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


def test_orientation_parse_yaw_int():
    o = Orientation.parse(3)
    assert o.to_yaw() == pytest.approx(3.0, abs=1e-6)


@pytest.mark.parametrize(
    ("axis", "yaw"),
    [("+x", 0.0), ("+y", math.pi / 2), ("-y", -math.pi / 2), ("xy", math.pi / 4)],
)
def test_orientation_parse_axis(axis, yaw):
    assert Orientation.parse(axis).to_yaw() == pytest.approx(yaw, abs=1e-6)


def test_orientation_parse_axis_neg_x_points_backwards():
    assert abs(Orientation.parse("-x").to_yaw()) == pytest.approx(math.pi, abs=1e-6)


def test_orientation_parse_unknown_axis_lists_axes():
    with pytest.raises(ValueError, match=r"\+x, \+y, -x, -y, xy"):
        Orientation.parse("+z")


def test_orientation_parse_rejects_bool():
    with pytest.raises(ValueError):
        Orientation.parse(True)


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


def test_orientation_xyzw_round_trip():
    q = [0.1, -0.2, 0.3, 0.9]
    o = Orientation.from_xyzw(q)
    assert list(o) == pytest.approx([0.9, 0.1, -0.2, 0.3])
    assert o.to_xyzw() == pytest.approx(q)


def test_orientation_matrix_round_trip():
    o = Orientation.from_xyzw([0.2, -0.3, 0.1, 0.9]).normalized()
    assert list(Orientation.from_matrix(o.to_matrix())) == pytest.approx(list(o), abs=1e-12)
    assert Orientation.from_yaw(math.pi / 2).to_matrix() @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)


def test_orientation_normalized():
    assert list(Orientation(2.0, 0.0, 0.0, 0.0).normalized()) == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert list(Orientation(0.0, 0.0, 0.0, 0.0).normalized()) == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_orientation_pow_turns_part_way_along_the_shorter_arc():
    turn = Orientation.from_yaw(1.0)
    assert (turn**0.5).to_yaw() == pytest.approx(0.5)
    negated = Orientation(-turn.w, -turn.x, -turn.y, -turn.z)
    assert (negated**0.5).to_yaw() == pytest.approx(0.5)
    assert list(Orientation(-1.0, 0.0, 0.0, 0.0) ** 0.5) == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert (turn**0.5 * turn**0.5).to_yaw() == pytest.approx(1.0)


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


def test_pose_parse_mapping_with_yaw_only():
    p = Pose.parse({"x": 1.0, "y": 2.0, "yaw": 1.5})
    assert (p.position.x, p.position.y, p.position.z) == pytest.approx((1.0, 2.0, 0.0))
    assert p.orientation.to_euler() == pytest.approx((0.0, 0.0, 1.5), abs=1e-6)
    assert Pose.parse({"x": 1.0, "y": 2.0, "z": 0.5}).orientation == Orientation.identity()


def test_pose_parse_mapping_with_roll_and_pitch():
    p = Pose.parse({"x": 1.0, "y": 2.0, "z": 3.0, "roll": 0.1, "pitch": 0.2, "yaw": 0.3})
    assert p == Pose.parse([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    assert Pose.parse({"x": 0.0, "y": 0.0, "pitch": 0.2}).orientation.to_euler() == pytest.approx((0.0, 0.2, 0.0), abs=1e-6)


@pytest.mark.parametrize("value", [{"x": 1.0}, {"x": 1.0, "y": 2.0, "heading": 0.3}, {"y": 2.0, "pitch": 0.3}])
def test_pose_parse_mapping_rejects_missing_and_unknown_keys(value):
    with pytest.raises(ValueError, match="Pose mapping must have keys"):
        Pose.parse(value)


def test_pose_times_pose_composes_in_the_left_frame():
    base = Pose(Position(4.0, 2.0, 0.0), Orientation.from_yaw(math.pi / 2))
    rel = Pose(Position(1.0, 0.0, 0.8), Orientation.from_euler((0.0, -math.pi / 4, 0.0)))
    out = base * rel
    assert (out.position.x, out.position.y, out.position.z) == pytest.approx((4.0, 3.0, 0.8))
    forward = out.orientation * Position(1.0, 0.0, 0.0)
    assert (forward.x, forward.y, forward.z) == pytest.approx((0.0, 2**-0.5, 2**-0.5))
    assert base * Pose() == base


def test_pose_times_point_carries_it_out_of_the_frame():
    base = Pose(Position(4.0, 2.0, 1.0), Orientation.from_yaw(math.pi / 2))
    point = base * Position(1.0, 0.0, 0.5)
    assert isinstance(point, Position)
    assert (point.x, point.y, point.z) == pytest.approx((4.0, 3.0, 1.5))


def test_pose_inverse_composes_to_the_identity_on_either_side():
    pose = Pose(Position(4.0, -2.0, 1.5), Orientation.from_euler((0.3, -0.4, 1.2)))
    inverse = pose.inverse()
    assert isinstance(inverse, Pose)
    assert isinstance(inverse.position, Position)
    assert isinstance(inverse.orientation, Orientation)
    for out in (pose * inverse, inverse * pose):
        assert tuple(out.position) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
        assert tuple(out.orientation) == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-9)


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
    assert (pr.x, pr.y, pr.z) == pytest.approx((1.0, 2.0, 0.0))
    assert pr.radius == pytest.approx(1.0)


def test_positionradius_parse_3d():
    pr = PositionRadius.parse([1.0, 2.0, 0.5])
    assert pr.x == pytest.approx(1.0)
    assert pr.y == pytest.approx(2.0)
    assert pr.z == pytest.approx(0.0)
    assert pr.radius == pytest.approx(0.5)


def test_positionradius_parse_4d():
    pr = PositionRadius.parse([1.0, 2.0, 0.3, 0.5])
    assert (pr.x, pr.y, pr.z) == pytest.approx((1.0, 2.0, 0.3))
    assert pr.radius == pytest.approx(0.5)


def test_positionradius_parse_rejects_other_lengths():
    with pytest.raises(ValueError):
        PositionRadius.parse([1.0, 2.0, 0.3, 0.5, 9.0])


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


# ---------------------------------------------------------------------------
# Facing
# ---------------------------------------------------------------------------


def test_facing_parse_single_axis():
    assert Facing.parse("-y").yaws == pytest.approx([-math.pi / 2], abs=1e-6)


def test_facing_parse_mixed_list():
    assert Facing.parse(["+x", 0, 0.7854]).yaws == pytest.approx([0.0, 0.0, 0.7854], abs=1e-6)


def test_facing_parse_rejects_empty_list():
    with pytest.raises(ValueError, match="orientations"):
        Facing.parse([])


def test_facing_structures_through_shared_converter():
    from arena_simulation_setup.utils.cattrs import converter

    assert converter.structure(["+y", "-x"], Facing).contains(math.pi)


def test_facing_serialize_unwraps_single_axis():
    assert Facing.parse("+y").serialize() == "+y"


def test_facing_serialize_names_exact_axes_and_keeps_angles():
    assert Facing.parse(["-x", math.pi / 2, 0.7854]).serialize() == ["-x", "+y", pytest.approx(0.7854)]


def test_facing_serialize_roundtrips():
    facing = Facing.parse(["xy", 1, -2.5])
    assert Facing.parse(facing.serialize()).yaws == pytest.approx(facing.yaws, abs=1e-12)


def test_facing_contains_wraps_around():
    facing = Facing.parse("-x")
    assert facing.contains(-math.pi)
    assert facing.contains(3 * math.pi)
    assert not facing.contains(0.0)


def test_facing_nearest_crosses_pi():
    facing = Facing.parse(["+x", "-x"])
    assert abs(facing.nearest(-3.0).to_yaw()) == pytest.approx(math.pi, abs=1e-6)


def test_facing_sample_is_seeded_and_covers_every_direction():
    facing = Facing.parse(["+x", "+y", "-y"])
    draws = [round(facing.sample(np.random.default_rng(seed)).to_yaw(), 4) for seed in range(64)]
    assert draws == [round(facing.sample(np.random.default_rng(seed)).to_yaw(), 4) for seed in range(64)]
    assert set(draws) == {0.0, 1.5708, -1.5708}
