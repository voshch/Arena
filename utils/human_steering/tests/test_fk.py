from __future__ import annotations

import math

import pytest
from human_steering import fk
from human_steering.fk import Joint, forward_kinematics, parse_urdf, preview, project_front, project_side


def _synthetic_tree() -> tuple[str, dict[str, Joint]]:
    """A tiny 3-link chain, independent of xacro/human_description: root -> torso
    (revolute about Z) -> hand (fixed offset), plus a fixed head link off the root."""
    joints = {
        "waist": Joint(name="waist", parent="root", child="torso", xyz=(0.0, 0.0, 1.0), rpy=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), kind="revolute"),
        "shoulder": Joint(name="shoulder", parent="torso", child="hand", xyz=(0.5, 0.0, 0.0), rpy=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), kind="fixed"),
        "neck": Joint(name="neck", parent="root", child="head", xyz=(0.0, 0.0, 0.3), rpy=(0.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), kind="fixed"),
    }
    return "root", joints


def test_fk_smoke_known_pose_zero_angle() -> None:
    root, joints = _synthetic_tree()
    positions = forward_kinematics(root, joints, {"waist": 0.0})
    assert positions["root"] == pytest.approx((0.0, 0.0, 0.0))
    assert positions["torso"] == pytest.approx((0.0, 0.0, 1.0))
    assert positions["hand"] == pytest.approx((0.5, 0.0, 1.0))
    assert positions["head"] == pytest.approx((0.0, 0.0, 0.3))


def test_fk_smoke_known_pose_quarter_turn() -> None:
    root, joints = _synthetic_tree()
    positions = forward_kinematics(root, joints, {"waist": math.pi / 2.0})
    # torso is at the joint origin, unaffected by its own joint's rotation.
    assert positions["torso"] == pytest.approx((0.0, 0.0, 1.0))
    # hand's local (0.5, 0, 0) offset rotates with the torso by +90 deg about Z.
    assert positions["hand"][0] == pytest.approx(0.0, abs=1e-9)
    assert positions["hand"][1] == pytest.approx(0.5, abs=1e-9)
    assert positions["hand"][2] == pytest.approx(1.0, abs=1e-9)
    # head hangs off root directly, unaffected by the waist joint entirely.
    assert positions["head"] == pytest.approx((0.0, 0.0, 0.3))


def test_fk_ignores_angle_on_fixed_joints() -> None:
    root, joints = _synthetic_tree()
    # a nonsense angle keyed to a fixed joint's name must be ignored.
    positions = forward_kinematics(root, joints, {"shoulder": math.pi})
    assert positions["hand"] == pytest.approx((0.5, 0.0, 1.0))


def test_project_front_uses_negated_y_and_z() -> None:
    projected = project_front({"p": (1.0, 2.0, 3.0)})
    assert projected["p"] == (-2.0, 3.0)


def test_project_side_uses_x_and_z() -> None:
    projected = project_side({"p": (1.0, 2.0, 3.0)})
    assert projected["p"] == (1.0, 3.0)


def test_parse_urdf_finds_root_and_joint_fields() -> None:
    urdf = """<?xml version="1.0"?>
    <robot name="test">
      <link name="root"/>
      <link name="torso"/>
      <link name="hand"/>
      <joint name="waist" type="revolute">
        <parent link="root"/>
        <child link="torso"/>
        <origin xyz="0 0 1.0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
      </joint>
      <joint name="shoulder" type="fixed">
        <parent link="torso"/>
        <child link="hand"/>
        <origin xyz="0.5 0 0" rpy="0 0 0"/>
      </joint>
    </robot>
    """
    root, joints = parse_urdf(urdf)
    assert root == "root"
    assert set(joints) == {"waist", "shoulder"}
    assert joints["waist"].kind == "revolute"
    assert joints["waist"].xyz == (0.0, 0.0, 1.0)
    assert joints["waist"].axis == (0.0, 0.0, 1.0)
    assert joints["shoulder"].kind == "fixed"
    assert joints["shoulder"].axis == (1.0, 0.0, 0.0)  # URDF default axis


def test_parse_urdf_defaults_missing_origin_to_identity() -> None:
    urdf = """<robot name="t"><link name="a"/><link name="b"/>
      <joint name="j" type="fixed"><parent link="a"/><child link="b"/></joint>
    </robot>"""
    _root, joints = parse_urdf(urdf)
    assert joints["j"].xyz == (0.0, 0.0, 0.0)
    assert joints["j"].rpy == (0.0, 0.0, 0.0)


def test_preview_projects_the_rviz_body_or_skips_without_rviz_utils() -> None:
    if fk.semantic_positions is None:
        pytest.skip("rviz_utils.hri.rig unavailable: source the ROS install to enable")
    from arena_simulation_setup.shared.body import SEMANTIC_DOFS

    rest_front, _ = preview(dict.fromkeys(SEMANTIC_DOFS, 0.0))
    bent_front, bent_side = preview(dict.fromkeys(SEMANTIC_DOFS, 0.0) | {"l_elbow": 1.2})
    assert len(rest_front) == len(bent_front) == len(bent_side) > 52
    assert rest_front != bent_front
