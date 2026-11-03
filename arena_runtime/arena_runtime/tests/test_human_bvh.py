import math

import attrs
import numpy as np
import pytest

from arena_simulation_setup.shared.body import axis_rotation, joint_angles

from arena_runtime.humans import bvh

TWO_BONES = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT LeftUpLeg
  {
    OFFSET 1 -1 0
    CHANNELS 3 Zrotation Yrotation Xrotation
    JOINT LeftLeg
    {
      OFFSET 0 -4 0
      CHANNELS 3 Zrotation Yrotation Xrotation
      End Site
      {
        OFFSET 0 -4 0
      }
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.5
0 10 0 0 0 0 0 0 30 0 0 0
0 6 0 0 20 0 0 0 45 0 0 0
"""

STRAIGHT = bvh.BindPose(
    orientations={"Hips": np.eye(3), "LeftUpLeg": np.eye(3), "LeftLeg": np.eye(3)},
    positions={"Hips": np.zeros(3), "LeftUpLeg": np.array([1.0, -1.0, 0.0]), "LeftLeg": np.array([1.0, -5.0, 0.0])},
)


def test_parse_reads_joints_channels_and_frames():
    motion = bvh.parse(TWO_BONES)
    assert list(motion.joints) == ["Hips", "LeftUpLeg", "LeftLeg"]
    assert motion.joints["Hips"].children == ("LeftUpLeg",)
    assert motion.joints["LeftUpLeg"].parent == "Hips"
    assert motion.frame_time == 0.5
    assert motion.values.shape == (2, 12)
    assert motion.joints["LeftLeg"].offset == (0.0, -4.0, 0.0)
    assert bvh.root_height(motion).tolist() == [10.0, 6.0]


def test_parse_rejects_a_short_motion_block():
    with pytest.raises(ValueError, match="need 24"):
        bvh.parse(TWO_BONES.replace("0 6 0 0 20 0 0 0 45 0 0 0\n", ""))


def test_local_rotations_compose_channels_in_file_order():
    rot = bvh.local_rotations(bvh.parse(TWO_BONES))["LeftUpLeg"][1]
    assert rot @ np.array([0.0, -1.0, 0.0]) == pytest.approx([0.0, -math.cos(math.radians(45)), -math.sin(math.radians(45))])


def test_upright_worlds_keep_the_root_tilt_and_drop_its_heading():
    motion = bvh.parse(TWO_BONES)
    rotations = bvh.local_rotations(motion)
    worlds = bvh.upright_worlds(motion, rotations)
    assert worlds["Hips"][1] == pytest.approx(np.eye(3))
    assert worlds["LeftLeg"][0] == pytest.approx(rotations["LeftUpLeg"][0])
    tilted = bvh.upright_worlds(motion, {**rotations, "Hips": np.stack([bvh._axis_rotation((1, 0, 0), 0.3)] * 2)})
    assert tilted["LeftUpLeg"][0] == pytest.approx(bvh._axis_rotation((1, 0, 0), 0.3) @ rotations["LeftUpLeg"][0])


def test_hip_angles_recover_a_composed_rotation():
    composed = axis_rotation((0.0, 0.0, -1.0), 0.3) @ axis_rotation((1.0, 0.0, 0.0), -0.2) @ axis_rotation((0.0, -1.0, 0.0), 1.1)
    assert joint_angles("left_hip", composed) == pytest.approx({"l_y_hip": 0.3, "l_p_hip": -0.2, "l_r_hip": 1.1})


def test_knee_angle_projects_onto_its_single_axis():
    turn = axis_rotation((0.0, -1.0, 0.0), -1.4) @ axis_rotation((0.0, 0.0, 1.0), 0.05)
    assert joint_angles("left_knee", turn)["l_knee"] == pytest.approx(-1.4, abs=0.01)


def test_wire_frames_resample_and_leave_unsolved_joints_at_zero():
    motion = bvh.parse(TWO_BONES)
    frames = bvh.wire_frames(motion, STRAIGHT, bones=["LeftUpLeg"], fps=2.0)
    assert [f["t"] for f in frames] == [0.0, 0.5]
    assert frames[0]["angles"]["l_r_hip"] == pytest.approx(-math.radians(30))
    assert frames[1]["angles"]["l_knee"] == 0.0
    assert set(frames[0]["angles"]) >= {"l_knee", "r_knee", "waist"}


def test_feet_left_out_keep_their_soles_flat():
    motion = bvh.parse(TWO_BONES.replace("0 10 0 0 0 0 0 0 30 0 0 0", "0 10 0 0 0 0 0 0 -40 0 0 70"))
    angles = bvh.wire_frames(motion, STRAIGHT, bones=["LeftUpLeg", "LeftLeg"], fps=2.0)[0]["angles"]
    assert angles["l_r_hip"] == pytest.approx(math.radians(40))
    assert angles["l_knee"] == pytest.approx(-math.radians(70))
    assert angles["l_ankle"] == pytest.approx(math.radians(30))


def test_a_bone_bind_roll_does_not_change_its_semantic_angles():
    motion = bvh.parse(TWO_BONES.replace("0 10 0 0 0 0 0 0 30 0 0 0", "0 10 0 0 0 0 0 25 0 0 0 0"))
    flipped = np.diag([1.0, -1.0, -1.0])
    turned = attrs.evolve(STRAIGHT, orientations={**STRAIGHT.orientations, "LeftUpLeg": flipped})
    plain = bvh.wire_frames(motion, STRAIGHT, bones=["LeftUpLeg"], fps=2.0)[0]["angles"]
    flip = bvh.wire_frames(motion, turned, bones=["LeftUpLeg"], fps=2.0)[0]["angles"]
    assert plain["l_y_hip"] == pytest.approx(-math.radians(25))
    assert flip["l_y_hip"] == pytest.approx(plain["l_y_hip"])


def test_parallel_legs_drop_hip_yaw_and_abduction_and_keep_flexion():
    motion = bvh.parse(TWO_BONES.replace("0 10 0 0 0 0 0 0 30 0 0 0", "0 10 0 0 0 0 10 25 -40 0 0 0"))
    open_ = bvh.wire_frames(motion, STRAIGHT, bones=["LeftUpLeg"], fps=2.0)[0]["angles"]
    parallel = bvh.wire_frames(motion, STRAIGHT, bones=["LeftUpLeg"], fps=2.0, parallel=True)[0]["angles"]
    assert abs(open_["l_y_hip"]) > 0.1 and abs(open_["l_p_hip"]) > 0.1
    assert parallel["l_y_hip"] == parallel["l_p_hip"] == 0.0
    assert parallel["l_r_hip"] == open_["l_r_hip"]


def test_bone_targets_point_a_differently_splayed_bind_bone_where_the_mocap_bone_points():
    motion = bvh.parse(TWO_BONES)
    splayed = attrs.evolve(STRAIGHT, positions={**STRAIGHT.positions, "LeftLeg": np.array([2.0, -5.0, 0.0])})
    target = bvh.bone_targets(motion, splayed, ["LeftUpLeg"])["LeftUpLeg"][0]
    bind_direction = splayed.positions["LeftLeg"] - splayed.positions["LeftUpLeg"]
    mocap = bvh.upright_worlds(motion, bvh.local_rotations(motion))["LeftUpLeg"][0] @ np.array([0.0, -4.0, 0.0])
    turned = target @ splayed.orientations["LeftUpLeg"].T @ bind_direction
    assert turned / np.linalg.norm(turned) == pytest.approx(mocap / np.linalg.norm(mocap))


DAE = """<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <library_controllers><controller id="body-skin"><skin source="#body-mesh">
    <source id="joints"><Name_array id="ja" count="2">Hips LeftUpLeg</Name_array></source>
    <source id="binds"><float_array id="ba" count="32">1 0 0 0  0 1 0 0  0 0 1 -0.9  0 0 0 1   1 0 0 0  0 -1 0 0.1  0 0 -1 0.8  0 0 0 1</float_array></source>
    <joints><input semantic="JOINT" source="#joints"/><input semantic="INV_BIND_MATRIX" source="#binds"/></joints>
  </skin></controller></library_controllers>
</COLLADA>
"""


def test_bind_frames_map_the_asset_axes_onto_bvh_axes(tmp_path):
    (tmp_path / "body.dae").write_text(DAE)
    bind = bvh.bind_pose(tmp_path / "body.dae")
    assert bind.orientations["Hips"] == pytest.approx(bvh.DAE_TO_BVH)
    assert bind.orientations["LeftUpLeg"] @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 0.0, 1.0])
    assert bind.orientations["LeftUpLeg"] @ np.array([0.0, 1.0, 0.0]) == pytest.approx([-1.0, 0.0, 0.0])
    assert bind.positions["LeftUpLeg"] == pytest.approx(bvh.DAE_TO_BVH @ np.array([0.0, 0.1, 0.8]))


def test_bind_legs_hang_straight_down_with_the_feet_keeping_their_orientation():
    tilt = bvh._axis_rotation((0.0, 0.0, 1.0), 0.2)
    foot = bvh._axis_rotation((1.0, 0.0, 0.0), 0.4)
    bind = bvh.BindPose(
        orientations={"LeftUpLeg": tilt, "LeftLeg": tilt, "LeftFoot": foot},
        positions={"LeftUpLeg": np.array([0.1, 1.0, 0.0]), "LeftLeg": np.array([0.2, 0.55, 0.0]), "LeftFoot": np.array([0.15, 0.1, 0.05])},
    )
    hung = bvh._hang_legs(bind)
    thigh, shin = np.linalg.norm(bind.positions["LeftLeg"] - bind.positions["LeftUpLeg"]), np.linalg.norm(bind.positions["LeftFoot"] - bind.positions["LeftLeg"])
    assert hung.positions["LeftLeg"] == pytest.approx(bind.positions["LeftUpLeg"] - [0.0, thigh, 0.0])
    assert hung.positions["LeftFoot"] == pytest.approx(hung.positions["LeftLeg"] - [0.0, shin, 0.0])
    assert hung.orientations["LeftFoot"] == pytest.approx(foot)
    knee_in_thigh = bind.orientations["LeftUpLeg"].T @ (bind.positions["LeftLeg"] - bind.positions["LeftUpLeg"])
    assert hung.orientations["LeftUpLeg"] @ knee_in_thigh == pytest.approx([0.0, -thigh, 0.0])
