import math

import numpy as np
import pytest
import yaml

from task_generator.simulators.human.pointing import skeleton as S

from task_generator.simulators.human.profile import JOINT_NAMES
from task_generator.simulators.human.support import (
    LEAN_S,
    PELVIS_MARGIN,
    RECLINE_S,
    SEATED_ARMS,
    SHIN_GAP,
    TORSO_JOINTS,
    BodyMeasurements,
    Legs,
    PelvisClearance,
    Support,
    _contacts,
    _leg_fk,
    ankle_reach,
    blend_pose,
    frames_until,
    leg_ik,
    load_body,
    plan_seat,
    planted_pelvis,
    rise_seconds,
    seat_depth,
    seat_pelvis,
    seat_stand,
    seated_legs,
    squat_profile,
    supported_pose,
    tilted_pose,
)

LEGS = Legs(hip=0.015, thigh=0.36, shin=0.375, ankle=0.08)
BODY = BodyMeasurements(clearance=PelvisClearance(sit=0.12, lie=0.11), legs=LEGS)
SLOPED = BodyMeasurements(clearance=PelvisClearance(sit=0.12, lie=0.11, seated=((0.0, 0.2), (math.pi, 0.0))), legs=LEGS)
SEATED = {"l_r_hip": math.pi / 2, "r_r_hip": math.pi / 2, "l_knee": -math.pi / 2, "r_knee": -math.pi / 2}


def rotate(xyzw: np.ndarray, v: tuple[float, float, float]) -> np.ndarray:
    x, y, z, w = xyzw
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return matrix @ np.array(v)


def test_planted_pelvis_stands_on_the_whole_leg_and_sits_on_the_shin():
    assert planted_pelvis({}, LEGS) == pytest.approx(0.015 + 0.36 + 0.375 + 0.08)
    assert planted_pelvis(SEATED, LEGS) == pytest.approx(0.015 + 0.375 + 0.08)


def test_sitter_is_lifted_onto_a_seat_above_its_planted_pelvis_only():
    position, orientation = supported_pose(Support("sit", 0.45), BODY, SEATED, 1.0, 2.0, math.pi / 2)
    assert position == pytest.approx([1.0, 2.0, 0.45 + 0.12 - 0.47])
    assert rotate(orientation, (1.0, 0.0, 0.0)) == pytest.approx([0.0, 1.0, 0.0])
    assert supported_pose(Support("sit", 0.3), BODY, SEATED, 1.0, 2.0, 0.0)[0][2] == 0.0
    assert supported_pose(Support("sit", 0.45), BODY, {}, 1.0, 2.0, 0.0)[0][2] == 0.0


@pytest.mark.parametrize("yaw", [0.0, 2.0])
def test_lying_ped_rests_its_pelvis_over_the_seat_on_its_back(yaw):
    position, orientation = supported_pose(Support("lie", 0.6), BODY, {}, 1.0, 2.0, yaw)
    pelvis = position + rotate(orientation, (0.0, 0.0, planted_pelvis({}, LEGS)))
    assert pelvis == pytest.approx([1.0, 2.0, 0.6 + 0.11])
    assert rotate(orientation, (1.0, 0.0, 0.0)) == pytest.approx([0.0, 0.0, 1.0])
    assert rotate(orientation, (0.0, 0.0, 1.0)) == pytest.approx([-math.cos(yaw), -math.sin(yaw), 0.0])


def test_body_is_read_from_the_annotation_beside_the_actor(tmp_path):
    sdf = tmp_path / "nurse" / "model.sdf"
    sdf.parent.mkdir()
    assert load_body(str(sdf)) is None
    (sdf.parent / "annotation.yaml").write_text(yaml.safe_dump({"name": "nurse", "pelvis_clearance": {"sit": 0.12, "lie": 0.11}}))
    assert load_body(str(sdf)) is None
    (sdf.parent / "annotation.yaml").write_text(yaml.safe_dump({"pelvis_clearance": {"sit": 0.12, "lie": 0.11}, "legs": {"hip": 0.015, "thigh": 0.36, "shin": 0.375, "ankle": 0.08}}))
    assert load_body(str(sdf)) == BODY
    assert load_body("") is None


@pytest.mark.parametrize("skin", ["not xml", "<COLLADA/>"])
def test_annotated_body_outlives_an_unreadable_skin(tmp_path, skin):
    sdf = tmp_path / "nurse" / "model.sdf"
    (sdf.parent / "meshes").mkdir(parents=True)
    (sdf.parent / "meshes" / "model.dae").write_text(skin)
    (sdf.parent / "annotation.yaml").write_text(yaml.safe_dump({"pelvis_clearance": {"sit": 0.12, "lie": 0.11}, "legs": {"hip": 0.015, "thigh": 0.36, "shin": 0.375, "ankle": 0.08}}))
    assert load_body(str(sdf)) == BODY


SQUAT = [{"t": k / 20.0, "angles": {**dict.fromkeys(JOINT_NAMES, 0.0), "l_r_hip": h, "r_r_hip": h, "l_knee": -2.0 * h, "r_knee": -2.0 * h, "waist": 0.3 * h}} for k, h in enumerate(np.linspace(0.0, 1.5, 21))]


def test_leg_ik_reaches_the_ankle_it_is_given():
    hip, knee = leg_ik(0.3, 0.5, LEGS)
    pos, _ = _leg_fk({"l_r_hip": hip, "l_knee": knee}, LEGS)
    assert pos["l_ankle"][0] - pos["l_hip"][0] == pytest.approx(0.3)
    assert pos["l_hip"][2] - pos["l_ankle"][2] == pytest.approx(0.5)
    assert leg_ik(0.0, LEGS.thigh + LEGS.shin, LEGS) == pytest.approx((0.0, 0.0), abs=1e-3)


def test_seated_legs_put_vertical_shins_on_the_floor_or_give_up():
    hip, reach = seated_legs(0.57, LEGS)
    pos, _ = _leg_fk({"l_r_hip": hip, "l_knee": -hip}, LEGS)
    assert pos["l_knee"][0] == pytest.approx(pos["l_ankle"][0])
    assert pos["l_knee"][0] - pos["pelvis"][0] == pytest.approx(reach)
    assert planted_pelvis({"l_r_hip": hip, "r_r_hip": hip, "l_knee": -hip, "r_knee": -hip}, LEGS) == pytest.approx(0.57)
    assert seated_legs(1.0, LEGS) is None


def test_sitting_down_keeps_the_feet_planted_and_ends_upright_on_the_seat():
    plan = plan_seat(SQUAT, BODY, 0.45, 0.25, None, 20.0)
    target = 0.45 + BODY.clearance.sit
    down = [f["angles"] for f in plan.frames]
    feet = [seat_depth(plan, a, LEGS) + ankle_reach(a, LEGS) for a in down]
    assert feet == pytest.approx([plan.feet] * len(feet), abs=1e-6)
    assert plan.stand == pytest.approx(plan.feet) == pytest.approx(seat_stand(0.25))
    assert seat_depth(plan, down[0], LEGS) == pytest.approx(plan.stand, abs=0.01)
    assert planted_pelvis(down[0], LEGS) == pytest.approx(LEGS.hip + LEGS.thigh + LEGS.shin + LEGS.ankle, abs=0.02)
    assert planted_pelvis(down[-1], LEGS) == pytest.approx(target)
    assert seat_depth(plan, down[-1], LEGS) == pytest.approx(plan.depth)
    assert all(down[-1][j] == 0.0 for j in TORSO_JOINTS)
    contact = len(down) - round(RECLINE_S * 20.0) - 1
    assert down[contact]["waist"] > 0.2
    pos, body = _leg_fk(down[-1], LEGS)
    assert pos["l_knee"][0] == pytest.approx(pos["l_ankle"][0])
    assert pos["l_knee"][0] - pos["pelvis"][0] + plan.depth - body.r_leg == pytest.approx(0.25 + SHIN_GAP)


@pytest.mark.parametrize("height", [0.3, 0.45, 0.6])
def test_seated_pelvis_rests_at_the_clearance_of_the_hip_flexion_it_sits_with(height):
    plan = plan_seat(SQUAT, SLOPED, height, 0.25, None, 20.0)
    seated = plan.frames[-1]["angles"]
    assert planted_pelvis(seated, LEGS) == pytest.approx(height + SLOPED.clearance.sit_at(plan.fold), abs=1e-3)
    assert 0.5 * (seated["l_r_hip"] + seated["r_r_hip"]) == pytest.approx(plan.fold, abs=1e-3)


def test_sitter_lift_takes_the_clearance_at_its_current_hip_flexion():
    position, _ = supported_pose(Support("sit", 0.45), SLOPED, SEATED, 1.0, 2.0, 0.0)
    assert position[2] == pytest.approx(0.45 + 0.1 - 0.47)


def test_rising_leans_forward_faster_than_the_recline_then_reverses_the_way_down():
    plan = plan_seat(SQUAT, BODY, 0.45, None, None, 20.0)
    up = [f["angles"] for f in plan.release]
    lean = round(LEAN_S * 20.0)
    assert lean < round(RECLINE_S * 20.0)
    assert up[0]["waist"] == 0.0
    assert up[lean]["waist"] == pytest.approx(plan.frames[-round(RECLINE_S * 20.0) - 1]["angles"]["waist"])
    assert up[-1] == pytest.approx(plan.frames[0]["angles"])


def test_a_seat_out_of_reach_of_the_floor_leaves_the_shins_dangling():
    plan = plan_seat(SQUAT, BODY, 0.88, 0.2, 0.2, 20.0)
    assert plan.feet is None
    last = plan.frames[-1]["angles"]
    assert (last["l_r_hip"], last["l_knee"], last["l_ankle"]) == pytest.approx((math.pi / 2, -math.pi / 2, 0.0))
    assert seat_depth(plan, last, LEGS) == pytest.approx(plan.depth)
    assert plan.stand == pytest.approx(seat_stand(0.2))
    assert seat_depth(plan, plan.frames[0]["angles"], LEGS) == pytest.approx(plan.stand)
    middle = seat_depth(plan, plan.frames[len(plan.frames) // 3]["angles"], LEGS)
    assert plan.depth < middle < plan.stand


@pytest.mark.parametrize("height", [0.3, 0.38, 0.45, 0.6, 0.88])
def test_seated_arms_rise_mirrored_with_the_fold_and_clear_the_thighs_and_hips(height):
    plan = plan_seat(SQUAT, BODY, height, 0.25, None, 20.0)
    first, last = plan.frames[0]["angles"], plan.frames[-1]["angles"]
    assert all(first[f"{side}_{joint}"] == 0.0 for side in "lr" for joint in SEATED_ARMS)
    assert all(last[f"l_{joint}"] == value for joint, value in SEATED_ARMS.items())
    assert last["r_y_shoulder"] == -last["l_y_shoulder"] and last["r_elbow"] == last["l_elbow"]
    body = S.Body()
    body.thigh, body.tibia = LEGS.thigh, LEGS.shin
    pos, _ = S.fk(dict.fromkeys(JOINT_NAMES, 0.0) | last, body)
    for side in "lr":
        clearances = S.collision_clearances(pos, body, side)
        assert min(v for k, v in clearances.items() if k.startswith("forearm/") and ("thigh" in k or "pelvis" in k)) > 0.0


def test_pelvis_sits_back_against_the_rear_edge_unless_the_calves_need_room():
    r_leg = S.Body().r_leg
    assert seat_pelvis(BODY, 0.36, 0.15, 0.1) == pytest.approx(BODY.clearance.lie - 0.1)
    assert seat_pelvis(BODY, 0.3, 0.3, 0.2) == pytest.approx(0.3 + SHIN_GAP + r_leg - 0.3)
    assert seat_pelvis(BODY, 0.0, 0.1, 0.4) == pytest.approx(0.1 - PELVIS_MARGIN)
    assert seat_pelvis(BODY, 0.36, None, 0.2) == 0.0


def test_rise_seconds_is_the_release_length():
    plan = plan_seat(SQUAT, BODY, 0.45, None, None, 20.0)
    assert rise_seconds(plan) == pytest.approx((len(plan.release) - 1) / 20.0)


def test_blend_pose_eases_position_and_turns_the_short_way():
    start = (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    end = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, math.sin(math.pi / 2), math.cos(math.pi / 2)]))
    assert blend_pose(start, end, 0.0)[0] == pytest.approx(np.zeros(3))
    assert blend_pose(start, end, 2.0)[1] == pytest.approx(end[1])
    position, q = blend_pose(start, end, 0.5)
    assert position == pytest.approx([0.5, 0.0, 0.0])
    assert 2.0 * math.atan2(q[2], q[3]) == pytest.approx(math.pi / 2)


SKIN = """<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <library_geometries><geometry id="m"><mesh>
    <source id="p"><float_array id="pa" count="9">-0.1 0.1 0.85  0.1 0.1 0.0  -0.1 -0.1 0.85</float_array>
      <technique_common><accessor source="#pa" count="3" stride="3"/></technique_common></source>
    <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  </mesh></geometry></library_geometries>
  <library_controllers><controller id="c"><skin source="#m">
    <source id="j"><Name_array id="ja" count="7">Hips LeftUpLeg LeftLeg LeftFoot RightUpLeg RightLeg RightFoot</Name_array></source>
    <source id="b"><float_array id="ba" count="112">1 0 0 0 0 1 0 0 0 0 1 -0.95 0 0 0 1  1 0 0 0 0 1 0 -0.1 0 0 1 -0.93 0 0 0 1  1 0 0 0 0 1 0 -0.1 0 0 1 -0.5 0 0 0 1  1 0 0 0 0 1 0 -0.1 0 0 1 -0.08 0 0 0 1  1 0 0 0 0 1 0 0.1 0 0 1 -0.93 0 0 0 1  1 0 0 0 0 1 0 0.1 0 0 1 -0.5 0 0 0 1  1 0 0 0 0 1 0 0.1 0 0 1 -0.08 0 0 0 1</float_array>
      <technique_common><accessor source="#ba" count="7" stride="16"/></technique_common></source>
    <source id="w"><float_array id="wa" count="1">1</float_array><technique_common><accessor source="#wa" count="1" stride="1"/></technique_common></source>
    <joints><input semantic="JOINT" source="#j"/><input semantic="INV_BIND_MATRIX" source="#b"/></joints>
    <vertex_weights count="3"><input semantic="JOINT" source="#j" offset="0"/><input semantic="WEIGHT" source="#w" offset="1"/>
      <vcount>1 1 1</vcount><v>1 0  3 0  4 0</v></vertex_weights>
  </skin></controller></library_controllers>
</COLLADA>
"""


def test_body_is_measured_off_the_skin_when_the_annotation_has_none(tmp_path):
    sdf = tmp_path / "tall" / "tall.sdf"
    (sdf.parent / "meshes").mkdir(parents=True)
    (sdf.parent / "annotation.yaml").write_text(yaml.safe_dump({"name": "tall", "face": "+x"}))
    (sdf.parent / "meshes" / "tall.dae").write_text(SKIN)
    body = load_body(str(sdf))
    assert body is not None
    assert (body.legs.hip, body.legs.thigh, body.legs.shin, body.legs.ankle) == pytest.approx((0.02, 0.43, 0.42, 0.08))
    assert body.clearance.sit == pytest.approx(0.02 + 0.1)


def test_sitter_moves_its_pelvis_depth_ahead_along_its_facing():
    position, _ = supported_pose(Support("sit", 0.45), BODY, SEATED, 1.0, 2.0, math.pi / 2, depth=-0.1)
    assert position[:2] == pytest.approx([1.0, 1.9])


def _yaw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


@pytest.mark.parametrize("yaw", [0.0, 1.2])
def test_an_untilted_clip_leaves_the_ped_standing_where_it_is(yaw: float) -> None:
    position, orientation = tilted_pose(BODY, {}, 1.0, 2.0, 0.3, yaw, (0.0, 0.0, 0.0, 1.0))
    assert position == pytest.approx([1.0, 2.0, 0.3])
    assert orientation == pytest.approx(_yaw(yaw))


@pytest.mark.parametrize("yaw", [0.0, -2.0])
def test_a_body_tilted_onto_its_back_lies_with_its_pelvis_over_the_ped(yaw: float) -> None:
    supine = (0.0, -math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))
    position, orientation = tilted_pose(BODY, {}, 1.0, 2.0, 0.0, yaw, supine)
    pelvis = position + rotate(orientation, (0.0, 0.0, planted_pelvis({}, LEGS)))
    assert pelvis == pytest.approx([1.0, 2.0, BODY.clearance.lie])  # the pelvis rests on its back, as on a bed
    head = rotate(orientation, (0.0, 0.0, 1.0))
    assert head == pytest.approx([-math.cos(yaw), -math.sin(yaw), 0.0], abs=1e-9)  # the head behind the heading


def test_a_tilted_body_rests_its_lowest_skin_on_the_floor_with_its_pelvis_over_the_ped() -> None:
    kneel = {"l_r_hip": 1.4, "r_r_hip": 1.4, "l_knee": -2.2, "r_knee": -2.2, "l_ankle": 0.6, "r_ankle": 0.6, "waist": 0.5}
    tilt = np.array([0.15, 0.35, 0.05, 0.92])
    tilt /= np.linalg.norm(tilt)
    position, orientation = tilted_pose(BODY, kneel, 1.0, 2.0, 0.2, 0.4, tilt)
    points, flesh, soles, pelvis = _contacts(kneel, BODY)
    lift = np.array([0.0, 0.0, pelvis[2] - soles[:, 2].min()])  # the renderer plants the lowest sole at body height 0
    world = lambda p: position + rotate(orientation, tuple(p - pelvis + lift))  # noqa: E731
    assert world(pelvis)[:2] == pytest.approx([1.0, 2.0])
    skin = [world(p)[2] - f for p, f in zip(points, flesh, strict=True)] + [world(p)[2] for p in soles]
    assert min(skin) == pytest.approx(0.2)
