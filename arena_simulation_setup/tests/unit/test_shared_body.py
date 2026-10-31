from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from arena_simulation_setup.shared.body import (
    BODY_JOINTS,
    DEFAULT_STANCE,
    DIGIT_JOINTS,
    FINGERS,
    JOINT_DOFS,
    PARENTS,
    SEMANTIC_DOFS,
    Template,
    axis_rotation,
    body_to_semantic,
    forward,
    from_smplx_rotation,
    from_smplx_vector,
    load_stance,
    load_template,
    matrix_to_quat,
    quat_to_matrix,
    rest_offsets,
    semantic_to_body,
)
from arena_assets.rig import write_rig
from arena_simulation_setup.testing import make_rig

TEMPLATE = Path(__file__).resolve().parents[2] / "configs" / "body" / "template.yaml"


@pytest.fixture(scope="module")
def template() -> Template:
    return load_template(TEMPLATE)


def _random_rotation(rng: np.random.Generator, scale: float = math.pi) -> np.ndarray:
    return axis_rotation(rng.normal(size=3), float(rng.uniform(-scale, scale)))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _pose(angles: dict[str, float], template: Template) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return forward(template, semantic_to_body(angles))


def test_quaternions_round_trip_through_rotation_matrices() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        m = _random_rotation(rng)
        q = matrix_to_quat(m)
        assert q[3] >= 0.0
        assert quat_to_matrix(q) == pytest.approx(m, abs=1e-12)


def test_smplx_left_up_forward_become_arena_y_z_x() -> None:
    assert from_smplx_vector((1.0, 0.0, 0.0)) == pytest.approx([0.0, 1.0, 0.0])
    assert from_smplx_vector((0.0, 1.0, 0.0)) == pytest.approx([0.0, 0.0, 1.0])
    assert from_smplx_vector((0.0, 0.0, 1.0)) == pytest.approx([1.0, 0.0, 0.0])
    assert from_smplx_rotation(axis_rotation((1.0, 0.0, 0.0), 0.7)) == pytest.approx(axis_rotation((0.0, 1.0, 0.0), 0.7))


def test_template_loads_every_body_and_digit_joint_parents_first(template: Template) -> None:
    assert tuple(template.parents)[: len(BODY_JOINTS)] == BODY_JOINTS
    assert set(template.parents) == {*BODY_JOINTS, *DIGIT_JOINTS}
    assert all(parent is None or list(template.parents).index(parent) < index for index, parent in enumerate(template.parents.values()))


def test_template_with_a_wrong_parent_or_format_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "template.yaml"
    bad.write_text("format: arena-body-template/1\njoints:\n  pelvis: {parent: null, offset: [0, 0, 1]}\n  left_hip: {parent: spine1, offset: [0, 0.1, 0]}\n")
    with pytest.raises(ValueError, match="left_hip"):
        load_template(bad)
    bad.write_text("format: arena-body-template/9\njoints: {}\n")
    with pytest.raises(ValueError, match="format"):
        load_template(bad)


def test_zero_pose_holds_the_arms_out_sideways(template: Template) -> None:
    positions, frames = forward(template, {})
    assert _unit(positions["left_wrist"] - positions["left_shoulder"]) == pytest.approx([0.0, 1.0, 0.0])
    assert _unit(positions["right_wrist"] - positions["right_shoulder"]) == pytest.approx([0.0, -1.0, 0.0])
    assert all(frame == pytest.approx(np.eye(3)) for frame in frames.values())


def test_semantic_rest_hangs_the_arms_beside_straight_legs(template: Template) -> None:
    positions, _ = _pose({}, template)
    for side in ("left", "right"):
        arm = positions[f"{side}_wrist"] - positions[f"{side}_shoulder"]
        assert _unit(arm) == pytest.approx([0.0, 0.0, -1.0])
        reach = np.linalg.norm(template.offsets[f"{side}_elbow"]) + np.linalg.norm(template.offsets[f"{side}_wrist"])
        assert np.linalg.norm(arm) == pytest.approx(reach)
        assert _unit(positions[f"{side}_ankle"] - positions[f"{side}_hip"]) == pytest.approx([0.0, 0.0, -1.0])


@pytest.mark.parametrize("side", ["l", "r"])
def test_semantic_flexions_move_limbs_the_way_joints_md_names_them(template: Template, side: str) -> None:
    name = {"l": "left", "r": "right"}[side]
    outward = 1.0 if side == "l" else -1.0
    rest, rest_frames = _pose({}, template)

    def moved(dof: str, angle: float, joint: str) -> np.ndarray:
        positions, _ = _pose({f"{side}_{dof}": angle}, template)
        return positions[joint] - rest[joint]

    assert moved("p_shoulder", 0.8, f"{name}_wrist")[0] > 0.1
    assert moved("elbow", 1.0, f"{name}_wrist")[0] > 0.1
    assert moved("r_hip", 0.6, f"{name}_knee")[0] > 0.1
    assert moved("knee", -1.0, f"{name}_ankle")[0] < -0.1
    assert moved("ankle", 0.5, f"{name}_foot")[2] > 0.02
    assert outward * moved("p_hip", 0.4, f"{name}_knee")[1] > 0.05
    assert moved("p_collar", 0.4, f"{name}_shoulder")[2] > 0.02
    assert moved("y_collar", 0.4, f"{name}_shoulder")[0] > 0.02
    _, frames = _pose({f"{side}_wrist": 0.6}, template)
    hand = frames[f"{name}_wrist"] @ np.array([0.0, outward, 0.0])
    assert hand == pytest.approx(rest_frames[f"{name}_wrist"] @ np.array([0.0, outward, 0.0]) + np.array([0.0, outward * math.sin(0.6), 1.0 - math.cos(0.6)]), abs=1e-9)


def test_semantic_waist_flexion_leans_the_chest_forward(template: Template) -> None:
    rest, _ = _pose({}, template)
    leaned, _ = _pose({"waist": 0.5}, template)
    assert leaned["spine3"][0] - rest["spine3"][0] > 0.05


def test_semantic_dofs_round_trip_through_the_body_pose() -> None:
    rng = np.random.default_rng(7)
    for _ in range(300):
        angles = {dof: float(rng.uniform(-0.5, 0.5)) for dof in SEMANTIC_DOFS}
        for s in "lr":
            angles[f"{s}_p_shoulder"] = float(rng.uniform(0.3, 1.5))
            angles[f"{s}_elbow"] = float(rng.uniform(0.0, 1.5))
            angles[f"{s}_knee"] = float(rng.uniform(-1.5, 0.0))
        back = body_to_semantic(semantic_to_body(angles))
        assert back == pytest.approx(angles, abs=1e-9)


def test_backward_arm_extension_keeps_its_sign() -> None:
    back = body_to_semantic(semantic_to_body({"l_p_shoulder": -0.3, "r_p_shoulder": -0.3}))
    assert back["l_p_shoulder"] == pytest.approx(-0.3)
    assert back["l_y_shoulder"] == pytest.approx(0.0, abs=1e-9)
    assert back["r_p_shoulder"] == pytest.approx(-0.3)


def _stance(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {joint: axis_rotation(rng.normal(size=3), float(rng.uniform(-0.6, 0.6))) for joint in BODY_JOINTS}


def test_semantic_zero_poses_the_stance(template: Template) -> None:
    stance = _stance(5)
    _, frames = forward(template, semantic_to_body({}, stance))
    for joint, world in stance.items():
        assert frames[joint] == pytest.approx(world, abs=1e-12)


def test_semantic_dofs_round_trip_over_a_stance() -> None:
    stance = _stance(9)
    rng = np.random.default_rng(2)
    for _ in range(100):
        angles = {dof: float(rng.uniform(-0.5, 0.5)) for dof in SEMANTIC_DOFS}
        for s in "lr":
            angles[f"{s}_p_shoulder"] = float(rng.uniform(0.3, 1.5))
            angles[f"{s}_elbow"] = float(rng.uniform(0.0, 1.5))
            angles[f"{s}_knee"] = float(rng.uniform(-1.5, 0.0))
        assert body_to_semantic(semantic_to_body(angles, stance), stance) == pytest.approx(angles, abs=1e-9)


def test_a_semantic_turn_acts_about_the_body_axes_over_any_stance(template: Template) -> None:
    stance = _stance(4)
    _, rest = forward(template, semantic_to_body({}, stance))
    _, turned = forward(template, semantic_to_body({"l_knee": -0.7}, stance))
    assert turned["left_knee"] == pytest.approx(rest["left_hip"] @ stance["left_hip"].T @ axis_rotation((0.0, -1.0, 0.0), -0.7) @ stance["left_knee"], abs=1e-12)


def test_the_head_turn_spreads_over_neck_and_head(template: Template) -> None:
    rest = semantic_to_body({})
    turned = semantic_to_body({"y_head": 1.0})
    assert quat_to_matrix(turned["neck"]) == pytest.approx(axis_rotation((0.0, 0.0, 1.0), 0.6), abs=1e-12)
    assert quat_to_matrix(turned["head"]) == pytest.approx(axis_rotation((0.0, 0.0, 1.0), 0.4), abs=1e-12)
    assert quat_to_matrix(rest["neck"]) == pytest.approx(np.eye(3))


def test_a_rig_stance_loads_beside_the_actor_and_defaults_without_one(tmp_path: Path) -> None:
    sdf = tmp_path / "alice.sdf"
    assert load_stance(str(sdf)) is DEFAULT_STANCE
    assert load_stance("") is DEFAULT_STANCE
    write_rig(tmp_path / "alice.rig", make_rig({"pelvis": (None, []), "head": ("pelvis", [])}, stance={"head": [0.0, 0.0, 1.0, 0.0]}))
    stance = load_stance(str(sdf))
    assert stance["head"] == pytest.approx(axis_rotation((0.0, 0.0, 1.0), math.pi))
    assert stance["pelvis"] == pytest.approx(np.eye(3))


def test_unknown_dofs_and_joints_are_rejected(template: Template) -> None:
    with pytest.raises(ValueError, match="l_wheel"):
        semantic_to_body({"l_wheel": 1.0})
    with pytest.raises(ValueError, match="left_tail"):
        body_to_semantic({"left_tail": np.array([0.0, 0.0, 0.0, 1.0])})
    with pytest.raises(ValueError, match="left_tail"):
        forward(template, {"left_tail": np.array([0.0, 0.0, 0.0, 1.0])})


def _bind_skeleton(template: Template, seed: int) -> tuple[dict[str, list[str]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """A bind skeleton: the template under a random pose, every bone with its own random roll, the neck spread over two bones."""
    rng = np.random.default_rng(seed)
    pose = {joint: matrix_to_quat(_random_rotation(rng, 0.5)) for joint in BODY_JOINTS}
    positions, frames = forward(template, pose)
    bones = {joint: [f"{joint}_bone"] for joint in BODY_JOINTS}
    bones["neck"] = ["neck_a", "neck_b"]
    heads = {f"{joint}_bone": positions[joint] for joint in BODY_JOINTS}
    heads["neck_a"] = positions["neck"]
    heads["neck_b"] = 0.5 * (positions["neck"] + positions["head"])
    rolls = {bone: frames[joint] @ _random_rotation(rng) for joint, names in bones.items() for bone in names}
    return bones, heads, rolls


CHILD = {"pelvis": "spine1", "spine3": "neck"}


def _children(template: Template) -> dict[str, str]:
    out: dict[str, str] = {}
    for joint, parent in template.parents.items():
        if parent is not None and joint in BODY_JOINTS:
            out.setdefault(parent, CHILD.get(parent, joint))
    return out


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_rest_offsets_put_every_bone_on_its_template_direction(template: Template, seed: int) -> None:
    bones, heads, rolls = _bind_skeleton(template, seed)
    offsets = rest_offsets(template, bones, heads, rolls)
    joint_head = {joint: heads[names[0]] for joint, names in bones.items()}
    for joint, child in _children(template).items():
        for bone, offset in offsets[joint]:
            local = rolls[bone].T @ _unit(joint_head[child] - joint_head[joint])
            assert quat_to_matrix(offset) @ local == pytest.approx(_unit(template.offsets[child]), abs=1e-9)


@pytest.mark.parametrize("seed", [3, 11])
def test_bones_posed_as_joint_rotation_times_offset_follow_the_body_pose(template: Template, seed: int) -> None:
    bones, heads, rolls = _bind_skeleton(template, seed)
    offsets = dict(rest_offsets(template, bones, heads, rolls))
    joint_head = {joint: heads[names[0]] for joint, names in bones.items()}
    pose = semantic_to_body({"l_p_shoulder": 0.9, "r_elbow": 1.2, "l_r_hip": 0.4, "r_knee": -0.8, "waist": 0.3, "y_head": 0.5})
    positions, frames = forward(template, pose)
    for joint, child in _children(template).items():
        for bone, offset in offsets[joint]:
            local = rolls[bone].T @ _unit(joint_head[child] - joint_head[joint])
            assert frames[joint] @ quat_to_matrix(offset) @ local == pytest.approx(_unit(positions[child] - positions[joint]), abs=1e-9)


def test_rest_offsets_keep_leaf_bones_at_their_bind_turn_from_the_parent(template: Template) -> None:
    bones, heads, rolls = _bind_skeleton(template, 5)
    offsets = {joint: dict(pairs) for joint, pairs in rest_offsets(template, bones, heads, rolls).items()}
    for leaf, parent in (("left_wrist", "left_elbow"), ("head", "neck"), ("right_foot", "right_ankle")):
        parent_bone = bones[parent][-1]
        relative = quat_to_matrix(offsets[parent][parent_bone]).T @ quat_to_matrix(offsets[leaf][f"{leaf}_bone"])
        assert relative == pytest.approx(rolls[parent_bone].T @ rolls[f"{leaf}_bone"], abs=1e-9)


def test_rest_offsets_need_every_mapped_joints_parent(template: Template) -> None:
    bones, heads, rolls = _bind_skeleton(template, 3)
    del bones["left_knee"]
    with pytest.raises(ValueError, match="left_ankle"):
        rest_offsets(template, bones, heads, rolls)


def test_the_semantic_vocabulary_ends_with_curl_and_spread_per_hand() -> None:
    assert len(SEMANTIC_DOFS) == 52
    assert SEMANTIC_DOFS[40:] == (
        *(f"l_curl_{digit}" for digit in FINGERS),
        "l_spread",
        *(f"r_curl_{digit}" for digit in FINGERS),
        "r_spread",
    )
    assert JOINT_DOFS["left_index1"] == ("l_curl_index", "l_spread")
    assert JOINT_DOFS["right_thumb1"] == ("r_curl_thumb",)
    assert JOINT_DOFS["left_pinky3"] == ("l_curl_pinky",)


def test_the_default_stance_hangs_the_digits_like_their_wrist() -> None:
    for name in DIGIT_JOINTS:
        assert DEFAULT_STANCE[name] == pytest.approx(DEFAULT_STANCE[f"{name.split('_')[0]}_wrist"])


def test_digits_at_zero_pose_the_stance_and_a_stance_without_digits_hangs_them_like_the_wrist(template: Template) -> None:
    _, frames = forward(template, semantic_to_body({}))
    for name in template.parents:
        if name in DEFAULT_STANCE:
            assert frames[name] == pytest.approx(DEFAULT_STANCE[name], abs=1e-12)
    wristed = {joint: world for joint, world in _stance(6).items()}
    _, frames = forward(template, semantic_to_body({}, wristed))
    assert frames["left_index2"] == pytest.approx(wristed["left_wrist"], abs=1e-12)
    assert frames["right_thumb3"] == pytest.approx(wristed["right_wrist"], abs=1e-12)


@pytest.mark.parametrize("side", ["l", "r"])
@pytest.mark.parametrize("digit", FINGERS[1:])
def test_positive_curl_moves_the_fingertip_toward_the_palm(template: Template, side: str, digit: str) -> None:
    name = {"l": "left", "r": "right"}[side]
    outward = 1.0 if side == "l" else -1.0
    tip = f"{name}_{digit}3"
    rest, _ = _pose({}, template)
    curled, _ = _pose({f"{side}_curl_{digit}": 1.0}, template)
    move = curled[tip] - rest[tip]
    assert outward * move[1] < -0.01
    assert abs(move[0]) < 1e-9


@pytest.mark.parametrize("side", ["l", "r"])
def test_positive_thumb_curl_swings_the_tip_across_the_palm(template: Template, side: str) -> None:
    name = {"l": "left", "r": "right"}[side]
    outward = 1.0 if side == "l" else -1.0
    rest, _ = _pose({}, template)
    curled, _ = _pose({f"{side}_curl_thumb": 1.0}, template)
    move = curled[f"{name}_thumb3"] - rest[f"{name}_thumb3"]
    assert outward * move[1] < 0.0
    assert move[0] < -0.01
    assert abs(move[2]) < 1e-9


@pytest.mark.parametrize("side", ["l", "r"])
def test_curl_in_the_t_pose_frame_closes_the_fingers_toward_the_palm_side_down(template: Template, side: str) -> None:
    name = {"l": "left", "r": "right"}[side]
    identity = {joint: np.eye(3) for joint in PARENTS}
    for digit in FINGERS[1:]:
        rest, _ = forward(template, semantic_to_body({}, identity))
        curled, _ = forward(template, semantic_to_body({f"{side}_curl_{digit}": 1.0}, identity))
        move = curled[f"{name}_{digit}3"] - rest[f"{name}_{digit}3"]
        assert move[2] < -0.01


def test_curl_flexes_the_base_and_middle_knuckles_fully_and_the_tip_at_0_7(template: Template) -> None:
    pose = semantic_to_body({"l_curl_middle": 1.0})
    angles = [2.0 * math.atan2(float(np.linalg.norm(pose[f"left_middle{k}"][:3])), float(pose[f"left_middle{k}"][3])) for k in (1, 2, 3)]
    assert angles == pytest.approx([1.0, 1.0, 0.7])


@pytest.mark.parametrize("side", ["l", "r"])
def test_positive_spread_fans_the_fingers_apart_in_the_palm_plane(template: Template, side: str) -> None:
    name = {"l": "left", "r": "right"}[side]
    rest, _ = _pose({}, template)
    fanned, _ = _pose({f"{side}_spread": 0.3}, template)
    shift = {digit: fanned[f"{name}_{digit}3"] - rest[f"{name}_{digit}3"] for digit in FINGERS}
    assert shift["index"][0] > shift["middle"][0] > 0.0 > shift["ring"][0] > shift["pinky"][0]
    assert shift["thumb"] == pytest.approx(np.zeros(3), abs=1e-12)
    for digit in FINGERS[1:]:
        assert abs(shift[digit][1]) < 1e-9


def test_digit_dofs_leave_every_other_joint_alone() -> None:
    plain = semantic_to_body({})
    fisted = semantic_to_body({f"{s}_{dof}": 0.5 for s in "lr" for dof in ("curl_thumb", "curl_index", "curl_middle", "curl_ring", "curl_pinky", "spread")})
    for joint in BODY_JOINTS:
        assert fisted[joint] == pytest.approx(plain[joint])


def test_digit_dofs_round_trip_through_the_body_pose_over_a_stance() -> None:
    rng = np.random.default_rng(12)
    for stance in (DEFAULT_STANCE, _stance(3)):
        for _ in range(50):
            angles = {dof: float(rng.uniform(-0.4, 0.4) if dof.endswith("spread") else rng.uniform(-0.3, 1.5)) for dof in SEMANTIC_DOFS[40:]}
            assert body_to_semantic(semantic_to_body(angles, stance), stance) == pytest.approx({**body_to_semantic(semantic_to_body({}, stance), stance), **angles}, abs=1e-9)


def test_a_missing_digit_dof_is_straight() -> None:
    assert body_to_semantic(semantic_to_body({}))["l_curl_index"] == pytest.approx(0.0)
    assert semantic_to_body({"r_spread": 0.2})["left_index1"] == pytest.approx(semantic_to_body({})["left_index1"])
