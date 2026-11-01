from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from arena_simulation_setup.shared.body import BODY_PARENTS as PARENTS
from arena_simulation_setup.shared.body import DIGIT_JOINTS, FINGERS, OPTIONAL_PARENTS

DIGIT_PARENTS = {name: parent for name, parent in OPTIONAL_PARENTS.items() if name in DIGIT_JOINTS}
TEMPLATE = Path(__file__).resolve().parents[2] / "configs" / "body" / "template.yaml"


@pytest.fixture(scope="module")
def joints() -> dict[str, dict]:
    doc = yaml.safe_load(TEMPLATE.read_text())
    assert doc["format"] == "arena-body-template/1"
    return doc["joints"]


def _offset(joints: dict[str, dict], name: str) -> np.ndarray:
    return np.array(joints[name]["offset"], dtype=float)


def _position(joints: dict[str, dict], name: str) -> np.ndarray:
    parent = joints[name]["parent"]
    return _offset(joints, name) + (np.zeros(3) if parent is None else _position(joints, parent))


def test_template_joints_follow_the_smplx_body_hierarchy(joints: dict[str, dict]) -> None:
    assert {name: spec["parent"] for name, spec in joints.items()} == {**PARENTS, **DIGIT_PARENTS}


def test_template_right_side_mirrors_the_left_across_the_sagittal_plane(joints: dict[str, dict]) -> None:
    for name in [*PARENTS, *DIGIT_PARENTS]:
        offset = _offset(joints, name)
        if name.startswith("right_"):
            left = _offset(joints, name.replace("right_", "left_", 1))
            assert offset == pytest.approx(left * np.array([1.0, -1.0, 1.0]))
        elif not name.startswith("left_"):
            assert offset[1] == 0.0


def test_template_zero_pose_lays_arms_out_sideways_and_legs_straight_down(joints: dict[str, dict]) -> None:
    for side, lateral in (("left", 1.0), ("right", -1.0)):
        for name in (f"{side}_elbow", f"{side}_wrist"):
            offset = _offset(joints, name)
            assert offset / np.linalg.norm(offset) == pytest.approx([0.0, lateral, 0.0])
        for name in (f"{side}_knee", f"{side}_ankle"):
            offset = _offset(joints, name)
            assert offset / np.linalg.norm(offset) == pytest.approx([0.0, 0.0, -1.0])
        toe = _offset(joints, f"{side}_foot")
        assert toe[0] > 0.0 and toe[1] == 0.0


def test_template_stands_on_the_body_frame_origin(joints: dict[str, dict]) -> None:
    pelvis = _offset(joints, "pelvis")
    assert pelvis[:2] == pytest.approx([0.0, 0.0])
    heights = {name: _position(joints, name)[2] for name in PARENTS}
    assert min(heights.values()) >= 0.0
    assert heights["head"] > heights["neck"] > heights["spine3"] > heights["spine2"] > heights["spine1"] > heights["pelvis"]


def test_template_digits_lie_flat_and_straight_along_the_hand_with_the_thumb_toward_the_palm(joints: dict[str, dict]) -> None:
    for side, lateral in (("left", 1.0), ("right", -1.0)):
        for digit in FINGERS:
            first = _offset(joints, f"{side}_{digit}1")
            assert lateral * first[1] > 0.0
            for k in (2, 3):
                offset = _offset(joints, f"{side}_{digit}{k}")
                if digit == "thumb":
                    assert offset[0] > 0.0 and offset[2] < 0.0
                else:
                    assert offset / np.linalg.norm(offset) == pytest.approx([0.0, lateral, 0.0])
            if digit != "thumb":
                assert first[2] == 0.0


def test_template_fingers_fan_from_the_thumb_side_to_the_little_finger(joints: dict[str, dict]) -> None:
    knuckles = [_offset(joints, f"left_{digit}1")[0] for digit in ("index", "middle", "ring", "pinky")]
    assert knuckles == sorted(knuckles, reverse=True)
    assert _offset(joints, "left_thumb1")[0] > knuckles[-1]


def test_template_digits_stay_within_a_hand_length_of_the_wrist(joints: dict[str, dict]) -> None:
    for name in DIGIT_PARENTS:
        tip = _position(joints, name) - _position(joints, name.split("_")[0] + "_wrist")
        assert 0.0 < np.linalg.norm(tip) < 0.19
