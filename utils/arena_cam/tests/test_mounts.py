from __future__ import annotations

import math

import pytest

from arena_cam import curves
from arena_cam.mounts import Mount, parse_mount


def test_a_bare_frame_mounts_at_its_origin_looking_level():
    assert parse_mount("env_0/jackal/base_link", 1.0) == Mount("env_0/jackal/base_link", 1.0)


def test_a_position_after_the_frame_places_the_mount():
    assert parse_mount("env_0/jackal/base_link@0.2,0,0.6", 1.2) == Mount("env_0/jackal/base_link", 1.2, (0.2, 0.0, 0.6))


def test_a_fourth_number_is_the_downward_pitch():
    assert parse_mount("env_0/jackal/base_link@0.2,0,0.6,0.1", 1.2).pitch == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("@0,0,1", "names no frame"),
        ("base_link@0,0", "takes 3 or 4 numbers"),
        ("base_link@0,0,1,0,0", "takes 3 or 4 numbers"),
        ("base_link@a,0,1", "non-number"),
    ],
)
def test_malformed_specs_name_the_expected_form(spec, message):
    with pytest.raises(ValueError, match=message) as error:
        parse_mount(spec, 1.0)
    assert "FRAME[@X,Y,Z[,PITCH]]" in str(error.value)


def test_the_file_tag_flattens_the_frame():
    assert Mount("/env_0/jackal/base_link", 1.0).tag == "env_0_jackal_base_link"


def test_a_level_mount_looks_along_the_frame_x_axis():
    position, quat = Mount("base_link", 1.0, (0.2, 0.0, 0.6)).local_pose()
    assert position == (0.2, 0.0, 0.6)
    assert curves.quat_forward(quat) == pytest.approx((1.0, 0.0, 0.0))


def test_a_positive_pitch_looks_down():
    _position, quat = Mount("base_link", 1.0, pitch=0.3).local_pose()
    assert curves.quat_forward(quat) == pytest.approx((math.cos(0.3), 0.0, -math.sin(0.3)))
