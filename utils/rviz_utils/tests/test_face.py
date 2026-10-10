from __future__ import annotations

import math

import numpy as np
import pytest
import xacro
from ament_index_python.packages import get_package_share_directory

from rviz_utils.hri.face import GAZE_IN_FACE, FaceChain, matrix_from_pose, pinhole_k, quat_from_matrix, root_drop

_HEIGHT = 1.65
_BODY = "env_0_agent_7"


def _urdf(height: float = _HEIGHT) -> str:
    path = f"{get_package_share_directory('human_description')}/urdf/human-tpl.xacro"
    return xacro.process_file(path, mappings={"id": _BODY, "height": str(height)}).toprettyxml(indent="  ")


def _proportions(height: float) -> tuple[float, float, float]:
    """(torso top z, neck length, head radius) from the xacro's figure-drawing units."""
    unit = height / 7.5
    return unit * 2.75, unit * 0.25, unit / 2


@pytest.fixture(scope="module")
def chain() -> FaceChain:
    return FaceChain.from_urdf(_urdf(), _BODY)


def test_rest_face_sits_on_the_eye_midpoint_looking_forward(chain: FaceChain):
    torso, neck, r = _proportions(_HEIGHT)
    face = chain.pose({})
    np.testing.assert_allclose(face[:3, 3], (r / 2 + 0.4 * r, 0.0, torso + neck + r), atol=1e-9)
    np.testing.assert_allclose(face[:3, :3], np.eye(3), atol=1e-9)


def test_head_yaw_swings_the_face_around_the_neck(chain: FaceChain):
    torso, neck, r = _proportions(_HEIGHT)
    face = chain.pose({f"y_head_{_BODY}": math.pi / 2})
    np.testing.assert_allclose(face[:3, 3], (0.0, r / 2 + 0.4 * r, torso + neck + r), atol=1e-9)
    np.testing.assert_allclose(face[:3, 0], (0.0, 1.0, 0.0), atol=1e-9)


def test_head_pitch_about_minus_y_tilts_the_view_up(chain: FaceChain):
    face = chain.pose({f"p_head_{_BODY}": 0.4})
    np.testing.assert_allclose(face[:3, 0], (math.cos(0.4), 0.0, math.sin(0.4)), atol=1e-9)


def test_spine_bend_carries_the_face(chain: FaceChain):
    rest = chain.pose({})
    bent = chain.pose({f"waist_{_BODY}": 0.3})
    assert bent[0, 3] > rest[0, 3]
    assert bent[2, 3] < rest[2, 3]


def test_taller_body_raises_the_face():
    tall = FaceChain.from_urdf(_urdf(1.9), _BODY).pose({})
    torso, neck, r = _proportions(1.9)
    assert tall[2, 3] == pytest.approx(torso + neck + r)


@pytest.mark.parametrize(
    "wxyz",
    [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.5, -0.5, 0.5, -0.5), (0.1, 0.7, -0.3, 0.64)],
)
def test_quaternion_round_trips_through_a_matrix(wxyz: tuple[float, float, float, float]):
    q = np.asarray(wxyz) / np.linalg.norm(wxyz)
    back = np.asarray(quat_from_matrix(matrix_from_pose((1.0, 2.0, 3.0), tuple(q))))
    assert min(np.linalg.norm(back - q), np.linalg.norm(back + q)) < 1e-9


def test_gaze_is_the_optical_frame_of_the_face():
    rot = matrix_from_pose((0.0, 0.0, 0.0), GAZE_IN_FACE)[:3, :3]
    np.testing.assert_allclose(rot @ (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(rot @ (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(rot @ (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), atol=1e-9)


def test_pinhole_k_spans_the_requested_fov():
    k = pinhole_k(math.radians(60.0), 512)
    assert 2.0 * math.atan(256.0 / k[0]) == pytest.approx(math.radians(60.0))
    assert k[0] == k[4]
    assert (k[2], k[5]) == (256.0, 256.0)


def test_root_drop_reads_the_rig_beside_the_actor_sdf(tmp_path):
    (tmp_path / "rig.yaml").write_text("skeleton: cmu\nroot_offset: [0.0, 0.0, -0.3316]\n")
    assert root_drop(str(tmp_path / "seated.sdf")) == pytest.approx(-0.3316)


def test_root_drop_is_zero_for_a_standing_bundle(tmp_path):
    (tmp_path / "rig.yaml").write_text("skeleton: cmu\n")
    assert root_drop(str(tmp_path / "standing.sdf")) == 0.0
    assert root_drop(str(tmp_path / "missing" / "actor.sdf")) == 0.0
    assert root_drop("") == 0.0
