import math

import mujoco
import numpy as np
import pytest

from arena_mujoco.services import SpawnUrdf
from arena_mujoco.viewport.camera import Lens, orbit_of, pose_of, render
from arena_mujoco.viewport.controller import FULL, POSITION_ONLY, YAW_ONLY, Keyframe, Pose, ViewportController, look_at, q_from_euler, q_rotate

_WIDTH = 320
_HEIGHT = 240
_RED = (220, 30, 30)
_GREEN = (30, 220, 30)
_SCENE = """
<mujoco>
  <visual><global offwidth="320" offheight="240"/><headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
  <worldbody>
    <geom name="red" type="box" size="0.5 0.5 0.5" pos="5 0 0" rgba="0.863 0.118 0.118 1"/>
    <geom name="green" type="box" size="0.25 0.25 0.25" pos="5 2 0" rgba="0.118 0.863 0.118 1"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def stage() -> tuple[mujoco.Renderer, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        renderer = mujoco.Renderer(model, _HEIGHT, _WIDTH)
    except Exception as exc:
        pytest.skip(f'no offscreen GL: {exc}')
    yield renderer, data
    renderer.close()


def _shot(stage: tuple[mujoco.Renderer, mujoco.MjData], pose: Pose, lens: Lens | None = None) -> np.ndarray:
    renderer, data = stage
    return render(renderer, data, pose, lens or Lens(), mujoco.MjvOption())


def _is(pixel: np.ndarray, color: tuple[int, int, int]) -> bool:
    return bool(np.all(np.abs(pixel.astype(int) - color) < 40))


def _columns(frame: np.ndarray, color: tuple[int, int, int]) -> int:
    return int(np.all(np.abs(frame[_HEIGHT // 2].astype(int) - color) < 40, axis=1).sum())


def test_camera_looks_along_its_x_axis_with_z_up(stage):
    frame = _shot(stage, Pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
    assert _is(frame[_HEIGHT // 2, _WIDTH // 2], _RED)
    green = np.argwhere(np.all(np.abs(frame.astype(int) - _GREEN) < 40, axis=2))
    assert len(green) > 0
    assert green[:, 1].max() < _WIDTH // 2
    away = _shot(stage, Pose((0.0, 0.0, 0.0), q_from_euler(0.0, 0.0, math.pi)))
    assert not _is(away[_HEIGHT // 2, _WIDTH // 2], _RED)


def test_roll_turns_the_image_about_the_view_axis(stage):
    frame = _shot(stage, Pose((0.0, 0.0, 0.0), q_from_euler(math.pi / 2, 0.0, 0.0)))
    assert _is(frame[_HEIGHT // 2, _WIDTH // 2], _RED)
    green = np.argwhere(np.all(np.abs(frame.astype(int) - _GREEN) < 40, axis=2))
    assert len(green) > 0
    assert abs(green[:, 1].mean() - _WIDTH / 2) < 12
    assert abs(green[:, 0].mean() - _HEIGHT / 2) > 40


def test_horizontal_fov_sets_how_wide_the_box_shows(stage):
    pose = Pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    for hfov in (0.6, 1.2):
        expected = _WIDTH * (0.5 / 4.5) / math.tan(hfov / 2.0)
        assert _columns(_shot(stage, pose, Lens(hfov=hfov)), _RED) == pytest.approx(expected, abs=3)


def test_orthographic_width_is_the_perspective_width_at_the_ground(stage):
    lens = Lens(hfov=1.0, orthographic=True)
    down = q_from_euler(0.0, math.pi / 2, 0.0)
    near = _shot(stage, Pose((5.0, 0.0, 4.0), down), lens)
    far = _shot(stage, Pose((5.0, 0.0, 8.0), down), lens)
    assert _columns(near, _RED) == pytest.approx(_WIDTH * 0.5 / (4.0 * math.tan(0.5)), abs=3)
    assert _columns(far, _RED) == pytest.approx(_WIDTH * 0.5 / (8.0 * math.tan(0.5)), abs=3)


def test_free_camera_orbit_and_pose_convert_both_ways():
    pose = pose_of((1.0, 2.0, 0.5), 4.0, 30.0, -20.0)
    forward = q_rotate(pose.orientation, (1.0, 0.0, 0.0))
    assert np.add(pose.position, np.multiply(forward, 4.0)) == pytest.approx((1.0, 2.0, 0.5))
    lookat, azimuth, elevation = orbit_of(pose, 4.0)
    assert lookat == pytest.approx((1.0, 2.0, 0.5))
    assert (azimuth, elevation) == pytest.approx((30.0, -20.0))
    assert q_rotate(pose.orientation, (0.0, 1.0, 0.0))[2] == pytest.approx(0.0, abs=1e-9)


def test_set_view_snaps_once_then_hands_the_camera_back():
    controller = ViewportController()
    controller.set_view((0.0, 0.0, 2.0), (4.0, 0.0, 2.0), 0.8)
    frame = controller.apply(0.0, Pose())
    assert frame.pose.position == (0.0, 0.0, 2.0)
    assert q_rotate(frame.pose.orientation, (1.0, 0.0, 0.0)) == pytest.approx((1.0, 0.0, 0.0))
    assert frame.fov == 0.8
    assert controller.apply(0.1, frame.pose).pose is None


@pytest.mark.parametrize(
    ('mode', 'forward'),
    [(FULL, (0.0, 1.0, 0.0)), (YAW_ONLY, (0.0, 1.0, 0.0)), (POSITION_ONLY, (1.0, 0.0, 0.0))],
)
def test_tracked_entity_carries_the_camera_by_the_chosen_channels(mode: int, forward: tuple[float, float, float]):
    controller = ViewportController()
    controller.set_reference_frame('robot', Pose(), False, mode)
    controller.set_reference_target(Pose((10.0, 0.0, 0.0), q_from_euler(0.0, 0.0, math.pi / 2)))
    controller.set_local(Pose((-3.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0)), False, 0.0)
    first = controller.apply(0.0, Pose())
    offset = (10.0, -3.0, 1.0) if mode != POSITION_ONLY else (7.0, 0.0, 1.0)
    assert first.pose.position == pytest.approx(offset)
    assert q_rotate(first.pose.orientation, (1.0, 0.0, 0.0)) == pytest.approx(forward, abs=1e-9)
    controller.set_reference_target(Pose((11.0, 0.0, 0.0), q_from_euler(0.0, 0.0, math.pi / 2)))
    followed = controller.apply(0.1, first.pose)
    assert followed.pose.position[0] == pytest.approx(offset[0] + 1.0)


def test_moving_the_camera_by_hand_releases_the_tracked_entity():
    controller = ViewportController()
    controller.set_reference_frame('robot', Pose(), False, FULL)
    controller.set_reference_target(Pose((10.0, 0.0, 0.0)))
    controller.set_local(Pose((-3.0, 0.0, 1.0)), False, 0.0)
    held = controller.apply(0.0, Pose()).pose
    controller.apply(0.1, held)
    assert controller.released is None
    assert controller.apply(0.2, Pose((0.0, 5.0, 1.0))).pose is None
    assert controller.released[0] == pytest.approx(math.dist((7.0, 0.0, 1.0), (0.0, 5.0, 1.0)))
    assert controller.tracked_entity == ''


def test_streamed_keyframes_interpolate_then_release_after_the_stream_goes_quiet():
    controller = ViewportController()
    aim = look_at((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    controller.push_keyframe(Keyframe(1.0, Pose((0.0, 0.0, 0.0), aim), False, 0.0), 0.9)
    controller.push_keyframe(Keyframe(2.0, Pose((4.0, 0.0, 0.0), aim), False, 0.7), 0.9)
    assert controller.apply(0.95, Pose()).pose.position == pytest.approx((0.0, 0.0, 0.0))
    middle = controller.apply(1.25, Pose())
    assert middle.pose.position == pytest.approx((1.0, 0.0, 0.0))
    assert middle.fov == 0.7
    assert controller.apply(1.4, Pose()).pose is None


def test_latching_freezes_the_reference_where_the_entity_stood():
    controller = ViewportController()
    controller.set_reference_frame('robot', Pose(), False, FULL)
    controller.set_reference_target(Pose((10.0, 0.0, 0.0)))
    assert controller.set_reference_frame('', Pose(), False, FULL) == 'latched current pose'
    controller.set_local(Pose((-3.0, 0.0, 1.0)), False, 0.0)
    assert controller.apply(0.0, Pose()).pose.position == pytest.approx((7.0, 0.0, 1.0))


def test_robot_link_frames_name_the_bodies_of_that_robot():
    entry = SpawnUrdf.RobotEntry(
        robot_model='jackal',
        tf_prefix='env_0/jackal/',
        base_frame='base_link',
        prim_name='env_0_Robots_jackal_base_link',
        localization=True,
        joint_names=(),
        joint_states_topic='',
        cmd_vel_topic='',
        odom_topic='',
        command_interfaces={},
        urdf_sensors={},
        sensors=(),
    )
    SpawnUrdf._robots[(0, 'env_0/Robots/jackal')] = entry
    try:
        assert SpawnUrdf.frame_body(0, 'env_0/jackal/base_link') == 'env_0_Robots_jackal_base_link'
        assert SpawnUrdf.frame_body(0, 'env_0/jackal/front-laser') == 'env_0_Robots_jackal_front_laser'
        assert SpawnUrdf.frame_body(1, 'env_0/jackal/base_link') is None
        assert SpawnUrdf.frame_body(0, 'env_0/map') is None
    finally:
        del SpawnUrdf._robots[(0, 'env_0/Robots/jackal')]
