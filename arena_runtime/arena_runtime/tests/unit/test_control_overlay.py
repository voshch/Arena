from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from ament_index_python.packages import get_package_share_directory

from arena_bringup.constants import SimSimulator
import arena_robots.Robot
from arena_robots.caps import PlanarSpec
from arena_robots.fleet import instance_caps
from arena_runtime.sim._control import effective_control_yaml, overlay_drive_frames, overlay_drive_geometry, overlay_planar_limits

JACKAL = arena_robots.Robot.RobotIdentifier('jackal').resolve_sync()
JACKAL_URDF = JACKAL.effective_caps({}).rendered_urdf()
AXIS_PREFIXES = {'linear': 'linear.x', 'lateral': 'linear.y', 'angular': 'angular.z'}
LIMIT_VALUE_SUFFIXES = ('min_velocity', 'max_velocity', 'min_acceleration', 'max_acceleration')
JACKAL_GEOMETRY = {'wheel_radius': 0.098, 'wheel_separation': 0.37559}
JACKAL_FRAMES = {'base_frame': 'base_link', 'odom_frame': 'odom'}


def _controller(sim: SimSimulator) -> dict:
    return JACKAL.control_config(sim.value)['jackal_velocity_controller']['ros__parameters']


def test_rendered_control_yaml_carries_the_overlay_and_prefixed_frames():
    out = Path(effective_control_yaml(None, JACKAL, 'test_overlay/jackal', 'env_0/r/', sim=SimSimulator.GAZEBO, urdf=JACKAL_URDF, **JACKAL_FRAMES))
    params = yaml.safe_load(out.read_text())['/**/jackal_velocity_controller']['ros__parameters']
    out.unlink()
    assert params['wheel_separation_multiplier'] == 1.106
    assert params['odom_frame_id'] == 'env_0/r/odom'
    assert params['base_frame_id'] == 'env_0/r/base_link'


def test_jackal_turns_on_its_gazebo_track_only_in_gazebo():
    multiplier = {sim: _controller(sim)['wheel_separation_multiplier'] for sim in SimSimulator}
    assert multiplier.pop(SimSimulator.GAZEBO) == 1.106
    assert set(multiplier.values()) == {1.5}


def _planar(**fields: object) -> PlanarSpec:
    return PlanarSpec.from_dict({'velocity': {'linear': [-0.5, 0.5], 'lateral': [0.0, 0.0], 'angular': [-1.0, 1.0]}, 'acceleration': {'linear': 5.0, 'lateral': float('inf'), 'angular': float('inf')}, **fields})


FLAGS = {
    'wheel_separation': 0.4,
    'linear.x.has_velocity_limits': True,
    'linear.x.has_acceleration_limits': True,
    'angular.z.has_velocity_limits': True,
    'angular.z.has_acceleration_limits': False,
    'linear.y.has_velocity_limits': False,
}


def test_planar_limits_write_the_keys_of_every_flagged_axis_and_nothing_else():
    out = overlay_planar_limits(FLAGS, _planar())
    assert out == {
        **FLAGS,
        'linear.x.min_velocity': -0.5,
        'linear.x.max_velocity': 0.5,
        'linear.x.min_acceleration': -5.0,
        'linear.x.max_acceleration': 5.0,
        'angular.z.min_velocity': -1.0,
        'angular.z.max_velocity': 1.0,
    }
    assert 'linear.x.max_velocity' not in FLAGS


def test_infinite_planar_limits_leave_the_controller_keys_unset():
    out = overlay_planar_limits(FLAGS, _planar(acceleration={'linear': float('inf')}))
    assert 'linear.x.max_acceleration' not in out
    assert 'linear.x.min_acceleration' not in out
    assert out['linear.x.max_velocity'] == 0.5


def test_a_controller_without_limit_flags_gets_no_limit_keys():
    params = {'wheel_separation': 0.4}
    assert overlay_planar_limits(params, _planar()) == params


def test_an_axis_the_planar_spec_leaves_out_stays_unset():
    out = overlay_planar_limits({'linear.y.has_velocity_limits': True, 'linear.y.has_acceleration_limits': True}, PlanarSpec.from_dict({}))
    assert out == {'linear.y.has_velocity_limits': True, 'linear.y.has_acceleration_limits': True}


def test_rendered_control_yaml_carries_the_planar_override_and_leaves_the_rest():
    out = Path(effective_control_yaml(None, JACKAL, 'test_planar/jackal', 'env_0/r/', sim=SimSimulator.GAZEBO, urdf=JACKAL_URDF, planar=_planar(), **JACKAL_FRAMES))
    rendered = yaml.safe_load(out.read_text())
    out.unlink()
    assert rendered['/**/jackal_velocity_controller']['ros__parameters']['linear.x.max_velocity'] == 0.5
    assert rendered['/**/controller_manager'] == JACKAL.control_config(SimSimulator.GAZEBO.value)['controller_manager']


def test_rendered_control_yaml_without_a_planar_override_keeps_the_controller_as_authored():
    out = Path(effective_control_yaml(None, JACKAL, 'test_no_planar/jackal', 'env_0/r/', sim=SimSimulator.GAZEBO, urdf=JACKAL_URDF, **JACKAL_FRAMES))
    rendered = yaml.safe_load(out.read_text())
    out.unlink()
    assert rendered['/**/jackal_velocity_controller']['ros__parameters'] == {**_controller(SimSimulator.GAZEBO), **JACKAL_GEOMETRY, 'base_frame_id': 'env_0/r/base_link', 'odom_frame_id': 'env_0/r/odom'}


def _shipped_control_files() -> list[tuple[str, str]]:
    root = Path(get_package_share_directory('arena_robots')) / 'robots'
    return [(path.parent.name, str(path)) for path in sorted(root.glob('*/control*.yaml'))]


@pytest.mark.parametrize(('model', 'path'), _shipped_control_files())
def test_a_shipped_control_file_declares_flags_and_no_limit_values(model, path):
    control = yaml.safe_load(Path(path).read_text())
    for name, section in control.items():
        if name == 'controller_manager' or not isinstance(section, dict) or not isinstance(section.get('ros__parameters'), dict):
            continue
        assert not [key for key in section['ros__parameters'] if key.endswith(LIMIT_VALUE_SUFFIXES)], f'{path}: {name}'


@pytest.mark.parametrize(('model', 'path'), _shipped_control_files())
def test_the_planar_limits_of_a_shipped_robot_reach_every_flagged_axis(model, path):
    caps = arena_robots.Robot.RobotIdentifier(model).resolve_sync().caps
    if 'mobile' not in caps.available:
        pytest.skip(f'{model} has no mobile cap')
    planar = caps.mobile.planar
    for name, section in yaml.safe_load(Path(path).read_text()).items():
        if name == 'controller_manager' or not isinstance(section, dict) or not isinstance(section.get('ros__parameters'), dict):
            continue
        params = section['ros__parameters']
        out = overlay_planar_limits(params, planar)
        expected = {}
        for axis, prefix in AXIS_PREFIXES.items():
            if params.get(f'{prefix}.has_velocity_limits') is True and axis in planar.velocity:
                expected[f'{prefix}.min_velocity'] = planar.velocity[axis].min
                expected[f'{prefix}.max_velocity'] = planar.velocity[axis].max
            if params.get(f'{prefix}.has_acceleration_limits') is True and axis in planar.acceleration:
                expected[f'{prefix}.min_acceleration'] = -planar.acceleration[axis]
                expected[f'{prefix}.max_acceleration'] = planar.acceleration[axis]
        assert out == {**params, **{key: value for key, value in expected.items() if math.isfinite(value)}}, f'{path}: {name}'


def test_a_halving_limits_override_halves_the_jackal_controller_limits():
    full = overlay_planar_limits(JACKAL.control_config()['jackal_velocity_controller']['ros__parameters'], JACKAL.caps.mobile.planar)
    planar = instance_caps('jackal', {'cap:mobile.limits': '{"scale": 0.5}'}).mobile.planar
    halved = overlay_planar_limits(JACKAL.control_config()['jackal_velocity_controller']['ros__parameters'], planar)
    assert halved['linear.x.max_velocity'] == pytest.approx(full['linear.x.max_velocity'] / 2)
    assert halved['angular.z.max_velocity'] == pytest.approx(full['angular.z.max_velocity'] / 2)


def _rendered_drive(model: str, controller: str) -> dict:
    view = arena_robots.Robot.RobotIdentifier(model).resolve_sync()
    out = Path(effective_control_yaml(None, view, f'test_geometry/{model}', 'env_0/r/', sim=SimSimulator.GAZEBO, urdf=view.effective_caps({}).rendered_urdf(), base_frame=view.model_params.base_frame, odom_frame=view.model_params.odom_frame))
    rendered = yaml.safe_load(out.read_text())
    out.unlink()
    return rendered[f'/**/{controller}']['ros__parameters']


def test_the_drive_frames_overlay_writes_both_frames_and_leaves_the_input():
    params = {'wheel_radius': 0.5, 'base_frame_id': 'stale'}
    out = overlay_drive_frames(params, 'base_footprint', 'odom')
    assert out == {'wheel_radius': 0.5, 'base_frame_id': 'base_footprint', 'odom_frame_id': 'odom'}
    assert params == {'wheel_radius': 0.5, 'base_frame_id': 'stale'}


@pytest.mark.parametrize(('model', 'controller'), [('jackal', 'jackal_velocity_controller'), ('rbkairos', 'rbkairos_velocity_controller'), ('rbvogui', 'robotnik_base_controller'), ('rskomnidirectional', 'rskomni_velocity_controller')])
def test_every_drive_controller_is_rendered_with_the_robots_own_frames(model, controller):
    view = arena_robots.Robot.RobotIdentifier(model).resolve_sync()
    params = _rendered_drive(model, controller)
    assert params['base_frame_id'] == f'env_0/r/{view.model_params.base_frame}'
    assert params['odom_frame_id'] == f'env_0/r/{view.model_params.odom_frame}'


def test_a_shipped_control_file_leaves_the_frames_to_the_renderer():
    for _, path in _shipped_control_files():
        for name, section in yaml.safe_load(Path(path).read_text()).items():
            if name != 'controller_manager' and isinstance(section, dict) and isinstance(section.get('ros__parameters'), dict):
                assert not [key for key in section['ros__parameters'] if key in ('base_frame_id', 'odom_frame_id')], f'{path}: {name}'


def test_the_drive_geometry_overlay_writes_over_the_controller_and_leaves_the_input():
    params = {'wheel_radius': 0.5, 'wheel_separation_multiplier': 1.5}
    out = overlay_drive_geometry(params, {'wheel_radius': 0.098, 'wheel_separation': 0.37559})
    assert out == {'wheel_radius': 0.098, 'wheel_separation_multiplier': 1.5, 'wheel_separation': 0.37559}
    assert params['wheel_radius'] == 0.5


def test_jackal_rendered_control_yaml_carries_its_urdf_wheels():
    params = _rendered_drive('jackal', 'jackal_velocity_controller')
    assert params['wheel_radius'] == pytest.approx(0.098)
    assert params['wheel_separation'] == pytest.approx(0.37559)
    assert params['wheel_separation_multiplier'] == 1.106


def test_rbkairos_rendered_control_yaml_carries_the_mecanum_projection():
    params = _rendered_drive('rbkairos', 'rbkairos_velocity_controller')
    assert params['kinematics.wheels_radius'] == pytest.approx(0.127)
    assert params['kinematics.sum_of_robot_center_projection_on_X_Y_axis'] == pytest.approx(0.47428)


def test_rbvogui_rendered_control_yaml_carries_the_swerve_modules():
    params = _rendered_drive('rbvogui', 'robotnik_base_controller')
    assert params['wheel_radius'] == pytest.approx(0.1165)
    assert params['wheel_positions_x'] == pytest.approx([0.368, 0.368, -0.368, -0.368])
    assert params['wheel_positions_y'] == pytest.approx([-0.235, 0.235, -0.235, 0.235])


def test_rskomnidirectional_rendered_control_yaml_carries_the_omni_wheels():
    params = _rendered_drive('rskomnidirectional', 'rskomni_velocity_controller')
    assert params['wheel_radius'] == pytest.approx(0.0345)
    assert params['wheel_angles'] == pytest.approx([2.617995, 0.523599, 4.712393], abs=1e-5)
    assert params['direction'] == 'ccw'


def test_a_legged_robot_control_yaml_has_no_drive_geometry():
    view = arena_robots.Robot.RobotIdentifier('a1').resolve_sync()
    out = Path(effective_control_yaml(None, view, 'test_geometry/a1', 'env_0/r/', sim=SimSimulator.GAZEBO, urdf=view.effective_caps({}).rendered_urdf(), base_frame=view.model_params.base_frame, odom_frame=view.model_params.odom_frame))
    rendered = yaml.safe_load(out.read_text())
    out.unlink()
    assert rendered['/**/joint_group_position_controller'] == yaml.safe_load(yaml.safe_dump(view.control_config()['joint_group_position_controller']))
