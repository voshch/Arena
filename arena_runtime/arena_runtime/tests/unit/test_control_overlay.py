from __future__ import annotations

from pathlib import Path

import yaml

from arena_bringup.constants import SimSimulator
from arena_robots.caps import PlanarSpec
from arena_runtime.sim._control import effective_control_yaml, load_control_yaml, overlay_planar_limits

BASE = {
    'controller_manager': {'ros__parameters': {'update_rate': 50}},
    'base_controller': {
        'ros__parameters': {
            'wheel_separation': 0.4,
            'wheel_separation_multiplier': 1.5,
            'odom_frame_id': 'odom',
        }
    },
}


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def test_sim_overlay_replaces_only_the_keys_it_names(tmp_path):
    config = _write(tmp_path, 'control.yaml', BASE)
    _write(tmp_path, 'control.gazebo.yaml', {'base_controller': {'ros__parameters': {'wheel_separation_multiplier': 1.1}}})
    loaded = load_control_yaml(str(config), SimSimulator.GAZEBO)
    assert loaded['base_controller']['ros__parameters'] == {'wheel_separation': 0.4, 'wheel_separation_multiplier': 1.1, 'odom_frame_id': 'odom'}
    assert loaded['controller_manager'] == BASE['controller_manager']


def test_other_sims_and_no_sim_ignore_the_overlay(tmp_path):
    config = _write(tmp_path, 'control.yaml', BASE)
    _write(tmp_path, 'control.gazebo.yaml', {'base_controller': {'ros__parameters': {'wheel_separation_multiplier': 1.1}}})
    assert load_control_yaml(str(config), SimSimulator.MUJOCO) == BASE
    assert load_control_yaml(str(config)) == BASE


def test_missing_overlay_loads_the_base_config(tmp_path):
    config = _write(tmp_path, 'control.yaml', BASE)
    assert load_control_yaml(str(config), SimSimulator.GAZEBO) == BASE


def test_rendered_control_yaml_carries_the_overlay_and_prefixed_frames(tmp_path):
    config = _write(tmp_path, 'control.yaml', BASE)
    _write(tmp_path, 'control.gazebo.yaml', {'base_controller': {'ros__parameters': {'wheel_separation_multiplier': 1.1}}})
    out = Path(effective_control_yaml(None, str(config), f'test_overlay/{tmp_path.name}', 'env_0/r/', sim=SimSimulator.GAZEBO))
    params = yaml.safe_load(out.read_text())['/**/base_controller']['ros__parameters']
    out.unlink()
    assert params['wheel_separation_multiplier'] == 1.1
    assert params['odom_frame_id'] == 'env_0/r/odom'


def test_jackal_turns_on_its_gazebo_track_only_in_gazebo():
    config = 'package://arena_robots/robots/jackal/control.yaml'
    multiplier = {sim: load_control_yaml(config, sim)['jackal_velocity_controller']['ros__parameters']['wheel_separation_multiplier'] for sim in SimSimulator}
    assert multiplier.pop(SimSimulator.GAZEBO) == 1.106
    assert set(multiplier.values()) == {1.5}


DRIVE = {
    'drive_controller': {
        'ros__parameters': {
            'wheel_separation': 0.4,
            'linear.x.max_velocity': 2.0,
            'linear.x.min_velocity': -2.0,
            'linear.x.max_acceleration': 20.0,
            'angular.z.max_velocity': 4.0,
        }
    },
    'controller_manager': {'ros__parameters': {'update_rate': 50}},
}


def _planar(**fields: object) -> PlanarSpec:
    return PlanarSpec.from_dict({'velocity': {'linear': [-0.5, 0.5], 'lateral': [0.0, 0.0], 'angular': [-1.0, 1.0]}, 'acceleration': {'linear': 5.0, 'lateral': float('inf'), 'angular': float('inf')}, **fields})


def test_planar_limits_overlay_the_declared_keys_only():
    params = DRIVE['drive_controller']['ros__parameters']
    out = overlay_planar_limits(params, _planar())
    assert out == {
        'wheel_separation': 0.4,
        'linear.x.max_velocity': 0.5,
        'linear.x.min_velocity': -0.5,
        'linear.x.max_acceleration': 5.0,
        'angular.z.max_velocity': 1.0,
    }
    assert params['linear.x.max_velocity'] == 2.0


def test_infinite_planar_limits_leave_the_controller_keys_alone():
    params = {'linear.x.max_acceleration': 20.0, 'linear.x.max_velocity': 2.0}
    out = overlay_planar_limits(params, _planar(acceleration={'linear': float('inf')}))
    assert out['linear.x.max_acceleration'] == 20.0
    assert out['linear.x.max_velocity'] == 0.5


def test_rendered_control_yaml_carries_the_planar_override_and_leaves_the_rest(tmp_path):
    config = _write(tmp_path, 'control.yaml', DRIVE)
    out = Path(effective_control_yaml(None, str(config), f'test_planar/{tmp_path.name}', 'env_0/r/', sim=SimSimulator.GAZEBO, planar=_planar()))
    rendered = yaml.safe_load(out.read_text())
    assert rendered['/**/drive_controller']['ros__parameters']['linear.x.max_velocity'] == 0.5
    assert rendered['/**/controller_manager'] == DRIVE['controller_manager']


def test_rendered_control_yaml_without_a_planar_override_is_the_file_content(tmp_path):
    config = _write(tmp_path, 'control.yaml', DRIVE)
    out = Path(effective_control_yaml(None, str(config), f'test_no_planar/{tmp_path.name}', 'env_0/r/', sim=SimSimulator.GAZEBO))
    rendered = yaml.safe_load(out.read_text())
    assert rendered['/**/drive_controller']['ros__parameters'] == DRIVE['drive_controller']['ros__parameters']
