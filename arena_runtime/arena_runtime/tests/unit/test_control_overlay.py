from __future__ import annotations

from pathlib import Path

import yaml

from arena_bringup.constants import SimSimulator
from arena_runtime.sim._control import effective_control_yaml, load_control_yaml

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
