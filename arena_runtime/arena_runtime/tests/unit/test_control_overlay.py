from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from ament_index_python.packages import get_package_share_directory

from arena_bringup.constants import SimSimulator
import arena_robots.Robot
from arena_robots.caps import PlanarSpec
from arena_robots.fleet import instance_caps
from arena_runtime.sim._control import effective_control_yaml, overlay_planar_limits

JACKAL = arena_robots.Robot.RobotIdentifier('jackal').resolve_sync()


def _controller(sim: SimSimulator) -> dict:
    return JACKAL.control_config(sim.value)['jackal_velocity_controller']['ros__parameters']


def test_rendered_control_yaml_carries_the_overlay_and_prefixed_frames():
    out = Path(effective_control_yaml(None, JACKAL, 'test_overlay/jackal', 'env_0/r/', sim=SimSimulator.GAZEBO))
    params = yaml.safe_load(out.read_text())['/**/jackal_velocity_controller']['ros__parameters']
    out.unlink()
    assert params['wheel_separation_multiplier'] == 1.106
    assert params['odom_frame_id'] == 'env_0/r/odom'


def test_jackal_turns_on_its_gazebo_track_only_in_gazebo():
    multiplier = {sim: _controller(sim)['wheel_separation_multiplier'] for sim in SimSimulator}
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


def test_rendered_control_yaml_carries_the_planar_override_and_leaves_the_rest():
    out = Path(effective_control_yaml(None, JACKAL, 'test_planar/jackal', 'env_0/r/', sim=SimSimulator.GAZEBO, planar=_planar()))
    rendered = yaml.safe_load(out.read_text())
    out.unlink()
    assert rendered['/**/jackal_velocity_controller']['ros__parameters']['linear.x.max_velocity'] == 0.5
    assert rendered['/**/controller_manager'] == JACKAL.control_config(SimSimulator.GAZEBO.value)['controller_manager']


def test_rendered_control_yaml_without_a_planar_override_keeps_the_controller_limits():
    out = Path(effective_control_yaml(None, JACKAL, 'test_no_planar/jackal', 'env_0/r/', sim=SimSimulator.GAZEBO))
    rendered = yaml.safe_load(out.read_text())
    out.unlink()
    expected = {k: v for k, v in _controller(SimSimulator.GAZEBO).items() if not k.endswith('_frame_id')}
    assert {k: v for k, v in rendered['/**/jackal_velocity_controller']['ros__parameters'].items() if not k.endswith('_frame_id')} == expected


def _shipped_control_files() -> list[tuple[str, str]]:
    root = Path(get_package_share_directory('arena_robots')) / 'robots'
    return [(path.parent.name, str(path)) for path in sorted(root.glob('*/control*.yaml'))]


@pytest.mark.parametrize(('model', 'path'), _shipped_control_files())
def test_the_effective_planar_limits_of_a_shipped_robot_leave_its_controller_sections_unchanged(model, path):
    caps = arena_robots.Robot.RobotIdentifier(model).resolve_sync().caps
    if 'mobile' not in caps.available:
        pytest.skip(f'{model} has no mobile cap')
    control = yaml.safe_load(Path(path).read_text())
    for name, section in control.items():
        if name == 'controller_manager' or not isinstance(section, dict) or not isinstance(section.get('ros__parameters'), dict):
            continue
        params = section['ros__parameters']
        assert overlay_planar_limits(params, caps.mobile.planar) == params, f'{path}: {name}'


def test_a_halving_limits_override_halves_the_jackal_controller_limits():
    base = JACKAL.control_config()['jackal_velocity_controller']['ros__parameters']
    planar = instance_caps('jackal', {'cap:mobile.limits': '{"scale": 0.5}'}).mobile.planar
    halved = overlay_planar_limits(base, planar)
    assert halved['linear.x.max_velocity'] == pytest.approx(base['linear.x.max_velocity'] / 2)
    assert halved['angular.z.max_velocity'] == pytest.approx(base['angular.z.max_velocity'] / 2)
