from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import arena_robots.Robot
from arena_bringup.constants import SimSimulator
from arena_robots.caps import DriveSpec, check_effort_drives
from arena_runtime.sim._control import effective_control_yaml

GO1 = arena_robots.Robot.RobotIdentifier('go1').resolve_sync()
JTC = 'joint_trajectory_controller/JointTrajectoryController'


def _control(kind: str, params: dict) -> dict:
    return {'controller_manager': {'ros__parameters': {'update_rate': 200, 'legs': {'type': kind}}}, 'legs': {'ros__parameters': params}}


def _drives(command: str = 'effort') -> dict[str, DriveSpec]:
    return {'a': DriveSpec(kp=50.0, kd=2.0, effort=30.0, command=command), 'b': DriveSpec(kp=40.0, kd=1.0, effort=20.0, command=command)}


def test_position_commanded_drives_pass_any_controller():
    check_effort_drives(_control(JTC, {'joints': ['a', 'b'], 'command_interfaces': ['position']}), _drives('position'))


@pytest.mark.parametrize(
    ('kind', 'params'),
    [
        (JTC, {'joints': ['a', 'b'], 'command_interfaces': ['effort']}),
        ('forward_command_controller/ForwardCommandController', {'joints': ['a', 'b'], 'interface_name': 'effort'}),
        ('effort_controllers/JointGroupEffortController', {'joints': ['a', 'b']}),
    ],
)
def test_effort_drives_pass_a_controller_commanding_effort(kind: str, params: dict):
    check_effort_drives(_control(kind, params), _drives())


def test_effort_drives_refuse_a_controller_commanding_position():
    with pytest.raises(ValueError, match='does not command effort'):
        check_effort_drives(_control(JTC, {'joints': ['a', 'b'], 'command_interfaces': ['position']}), _drives())


def test_go1_soft_gains_render_its_position_controller_unchanged():
    caps = GO1.effective_caps({}, caps={'drives.*.kp': 20, 'drives.*.kd': 0.5})
    assert caps.drives is not None
    out = Path(effective_control_yaml(None, GO1, 'test_effort/go1', 'env_0/go1/', sim=SimSimulator.MUJOCO, urdf=caps.rendered_urdf(), base_frame='base', odom_frame='odom', drives=caps.drives.all()))
    params = yaml.safe_load(out.read_text())['/**/joint_group_position_controller']['ros__parameters']
    assert params['command_interfaces'] == ['position']
    assert 'command_joints' not in params


def test_go1_effort_commanded_legs_refuse_its_position_controller():
    caps = GO1.effective_caps({}, caps={'drives.*.command': 'effort'})
    assert caps.drives is not None
    with pytest.raises(ValueError, match='joint_group_position_controller'):
        effective_control_yaml(None, GO1, 'test_effort/go1', 'env_0/go1/', sim=SimSimulator.MUJOCO, urdf=caps.rendered_urdf(), base_frame='base', odom_frame='odom', drives=caps.drives.all())
