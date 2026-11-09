from __future__ import annotations

from arena_runtime.sim.flatsim_simulator.flatsim_simulator import mock_controllers


def test_an_arm_robot_runs_every_controller_but_the_base_one() -> None:
    names = ['joint_state_broadcaster', 'tiago_velocity_controller', 'torso_controller', 'arm_controller', 'gripper_controller']
    assert mock_controllers(names, 'tiago_velocity_controller') == ['joint_state_broadcaster', 'torso_controller', 'arm_controller', 'gripper_controller']


def test_a_base_only_robot_runs_no_controller_manager() -> None:
    assert mock_controllers(['joint_state_broadcaster', 'p3at_velocity_controller'], 'p3at_velocity_controller') == []
