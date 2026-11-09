from __future__ import annotations

import launch.actions

from arena_rclpy_mixins.launch_argument import declared_arguments


def test_every_argument_of_arena_runtime_launch_has_a_description():
    undescribed = launch.actions.DeclareLaunchArgument("probe").description
    arguments = declared_arguments("arena_bringup", "arena_runtime.launch.py")
    assert "sim" in {argument.name for argument in arguments}
    assert [argument.name for argument in arguments if argument.description == undescribed] == []


def test_declared_arguments_carry_defaults_and_choices():
    arguments = {argument.name: argument for argument in declared_arguments("arena_bringup", "arena_runtime.launch.py")}
    assert arguments["world.lighting"].default == "authored"
    assert arguments["world.lighting"].choices == ("authored", "auto")
    assert "Valid choices" not in arguments["world.lighting"].description
