from __future__ import annotations

import pytest

from arena_rclpy_mixins.launch_argument import LaunchArgument


def test_missing_description_is_rejected():
    with pytest.raises(TypeError):
        LaunchArgument("x", default_value="1")


def test_description_is_exposed():
    arg = LaunchArgument("x", default_value="1", description="the x value")
    assert arg.name == "x"
    assert arg.description == "the x value"
    assert arg.default_value is not None


def test_choices_are_kept():
    arg = LaunchArgument("x", default_value="a", description="d", choices=["a", "b"])
    assert arg.choices == ["a", "b"]
