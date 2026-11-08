from __future__ import annotations

import pytest

from arena_bringup.substitutions import LaunchArgument


def test_missing_description_is_rejected():
    with pytest.raises(TypeError):
        LaunchArgument("x", default_value="1")


@pytest.mark.parametrize("description", ["", "   "])
def test_blank_description_raises_value_error_naming_the_argument(description):
    with pytest.raises(ValueError, match="'x'"):
        LaunchArgument("x", default_value="1", description=description)


def test_description_is_exposed():
    arg = LaunchArgument("x", default_value="1", description="the x value")
    assert arg.name == "x"
    assert arg.description == "the x value"
    assert arg.default_value is not None


def test_choices_are_kept():
    arg = LaunchArgument("x", default_value="a", description="d", choices=["a", "b"])
    assert arg.choices == ["a", "b"]
