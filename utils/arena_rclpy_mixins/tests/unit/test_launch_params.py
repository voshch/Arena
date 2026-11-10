from __future__ import annotations

import pytest

pytest.importorskip("rcl_interfaces")

from arena_rclpy_mixins.launch_params import launch_str_to_value, value_to_launch_str


@pytest.mark.parametrize("value", [3, -2, 0.5, 1e-07, "default", "a b", [1, 2], [0.5, 1.5, 1e-07], ["x", "y z"], []])
def test_value_to_launch_str_round_trips_through_launch_str_to_value(value: object) -> None:
    assert launch_str_to_value(value_to_launch_str(value)) == value


def test_value_to_launch_str_writes_booleans_as_launch_booleans() -> None:
    assert value_to_launch_str(True) == "true"
    assert value_to_launch_str(False) == "false"
    assert value_to_launch_str([True, False]) == "[true, false]"
