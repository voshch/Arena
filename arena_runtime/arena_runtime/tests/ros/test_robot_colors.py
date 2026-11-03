from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("interactive_markers")


def test_robots_keep_their_color_when_another_robot_leaves():
    from arena_runtime.env.colors import PALETTE, RobotColors

    colors = RobotColors()
    first, second, third = colors.rgb("r1"), colors.rgb("r2"), colors.rgb("r3")
    assert len({first, second, third}) == 3
    colors.retain(["r1", "r3"])
    assert colors.rgb("r1") == first
    assert colors.rgb("r3") == third
    assert colors.rgb("r4") == second
    assert first == PALETTE[0]


def test_palette_wraps_once_every_slot_is_taken():
    from arena_runtime.env.colors import PALETTE, RobotColors

    colors = RobotColors()
    for index in range(len(PALETTE)):
        colors.rgb(f"r{index}")
    assert colors.rgb("extra") in PALETTE


def test_rgba_scales_to_unit_range():
    from arena_runtime.env.colors import PALETTE, RobotColors

    r, g, b = PALETTE[0]
    assert RobotColors().rgba("r1", 0.5) == (r / 255, g / 255, b / 255, 0.5)
