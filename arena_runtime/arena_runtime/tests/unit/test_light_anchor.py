"""Frame-anchored light placement and the move threshold."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("arena_runtime.sim._mechanism_shim")

from arena_runtime.sim._mechanism_shim import LIGHT_MOVE_EPSILON, _light_moved, light_anchor  # noqa: E402
from arena_simulation_setup.utils.geometry import Position  # noqa: E402


def test_offset_turns_with_frame_yaw() -> None:
    anchored = light_anchor(Position(2.0, 1.0, 0.1), math.pi / 2, Position(0.5, 0.0, 0.3))
    assert (anchored.x, anchored.y, anchored.z) == pytest.approx((2.0, 1.5, 0.4))


def test_zero_offset_sits_on_the_frame() -> None:
    anchored = light_anchor(Position(-3.0, 4.0, 0.0), 1.0, Position(0.0, 0.0, 0.0))
    assert (anchored.x, anchored.y, anchored.z) == pytest.approx((-3.0, 4.0, 0.0))


def test_first_pose_always_moves() -> None:
    assert _light_moved(None, Position(0.0, 0.0, 0.0), 0.0)


@pytest.mark.parametrize(
    ("position", "yaw", "moved"),
    [
        (Position(LIGHT_MOVE_EPSILON / 2, 0.0, 0.0), 0.0, False),
        (Position(LIGHT_MOVE_EPSILON * 2, 0.0, 0.0), 0.0, True),
        (Position(0.0, 0.0, 0.0), LIGHT_MOVE_EPSILON * 2, True),
        (Position(0.0, 0.0, 0.0), math.tau, False),
    ],
)
def test_move_threshold(position: Position, yaw: float, moved: bool) -> None:
    assert _light_moved((Position(0.0, 0.0, 0.0), 0.0), position, yaw) is moved
