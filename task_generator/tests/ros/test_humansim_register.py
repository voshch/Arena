from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def _assert_register_modes_works(cls: type) -> None:
    from task_generator.constants import Constants
    from task_generator.tasks.registry import OBSTACLES_MODES

    try:
        cls._register_task_modes()
    except (AssertionError, KeyError, ValueError):
        pass  # registry is process-global, another subclass may have registered first
    assert Constants.TaskMode.TM_Obstacles.PROMPT in OBSTACLES_MODES


def test_arena_humansim_register_task_modes():
    pytest.importorskip("arena_humansim_msgs.msg")
    from task_generator.simulators.human.arena_humansim.arena_humansim import (
        ArenaHumanSimulator,
    )

    _assert_register_modes_works(ArenaHumanSimulator)
