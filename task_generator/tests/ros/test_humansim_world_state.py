"""The bridge tells arena_humansim where the robot *is*, every feedback tick, not only where
the task generator put it."""
from __future__ import annotations

from types import SimpleNamespace

import attrs
import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("arena_humansim_msgs.msg")


@attrs.define
class _Robot:
    name: str
    pose: tuple[float, float]


def _bridge(managers):
    from task_generator.simulators.human.arena_humansim.arena_humansim import ArenaHumanSimulator

    fake = SimpleNamespace(node=SimpleNamespace(robots_manager=SimpleNamespace(managers=managers)), _tracked_robots={}, _dirty_robots={})
    fake.refresh = lambda: ArenaHumanSimulator._refresh_robot_poses(fake)
    return fake


def test_a_placed_robot_goes_out_with_its_live_pose_every_tick():
    manager = SimpleNamespace(pose=(3.0, 4.0))
    b = _bridge({"jackal": manager})
    b._tracked_robots["jackal"] = _Robot("jackal", pose=(0.0, 0.0))
    b.refresh()
    assert b._dirty_robots["jackal"].pose == (3.0, 4.0)
    b._dirty_robots.clear()  # what _publish_world_state does after sending
    manager.pose = (3.5, 4.0)
    b.refresh()
    assert b._dirty_robots["jackal"].pose == (3.5, 4.0), "the next tick carries the newer pose"


def test_a_robot_without_a_live_pose_keeps_its_placement():
    b = _bridge({"jackal": SimpleNamespace(pose=None)})
    b._tracked_robots["jackal"] = _Robot("jackal", pose=(1.0, 2.0))
    b.refresh()
    assert b._dirty_robots["jackal"].pose == (1.0, 2.0)


def test_no_managers_means_nothing_to_refresh():
    b = _bridge({})
    b._tracked_robots["jackal"] = _Robot("jackal", pose=(1.0, 2.0))
    b.refresh()
    assert b._dirty_robots == {}
