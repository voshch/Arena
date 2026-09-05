from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


@pytest.fixture()
def stub_manager():
    from arena_runtime.constants import SimSimulator

    class _FakeConf:
        class Robot:
            class TIMEOUT:
                value = 60
        class Arena:
            class SIM:
                value = SimSimulator.GAZEBO

    class _FakeLogger:
        def get_child(self, name):
            return self
        def debug(self, *a, **kw): ...
        def info(self, *a, **kw): ...
        def warn(self, *a, **kw): ...
        def error(self, *a, **kw): ...

    node = SimpleNamespace(
        conf=_FakeConf(),
        sim_time=SimpleNamespace(sec=0),
        get_logger=lambda: _FakeLogger(),
    )
    return SimpleNamespace(node=node, managers={})


@pytest.fixture()
def _seed_readiness():
    """Pin the readiness cache to a chosen value for the test, restoring after."""
    from task_generator.manager.robot_manager import robots_manager

    prior = robots_manager._READINESS_CACHE

    def seed(pending: dict[str, frozenset[str]]):
        robots_manager._READINESS_CACHE = robots_manager._Readiness(
            ready=frozenset(), pending=pending
        )

    yield seed
    robots_manager._READINESS_CACHE = prior


def test_uninstalled_entry_skipped_rest_kept(stub_manager, _seed_readiness):
    from task_generator.manager.robot_manager.robots_manager import RobotsManager

    _seed_readiness({"turtlebot3_burger": frozenset({"arena_robots/deps/tb3"})})
    diff = RobotsManager._parse_robot_configurations(stub_manager, "turtlebot3_burger,jackal")
    assert [r.model.name for r in diff.to_add.values()] == ["jackal"]


def test_all_entries_uninstalled_raises(stub_manager, _seed_readiness):
    from task_generator.manager.robot_manager.robots_manager import RobotsManager

    _seed_readiness({"turtlebot3_burger": frozenset({"arena_robots/deps/tb3"})})
    with pytest.raises(RuntimeError, match="no installed robot"):
        RobotsManager._parse_robot_configurations(stub_manager, "turtlebot3_burger")


def test_installed_roster_unaffected(stub_manager, _seed_readiness):
    from task_generator.manager.robot_manager.robots_manager import RobotsManager

    _seed_readiness({})
    diff = RobotsManager._parse_robot_configurations(stub_manager, "jackal")
    assert [r.model.name for r in diff.to_add.values()] == ["jackal"]
