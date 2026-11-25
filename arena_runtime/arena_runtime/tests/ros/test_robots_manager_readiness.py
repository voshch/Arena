from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture()
def stub_manager():
    from arena_bringup.constants import SimSimulator

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
    from arena_runtime.env.robot_manager import robots_manager

    prior = robots_manager._READINESS_CACHE

    def seed(pending: dict[str, frozenset[str]], components_pending: dict[str, frozenset[str]] | None = None):
        robots_manager._READINESS_CACHE = robots_manager._Readiness(ready=frozenset(), pending=pending, components_pending=components_pending or {})

    yield seed
    robots_manager._READINESS_CACHE = prior


def test_uninstalled_entry_skipped_rest_kept(stub_manager, _seed_readiness):
    from arena_runtime.env.robot_manager.robots_manager import RobotsManager

    _seed_readiness({"turtlebot3_burger": frozenset({"arena_robots/deps/tb3"})})
    diff = RobotsManager._parse_robot_configurations(stub_manager, "turtlebot3_burger,jackal")
    assert [r.model.name for r in diff.to_add.values()] == ["jackal"]


def test_all_entries_uninstalled_raises(stub_manager, _seed_readiness):
    from arena_runtime.env.robot_manager.robots_manager import RobotsManager

    _seed_readiness({"turtlebot3_burger": frozenset({"arena_robots/deps/tb3"})})
    with pytest.raises(RuntimeError, match="no installed robot"):
        RobotsManager._parse_robot_configurations(stub_manager, "turtlebot3_burger")


def test_installed_roster_unaffected(stub_manager, _seed_readiness):
    from arena_runtime.env.robot_manager.robots_manager import RobotsManager

    _seed_readiness({})
    diff = RobotsManager._parse_robot_configurations(stub_manager, "jackal")
    assert [r.model.name for r in diff.to_add.values()] == ["jackal"]


def test_robot_with_uninstalled_component_skipped_rest_kept(stub_manager, _seed_readiness):
    from arena_runtime.env.robot_manager.robots_manager import RobotsManager

    _seed_readiness({}, {"arm/ur5e": frozenset({"arena_robots/deps/ur_description"})})
    diff = RobotsManager._parse_robot_configurations(stub_manager, "jackal[name=plain],jackal[name=armed,top=arm/ur5e]")
    assert list(diff.to_add) == ["plain"]


def test_overridden_robot_with_uninstalled_component_skipped_rest_kept(stub_manager, _seed_readiness):
    from arena_runtime.env.robot_manager.robots_manager import RobotsManager

    _seed_readiness({}, {"arm/fr3": frozenset({"arena_robots/deps/franka_description"})})
    diff = RobotsManager._parse_robot_configurations(stub_manager, "jackal[name=plain],jackal[name=armed,top=arm/fr3[stall_time=2.0]]")
    assert list(diff.to_add) == ["plain"]


def test_only_robot_with_uninstalled_component_raises_with_install_command(stub_manager, _seed_readiness):
    from arena_runtime.env.robot_manager.robots_manager import RobotsManager

    _seed_readiness({}, {"arm/ur5e": frozenset({"arena_robots/deps/ur_description"})})
    with pytest.raises(RuntimeError, match="arena feature robots add arm/ur5e"):
        RobotsManager._parse_robot_configurations(stub_manager, "jackal[name=armed,top=arm/ur5e]")
