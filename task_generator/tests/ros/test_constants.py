from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def test_tm_obstacles_prefix_no_args():
    from task_generator.constants import Constants
    ns = Constants.TaskMode.TM_Obstacles.prefix()
    assert str(ns) == "tm_obstacles"


def test_tm_obstacles_prefix_one_arg():
    from task_generator.constants import Constants
    ns = Constants.TaskMode.TM_Obstacles.prefix("file")
    assert "tm_obstacles" in str(ns)
    assert "file" in str(ns)


def test_tm_obstacles_prefix_two_args():
    from task_generator.constants import Constants
    ns = Constants.TaskMode.TM_Obstacles.prefix("a", "b")
    s = str(ns)
    assert "tm_obstacles" in s
    assert "a" in s
    assert "b" in s


def test_tm_robots_prefix_no_args():
    from task_generator.constants import Constants
    ns = Constants.TaskMode.TM_Robots.prefix()
    assert str(ns) == "tm_robots"


def test_tm_robots_prefix_one_arg():
    from task_generator.constants import Constants
    ns = Constants.TaskMode.TM_Robots.prefix("foo")
    assert "tm_robots" in str(ns)
    assert "foo" in str(ns)


def test_tm_module_prefix_no_args():
    from task_generator.constants import Constants
    ns = Constants.TaskMode.TM_Module.prefix()
    assert str(ns) == "tm_module"


def test_tm_module_prefix_one_arg():
    from task_generator.constants import Constants
    ns = Constants.TaskMode.TM_Module.prefix("bar")
    s = str(ns)
    assert "tm_module" in s
    assert "bar" in s


def test_tm_obstacles_default_returns_random():
    from task_generator.constants import Constants
    d = Constants.TaskMode.TM_Obstacles.default()
    assert d is Constants.TaskMode.TM_Obstacles.RANDOM


def test_tm_robots_default_returns_random():
    from task_generator.constants import Constants
    d = Constants.TaskMode.TM_Robots.default()
    assert d is Constants.TaskMode.TM_Robots.RANDOM


def test_tm_module_default_returns_empty_set():
    from task_generator.constants import Constants
    d = Constants.TaskMode.TM_Module.default()
    assert isinstance(d, set)
    assert len(d) == 0


def test_tm_obstacles_values_stable():
    from task_generator.constants import Constants
    TM = Constants.TaskMode.TM_Obstacles
    assert TM.PARAMETRIZED.value == "parametrized"
    assert TM.RANDOM.value == "random"
    assert TM.SCENARIO.value == "scenario"
    assert TM.ENVIRONMENT.value == "environment"
    assert TM.PROMPT.value == "prompt"


def test_tm_robots_values_stable():
    from task_generator.constants import Constants
    TM = Constants.TaskMode.TM_Robots
    assert TM.GUIDED.value == "guided"
    assert TM.EXPLORE.value == "explore"
    assert TM.RANDOM.value == "random"
    assert TM.SCENARIO.value == "scenario"


def test_tm_module_values_stable():
    from task_generator.constants import Constants
    TM = Constants.TaskMode.TM_Module
    assert TM.STAGED.value == "staged"
    assert TM.DYNAMIC_MAP.value == "dynamic_map"
    assert TM.CLEAR_FORBIDDEN_ZONES.value == "clear_forbidden_zones"
    assert TM.RVIZ_UI.value == "rviz_ui"


def test_sim_simulator_values_stable():
    from arena_runtime.constants import SimSimulator
    assert SimSimulator.DUMMY.value == "dummy"
    assert SimSimulator.FLATLAND.value == "flatland"
    assert SimSimulator.GAZEBO.value == "gazebo"
    assert SimSimulator.UNITY.value == "unity"
    assert SimSimulator.ISAAC.value == "isaac"


def test_human_simulator_values_stable():
    from task_generator.constants import Constants
    HS = Constants.HumanSimulator
    assert HS.DUMMY.value == "dummy"
    assert HS.NONE.value == "none"
    assert HS.ISAAC.value == "isaac"
    assert HS.HUNAV.value == "hunav"
    assert HS.ARENA.value == "arena"
    # hunav and arena are both registered — alternative backends chosen by `human:=`,
    # never concurrent. Dropping either changes what `human:=<key>` resolves to.
    assert {m.value for m in HS} == {"dummy", "none", "isaac", "hunav", "arena"}
