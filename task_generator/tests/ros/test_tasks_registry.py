from __future__ import annotations

import pytest


def test_registry_has_obstacles_entries():
    from task_generator.constants import Constants
    from task_generator.tasks.registry import OBSTACLES_MODES

    assert Constants.TaskMode.TM_Obstacles.PARAMETRIZED in OBSTACLES_MODES
    assert Constants.TaskMode.TM_Obstacles.RANDOM in OBSTACLES_MODES
    assert Constants.TaskMode.TM_Obstacles.SCENARIO in OBSTACLES_MODES
    assert Constants.TaskMode.TM_Obstacles.ENVIRONMENT in OBSTACLES_MODES
    # PROMPT is registered per-BaseHumanSimulator subclass via _register_task_modes,
    # not centrally; see test_humansim_register.py.


def test_registry_has_robots_entries():
    from task_generator.constants import Constants
    from task_generator.tasks.registry import ROBOTS_MODES

    assert Constants.TaskMode.TM_Robots.GUIDED in ROBOTS_MODES
    assert Constants.TaskMode.TM_Robots.EXPLORE in ROBOTS_MODES
    assert Constants.TaskMode.TM_Robots.RANDOM in ROBOTS_MODES
    assert Constants.TaskMode.TM_Robots.SCENARIO in ROBOTS_MODES
    assert Constants.TaskMode.TM_Robots.STATIONARY in ROBOTS_MODES
    assert Constants.TaskMode.TM_Robots.CHARACTERIZATION in ROBOTS_MODES


def test_registry_has_module_entries():
    from task_generator.constants import Constants
    from task_generator.tasks.registry import MODULE_MODES

    assert Constants.TaskMode.TM_Module.CLEAR_FORBIDDEN_ZONES in MODULE_MODES
    assert Constants.TaskMode.TM_Module.RVIZ_UI in MODULE_MODES
    assert Constants.TaskMode.TM_Module.SOUNDS in MODULE_MODES
    assert Constants.TaskMode.TM_Module.ZONE_EDIT in MODULE_MODES


@pytest.mark.parametrize(
    ("modes", "axis", "member"),
    [
        ("OBSTACLES_MODES", "TM_Obstacles", "RANDOM"),
        ("ROBOTS_MODES", "TM_Robots", "RANDOM"),
        ("MODULE_MODES", "TM_Module", "CLEAR_FORBIDDEN_ZONES"),
    ],
)
def test_loader_returns_a_class(modes: str, axis: str, member: str):
    import task_generator.tasks.registry as registry
    from task_generator.constants import Constants

    assert isinstance(getattr(registry, modes).get(getattr(getattr(Constants.TaskMode, axis), member)), type)


@pytest.mark.parametrize(
    ("modes", "axis", "member", "value"),
    [
        ("OBSTACLES_MODES", "TM_Obstacles", "RANDOM", "random"),
        ("ROBOTS_MODES", "TM_Robots", "SCENARIO", "scenario"),
    ],
)
def test_namespace_contains_the_mode_value(modes: str, axis: str, member: str, value: str):
    import task_generator.tasks.registry as registry
    from task_generator.constants import Constants

    assert value in str(getattr(registry, modes).meta(getattr(getattr(Constants.TaskMode, axis), member)).namespace)
