from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


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
    assert Constants.TaskMode.TM_Module.STAGED in MODULE_MODES


def test_register_obstacles_duplicate_raises():
    from task_generator.constants import Constants
    from task_generator.tasks.registry import _REGISTRY_NAMESPACE, OBSTACLES_MODES
    with pytest.raises((AssertionError, KeyError, ValueError)):
        @OBSTACLES_MODES.register(Constants.TaskMode.TM_Obstacles.RANDOM, namespace=_REGISTRY_NAMESPACE("dup"))
        def _loader():
            pass


def test_register_robots_duplicate_raises():
    from task_generator.constants import Constants
    from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES
    with pytest.raises((AssertionError, KeyError, ValueError)):
        @ROBOTS_MODES.register(Constants.TaskMode.TM_Robots.GUIDED, namespace=_REGISTRY_NAMESPACE("dup"))
        def _loader():
            pass


def test_register_module_duplicate_raises():
    from task_generator.constants import Constants
    from task_generator.tasks.registry import _REGISTRY_NAMESPACE, MODULE_MODES
    with pytest.raises((AssertionError, KeyError, ValueError)):
        @MODULE_MODES.register(Constants.TaskMode.TM_Module.STAGED, namespace=_REGISTRY_NAMESPACE("dup"))
        def _loader():
            pass


def test_obstacles_loader_returns_class():
    from task_generator.constants import Constants
    from task_generator.tasks.registry import OBSTACLES_MODES
    cls = OBSTACLES_MODES.get(Constants.TaskMode.TM_Obstacles.RANDOM)
    assert isinstance(cls, type)


def test_robots_loader_returns_class():
    from task_generator.constants import Constants
    from task_generator.tasks.registry import ROBOTS_MODES
    cls = ROBOTS_MODES.get(Constants.TaskMode.TM_Robots.RANDOM)
    assert isinstance(cls, type)


def test_module_loader_returns_class():
    from task_generator.constants import Constants
    from task_generator.tasks.registry import MODULE_MODES
    cls = MODULE_MODES.get(Constants.TaskMode.TM_Module.CLEAR_FORBIDDEN_ZONES)
    assert isinstance(cls, type)


def test_obstacles_namespace_contains_value():
    from task_generator.constants import Constants
    from task_generator.tasks.registry import OBSTACLES_MODES
    meta = OBSTACLES_MODES.meta(Constants.TaskMode.TM_Obstacles.RANDOM)
    assert "random" in str(meta.namespace)


def test_robots_namespace_contains_value():
    from task_generator.constants import Constants
    from task_generator.tasks.registry import ROBOTS_MODES
    meta = ROBOTS_MODES.meta(Constants.TaskMode.TM_Robots.SCENARIO)
    assert "scenario" in str(meta.namespace)


def test_meta_accessible_without_invoking_loader():
    """Regression guard: .meta() reads registry-side, never invokes the loader.

    Canary: STAGED's impl.py imports `map_generator.constants` (ROS1-era, not on the
    runtime path). If .meta() invokes the loader, this raises ModuleNotFoundError.
    """
    from task_generator.constants import Constants
    from task_generator.tasks.registry import MODULE_MODES
    meta = MODULE_MODES.meta(Constants.TaskMode.TM_Module.STAGED)
    assert meta is not None
    assert meta.schema is None
