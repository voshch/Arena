from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def _register() -> None:
    """Register the HumanSim-backed obstacle modes. The registry is process-global, so a
    sibling test may already have done it."""
    from task_generator.simulators.human.arena_humansim.arena_humansim import (
        ArenaHumanSimulator,
    )

    try:
        ArenaHumanSimulator._register_task_modes()
    except (AssertionError, KeyError, ValueError):
        pass


def test_edge_case_is_in_the_enum():
    from task_generator.constants import Constants

    assert Constants.TaskMode.TM_Obstacles("edge_case") is Constants.TaskMode.TM_Obstacles.EDGE_CASE


def test_edge_case_registers_with_the_humansim_backend():
    """Registered from the adapter rather than at import, like PROMPT: the mode
    materialises arena_humansim agent types, so it must not be offered under other
    human backends where it would silently do nothing."""
    pytest.importorskip("arena_humansim_msgs.msg")
    from task_generator.constants import Constants
    from task_generator.tasks.registry import OBSTACLES_MODES

    _register()
    assert Constants.TaskMode.TM_Obstacles.EDGE_CASE in OBSTACLES_MODES


def test_edge_case_loader_returns_the_class():
    pytest.importorskip("arena_humansim_msgs.msg")
    from task_generator.constants import Constants
    from task_generator.tasks.obstacles.edge_case.impl import TM_EdgeCase
    from task_generator.tasks.registry import OBSTACLES_MODES

    _register()
    assert OBSTACLES_MODES.get(Constants.TaskMode.TM_Obstacles.EDGE_CASE) is TM_EdgeCase


def test_edge_case_namespace():
    pytest.importorskip("arena_humansim_msgs.msg")
    from task_generator.constants import Constants
    from task_generator.tasks.registry import OBSTACLES_MODES

    _register()
    meta = OBSTACLES_MODES.meta(Constants.TaskMode.TM_Obstacles.EDGE_CASE)
    assert "edge_case" in str(meta.namespace)
    assert meta.schema is not None


def test_schema_declares_every_param():
    """walk_schemas runs these before any impl is imported, so the param surface must be
    complete without touching TM_EdgeCase."""
    from arena_rclpy_mixins.shared import Namespace
    from task_generator.tasks.obstacles.edge_case import declare_schema

    declared: list[str] = []

    class _FakeRosParam:
        def declare_forward(self, name, default, *args, **kwargs):
            declared.append(str(name))

    class _FakeNode:
        rosparam = _FakeRosParam()

    declare_schema(_FakeNode(), Namespace("task")("edge_case"))
    leaves = {name.rsplit(".", 1)[-1].rsplit("/", 1)[-1] for name in declared}
    assert {"read_scenario_block", "record_dir", "score", "score_rate_hz", "score_scope", "objects", "objects_rate_hz"} <= leaves
    # The knob surface is gone: a case is its scenario's `edge_case:` block, nothing else.
    assert not {"knob", "level", "mode", "inject", "place", "approach_angle", "select", "catalogue"} & leaves


def test_schema_and_impl_agree_on_every_default():
    """The param surface is declared twice - once in `declare_schema` (which `walk_schemas`
    runs before any impl is imported) and once when `TM_EdgeCase.__init__` binds its ROSParams.
    If the two disagree, which value you get depends on load order, and a live `ros2 param get`
    would show something no source file says.

    (A live run did once report `place: scenario` against a declared `intercept`. That turned
    out to be a stale ROS daemon serving a previous node's cache, not a real disagreement -
    but nothing was pinning the two together, so it was not possible to rule out from source.)
    """
    import inspect

    from arena_rclpy_mixins.shared import Namespace
    from task_generator.tasks.obstacles.edge_case import declare_schema
    from task_generator.tasks.obstacles.edge_case.impl import TM_EdgeCase

    schema: dict[str, object] = {}

    class _FakeRosParam:
        def declare_forward(self, name, default, *args, **kwargs):
            schema[str(name).rsplit(".", 1)[-1].rsplit("/", 1)[-1]] = default

    class _FakeNode:
        rosparam = _FakeRosParam()

    declare_schema(_FakeNode(), Namespace("task")("edge_case"))

    # `__init__` binds each param as ROSParam[...](namespace("<leaf>"), value=<default>).
    src = inspect.getsource(TM_EdgeCase.__init__)
    impl: dict[str, str] = {}
    for line in src.splitlines():
        if 'self.namespace("' not in line or "value=" not in line:
            continue
        leaf = line.split('self.namespace("', 1)[1].split('"', 1)[0]
        impl[leaf] = line.split("value=", 1)[1].rstrip(" ),")

    consts = {
        "_MODE_INJECT": "inject",
        "_PLACE_INTERCEPT": "intercept",
        "_SELECT_NEAREST": "nearest_approach",
        "DEFAULT_CATALOGUE": "level_d",
        "_SCOPE_AUTO": "auto",
        "DEFAULT_RATE_HZ": "10.0",
        "OBJECT_RATE_HZ": "5.0",
    }
    mismatches = []
    for leaf, declared_default in schema.items():
        if leaf not in impl:
            continue
        raw = impl[leaf]
        resolved = consts.get(raw, raw.strip('"'))
        if str(resolved) != str(declared_default):
            mismatches.append(f"{leaf}: schema={declared_default!r} impl={resolved!r}")
    assert not mismatches, "schema and impl defaults disagree: " + "; ".join(mismatches)


def test_base_is_the_scenario_mode():
    """The base population always comes from a scenario file: resolution must reach the
    registered class without a live node."""
    pytest.importorskip("arena_humansim_msgs.msg")
    from task_generator.constants import Constants
    from task_generator.tasks.obstacles.edge_case.impl import TM_EdgeCase
    from task_generator.tasks.registry import OBSTACLES_MODES

    _register()
    assert Constants.TaskMode.TM_Obstacles.SCENARIO in OBSTACLES_MODES
    tm = object.__new__(TM_EdgeCase)
    tm._base_cache = "cached"
    assert TM_EdgeCase._base(tm) == "cached"
