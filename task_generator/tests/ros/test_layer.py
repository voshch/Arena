"""The task layer as arena_runtime finds and uses it."""

from __future__ import annotations

import importlib.metadata
import uuid

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_people_msgs.msg")

from arena_runtime.env.task_layer import TASK_LAYER_GROUP

from task_generator.modes import Modes
from task_generator.tasks import layer


def test_the_package_registers_this_module_as_the_task_layer() -> None:
    (entry,) = [entry for entry in importlib.metadata.entry_points(group=TASK_LAYER_GROUP) if entry.name == "task_generator"]
    assert entry.load() is layer


def _node(**declared: str) -> object:
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer

    node = ROSParamServer(f"layer_{uuid.uuid4().hex[:8]}", start_parameter_services=False)
    for name, value in declared.items():
        node.declare_parameter(name.replace("__", "."), value)
    return node


def test_task_params_parse_the_selected_modes(rclpy_context: object) -> None:
    node = _node()
    try:
        params = layer.declare_task_params(node)
        assert params.TM_ROBOTS.value is Modes.TM_Robots.default() is Modes.TM_Robots.EXPLORE
        assert params.TM_OBSTACLES.value is Modes.TM_Obstacles.default()
        assert params.TM_MODULES.value == Modes.TM_Module.default()
        assert params.TM_CONFIG.value == ""
        assert params.resolve({}) == {"robots": ["explore"], "obstacles": ["random"], "modules": ["clear_forbidden_zones", "rviz_ui"]}
    finally:
        node.destroy_node()


ROBOT_NAMES = ", ".join(mode.value for mode in Modes.TM_Robots)
OBSTACLE_NAMES = ", ".join(mode.value for mode in Modes.TM_Obstacles)
MODULE_NAMES = ", ".join(layer.mode_namespaces()["modules"])


def test_validate_modes_returns_canonical_names_for_the_listed_axes_only() -> None:
    assert layer.validate_modes({}) == {}
    assert layer.validate_modes({"robots": ["Random"]}) == {"robots": ["random"]}
    assert layer.validate_modes({"obstacles": ["SCENARIO"], "modules": ["Zone_Edit", "sounds", "ZONE_EDIT"]}) == {"obstacles": ["scenario"], "modules": ["zone_edit", "sounds"]}
    assert layer.validate_modes({"modules": []}) == {"modules": []}


def test_validate_modes_rejects_an_unknown_axis() -> None:
    with pytest.raises(ValueError) as error:
        layer.validate_modes({"planner": ["dwb"]})
    assert str(error.value) == "unknown task axis 'planner', allowed: robots, obstacles, modules"


@pytest.mark.parametrize(
    ("axis", "allowed"),
    [("robots", ROBOT_NAMES), ("obstacles", OBSTACLE_NAMES), ("modules", MODULE_NAMES)],
)
def test_validate_modes_rejects_an_unknown_mode_name(axis: str, allowed: str) -> None:
    with pytest.raises(ValueError) as error:
        layer.validate_modes({axis: ["nope"]})
    assert str(error.value) == f"unknown {axis} mode 'nope', allowed: {allowed}"


@pytest.mark.parametrize("name", ["staged", "dynamic_map"])
def test_validate_modes_rejects_a_mode_with_no_registered_module(name: str) -> None:
    with pytest.raises(ValueError) as error:
        layer.validate_modes({"modules": [name]})
    assert str(error.value) == f"unknown modules mode {name!r}, allowed: {MODULE_NAMES}"


@pytest.mark.parametrize(("axis", "allowed"), [("robots", ROBOT_NAMES), ("obstacles", OBSTACLE_NAMES)])
@pytest.mark.parametrize("names", [[], ["random", "scenario"]])
def test_validate_modes_requires_exactly_one_name_on_a_single_mode_axis(axis: str, allowed: str, names: list[str]) -> None:
    with pytest.raises(ValueError) as error:
        layer.validate_modes({axis: names})
    assert str(error.value) == f"task axis {axis!r} takes exactly one mode, got {names!r}, allowed: {allowed}"


def test_a_bad_launch_value_fails_with_the_validators_message(rclpy_context: object) -> None:
    from rclpy.parameter import Parameter

    from arena_rclpy_mixins.ROSParamServer import ROSParamServer

    cases = [
        ("task.robots", "nope", f"unknown robots mode 'nope', allowed: {ROBOT_NAMES}"),
        ("task.robots", "", f"task axis 'robots' takes exactly one mode, got [], allowed: {ROBOT_NAMES}"),
        ("task.obstacles", "random,scenario", f"task axis 'obstacles' takes exactly one mode, got ['random', 'scenario'], allowed: {OBSTACLE_NAMES}"),
        ("task.modules", "sounds,nope", f"unknown modules mode 'nope', allowed: {MODULE_NAMES}"),
    ]
    for name, value, message in cases:
        node = ROSParamServer(f"layer_{uuid.uuid4().hex[:8]}", start_parameter_services=False, parameter_overrides=[Parameter(name, value=value)])
        try:
            with pytest.raises(ValueError) as error:
                layer.declare_task_params(node)
            assert str(error.value) == message
        finally:
            node.destroy_node()


def test_the_launch_params_accept_any_case_and_an_empty_module_list(rclpy_context: object) -> None:
    from rclpy.parameter import Parameter

    from arena_rclpy_mixins.ROSParamServer import ROSParamServer

    overrides = [Parameter("task.robots", value="Random"), Parameter("task.obstacles", value="SCENARIO"), Parameter("task.modules", value="")]
    node = ROSParamServer(f"layer_{uuid.uuid4().hex[:8]}", start_parameter_services=False, parameter_overrides=overrides)
    try:
        params = layer.declare_task_params(node)
        assert params.TM_ROBOTS.value is Modes.TM_Robots.RANDOM
        assert params.TM_OBSTACLES.value is Modes.TM_Obstacles.SCENARIO
        assert params.TM_MODULES.value == set()
    finally:
        node.destroy_node()


def test_every_registered_mode_has_a_namespace_and_validates() -> None:
    from arena_runtime.env.task_layer import registered_modes

    registered = registered_modes(layer)
    assert set(registered) == {"robots", "obstacles", "modules"}
    for axis, names in registered.items():
        assert names
        assert set(layer.mode_namespaces()[axis]) == set(names)
        for name in names:
            selected = names if axis == "modules" else [name]
            assert layer.validate_modes({axis: selected}) == {axis: selected}


def test_the_robots_and_obstacles_scenario_modes_share_a_namespace() -> None:
    namespaces = layer.mode_namespaces()
    assert namespaces["robots"]["scenario"] == namespaces["obstacles"]["scenario"] == "task.scenario"


def test_the_idle_sink_is_a_fleet_only_robot_mode() -> None:
    assert "null" in layer.fleet_only_robots()


def test_the_modules_selection_adds_the_always_on_ones_and_leaves_the_param_alone(rclpy_context: object) -> None:
    from rclpy.parameter import Parameter

    from arena_rclpy_mixins.ROSParamServer import ROSParamServer

    node = ROSParamServer(f"layer_{uuid.uuid4().hex[:8]}", start_parameter_services=False, parameter_overrides=[Parameter("task.modules", value="zone_edit")])
    try:
        params = layer.declare_task_params(node)
        assert params.TM_MODULES.value == {Modes.TM_Module.ZONE_EDIT}
        assert params.resolve({})["modules"] == ["clear_forbidden_zones", "rviz_ui", "zone_edit"]
    finally:
        node.destroy_node()


@pytest.mark.parametrize(
    ("acoustics", "static_sounds", "expected"),
    [("none", "[]", False), ("none", "", False), ("arena", "[]", True), ("none", "[{name: radio}]", True), ("arena", "[{name: radio}]", True)],
)
def test_the_sounds_module_is_selected_with_acoustics_or_static_sounds(rclpy_context: object, acoustics: str, static_sounds: str, expected: bool) -> None:
    node = _node(acoustics=acoustics, auditory__static_sounds=static_sounds)
    try:
        params = layer.declare_task_params(node)
        assert ("sounds" in params.resolve({})["modules"]) is expected
        assert params.TM_MODULES.value == set()
    finally:
        node.destroy_node()


def test_the_sounds_module_is_off_when_the_env_declares_neither_param(rclpy_context: object) -> None:
    node = _node()
    try:
        assert "sounds" not in layer.declare_task_params(node).resolve({})["modules"]
    finally:
        node.destroy_node()


def test_select_writes_only_the_listed_axes_and_only_when_they_differ(rclpy_context: object) -> None:
    from rcl_interfaces.msg import SetParametersResult

    node = _node()
    writes: list[str] = []
    try:
        params = layer.declare_task_params(node)
        node.add_on_set_parameters_callback(lambda changed: writes.extend(parameter.name for parameter in changed) or SetParametersResult(successful=True))
        params.select({"robots": ["explore"]})
        params.select({"robots": ["random"]})
        assert writes == ["task.robots"]
        assert params.resolve({})["robots"] == ["random"]
        assert params.resolve({})["obstacles"] == ["random"]
        params.select({"obstacles": ["scenario"], "modules": ["zone_edit"]})
        assert params.resolve({}) == {"robots": ["random"], "obstacles": ["scenario"], "modules": ["clear_forbidden_zones", "rviz_ui", "zone_edit"]}
        params.select({"modules": []})
        assert params.resolve({})["modules"] == ["clear_forbidden_zones", "rviz_ui"]
        params.select(params.resolve({}))
        assert writes == ["task.robots", "task.obstacles", "task.modules", "task.modules"]
    finally:
        node.destroy_node()


def test_resolve_takes_the_given_name_and_keeps_the_live_one_for_a_missing_robots_or_obstacles_axis(rclpy_context: object) -> None:
    node = _node()
    try:
        params = layer.declare_task_params(node)
        assert params.resolve({}) == {"robots": ["explore"], "obstacles": ["random"], "modules": ["clear_forbidden_zones", "rviz_ui"]}
        assert params.resolve({"robots": ["random"]})["robots"] == ["random"]
        assert params.resolve({"robots": ["random"]})["obstacles"] == ["random"]
        assert params.resolve({"obstacles": ["scenario"]})["robots"] == ["explore"]
    finally:
        node.destroy_node()


def test_resolve_adds_the_always_on_modules_to_the_given_ones_and_keeps_the_live_set_when_missing(rclpy_context: object) -> None:
    from rclpy.parameter import Parameter

    from arena_rclpy_mixins.ROSParamServer import ROSParamServer

    node = ROSParamServer(f"layer_{uuid.uuid4().hex[:8]}", start_parameter_services=False, parameter_overrides=[Parameter("task.modules", value="zone_edit")])
    try:
        params = layer.declare_task_params(node)
        assert params.resolve({})["modules"] == ["clear_forbidden_zones", "rviz_ui", "zone_edit"]
        assert params.resolve({"modules": []})["modules"] == ["clear_forbidden_zones", "rviz_ui"]
        assert params.resolve({"modules": ["sounds"]})["modules"] == ["clear_forbidden_zones", "rviz_ui", "sounds"]
    finally:
        node.destroy_node()


def test_resolve_adds_sounds_to_the_given_modules_when_the_env_has_acoustics(rclpy_context: object) -> None:
    node = _node(acoustics="arena", auditory__static_sounds="[]")
    try:
        params = layer.declare_task_params(node)
        assert params.resolve({"modules": []})["modules"] == ["clear_forbidden_zones", "rviz_ui", "sounds"]
    finally:
        node.destroy_node()


def test_select_with_the_live_selection_writes_nothing(rclpy_context: object) -> None:
    from rcl_interfaces.msg import SetParametersResult

    node = _node()
    writes: list[str] = []
    try:
        params = layer.declare_task_params(node)
        node.add_on_set_parameters_callback(lambda changed: writes.extend(parameter.name for parameter in changed) or SetParametersResult(successful=True))
        params.select({"obstacles": ["random"], "modules": ["rviz_ui"]})
        params.select(params.resolve({}))
        assert writes == []
        params.select({"modules": []})
        assert writes == []
    finally:
        node.destroy_node()


def test_the_default_layer_stays_selected_with_a_second_runtime_registered(tmp_path, monkeypatch) -> None:
    import sys

    from arena_runtime.env.task_layer import TASK_LAYER_ENV, task_layer
    from arena_runtime.testing.task_layer import register_task_layer

    from task_generator.tasks import layer

    register_task_layer(tmp_path, "second_runtime", "MARK = 2\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv(TASK_LAYER_ENV, raising=False)
    assert task_layer() is layer
    assert task_layer("second_runtime").MARK == 2
    sys.modules.pop("second_runtime", None)


def test_the_default_layer_keeps_the_queued_episode() -> None:
    import asyncio

    from arena_runtime.env.task_layer import EpisodeRecord, TaskModeOverrides

    queued = TaskModeOverrides(modes={"robots": ["random"]}, world="map_empty", seed=4)
    assert asyncio.run(layer.next_episode(previous=EpisodeRecord(), queued=queued)) is queued
