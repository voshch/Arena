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


def test_task_params_parse_the_selected_modes(rclpy_context: object) -> None:
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer

    node = ROSParamServer(f"layer_{uuid.uuid4().hex[:8]}", start_parameter_services=False)
    try:
        params = layer.declare_task_params(node)
        assert params.TM_ROBOTS.value is Modes.TM_Robots.default()
        assert params.TM_OBSTACLES.value is Modes.TM_Obstacles.default()
        assert params.TM_MODULES.value == Modes.TM_Module.default()
        assert params.TM_CONFIG.value == ""
        assert (params.robots, params.obstacles, params.modules) == (Modes.TM_Robots.default().value, Modes.TM_Obstacles.default().value, [])
    finally:
        node.destroy_node()


def test_modules_are_a_comma_separated_list_of_known_names() -> None:
    assert layer._modules("sounds,,zone_edit") == {Modes.TM_Module.SOUNDS, Modes.TM_Module.ZONE_EDIT}
    with pytest.raises(ValueError, match="__bad__"):
        layer._modules("__bad__")


def test_every_registered_mode_is_in_the_vocabulary_and_has_a_namespace() -> None:
    names, vocabulary, namespaces = layer.mode_names(), layer.mode_vocabulary(), layer.mode_namespaces()
    for axis in ("robots", "obstacles", "modules"):
        assert set(getattr(names, axis)) <= set(getattr(vocabulary, axis))
        assert set(namespaces[axis]) == set(getattr(names, axis))


def test_the_idle_sink_is_a_fleet_only_robot_mode() -> None:
    assert "null" in layer.fleet_only_robots()
