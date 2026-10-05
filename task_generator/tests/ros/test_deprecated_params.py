"""Deprecated task-generator parameters land on their replacements."""

from __future__ import annotations

import pytest


@pytest.fixture
def make_node():
    rclpy = pytest.importorskip("rclpy")
    from rclpy.context import Context
    from rclpy.node import Node
    from rclpy.parameter import Parameter

    context = Context()
    rclpy.init(context=context)
    nodes = []

    def _make(overrides: dict[str, object]):
        node = Node(
            "deprecated_params_probe",
            context=context,
            parameter_overrides=[Parameter(name, value=value) for name, value in overrides.items()],
            automatically_declare_parameters_from_overrides=True,
        )
        nodes.append(node)
        return node

    yield _make
    for node in nodes:
        node.destroy_node()
    rclpy.shutdown(context=context)


def test_every_deprecated_name_lands_on_its_replacement(make_node):
    from task_generator.constants.runtime import DEPRECATED_PARAMS, migrate_deprecated_params

    values = {old: float(i) for i, old in enumerate(DEPRECATED_PARAMS)}
    node = make_node(values)

    assert migrate_deprecated_params(node) == list(DEPRECATED_PARAMS)
    for old, new in DEPRECATED_PARAMS.items():
        assert node.get_parameter(new).value == values[old]


def test_explicit_replacement_beats_deprecated_name(make_node):
    from task_generator.constants.runtime import migrate_deprecated_params

    node = make_node({"timeout": 30.0, "task.episode.timeout": 90.0})

    assert migrate_deprecated_params(node) == ["timeout"]
    assert node.get_parameter("task.episode.timeout").value == 90.0


def test_no_deprecated_names_means_no_changes(make_node):
    from task_generator.constants.runtime import migrate_deprecated_params

    node = make_node({"task.episode.goto_pose.tolerance.radius": 0.5})

    assert migrate_deprecated_params(node) == []
    assert not node.has_parameter("task.episode.timeout")
