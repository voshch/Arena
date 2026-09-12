"""Verify each typed declare helper builds the expected ParameterDescriptor.

Uses a recording stub on `node.rosparam.declare_forward` instead of a live
node, schema authors care about what the descriptor looks like, not about
the rclpy plumbing (covered separately by declare_safe tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest


@pytest.fixture(autouse=True)
def _ros_gate() -> None:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


@dataclass
class _Recorder:
    calls: list[tuple[str, object, object]] = field(default_factory=list)

    def declare_forward(self, name: str, value: object, *, descriptor: object) -> None:
        self.calls.append((name, value, descriptor))


@dataclass
class _StubNode:
    rosparam: _Recorder = field(default_factory=_Recorder)


def _last(node: _StubNode) -> tuple[str, object, object]:
    return node.rosparam.calls[-1]


def test_declare_int_pair() -> None:
    import rclpy
    from arena_rclpy_mixins.declarations import declare_int_pair

    node = _StubNode()
    declare_int_pair(node, "task.random.static.n", [5, 15], label="Static count", description="counts")
    name, value, desc = _last(node)
    assert name == "task.random.static.n"
    assert value == [5, 15]
    assert desc.type == rclpy.Parameter.Type.INTEGER_ARRAY.value
    assert desc.additional_constraints == "label:Static count;range:int_pair"
    assert desc.description == "counts"


def test_declare_float_pair() -> None:
    import rclpy
    from arena_rclpy_mixins.declarations import declare_float_pair

    node = _StubNode()
    declare_float_pair(node, "task.x.range", [0.0, 1.0])
    _, _, desc = _last(node)
    assert desc.type == rclpy.Parameter.Type.DOUBLE_ARRAY.value
    assert desc.additional_constraints == "range:float_pair"


def test_declare_catalog() -> None:
    import rclpy
    from arena_rclpy_mixins.declarations import declare_catalog

    node = _StubNode()
    declare_catalog(node, "task.scenario.file", "default", catalog="scenarios", label="Scenario file")
    _, value, desc = _last(node)
    assert value == "default"
    assert desc.type == rclpy.Parameter.Type.STRING.value
    assert desc.additional_constraints == "label:Scenario file;catalog:scenarios"


def test_declare_catalog_array_empty_default() -> None:
    import rclpy
    from arena_rclpy_mixins.declarations import declare_catalog_array

    node = _StubNode()
    declare_catalog_array(node, "task.random.static.models", [], catalog="objects")
    _, value, desc = _last(node)
    assert value == []
    assert desc.type == rclpy.Parameter.Type.STRING_ARRAY.value
    assert desc.additional_constraints == "catalog:objects"


def test_declare_enum() -> None:
    import rclpy
    from arena_rclpy_mixins.declarations import declare_enum

    node = _StubNode()
    declare_enum(node, "task.prompt.generation_mode", "arena", choices=["arena", "behavior_tree"], label="Generation mode")
    _, value, desc = _last(node)
    assert value == "arena"
    assert desc.type == rclpy.Parameter.Type.STRING.value
    assert desc.additional_constraints == "label:Generation mode;enum:arena,behavior_tree"


def test_declare_int_with_range() -> None:
    import rclpy
    from arena_rclpy_mixins.declarations import declare_int

    node = _StubNode()
    declare_int(node, "task.x.count", 5, lo=0, hi=10, step=2, label="Count")
    _, value, desc = _last(node)
    assert value == 5
    assert desc.type == rclpy.Parameter.Type.INTEGER.value
    assert len(desc.integer_range) == 1
    assert desc.integer_range[0].from_value == 0
    assert desc.integer_range[0].to_value == 10
    assert desc.integer_range[0].step == 2


def test_declare_int_without_range() -> None:
    from arena_rclpy_mixins.declarations import declare_int

    node = _StubNode()
    declare_int(node, "task.x.count", 5)
    _, _, desc = _last(node)
    assert desc.integer_range == []


def test_declare_double_with_range() -> None:
    import rclpy
    from arena_rclpy_mixins.declarations import declare_double

    node = _StubNode()
    declare_double(node, "task.x.gain", 0.5, lo=0.0, hi=1.0, step=0.1)
    _, _, desc = _last(node)
    assert desc.type == rclpy.Parameter.Type.DOUBLE.value
    assert desc.floating_point_range[0].from_value == 0.0
    assert desc.floating_point_range[0].to_value == 1.0
    assert desc.floating_point_range[0].step == 0.1


def test_declare_string() -> None:
    import rclpy
    from arena_rclpy_mixins.declarations import declare_string

    node = _StubNode()
    declare_string(node, "task.prompt.user_prompt", "default", label="Prompt")
    _, _, desc = _last(node)
    assert desc.type == rclpy.Parameter.Type.STRING.value
    assert desc.additional_constraints == "label:Prompt"


def test_declare_bool() -> None:
    import rclpy
    from arena_rclpy_mixins.declarations import declare_bool

    node = _StubNode()
    declare_bool(node, "task.x.enabled", True)
    _, value, desc = _last(node)
    assert value is True
    assert desc.type == rclpy.Parameter.Type.BOOL.value


def test_label_omitted_when_empty() -> None:
    from arena_rclpy_mixins.declarations import declare_int_pair

    node = _StubNode()
    declare_int_pair(node, "task.x.n", [0, 1])
    _, _, desc = _last(node)
    assert desc.additional_constraints == "range:int_pair"
