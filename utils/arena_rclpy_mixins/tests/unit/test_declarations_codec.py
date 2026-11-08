from __future__ import annotations

import pytest
from arena_rclpy_mixins import declarations
from arena_rclpy_mixins.declarations import parse_constraints
from arena_rclpy_mixins.launch_params import launch_str_to_value, value_to_launch_str
from rcl_interfaces.msg import ParameterDescriptor


class _Recorder:
    def __init__(self) -> None:
        self.descriptors: list[ParameterDescriptor] = []
        self.rosparam = self

    def declare_forward(self, name: str, default: object, descriptor: ParameterDescriptor) -> None:
        self.descriptors.append(descriptor)


def _declared(helper: str, **kwargs: object) -> ParameterDescriptor:
    node = _Recorder()
    getattr(declarations, helper)(node, "p", kwargs.pop("default"), label="Pick one", **kwargs)
    return node.descriptors[0]


@pytest.mark.parametrize(
    ("helper", "kwargs", "expected"),
    [
        ("declare_int_pair", {"default": [1, 2]}, {"pair": "int_pair"}),
        ("declare_float_pair", {"default": [1.0, 2.0]}, {"pair": "float_pair"}),
        ("declare_catalog", {"default": "a", "catalog": "scenarios"}, {"catalog": "scenarios"}),
        ("declare_catalog_array", {"default": ["a"], "catalog": "worlds"}, {"catalog": "worlds"}),
        ("declare_double_array", {"default": [1.0]}, {}),
        ("declare_enum", {"default": "a", "choices": ["a", "b", "c"]}, {"choices": ["a", "b", "c"]}),
        ("declare_string", {"default": "a"}, {}),
        ("declare_sketch", {"default": "a"}, {"widget": "sketch"}),
        ("declare_text", {"default": "a"}, {"widget": "text"}),
        ("declare_int", {"default": 1}, {}),
        ("declare_double", {"default": 1.0}, {}),
        ("declare_bool", {"default": True}, {}),
    ],
)
def test_parse_constraints_reads_back_what_every_declare_helper_writes(helper: str, kwargs: dict[str, object], expected: dict[str, object]) -> None:
    descriptor = _declared(helper, **kwargs)

    assert parse_constraints(descriptor.additional_constraints) == {"label": "Pick one", **expected}


def test_parse_constraints_leaves_out_tokens_that_are_absent() -> None:
    assert parse_constraints("") == {}
    assert parse_constraints("catalog:robots") == {"catalog": "robots"}
    assert parse_constraints("label:Time: a") == {"label": "Time: a"}


@pytest.mark.parametrize("value", [3, -2, 0.5, 1e-07, "default", "a b", [1, 2], [0.5, 1.5, 1e-07], ["x", "y z"], []])
def test_value_to_launch_str_round_trips_through_launch_str_to_value(value: object) -> None:
    assert launch_str_to_value(value_to_launch_str(value)) == value


def test_value_to_launch_str_writes_booleans_as_launch_booleans() -> None:
    assert value_to_launch_str(True) == "true"
    assert value_to_launch_str(False) == "false"
    assert value_to_launch_str([True, False]) == "[true, false]"
