from __future__ import annotations

import math


def _node(overrides):
    import rclpy

    from arena_rclpy_mixins.ROSParamServer import ROSParamServer

    return ROSParamServer(
        "param_overrides_probe",
        parameter_overrides=[rclpy.Parameter(name, value=value) for name, value in overrides.items()],
        automatically_declare_parameters_from_overrides=True,
    )


def test_integer_override_of_double_param_is_coerced():
    import rclpy

    from arena_rclpy_mixins.declarations import declare_double

    node = _node({"pos_x": 10})
    try:
        declare_double(node, "pos_x", math.nan)
        param = node.get_parameter("pos_x")
        assert param.type_ == rclpy.Parameter.Type.DOUBLE
        assert param.value == 10.0
    finally:
        node.destroy_node()


def test_double_override_keeps_its_value():
    from arena_rclpy_mixins.declarations import declare_double

    node = _node({"pos_y": 2.5})
    try:
        declare_double(node, "pos_y", math.nan)
        assert node.get_parameter("pos_y").value == 2.5
    finally:
        node.destroy_node()


def test_unset_double_takes_its_default():
    from arena_rclpy_mixins.declarations import declare_double

    node = _node({})
    try:
        declare_double(node, "pos_theta", 0.75)
        assert node.get_parameter("pos_theta").value == 0.75
    finally:
        node.destroy_node()
