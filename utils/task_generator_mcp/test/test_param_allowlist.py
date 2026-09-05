"""Test: SetParameters.Request is built correctly from the EPISODE_PARAMS allowlist."""
import pytest
import rclpy.parameter
from rcl_interfaces.msg import Parameter, ParameterValue
from rcl_interfaces.srv import SetParameters

from task_generator_mcp.params import EPISODE_PARAMS, STATIC_CONFIG_PARAMS
from task_generator_mcp.tools import _python_to_param_value


def _build_set_request(kwargs: dict) -> SetParameters.Request:
    """Mirror the logic in tools._dispatch for config_set_episode_params."""
    params: list[Parameter] = []
    for key in EPISODE_PARAMS:
        val = kwargs.get(key)
        if val is not None:
            params.append(_python_to_param_value(key, val))
    req = SetParameters.Request()
    req.parameters = params
    return req


def test_only_supplied_fields_are_included():
    req = _build_set_request({"timeout": 30.0, "episodes": 5})
    names = [p.name for p in req.parameters]
    assert "timeout" in names
    assert "episodes" in names
    assert "auto_reset" not in names
    assert "goal_tolerance_radius" not in names


def test_none_values_are_excluded():
    req = _build_set_request({"timeout": None, "auto_reset": True})
    names = [p.name for p in req.parameters]
    assert "timeout" not in names
    assert "auto_reset" in names


def test_float_param_type():
    req = _build_set_request({"timeout": 60.0})
    p = req.parameters[0]
    assert p.name == "timeout"
    assert p.value.type == rclpy.parameter.Parameter.Type.DOUBLE.value
    assert p.value.double_value == pytest.approx(60.0)


def test_int_param_type():
    req = _build_set_request({"episodes": 10})
    p = req.parameters[0]
    assert p.name == "episodes"
    assert p.value.type == rclpy.parameter.Parameter.Type.INTEGER.value
    assert p.value.integer_value == 10


def test_bool_param_type():
    req = _build_set_request({"auto_reset": False})
    p = req.parameters[0]
    assert p.name == "auto_reset"
    assert p.value.type == rclpy.parameter.Parameter.Type.BOOL.value
    assert p.value.bool_value is False


def test_allowlist_does_not_contain_train_mode():
    assert "train_mode" not in EPISODE_PARAMS


def test_static_config_params_coverage():
    required = {"sim", "human", "robot.mobile_adapter", "robot.arm_adapter"}
    assert required <= set(STATIC_CONFIG_PARAMS)
