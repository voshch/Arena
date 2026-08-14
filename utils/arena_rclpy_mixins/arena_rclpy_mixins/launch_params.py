"""Two-way codec between ROS ParameterValue and `key:=value` launch strings.

Booleans are not representable in either direction: the launch side keeps them
as strings, which would declare the parameter with the wrong type.
"""

from __future__ import annotations

from collections.abc import Callable

import yaml
from rcl_interfaces.msg import ParameterType, ParameterValue

_ENCODERS: dict[int, Callable[[ParameterValue], str]] = {
    ParameterType.PARAMETER_STRING: lambda pv: pv.string_value,
    ParameterType.PARAMETER_INTEGER: lambda pv: str(pv.integer_value),
    ParameterType.PARAMETER_DOUBLE: lambda pv: repr(pv.double_value),
    ParameterType.PARAMETER_INTEGER_ARRAY: lambda pv: "[" + ", ".join(str(v) for v in pv.integer_array_value) + "]",
    ParameterType.PARAMETER_DOUBLE_ARRAY: lambda pv: "[" + ", ".join(repr(float(v)) for v in pv.double_array_value) + "]",
    ParameterType.PARAMETER_STRING_ARRAY: lambda pv: "[" + ", ".join(str(v) for v in pv.string_array_value) + "]",
}


def param_value_to_launch_str(value: ParameterValue) -> str | None:
    """Serialise a ParameterValue to a yaml-parseable launch value, None if not representable."""
    encode = _ENCODERS.get(value.type)
    return None if encode is None else encode(value)


def launch_str_to_value(raw: str) -> object:
    """Parse a launch value into list/dict/numeric, leaving anything else as the raw string."""
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    if isinstance(parsed, (list, dict, int, float)) and not isinstance(parsed, bool):
        return parsed
    return raw
