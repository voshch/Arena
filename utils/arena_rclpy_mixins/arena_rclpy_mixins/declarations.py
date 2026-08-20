"""Typed schema declaration helpers.

Wrap node.rosparam.declare_forward() with the Arena descriptor mini-DSL
(label/catalog/enum/range tokens in additional_constraints), so schema
authors don't hand-build ParameterDescriptor each time.

The mini-DSL is parsed by the rviz panel ([rebuildParamTree]); keep this
module and the panel parser in sync if tokens change.
"""

import typing
from collections.abc import Sequence

from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor
from rclpy import Parameter

if typing.TYPE_CHECKING:
    from .ROSParamServer import ROSParamServer


def _constraints(*tokens: str) -> str:
    return ";".join(t for t in tokens if t)


def _label(label: str) -> str:
    return f"label:{label}" if label else ""


def declare_int_pair(
    node: "ROSParamServer",
    name: str,
    default: Sequence[int],
    *,
    label: str = "",
    description: str = "",
) -> None:
    node.rosparam.declare_forward(
        name,
        list(default),
        descriptor=ParameterDescriptor(
            type=Parameter.Type.INTEGER_ARRAY.value,
            additional_constraints=_constraints(_label(label), "range:int_pair"),
            description=description,
        ),
    )


def declare_float_pair(
    node: "ROSParamServer",
    name: str,
    default: Sequence[float],
    *,
    label: str = "",
    description: str = "",
) -> None:
    node.rosparam.declare_forward(
        name,
        list(default),
        descriptor=ParameterDescriptor(
            type=Parameter.Type.DOUBLE_ARRAY.value,
            additional_constraints=_constraints(_label(label), "range:float_pair"),
            description=description,
        ),
    )


def declare_catalog(
    node: "ROSParamServer",
    name: str,
    default: str,
    *,
    catalog: str,
    label: str = "",
    description: str = "",
) -> None:
    node.rosparam.declare_forward(
        name,
        default,
        descriptor=ParameterDescriptor(
            type=Parameter.Type.STRING.value,
            additional_constraints=_constraints(_label(label), f"catalog:{catalog}"),
            description=description,
        ),
    )


def declare_catalog_array(
    node: "ROSParamServer",
    name: str,
    default: Sequence[str],
    *,
    catalog: str,
    label: str = "",
    description: str = "",
) -> None:
    node.rosparam.declare_forward(
        name,
        list(default),
        descriptor=ParameterDescriptor(
            type=Parameter.Type.STRING_ARRAY.value,
            additional_constraints=_constraints(_label(label), f"catalog:{catalog}"),
            description=description,
        ),
    )


def declare_enum(
    node: "ROSParamServer",
    name: str,
    default: str,
    *,
    choices: Sequence[str],
    label: str = "",
    description: str = "",
) -> None:
    node.rosparam.declare_forward(
        name,
        default,
        descriptor=ParameterDescriptor(
            type=Parameter.Type.STRING.value,
            additional_constraints=_constraints(_label(label), "enum:" + ",".join(choices)),
            description=description,
        ),
    )


def declare_string(
    node: "ROSParamServer",
    name: str,
    default: str,
    *,
    label: str = "",
    description: str = "",
) -> None:
    node.rosparam.declare_forward(
        name,
        default,
        descriptor=ParameterDescriptor(
            type=Parameter.Type.STRING.value,
            additional_constraints=_label(label),
            description=description,
        ),
    )


def declare_sketch(
    node: "ROSParamServer",
    name: str,
    default: str,
    *,
    label: str = "",
    description: str = "",
) -> None:
    node.rosparam.declare_forward(
        name,
        default,
        descriptor=ParameterDescriptor(
            type=Parameter.Type.STRING.value,
            additional_constraints=_constraints(_label(label), "sketch"),
            description=description,
        ),
    )


def declare_text(
    node: "ROSParamServer",
    name: str,
    default: str,
    *,
    label: str = "",
    description: str = "",
) -> None:
    node.rosparam.declare_forward(
        name,
        default,
        descriptor=ParameterDescriptor(
            type=Parameter.Type.STRING.value,
            additional_constraints=_constraints(_label(label), "text"),
            description=description,
        ),
    )


def declare_int(
    node: "ROSParamServer",
    name: str,
    default: int,
    *,
    label: str = "",
    description: str = "",
    lo: int | None = None,
    hi: int | None = None,
    step: int = 1,
) -> None:
    desc = ParameterDescriptor(
        type=Parameter.Type.INTEGER.value,
        additional_constraints=_label(label),
        description=description,
    )
    if lo is not None and hi is not None:
        desc.integer_range = [IntegerRange(from_value=lo, to_value=hi, step=step)]
    node.rosparam.declare_forward(name, default, descriptor=desc)


def declare_double(
    node: "ROSParamServer",
    name: str,
    default: float,
    *,
    label: str = "",
    description: str = "",
    lo: float | None = None,
    hi: float | None = None,
    step: float = 0.0,
) -> None:
    desc = ParameterDescriptor(
        type=Parameter.Type.DOUBLE.value,
        additional_constraints=_label(label),
        description=description,
    )
    if lo is not None and hi is not None:
        desc.floating_point_range = [FloatingPointRange(from_value=lo, to_value=hi, step=step)]
    node.rosparam.declare_forward(name, default, descriptor=desc)


def declare_bool(
    node: "ROSParamServer",
    name: str,
    default: bool,
    *,
    label: str = "",
    description: str = "",
) -> None:
    node.rosparam.declare_forward(
        name,
        default,
        descriptor=ParameterDescriptor(
            type=Parameter.Type.BOOL.value,
            additional_constraints=_label(label),
            description=description,
        ),
    )
