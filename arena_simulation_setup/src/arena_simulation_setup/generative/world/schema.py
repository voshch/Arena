"""Pydantic-to-ROS-parameter bridge for world generator configurations.

Dispatches each pydantic field to the shared `arena_rclpy_mixins.declarations`
helpers, so the descriptor mini-DSL (label/range tokens) has a single producer.
"""

from __future__ import annotations

import enum
import types
import typing

import pydantic
from arena_rclpy_mixins.declarations import (
    declare_bool,
    declare_double,
    declare_float_pair,
    declare_int,
    declare_int_pair,
    declare_sketch,
    declare_string,
    declare_text,
)

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer


def _title(field_name: str) -> str:
    return field_name.replace("_", " ").title()


def _find_ge(metadata: list) -> float | None:
    for m in metadata:
        try:
            v = m.ge
        except AttributeError:
            continue
        if v is not None:
            return v
    return None


def _find_le(metadata: list) -> float | None:
    for m in metadata:
        try:
            v = m.le
        except AttributeError:
            continue
        if v is not None:
            return v
    return None


def _unset_int(field_info: pydantic.fields.FieldInfo) -> int | None:
    """The int that stands in for None of an `int | None` field on the ROS side, one below its lower bound."""
    annotation = field_info.annotation
    if typing.get_origin(annotation) not in (typing.Union, types.UnionType) or set(typing.get_args(annotation)) != {int, type(None)}:
        return None
    ge = _find_ge(list(field_info.metadata))
    if ge is None:
        raise ValueError(f"Optional int field needs a lower bound to stand None in for, got {annotation!r}")
    return int(ge) - 1


def _is_optional_str(field_info: pydantic.fields.FieldInfo) -> bool:
    """True if field is typed `str | None` with default None."""
    annotation = field_info.annotation
    if typing.get_origin(annotation) not in (typing.Union, types.UnionType) or set(typing.get_args(annotation)) != {str, type(None)}:
        return False
    return field_info.default is None


def config_from_params(model_cls: type[pydantic.BaseModel], config: dict) -> dict:
    """Map the ROS stand-ins of `int | None` and `str | None` fields in config back to None."""
    unset = {name: _unset_int(info) for name, info in model_cls.model_fields.items()}
    optional_strs = {name for name, info in model_cls.model_fields.items() if _is_optional_str(info)}
    out: dict[str, object] = {}
    for leaf, value in config.items():
        if unset.get(leaf) is not None and value == unset[leaf]:
            out[leaf] = None
        elif leaf in optional_strs and value == "":
            out[leaf] = None
        else:
            out[leaf] = value
    return out


def declare_config_params(node: ROSParamServer, namespace_prefix: str, model_cls: type[pydantic.BaseModel]) -> None:
    """Declare one ROS param per scalar/pair field of model_cls under namespace_prefix.<field>."""
    for field_name, field_info in model_cls.model_fields.items():
        name = f"{namespace_prefix}.{field_name}"
        label = _title(field_name)
        annotation = field_info.annotation
        default = field_info.default
        metadata = list(field_info.metadata) if field_info.metadata else []

        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        if origin in (typing.Union, types.UnionType):
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                annotation = non_none[0]
                origin = typing.get_origin(annotation)
                args = typing.get_args(annotation)

        if origin in (tuple, list) and len(args) == 2 and all(a in (int, float) for a in args):
            if args[0] is int and args[1] is int:
                declare_int_pair(node, name, [int(v) for v in default], label=label)
            else:
                declare_float_pair(node, name, [float(v) for v in default], label=label)
            continue

        ge = _find_ge(metadata)
        le = _find_le(metadata)

        if annotation is bool:
            declare_bool(node, name, bool(default), label=label)
        elif annotation is int:
            unset = _unset_int(field_info)
            if unset is None:
                declare_int(node, name, default, label=label, lo=int(ge) if ge is not None else None, hi=int(le) if le is not None else None)
            else:
                declare_int(node, name, unset if default is None else default, label=label, description=f"{unset} = unset", lo=unset, hi=int(le) if le is not None else None)
        elif annotation is float:
            declare_double(node, name, float(default), label=label, lo=float(ge) if ge is not None else None, hi=float(le) if le is not None else None)
        elif annotation is str or (isinstance(annotation, type) and issubclass(annotation, str)):
            extra = field_info.json_schema_extra
            widget = extra.get("widget") if isinstance(extra, dict) else None
            default_str = str(default.value if isinstance(default, enum.Enum) else (default or ""))
            if widget == "sketch":
                declare_sketch(node, name, default_str, label=label)
            elif widget == "text":
                declare_text(node, name, default_str, label=label)
            else:
                declare_string(node, name, default_str, label=label)
        elif origin in (list, tuple):
            continue
        else:
            raise ValueError(f"Unsupported annotation {annotation!r} for field '{field_name}'; only bool/int/float/str, 2-tuples thereof, and skipped list/tuple fields are supported")
