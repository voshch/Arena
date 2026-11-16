"""Pydantic-to-ROS-parameter bridge for world generator configurations."""

from __future__ import annotations

import enum
import types
import typing

import pydantic
from arena_rclpy_mixins.params import Choices, Constraint, Pair, Param, Range, Text, declare

if typing.TYPE_CHECKING:
    import rclpy.node


def _describe(field_info: pydantic.fields.FieldInfo, field_name: str, namespace_prefix: str) -> str:
    if field_info.description:
        return field_info.description
    return f"{field_name.replace('_', ' ').capitalize()} of the {namespace_prefix.rpartition('.')[2]} world generator."


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


def declare_config_params(node: rclpy.node.Node, namespace_prefix: str, model_cls: type[pydantic.BaseModel]) -> None:
    """Declare one ROS param per scalar/pair field of model_cls under namespace_prefix.<field>."""
    for field_name, field_info in model_cls.model_fields.items():
        name = f"{namespace_prefix}.{field_name}"
        description = _describe(field_info, field_name, namespace_prefix)
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
                declare(node, Param[list[int]](name, [int(v) for v in default], description, constraint=Pair()))
            else:
                declare(node, Param[list[float]](name, [float(v) for v in default], description, constraint=Pair()))
            continue

        ge = _find_ge(metadata)
        le = _find_le(metadata)

        if annotation is bool:
            declare(node, Param[bool](name, bool(default), description))
        elif annotation is int:
            unset = _unset_int(field_info)
            hi = int(le) if le is not None else None
            if unset is None:
                lo = int(ge) if ge is not None else None
                declare(node, Param[int](name, default, description, constraint=Range(lo, hi) if lo is not None or hi is not None else None))
            else:
                declare(node, Param[int](name, unset if default is None else default, f"{description.rstrip('.')} ({unset} = unset).", constraint=Range(unset, hi)))
        elif annotation is float:
            lo = float(ge) if ge is not None else None
            hi = float(le) if le is not None else None
            declare(node, Param[float](name, float(default), description, constraint=Range(lo, hi) if lo is not None or hi is not None else None))
        elif annotation is str or (isinstance(annotation, type) and issubclass(annotation, str)):
            extra = field_info.json_schema_extra
            widget = extra.get("widget") if isinstance(extra, dict) else None
            default_str = str(default.value if isinstance(default, enum.Enum) else (default or ""))
            constraint: Constraint | None
            if widget == "sketch":
                constraint = Text(sketch=True)
            elif widget == "text":
                constraint = Text()
            elif isinstance(annotation, type) and issubclass(annotation, enum.Enum):
                constraint = Choices(*(str(member.value) for member in annotation))
            else:
                constraint = None
            declare(node, Param[str](name, default_str, description, constraint=constraint))
        elif origin in (list, tuple):
            continue
        else:
            raise ValueError(f"Unsupported annotation {annotation!r} for field '{field_name}'; only bool/int/float/str, 2-tuples thereof, and skipped list/tuple fields are supported")
