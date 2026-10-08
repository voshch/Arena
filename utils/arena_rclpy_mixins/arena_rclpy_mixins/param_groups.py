"""Typed parameter groups: declarations with defaults, coercers and descriptions, declared per group on a ROSParamServer node."""

from __future__ import annotations

import math
import typing
from collections.abc import Callable, Iterable, Mapping

import attrs
import yaml

if typing.TYPE_CHECKING:
    from launch.actions import DeclareLaunchArgument

    from arena_rclpy_mixins.ROSParamServer import ROSParamServer, ROSParamT


@attrs.frozen(init=False)
class PerRole[T]:
    """Default that differs between roles, by role name."""

    by_role: tuple[tuple[str, T], ...]

    def __init__(self, **by_role: T) -> None:
        if not by_role:
            raise ValueError("PerRole needs a default for at least one role")
        self.__attrs_init__(by_role=tuple(by_role.items()))

    def pick(self, role: str) -> T:
        """Raises KeyError for a role without a default."""
        for name, value in self.by_role:
            if name == role:
                return value
        raise KeyError(f"no default for role {role!r}, expected one of {[name for name, _ in self.by_role]}")

    @property
    def first(self) -> T:
        return self.by_role[0][1]


def finite(value: object) -> float:
    result = float(typing.cast(float, value))
    if not math.isfinite(result):
        raise ValueError(f"{value!r} is not finite")
    return result


def within(lo: float, hi: float, *, lo_open: bool = False) -> Callable[[object], float]:
    def parse(value: object) -> float:
        result = finite(value)
        if result < lo or result > hi or (lo_open and result == lo):
            raise ValueError(f"{result} is outside {'(' if lo_open else '['}{lo}, {hi}]")
        return result

    return parse


def positive(value: object) -> float:
    result = finite(value)
    if result <= 0.0:
        raise ValueError(f"{result} must be positive")
    return result


def non_negative(value: object) -> float:
    result = finite(value)
    if result < 0.0:
        raise ValueError(f"{result} must not be negative")
    return result


def count(value: object) -> int:
    result = int(typing.cast(int, value))
    if result < 0:
        raise ValueError(f"{result} must not be negative")
    return result


def positive_count(value: object) -> int:
    result = count(value)
    if result == 0:
        raise ValueError(f"{result} must be positive")
    return result


def floats(value: object) -> tuple[float, ...]:
    return tuple(finite(v) for v in typing.cast(Iterable[float], value))


def names(value: object) -> tuple[str, ...]:
    return tuple(str(v).strip().lower() for v in typing.cast(Iterable[str], value) if str(v).strip())


def _coerce_value(default: object, raw: str) -> object:
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    loaded = yaml.safe_load(raw)
    items = loaded if isinstance(loaded, list) else [loaded]
    element = type(default[0]) if default else str
    return [element(item) for item in items]


class Param[T]:
    """Parameter declaration. On a group class it is the declaration, on a group instance the live ROSParam."""

    def __init__(self, name: str, default: T | PerRole[T], *, parse: Callable[[object], T] | None = None, description: str = "") -> None:
        self.name = name
        self.default = default
        self.parse = parse
        self.description = description
        self.attr = ""

    def __set_name__(self, owner: type, attr: str) -> None:
        self.attr = attr

    @property
    def field(self) -> str:
        """Config field name, the lowercase declaration name."""
        return self.attr.lower()

    def coerce(self, raw: str) -> object:
        """Launch string as this parameter's type, list elements as the default's element type."""
        default = self.default.first if isinstance(self.default, PerRole) else self.default
        match default:
            case bool():
                value = yaml.safe_load(raw)
                if not isinstance(value, bool):
                    raise ValueError(f"{self.name} expects true or false, got {raw!r}")
                return value
            case int() | float() | list() | tuple():
                try:
                    value = _coerce_value(default, raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{self.name} expects {type(default).__name__}, got {raw!r}") from exc
                if value == []:
                    raise ValueError(f"{self.name} cannot be an empty list, a ROS parameter carries no empty array")
                return value
            case _:
                return raw

    @typing.overload
    def __get__(self, instance: None, owner: type) -> Param[T]: ...

    @typing.overload
    def __get__(self, instance: ParamGroup, owner: type) -> ROSParamT[T]: ...

    def __get__(self, instance: ParamGroup | None, owner: type) -> Param[T] | ROSParamT[T]:
        if instance is None:
            return self
        return instance._live[self.attr]


class ParamGroup:
    """Parameter group, declares its parameters on the node at construction."""

    def __init__(self, server: ROSParamServer, *, role: str | None = None) -> None:
        self._server = server
        self._role = role
        self._live: dict[str, ROSParamT[object]] = {}
        for param in type(self).params():
            default = param.default
            if isinstance(default, PerRole):
                default = default.pick(self.role())
            self._live[param.attr] = server.ROSParam(param.name, default, parse=param.parse)

    def role(self) -> str:
        if self._role is None:
            raise TypeError(f"{type(self).__name__} has role-dependent defaults and needs a role")
        return self._role

    @classmethod
    def params(cls) -> tuple[Param[object], ...]:
        return tuple(value for klass in reversed(cls.__mro__) for value in vars(klass).values() if isinstance(value, Param))

    def values(self) -> dict[str, object]:
        """Live value of every parameter by field name."""
        return {param.field: self._live[param.attr].value for param in type(self).params()}

    @classmethod
    def defaults(cls) -> dict[str, object]:
        """Parsed default of every parameter by field name, role-dependent ones left out."""
        return {param.field: param.parse(param.default) if param.parse is not None else param.default for param in cls.params() if not isinstance(param.default, PerRole)}


def configure[C](config: type[C], *groups: ParamGroup | type[ParamGroup], **given: object) -> C:
    """An attrs config with every field not given taken from the group parameter of that field name, live from a group instance, the default from a group class."""
    values: dict[str, object] = {}
    for group in groups:
        values.update(group.values() if isinstance(group, ParamGroup) else group.defaults())
    values.update(given)
    return config(**{field.name: values[field.name] for field in attrs.fields(config) if field.init})


def declare_launch_arguments(params: Mapping[str, Param[object]], *, prefix: str = "", reserved: Iterable[str] = ()) -> list[DeclareLaunchArgument]:
    """An empty-default launch argument prefix + name per described parameter, names in reserved left out."""
    from launch.actions import DeclareLaunchArgument

    skipped = frozenset(reserved)
    return [DeclareLaunchArgument(prefix + name, default_value="", description=f"{param.description} Empty = node default.") for name, param in params.items() if param.description and name not in skipped]
