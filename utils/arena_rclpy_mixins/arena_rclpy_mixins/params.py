"""Node parameter declarations: a Param[T] carries its description and constraint, declared on any Node or through a ParamGroup."""

from __future__ import annotations

import array
import copy
import enum
import math
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence

import attrs
import yaml

if typing.TYPE_CHECKING:
    import rclpy
    import rclpy.node
    from launch.actions import DeclareLaunchArgument
    from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor, SetParametersResult

    from .ROSParamServer import ROSParamT

_INT64_MAX = 2**63 - 1
_INT64_MIN = -(2**63)
_ROS_TYPE_NAMES: dict[str, str] = {
    "bool": "BOOL",
    "int": "INTEGER",
    "float": "DOUBLE",
    "str": "STRING",
    "bool[]": "BOOL_ARRAY",
    "int[]": "INTEGER_ARRAY",
    "float[]": "DOUBLE_ARRAY",
    "str[]": "STRING_ARRAY",
}
_KIND_BY_ROS_TYPE_NAME = {ros: kind for kind, ros in _ROS_TYPE_NAMES.items()}
_SCALARS = ("bool", "int", "float", "str")


def _kind_of(value: object) -> str | None:
    match value:
        case bool():
            return "bool"
        case int():
            return "int"
        case float():
            return "float"
        case str():
            return "str"
        case list() | tuple():
            kinds = {_kind_of(item) for item in value}
            if len(kinds) == 1 and (kind := kinds.pop()) in _SCALARS:
                return f"{kind}[]"
            if kinds == {"int", "float"}:
                return "float[]"
    return None


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_SEQUENCES = (list, tuple, array.array)


def _matches(kind: str, value: object) -> bool:
    if kind.endswith("[]"):
        return isinstance(value, _SEQUENCES) and all(_matches(kind[:-2], item) for item in value)
    match kind:
        case "bool":
            return isinstance(value, bool)
        case "int":
            return isinstance(value, int) and not isinstance(value, bool)
        case "float":
            return _is_number(value)
        case _:
            return isinstance(value, str)


class Constraint:
    """Shape of the values a parameter accepts, with one descriptor encoding and one validator."""

    def check(self, value: object) -> None:
        del value

    def tokens(self, kind: str) -> list[str]:
        del kind
        return []


def _bound(value: float | None, open_: bool, left: bool) -> str:
    if value is None:
        return "(-inf" if left else "inf)"
    if left:
        return f"{'(' if open_ else '['}{value}"
    return f"{value}{')' if open_ else ']'}"


@attrs.frozen
class Range(Constraint):
    """Numeric bounds, None = unbounded on that side."""

    lo: float | None = None
    hi: float | None = None
    step: float = 0.0
    lo_open: bool = False
    hi_open: bool = False

    @property
    def infinite(self) -> bool:
        return any(bound is not None and math.isinf(bound) for bound in (self.lo, self.hi))

    def interval(self) -> str:
        return f"{_bound(self.lo, self.lo_open, True)}, {_bound(self.hi, self.hi_open, False)}"

    def check(self, value: object) -> None:
        if not _is_number(value):
            raise ValueError(f"{value!r} is not a number")
        number = typing.cast(float, value)
        below = self.lo is not None and (number < self.lo or (self.lo_open and number == self.lo))
        above = self.hi is not None and (number > self.hi or (self.hi_open and number == self.hi))
        if below or above:
            raise ValueError(f"{number} is outside {self.interval()}")

    def tokens(self, kind: str) -> list[str]:
        del kind
        return [token for token, flag in (("open:lo", self.lo_open), ("open:hi", self.hi_open)) if flag]


@attrs.frozen(init=False)
class Choices(Constraint):
    """Allowed string values, comma-separated several at once with many."""

    values: tuple[str, ...]
    many: bool

    def __init__(self, *values: str, many: bool = False) -> None:
        if not values:
            raise ValueError("Choices needs at least one value")
        self.__attrs_init__(values=tuple(values), many=many)  # pyright: ignore[reportAttributeAccessIssue]

    def check(self, value: object) -> None:
        if not isinstance(value, str):
            raise ValueError(f"{value!r} is not a string")
        names = [name for name in value.split(",") if name] if self.many else [value]
        unknown = [name for name in names if name not in self.values]
        if unknown:
            raise ValueError(f"{unknown if self.many else value!r} not in {list(self.values)}")

    def tokens(self, kind: str) -> list[str]:
        del kind
        return ["enum:" + ",".join(self.values), *(["many"] if self.many else [])]


@attrs.frozen
class Catalog(Constraint):
    """Values come from a live catalog of that name, not validated on the node."""

    name: str

    def tokens(self, kind: str) -> list[str]:
        del kind
        return [f"catalog:{self.name}"]


@attrs.frozen
class Pair(Constraint):
    """A [min, max] array, one element reads as both, two elements read sorted."""

    lo: float | None = None
    hi: float | None = None

    def check(self, value: object) -> None:
        if not isinstance(value, _SEQUENCES) or not 1 <= len(value) <= 2 or not all(_is_number(item) for item in value):
            raise ValueError(f"{value!r} is not a [min, max] pair")
        bounds = Range(self.lo, self.hi)
        for item in value:
            bounds.check(item)

    def tokens(self, kind: str) -> list[str]:
        tokens = ["range:int_pair" if kind == "int[]" else "range:float_pair"]
        if self.lo is not None:
            tokens.append(f"lo:{self.lo}")
        if self.hi is not None:
            tokens.append(f"hi:{self.hi}")
        return tokens


@attrs.frozen
class Text(Constraint):
    """Multiline text, a sketch when the text is a drawing."""

    sketch: bool = False

    def tokens(self, kind: str) -> list[str]:
        del kind
        return ["sketch" if self.sketch else "text"]


def _number_token(value: str) -> float:
    number = float(value)
    return int(number) if number.is_integer() else number


def parse_constraint(descriptor: ParameterDescriptor) -> Constraint | None:
    """The Constraint a descriptor written by Param.descriptor encodes, None for a free parameter."""
    tokens = [token for token in descriptor.additional_constraints.split(";") if token]
    keyed = {key: value for key, _, value in (token.partition(":") for token in tokens)}
    if "enum" in keyed:
        return Choices(*keyed["enum"].split(","), many="many" in keyed)
    if "catalog" in keyed:
        return Catalog(keyed["catalog"])
    if "range" in keyed:
        return Pair(lo=_number_token(keyed["lo"]) if "lo" in keyed else None, hi=_number_token(keyed["hi"]) if "hi" in keyed else None)
    if "text" in keyed or "sketch" in keyed:
        return Text(sketch="sketch" in keyed)
    opens = {token for token in tokens if token.startswith("open:")}
    if descriptor.integer_range:
        bounds = typing.cast("Sequence[IntegerRange]", descriptor.integer_range)[0]
        return Range(None if bounds.from_value == _INT64_MIN else bounds.from_value, None if bounds.to_value == _INT64_MAX else bounds.to_value, bounds.step, "open:lo" in opens, "open:hi" in opens)
    if descriptor.floating_point_range:
        bounds = typing.cast("Sequence[FloatingPointRange]", descriptor.floating_point_range)[0]
        return Range(None if math.isinf(bounds.from_value) else bounds.from_value, None if math.isinf(bounds.to_value) else bounds.to_value, bounds.step, "open:lo" in opens, "open:hi" in opens)
    return None


def _finite_or_none(value: float | None) -> float | None:
    return None if value is None or math.isinf(value) else value


def constraint_json(constraint: Constraint) -> dict[str, object]:
    """The constraint as plain JSON data, unbounded ends as null."""
    match constraint:
        case Range():
            return {"kind": "range", "lo": _finite_or_none(constraint.lo), "hi": _finite_or_none(constraint.hi), "step": constraint.step, "lo_open": constraint.lo_open, "hi_open": constraint.hi_open}
        case Choices():
            return {"kind": "choices", "values": list(constraint.values), "many": constraint.many}
        case Catalog():
            return {"kind": "catalog", "name": constraint.name}
        case Pair():
            return {"kind": "pair", "lo": constraint.lo, "hi": constraint.hi}
        case Text():
            return {"kind": "text", "sketch": constraint.sketch}
    raise TypeError(f"not a constraint: {constraint!r}")


@attrs.frozen(init=False)
class PerRole[T]:
    """Default that differs between roles, by role name."""

    by_role: tuple[tuple[str, T], ...]

    def __init__(self, **by_role: T) -> None:
        if not by_role:
            raise ValueError("PerRole needs a default for at least one role")
        self.__attrs_init__(by_role=tuple(by_role.items()))  # pyright: ignore[reportAttributeAccessIssue]

    def pick(self, role: str) -> T:
        """Raises KeyError for a role without a default."""
        for name, value in self.by_role:
            if name == role:
                return value
        raise KeyError(f"no default for role {role!r}, expected one of {[name for name, _ in self.by_role]}")

    @property
    def first(self) -> T:
        return self.by_role[0][1]


def pair(value: object) -> tuple[float, float]:
    """A [min, max] parameter as a sorted tuple, one element as both ends."""
    items = [typing.cast(float, item) for item in typing.cast(Sequence[float], value)]
    lo, hi = (items[0], items[0]) if len(items) == 1 else sorted(items[:2])
    return lo, hi


def floats(value: object) -> tuple[float, ...]:
    return tuple(float(item) for item in typing.cast(Iterable[float], value))


def names(value: object) -> tuple[str, ...]:
    return tuple(str(item).strip().lower() for item in typing.cast(Iterable[str], value) if str(item).strip())


_CONSTRAINT_KINDS: dict[type[Constraint], tuple[str, ...]] = {
    Range: ("int", "float"),
    Choices: ("str",),
    Catalog: ("str", "str[]"),
    Pair: ("int[]", "float[]"),
    Text: ("str",),
}


class Param[T]:
    """Parameter declaration. On a group class it is the declaration, on a group instance the live handle."""

    @typing.overload
    def __init__(self, name: str, default: T | PerRole[T] | None, description: str, *, constraint: Constraint | None = None, parse: Callable[[object], T] | None = None, type_: rclpy.Parameter.Type | None = None, read_only: bool = False) -> None: ...

    @typing.overload
    def __init__(self, name: str, default: object, description: str, *, constraint: Constraint | None = None, parse: Callable[[object], T], type_: rclpy.Parameter.Type | None = None, read_only: bool = False) -> None: ...

    def __init__(self, name: str, default: object, description: str, *, constraint: Constraint | None = None, parse: Callable[[object], T] | None = None, type_: rclpy.Parameter.Type | None = None, read_only: bool = False) -> None:
        if not name:
            raise ValueError("a parameter needs a name")
        sample = default.first if isinstance(default, PerRole) else default
        if isinstance(sample, enum.Enum):
            kind = "str"
            constraint = constraint if constraint is not None else Choices(*(str(member.value) for member in type(sample)))
            parse = parse if parse is not None else typing.cast(Callable[[object], T], type(sample))
        else:
            kind = _kind_of(sample)
            if type_ is not None:
                kind = _KIND_BY_ROS_TYPE_NAME[type_.name]
            elif kind is None or (isinstance(sample, (list, tuple)) and not sample):
                raise ValueError(f"parameter {name}: the type cannot be read from default {sample!r}, pass type_")
        if constraint is not None and kind not in _CONSTRAINT_KINDS[type(constraint)]:
            raise ValueError(f"parameter {name}: {type(constraint).__name__} does not apply to a {kind} parameter")
        self.name = name
        self.default = typing.cast("T | PerRole[T] | None", default)
        self.description = description.strip()
        self.constraint = constraint
        self.parse = parse
        self.kind = kind
        self.read_only = read_only
        self.attr = ""
        infinite = isinstance(constraint, Range) and constraint.infinite
        self.finite = kind in ("float", "float[]") and not infinite and (not isinstance(sample, float) or math.isfinite(sample))

    def __set_name__(self, owner: type, attr: str) -> None:
        self.attr = attr

    def __repr__(self) -> str:
        return f"Param({self.name!r}, {self.default!r})"

    @property
    def field(self) -> str:
        """Config field name, the lowercase declaration name."""
        return self.attr.lower()

    @property
    def required(self) -> bool:
        return self.default is None

    def at(self, namespace: str | None) -> Param[T]:
        """The same declaration under a dotted namespace prefix."""
        if not namespace:
            return self
        prefixed = copy.copy(self)
        prefixed.name = f"{namespace}.{self.name}"
        return prefixed

    def ros_default(self, role: str | None = None) -> object:
        """The declared default as its ROS value, picked by role for a PerRole default."""
        default = self.default
        if isinstance(default, PerRole):
            if role is None:
                raise TypeError(f"parameter {self.name} has role-dependent defaults and needs a role")
            default = default.pick(role)
        if isinstance(default, enum.Enum):
            return default.value
        if isinstance(default, tuple):
            default = list(default)
        if self.kind == "float" and isinstance(default, int) and not isinstance(default, bool):
            return float(default)
        if self.kind == "float[]" and isinstance(default, list):
            return [float(item) for item in default]
        return default

    def parsed_default(self, role: str | None = None) -> T:
        value = self.ros_default(role)
        return self.parse(value) if self.parse is not None else typing.cast(T, value)

    def check(self, value: object) -> None:
        """Raises ValueError when the ROS value does not fit the type, the finiteness rule or the constraint."""
        if not _matches(self.kind, value):
            raise ValueError(f"{value!r} is not {self.kind}")
        if self.finite:
            numbers = typing.cast(Iterable[float], value) if self.kind.endswith("[]") else (typing.cast(float, value),)
            if not all(math.isfinite(number) for number in numbers):
                raise ValueError(f"{value!r} is not finite")
        if self.constraint is not None:
            self.constraint.check(value)

    def descriptor(self) -> ParameterDescriptor:
        """The ParameterDescriptor of this declaration, no node needed."""
        from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor
        from rclpy import Parameter

        constraint = self.constraint
        descriptor = ParameterDescriptor(
            type=Parameter.Type[_ROS_TYPE_NAMES[self.kind]].value,
            description=self.description,
            read_only=self.read_only,
            additional_constraints=";".join(constraint.tokens(self.kind)) if constraint is not None else "",
        )
        if isinstance(constraint, Range):
            if self.kind == "int":
                lo = _INT64_MIN if constraint.lo is None else int(constraint.lo)
                hi = _INT64_MAX if constraint.hi is None else int(constraint.hi)
                descriptor.integer_range = [IntegerRange(from_value=lo, to_value=hi, step=int(constraint.step))]
            else:
                lo = -math.inf if constraint.lo is None else float(constraint.lo)
                hi = math.inf if constraint.hi is None else float(constraint.hi)
                descriptor.floating_point_range = [FloatingPointRange(from_value=lo, to_value=hi, step=float(constraint.step))]
        return descriptor

    def coerce(self, raw: str) -> object:
        """Launch string as this parameter's ROS value."""
        kind = self.kind
        if kind == "str":
            return raw
        if kind == "bool":
            value = yaml.safe_load(raw)
            if not isinstance(value, bool):
                raise ValueError(f"{self.name} expects true or false, got {raw!r}")
            return value
        try:
            if kind == "int":
                return int(raw)
            if kind == "float":
                return float(raw)
            loaded = yaml.safe_load(raw)
            items = loaded if isinstance(loaded, list) else [loaded]
            element = {"bool": bool, "int": int, "float": float, "str": str}[kind[:-2]]
            value = [element(item) for item in items]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self.name} expects {kind}, got {raw!r}") from exc
        if value == []:
            raise ValueError(f"{self.name} cannot be an empty list, a ROS parameter carries no empty array")
        return value

    @typing.overload
    def __get__(self, instance: None, owner: type) -> Param[T]: ...

    @typing.overload
    def __get__(self, instance: ParamGroup, owner: type) -> ParamHandle[T]: ...

    def __get__(self, instance: ParamGroup | None, owner: type) -> Param[T] | ParamHandle[T]:
        if instance is None:
            return self
        return typing.cast("ParamHandle[T]", instance._live[self.attr])


class ParamHandle[T](typing.Protocol):
    """A declared parameter on a node: its name, parsed live value and removal."""

    @property
    def name(self) -> str: ...

    @property
    def value(self) -> T: ...

    @value.setter
    def value(self, value: object) -> None: ...

    def destroy(self) -> None: ...


class _Lazy[T]:
    _raw: object
    _parsed: T
    _has: bool

    def __init__(self, param: Param[T]) -> None:
        self._param = param
        self._reset_cache()

    def _reset_cache(self) -> None:
        self._raw = _UNSET
        self._has = False

    def _value_of(self, raw: object) -> T:
        if raw is None:
            raise ValueError(f"parameter {self._param.name} is unset")
        if not self._has or raw is not self._raw:
            self._parsed = self._param.parse(raw) if self._param.parse is not None else typing.cast(T, raw)
            self._raw = raw
            self._has = True
        return self._parsed


class _UNSET: ...


class _ServerHandle[T](_Lazy[T]):
    def __init__(self, param: Param[T], live: ROSParamT[object]) -> None:
        super().__init__(param)
        self._live = live

    @property
    def name(self) -> str:
        return self._param.name

    @property
    def value(self) -> T:
        return self._value_of(self._live.param)  # pyright: ignore[reportAttributeAccessIssue]

    @value.setter
    def value(self, value: object) -> None:
        self._live.value = value

    def callback(self, value: object) -> bool:
        return self._live.callback(value)

    def destroy(self) -> None:
        self._live.destroy()


class _NodeHandle[T](_Lazy[T]):
    def __init__(self, param: Param[T], node: rclpy.node.Node) -> None:
        super().__init__(param)
        self._node = node

    @property
    def name(self) -> str:
        return self._param.name

    @property
    def value(self) -> T:
        return self._value_of(_current(self._node, self._param.name))

    @value.setter
    def value(self, value: object) -> None:
        from rclpy import Parameter

        result = self._node.set_parameters([Parameter(self._param.name, value=value)])[0]
        if not result.successful:
            raise ValueError(result.reason)

    def destroy(self) -> None:
        import rclpy.exceptions

        try:
            self._node.undeclare_parameter(self._param.name)
        except rclpy.exceptions.ParameterNotDeclaredException:
            pass


class _Registry:
    def __init__(self) -> None:
        self.params: dict[str, Param[object]] = {}
        self.handles: dict[str, ParamHandle[object]] = {}
        self.groups: dict[tuple[type, str, str | None], ParamGroup] = {}
        self.installed = False

    def on_set(self, parameters: list[rclpy.Parameter]) -> SetParametersResult:
        from rcl_interfaces.msg import SetParametersResult

        for parameter in parameters:
            param = self.params.get(parameter.name)
            if param is None or parameter.value is None:
                continue
            try:
                param.check(parameter.value)
            except ValueError as exc:
                return SetParametersResult(successful=False, reason=f"{parameter.name}: {exc}")
        return SetParametersResult(successful=True)


def _registry(node: rclpy.node.Node) -> _Registry:
    registry = vars(node).get("_arena_params")
    if registry is None:
        registry = _Registry()
        vars(node)["_arena_params"] = registry
    return registry


def declared(node: rclpy.node.Node) -> dict[str, Param[object]]:
    """Every parameter declared through declare on this node, by name."""
    return _registry(node).params


def _current(node: rclpy.node.Node, name: str) -> object:
    import rclpy.exceptions
    from rclpy import Parameter

    try:
        parameter = node.get_parameter(name)
    except rclpy.exceptions.ParameterUninitializedException:
        return None
    return None if parameter.type_ is Parameter.Type.NOT_SET else parameter.value


def declare_safe(node: rclpy.node.Node, name: str, value: object, *, descriptor: ParameterDescriptor | None = None, ignore_override: bool = False) -> None:
    """Declare on any node, keeping an override already declared (its descriptor set, an int override of a double redeclared as float)."""
    import rclpy.exceptions
    from rclpy import Parameter

    if node.has_parameter(name):
        if descriptor is not None:
            existing = node.get_parameter(name)
            if descriptor.type == Parameter.Type.NOT_SET.value:
                descriptor.type = existing.type_.value
            elif descriptor.type == Parameter.Type.DOUBLE.value and existing.type_ == Parameter.Type.INTEGER:
                node.undeclare_parameter(name)
                node.declare_parameter(name, float(typing.cast(int, existing.value)), descriptor=descriptor, ignore_override=True)
                return
            node.set_descriptor(name, descriptor)
        return
    declared_type = Parameter.Type.NOT_SET.value if descriptor is None else descriptor.type
    typed = declared_type != Parameter.Type.NOT_SET.value and not isinstance(value, Parameter.Type)
    try:
        if typed and (value is None or value == []):
            type_enum = Parameter.Type(declared_type)
            node.declare_parameter(name, type_enum, descriptor=descriptor, ignore_override=ignore_override)
            if value is not None:
                try:
                    already_set = node.get_parameter(name).type_ != Parameter.Type.NOT_SET
                except rclpy.exceptions.ParameterUninitializedException:
                    already_set = False
                if not already_set:
                    node.set_parameters([Parameter(name=name, type_=type_enum, value=value)])
        else:
            node.declare_parameter(name, value, descriptor=descriptor, ignore_override=ignore_override)
    except rclpy.exceptions.ParameterAlreadyDeclaredException:
        pass


@typing.overload
def declare[T](node: rclpy.node.Node, param: Param[T], *, role: str | None = None) -> ParamHandle[T]: ...


@typing.overload
def declare(node: rclpy.node.Node, param: Param[typing.Any], *, role: str | None = None) -> ParamHandle[typing.Any]: ...


def declare(node: rclpy.node.Node, param: Param[typing.Any], *, role: str | None = None) -> ParamHandle[typing.Any]:
    """Declare param on node with its descriptor, validate the current value, return the live handle (the same one on a repeat)."""
    import rclpy.exceptions

    from .ROSParamServer import ROSParamServer

    registry = _registry(node)
    existing = registry.handles.get(param.name)
    if existing is not None:
        return existing
    try:
        declare_safe(node, param.name, param.ros_default(role), descriptor=param.descriptor())
        current = _current(node, param.name)
        if current is not None:
            param.check(current)
    except (ValueError, rclpy.exceptions.InvalidParameterValueException, rclpy.exceptions.InvalidParameterTypeException) as exc:
        raise RuntimeError(f"parameter {param.name}: {exc}") from exc
    handle: ParamHandle[typing.Any]
    if isinstance(node, ROSParamServer):
        node.add_param_callback(param.name, _validator(param))
        handle = _ServerHandle(param, node.ROSParam[object](param.name))
    else:
        if not registry.installed:
            node.add_on_set_parameters_callback(registry.on_set)
            registry.installed = True
        handle = _NodeHandle(param, node)
    registry.params[param.name] = param
    registry.handles[param.name] = handle
    return handle


def _validator[T](param: Param[T]) -> Callable[[object], bool]:
    def check(value: object) -> bool:
        if value is not None:
            param.check(value)
        return True

    return check


class ParamGroup:
    """Parameter group, declares its parameters on the node at construction, one instance per node, namespace and role."""

    _ready: bool

    def __new__(cls, server: rclpy.node.Node, *, role: str | None = None, namespace: str | None = None) -> typing.Self:
        registry = _registry(server)
        key = (cls, str(namespace or ""), role)
        instance = registry.groups.get(key)
        if instance is None:
            instance = super().__new__(cls)
            instance._ready = False
            registry.groups[key] = instance
        return typing.cast(typing.Self, instance)

    def __init__(self, server: rclpy.node.Node, *, role: str | None = None, namespace: str | None = None) -> None:
        if self._ready:
            return
        self._server = server
        self._role = role
        self._namespace = str(namespace or "")
        self._live: dict[str, ParamHandle[object]] = {param.attr: declare(server, param.at(self._namespace), role=role) for param in type(self).params()}
        self._ready = True

    def role(self) -> str:
        if self._role is None:
            raise TypeError(f"{type(self).__name__} has role-dependent defaults and needs a role")
        return self._role

    @property
    def namespace(self) -> str:
        return self._namespace

    @classmethod
    def params(cls) -> tuple[Param[object], ...]:
        return tuple(value for klass in reversed(cls.__mro__) for value in vars(klass).values() if isinstance(value, Param))

    @classmethod
    def by_name(cls, namespace: str | None = None) -> dict[str, Param[object]]:
        """Every parameter by its full name under the namespace."""
        return {param.at(namespace).name: param.at(namespace) for param in cls.params()}

    def values(self) -> dict[str, object]:
        """Live value of every parameter by field name."""
        return {param.field: self._live[param.attr].value for param in type(self).params()}

    @classmethod
    def defaults(cls) -> dict[str, object]:
        """Parsed default of every parameter by field name, role-dependent ones left out."""
        return {param.field: param.parsed_default() for param in cls.params() if not isinstance(param.default, PerRole)}


def configure[C](config: type[C], *groups: ParamGroup | type[ParamGroup], **given: object) -> C:
    """An attrs config with every field not given taken from the group parameter of that field name, live from a group instance, the default from a group class."""
    values: dict[str, object] = {}
    for group in groups:
        values.update(group.values() if isinstance(group, ParamGroup) else group.defaults())
    values.update(given)
    return config(**{field.name: values[field.name] for field in attrs.fields(config) if field.init})


def declare_launch_arguments(params: Mapping[str, Param[object]], *, prefix: str = "", reserved: Iterable[str] = ()) -> list[DeclareLaunchArgument]:
    """An empty-default launch argument prefix + name per parameter, names in reserved left out, choices from Choices and enum defaults."""
    from launch.actions import DeclareLaunchArgument

    skipped = frozenset(reserved)
    arguments: list[DeclareLaunchArgument] = []
    for name, param in params.items():
        if name in skipped:
            continue
        choices = ["", *param.constraint.values] if isinstance(param.constraint, Choices) and not param.constraint.many else None
        arguments.append(DeclareLaunchArgument(prefix + name, default_value="", description=f"{param.description} Empty = node default.", choices=choices))
    return arguments
