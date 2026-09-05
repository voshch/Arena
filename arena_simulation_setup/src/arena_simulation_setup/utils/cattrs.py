from __future__ import annotations

import abc
import contextvars
import traceback
import typing
from collections.abc import Mapping
from copy import deepcopy

import attrs
import cattrs

from arena_simulation_setup.utils.resolution import activate_resolver

if typing.TYPE_CHECKING:
    from arena_simulation_setup.utils.geometry import PointResolver

_active_converter: contextvars.ContextVar[ArenaConverter] = contextvars.ContextVar('_active_converter')


class ArenaConverter(cattrs.Converter):
    """Custom converter for Arena types that exposes an API for retrieving registered structure and unstructure hooks."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._encoders: dict[type, typing.Callable] = {}
        self._decoders: list[tuple[typing.Callable[[type], bool], typing.Callable]] = []
        self._resolver: PointResolver | None = None

    def set_resolver(self, resolver: PointResolver | None) -> None:
        """Bind a zone resolver, activated while this converter structures values."""
        self._resolver = resolver

    @property
    def type_encoders(self) -> dict[type, typing.Callable]:
        return self._encoders

    @property
    def type_decoders(self) -> list[tuple[typing.Callable[[type], bool], typing.Callable]]:
        return self._decoders

    def register_unstructure_hook(self, cl: type, *args: object, **kwargs: object) -> None:
        """
        Register an unstructure hook for a given type.
        """
        if args:
            func = args[0]
            self._encoders[cl] = func
        return super().register_unstructure_hook(cl, *args, **kwargs)

    def register_structure_hook(self, cl: type, *args: object, **kwargs: object) -> None:
        """
        Register a structure hook for a given type.
        """
        if args:

            def predicate(t: type, c: type = cl) -> bool:
                try:
                    return isinstance(t, type) and issubclass(t, c)
                except TypeError:
                    return False

            func = args[0]
            self._decoders.insert(0, (predicate, func))
            # return super().register_structure_hook_func(predicate, func)
        return super().register_structure_hook(cl, *args, **kwargs)

    def structure(self, obj: object, cl: type[T]) -> T:
        token = _active_converter.set(self)
        try:
            with activate_resolver(self._resolver):
                return super().structure(obj, cl)
        finally:
            _active_converter.reset(token)

    def structure_attrs_fromdict(self, obj: Mapping[str, object], cl: type) -> object:
        token = _active_converter.set(self)
        try:
            with activate_resolver(self._resolver):
                return super().structure_attrs_fromdict(obj, cl)
        finally:
            _active_converter.reset(token)

    @staticmethod
    def current() -> ArenaConverter:
        """Return the converter currently performing structuring, or the module default."""
        return _active_converter.get(converter)


converter = ArenaConverter()

IdempotentT = typing.TypeVar('IdempotentT', bound='Idempotent')


class Idempotent:
    """
    A class that ensures its instances are idempotent.
    """

    @classmethod
    def _compatible(cls, obj: object) -> typing.Self | None:
        """
        Check if the object is compatible with the class.
        """
        if isinstance(obj, cls) or issubclass(type(obj), cls) or typing.get_origin(type(obj)) is cls:
            return obj
        return None

    @classmethod
    def instance_or(cls, *chain: typing.Callable) -> typing.Callable[[object], typing.Self]:
        def inner(v: object) -> typing.Self:
            for fn in (cls._compatible, *chain):
                if (inst := fn(v)) is not None:
                    return inst
            raise ValueError(f'Cannot convert {v} to {cls}')

        return inner

    @classmethod
    def converter(cls, *args: object, **kwargs: object) -> typing.Self:
        """
        If the value is already an instance of the class , return it.
        Otherwise, create a new instance of the class with the value.
        """
        if args and (inst := cls._compatible(args[0])):
            return inst
        return cls(*args, **kwargs)

    @classmethod
    def converter_clone(cls, *args: object, **kwargs: object) -> typing.Self:
        """
        If the value is already an instance of the class , return a deepcopy of it.
        If not , create a new instance of the class with the value.
        """
        if args and (inst := cls._compatible(args[0])):
            return deepcopy(inst)
        return cls(*args, **kwargs)


T = typing.TypeVar('T')


def idempotent(cls: type[T]) -> type[T]:
    """
    Make class idempotent.
    """
    if not issubclass(cls, Idempotent):
        return type(cls.__name__, (Idempotent, cls), {})
    return cls


# Serialization and Deserialization


class Serializable(abc.ABC):
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

        if not getattr(cls, "__abstractmethods__", set()):

            def unstructure_hook(obj: Serializable) -> object:
                return obj.serialize()

            converter.register_unstructure_hook(cls, unstructure_hook)

    @abc.abstractmethod
    def serialize(self) -> object:
        """
        Define the custom serialization logic for this object.
        """
        raise NotImplementedError


ParseableT = typing.TypeVar('ParseableT', bound='Parseable')


class Parseable:
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

        if not getattr(cls, "__abstractmethods__", set()):

            def try_parse(value: object, target_type: type) -> object:
                errors: list[Exception] = []
                c = ArenaConverter.current()
                # check idempotence
                try:
                    if isinstance(value, target_type):
                        return value
                except TypeError as e:
                    errors.append(e)

                # try Parseable.parse
                try:
                    return target_type.parse(value)
                except Exception as e:
                    errors.append(e)

                # try "normal" attrs structuring
                try:
                    if isinstance(value, dict):
                        unknown = set(value) - {f.name for f in attrs.fields(target_type)}
                        if unknown:
                            raise ValueError(f'unknown keys {sorted(unknown)} for {target_type}')
                        return c.structure_attrs_fromdict(value, target_type)
                except Exception as e:
                    errors.append(e)

                errors_repr = ''
                for error in errors:
                    errors_repr += ''.join(traceback.format_exception(type(error), error, error.__traceback__))
                raise ValueError(f'could not parse into {target_type} from {value}:{errors_repr}')

            converter.register_structure_hook(cls, try_parse)

    @classmethod
    def parse(cls, value: object) -> typing.Self:
        if isinstance(value, Mapping):
            unknown = set(value) - {f.name for f in attrs.fields(cls)}
            if unknown:
                raise ValueError(f'unknown keys {sorted(unknown)} for {cls}')
        return converter.structure_attrs_fromdict(deepcopy(value), cls)


converter.register_structure_hook(dict, lambda v, t: v if isinstance(v, dict) else dict(v))

__all__ = [
    "Serializable",
    "Parseable",
    "converter",
    "Idempotent",
]
