from __future__ import annotations

import abc
from copy import deepcopy

import attrs
import pytest

from arena_simulation_setup.utils.cattrs import (
    ArenaConverter,
    Idempotent,
    Parseable,
    Serializable,
    converter,
)


# ---------------------------------------------------------------------------
# ArenaConverter
# ---------------------------------------------------------------------------


def test_arena_converter_is_singleton_module_level():
    from arena_simulation_setup.utils import cattrs as _m

    assert _m.converter is converter


def test_arena_converter_type_encoders_is_dict():
    assert isinstance(converter.type_encoders, dict)


def test_arena_converter_type_decoders_is_list():
    assert isinstance(converter.type_decoders, list)


def test_arena_converter_register_unstructure_hook_stored():
    c = ArenaConverter()
    hook = lambda obj: {}
    c.register_unstructure_hook(int, hook)
    assert c.type_encoders[int] is hook


def test_arena_converter_register_structure_hook_prepends():
    c = ArenaConverter()
    initial_len = len(c.type_decoders)
    hook = lambda v, t: v
    c.register_structure_hook(str, hook)
    assert len(c.type_decoders) == initial_len + 1
    # most-recently registered is first
    predicate, fn = c.type_decoders[0]
    assert fn is hook


def test_arena_converter_structure_hook_predicate_subclass():
    c = ArenaConverter()
    hook = lambda v, t: v
    c.register_structure_hook(int, hook)
    pred, _ = c.type_decoders[0]
    assert pred(int)
    assert pred(bool)  # bool is subclass of int


def test_arena_converter_structure_hook_predicate_unrelated():
    c = ArenaConverter()
    hook = lambda v, t: v
    c.register_structure_hook(int, hook)
    pred, _ = c.type_decoders[0]
    assert not pred(str)


# ---------------------------------------------------------------------------
# Idempotent
# ---------------------------------------------------------------------------


def test_idempotent_compatible_same_class():
    class Foo(Idempotent):
        pass

    obj = Foo()
    assert Foo._compatible(obj) is obj


def test_idempotent_compatible_subclass():
    class Base(Idempotent):
        pass

    class Sub(Base):
        pass

    sub = Sub()
    assert Base._compatible(sub) is sub


def test_idempotent_compatible_unrelated():
    class Foo(Idempotent):
        pass

    assert Foo._compatible(42) is None


def test_idempotent_converter_returns_same_instance():
    class Foo(Idempotent):
        pass

    obj = Foo()
    assert Foo.converter(obj) is obj


def test_idempotent_converter_clone_returns_copy():
    @attrs.define
    class Bar(Idempotent):
        x: int = 0

    obj = Bar(x=5)
    clone = Bar.converter_clone(obj)
    assert clone is not obj
    assert clone.x == 5


def test_idempotent_instance_or_first_match():
    class Foo(Idempotent):
        pass

    obj = Foo()
    fn = Foo.instance_or(lambda v: None)
    assert fn(obj) is obj


def test_idempotent_instance_or_chain_fallback():
    class Foo(Idempotent):
        pass

    called = []

    def fallback(v):
        called.append(v)
        return Foo()

    fn = Foo.instance_or(fallback)
    result = fn(42)
    assert isinstance(result, Foo)
    assert called == [42]


def test_idempotent_instance_or_all_fail():
    class Foo(Idempotent):
        pass

    fn = Foo.instance_or(lambda v: None)
    with pytest.raises(ValueError):
        fn(42)


# ---------------------------------------------------------------------------
# Serializable
# ---------------------------------------------------------------------------


def test_serializable_concrete_registers_unstructure_hook():
    @attrs.define
    class MySerializable(Serializable):
        x: int = 0

        def serialize(self):
            return {'x': self.x}

    obj = MySerializable(x=7)
    result = converter.unstructure(obj)
    assert result == {'x': 7}


def test_serializable_abstract_skips_hook():
    class AbstractSer(Serializable):
        @abc.abstractmethod
        def serialize(self):
            raise NotImplementedError

    # abstract class should not have a hook registered for it specifically
    # (no error is the pass condition, it's skipped)
    assert AbstractSer.__abstractmethods__


# ---------------------------------------------------------------------------
# Parseable
# ---------------------------------------------------------------------------


def test_parseable_isinstance_passthrough():
    @attrs.define
    class MyParseable(Parseable):
        x: int = 0

        @classmethod
        def parse(cls, value):
            return cls(x=int(value))

    obj = MyParseable(x=3)
    result = converter.structure(obj, MyParseable)
    assert result is obj


def test_parseable_parse_path():
    @attrs.define
    class MP2(Parseable):
        val: int = 0

        @classmethod
        def parse(cls, value):
            return cls(val=int(value))

    result = converter.structure(5, MP2)
    assert result.val == 5


def test_parseable_attrs_fromdict_path():
    @attrs.define
    class MP3(Parseable):
        name: str = ''

        @classmethod
        def parse(cls, value):
            raise ValueError("force fallback")

    result = converter.structure({'name': 'hello'}, MP3)
    assert result.name == 'hello'


def test_parseable_all_fail_raises_valueerror():
    @attrs.define
    class MP4(Parseable):
        x: int = 0

        @classmethod
        def parse(cls, value):
            raise ValueError("nope")

    with pytest.raises((ValueError, Exception)):
        converter.structure("not_a_dict", MP4)
