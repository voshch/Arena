from __future__ import annotations

import os
from pathlib import Path

import attrs
import pytest

from arena_simulation_setup import DOMAIN_DEFAULT
from arena_simulation_setup.tree import (
    DomainAssetIdentifier,
    FallbackResolver,
    Identifier,
    ModifiersDomainAssetIdentifier,
    SimplePathResolver,
)


@attrs.define(eq=False, hash=False)
class _FooIdentifier(Identifier[None]):
    _resolvers = []

    def load(self, path: Path, /, **kwargs: object) -> None:
        return None


@attrs.define(eq=False, hash=False)
class _DomainFoo(DomainAssetIdentifier[None]):
    _asset_type = 'foo'
    _resolvers = []

    def load(self, path: Path, /, **kwargs: object) -> None:
        return None


@attrs.define(eq=False, hash=False)
class _ModFoo(ModifiersDomainAssetIdentifier[None]):
    _asset_type = 'foo'
    _resolvers = []

    def load(self, path: Path, /, **kwargs: object) -> None:
        return None


def test_identifier_parse_str():
    ident = _FooIdentifier.parse('hello')
    assert ident.name == 'hello'


def test_identifier_parse_idempotent():
    ident = _FooIdentifier.parse('hello')
    assert _FooIdentifier.parse(ident) is ident


def test_identifier_parse_non_str_raises():
    with pytest.raises(TypeError):
        _FooIdentifier.parse(42)


def test_identifier_relpath():
    ident = _FooIdentifier(name='some/model')
    assert ident.relpath() == Path('some/model')


def test_identifier_serialize():
    ident = _FooIdentifier(name='asset')
    assert ident.serialize() == 'asset'


def test_identifier_from_relpath_single():
    ident = _FooIdentifier.from_relpath(Path('myasset'))
    assert ident.name == 'myasset'


def test_identifier_from_relpath_nested():
    ident = _FooIdentifier.from_relpath(Path('a/b/c'))
    assert ident.name == 'a/b/c'


def test_identifier_from_relpath_empty_raises():
    with pytest.raises(AssertionError):
        _FooIdentifier.from_relpath(Path(''))


def test_identifier_label():
    assert _FooIdentifier.label() == '_FooIdentifier'


def test_domain_parse_name_only():
    ident = _DomainFoo.parse('mymodel')
    assert ident.name == 'mymodel'
    assert ident.domain == DOMAIN_DEFAULT


def test_domain_parse_domain_slash_name():
    ident = _DomainFoo.parse('MyDomain/mymodel')
    assert ident.domain == 'MyDomain'
    assert ident.name == 'mymodel'


def test_domain_parse_with_asset_type_infix():
    ident = _DomainFoo.parse('MyDomain/foo/mymodel')
    assert ident.domain == 'MyDomain'
    assert ident.name == 'mymodel'


def test_domain_parse_idempotent():
    ident = _DomainFoo.parse('Dom/name')
    assert _DomainFoo.parse(ident) is ident


def test_domain_parse_non_str_raises():
    with pytest.raises(TypeError):
        _DomainFoo.parse(123)


def test_domain_parse_multipart_name():
    ident = _DomainFoo.parse('Dom/a/b/c')
    assert ident.domain == 'Dom'
    assert ident.name == 'a/b/c'


def test_domain_relpath():
    ident = _DomainFoo(name='mymodel', domain='MyDomain')
    assert ident.relpath() == Path('MyDomain/foo/mymodel')


def test_domain_shortname():
    ident = _DomainFoo(name='mymodel', domain='MyDomain')
    assert ident.shortname == 'MyDomain/foo/mymodel'


def test_domain_eq_same():
    a = _DomainFoo(name='m', domain='D')
    b = _DomainFoo(name='m', domain='D')
    assert a == b


def test_domain_eq_diff_name():
    assert _DomainFoo(name='a', domain='D') != _DomainFoo(name='b', domain='D')


def test_domain_eq_diff_domain():
    assert _DomainFoo(name='m', domain='A') != _DomainFoo(name='m', domain='B')


def test_domain_eq_wrong_type():
    ident = _DomainFoo(name='m', domain='D')
    assert ident != 'not-an-identifier'


def test_domain_hash_equal_objects():
    a = _DomainFoo(name='m', domain='D')
    b = _DomainFoo(name='m', domain='D')
    assert hash(a) == hash(b)


def test_domain_hash_usable_in_set():
    a = _DomainFoo(name='m', domain='D')
    b = _DomainFoo(name='m', domain='D')
    c = _DomainFoo(name='n', domain='D')
    s = {a, b, c}
    assert len(s) == 2


def test_domain_from_relpath_roundtrip():
    ident = _DomainFoo(name='mymodel', domain='MyDomain')
    assert _DomainFoo.from_relpath(ident.relpath()) == ident


def test_domain_from_relpath_too_short_raises():
    with pytest.raises(AssertionError):
        _DomainFoo.from_relpath(Path('only_one_part'))


def test_domain_from_relpath_wrong_asset_type_raises():
    with pytest.raises(AssertionError):
        _DomainFoo.from_relpath(Path('MyDomain/bar/model'))


def test_domain_from_relpath_nested_name():
    ident = _DomainFoo.from_relpath(Path('D/foo/a/b'))
    assert ident.domain == 'D'
    assert ident.name == 'a/b'


def test_mod_parse_str_no_modifiers():
    ident = _ModFoo.parse('Dom/model')
    assert ident.domain == 'Dom'
    assert ident.name == 'model'
    assert ident.modifiers == {}


def test_mod_parse_tuple():
    ident = _ModFoo.parse(('Dom/model', {'scale': 2.0}))
    assert ident.modifiers == {'scale': 2.0}
    assert ident.domain == 'Dom'


def test_mod_parse_idempotent():
    ident = _ModFoo.parse('Dom/model')
    assert _ModFoo.parse(ident) is ident


def test_mod_parse_wrong_identifier_type_raises():
    other = _DomainFoo.parse('Dom/model')
    with pytest.raises(TypeError):
        _ModFoo.parse(other)


def test_mod_parse_tuple_wrong_length_raises():
    with pytest.raises((ValueError, TypeError)):
        _ModFoo.parse(('a', 'b', 'c'))


def test_mod_parse_tuple_non_str_value_raises():
    with pytest.raises(TypeError):
        _ModFoo.parse((42, {}))


def test_mod_parse_tuple_non_dict_modifiers_raises():
    with pytest.raises(TypeError):
        _ModFoo.parse(('Dom/model', 'not-a-dict'))


def test_mod_eq_same_no_modifiers():
    a = _ModFoo(name='m', domain='D')
    b = _ModFoo(name='m', domain='D')
    assert a == b


def test_mod_eq_same_with_modifiers():
    a = _ModFoo(name='m', domain='D', modifiers={'k': 1})
    b = _ModFoo(name='m', domain='D', modifiers={'k': 1})
    assert a == b


def test_mod_eq_diff_modifiers():
    a = _ModFoo(name='m', domain='D', modifiers={'k': 1})
    b = _ModFoo(name='m', domain='D', modifiers={'k': 2})
    assert a != b


def test_mod_eq_wrong_type():
    ident = _ModFoo(name='m', domain='D')
    assert ident != _DomainFoo(name='m', domain='D')


def test_mod_hash_with_modifiers():
    a = _ModFoo(name='m', domain='D', modifiers={'x': 1})
    b = _ModFoo(name='m', domain='D', modifiers={'x': 1})
    assert hash(a) == hash(b)


def test_mod_serialize_no_modifiers():
    ident = _ModFoo(name='m', domain='D')
    result = ident.serialize()
    assert isinstance(result, tuple)
    assert result[1] == {}


def test_mod_serialize_with_modifiers():
    ident = _ModFoo(name='m', domain='D', modifiers={'k': 'v'})
    result = ident.serialize()
    assert isinstance(result, tuple)
    assert result[1] == {'k': 'v'}


def test_mod_relpath_roundtrip():
    ident = _ModFoo(name='model', domain='TestDomain', modifiers={'scale': 1})
    back = _ModFoo.from_relpath(ident.relpath())
    assert back.domain == ident.domain
    assert back.name == ident.name


def test_simple_path_resolver_listall_empty_dir(tmp_path):
    resolver = SimplePathResolver(_DomainFoo, path=tmp_path)
    assert list(resolver.listall()) == []


def test_simple_path_resolver_listall_nonexistent(tmp_path):
    resolver = SimplePathResolver(_DomainFoo, path=tmp_path / 'ghost')
    assert list(resolver.listall()) == []


def test_simple_path_resolver_listall_finds_assets(tmp_path):
    (tmp_path / 'MyDomain' / 'foo' / 'chair').mkdir(parents=True)
    resolver = SimplePathResolver(_DomainFoo, path=tmp_path)
    results = list(resolver.listall())
    assert any(r.name == 'chair' and r.domain == 'MyDomain' for r in results)


def test_simple_path_resolver_listall_multiple_assets(tmp_path):
    for name in ('chair', 'table'):
        (tmp_path / 'D' / 'foo' / name).mkdir(parents=True)
    resolver = SimplePathResolver(_DomainFoo, path=tmp_path)
    results = list(resolver.listall())
    names = {r.name for r in results}
    assert 'chair' in names
    assert 'table' in names


def test_simple_path_resolver_listall_skips_invalid_paths(tmp_path):
    (tmp_path / 'random.txt').write_text('x')
    resolver = SimplePathResolver(_DomainFoo, path=tmp_path)
    results = list(resolver.listall())
    assert all(isinstance(r, _DomainFoo) for r in results)


def test_fallback_resolver_is_write_only():
    import asyncio

    base = Path('/some/base')
    resolver = FallbackResolver(_DomainFoo, path=base)
    ident = _DomainFoo(name='mymodel', domain='D')
    assert resolver.destination(ident) == base / ident.relpath()
    assert asyncio.run(resolver.resolve(ident)) is None


def test_fallback_resolver_repr():
    resolver = FallbackResolver(_DomainFoo, path=Path('/base'))
    assert '/base' in repr(resolver)


def test_fallback_resolver_listall_empty():
    resolver = FallbackResolver(_DomainFoo, path=Path('/base'))
    assert list(resolver.listall()) == []


def test_simple_resolver_invalidate_clears_cache(tmp_path):
    (tmp_path / 'D' / 'foo' / 'model').mkdir(parents=True)
    resolver = SimplePathResolver(_DomainFoo, path=tmp_path)
    ident = _DomainFoo(name='model', domain='D')
    import asyncio
    asyncio.run(resolver.resolve(ident))
    assert len(resolver._cache) > 0
    resolver.invalidate()
    assert resolver._cache == {}


def test_simple_resolver_repr(tmp_path):
    resolver = SimplePathResolver(_DomainFoo, path=tmp_path)
    r = repr(resolver)
    assert 'SimplePathResolver' in r
    assert str(tmp_path) in r
