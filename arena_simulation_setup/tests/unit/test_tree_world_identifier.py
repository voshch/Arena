"""WorldIdentifier value equality, offline: no touching the class's real (network) resolvers."""

from __future__ import annotations

import asyncio

from arena_simulation_setup.tree import SimplePathResolver
from arena_simulation_setup.tree.World.World import WorldIdentifier


def test_world_identifier_eq_by_name():
    a = WorldIdentifier('x')
    b = WorldIdentifier('x')
    assert a == b


def test_world_identifier_hash_by_name():
    a = WorldIdentifier('x')
    b = WorldIdentifier('x')
    assert hash(a) == hash(b)


def test_world_identifier_neq_different_name():
    assert WorldIdentifier('x') != WorldIdentifier('y')


def test_world_identifier_usable_in_set():
    a = WorldIdentifier('x')
    b = WorldIdentifier('x')
    c = WorldIdentifier('y')
    assert {a, b, c} == {a, c}
    assert len({a, b, c}) == 2


def test_world_identifier_value_equality_drives_resolver_cache(tmp_path):
    """Separate instances sharing a name hit the same resolver._cache entry."""
    (tmp_path / 'x').mkdir()
    resolver = SimplePathResolver(WorldIdentifier, path=tmp_path)

    a = WorldIdentifier('x')
    b = WorldIdentifier('x')
    assert a is not b

    resolved = asyncio.run(resolver.resolve(a))
    assert resolved == tmp_path / 'x'
    assert b in resolver._cache
    assert resolver._cache[b] == tmp_path / 'x'
