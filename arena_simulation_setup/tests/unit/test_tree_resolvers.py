"""Offline coverage for resolver ordering, probe verdicts, and write-path selection."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import attrs
import pytest

from arena_simulation_setup.tree import (
    FallbackResolver,
    Identifier,
    SimplePathResolver,
    Verdict,
    providers,
)


def _make_identifier_cls() -> type[Identifier]:
    """Fresh Identifier subclass per test, so ``_resolvers`` never leaks across tests."""

    @attrs.define(eq=False, hash=False)
    class _TestIdentifier(Identifier[None]):
        _resolvers = []

        def load(self, path: Path, /, **kwargs: object) -> None:
            del path, kwargs
            return None

    return _TestIdentifier


# ---------------------------------------------------------------------------
# Resolver ordering
# ---------------------------------------------------------------------------


def test_earlier_simple_path_resolver_wins(tmp_path):
    cls = _make_identifier_cls()
    root_a = tmp_path / 'a'
    root_b = tmp_path / 'b'
    (root_a / 'thing').mkdir(parents=True)
    (root_b / 'thing').mkdir(parents=True)
    cls.use(SimplePathResolver(cls, path=root_a))
    cls.use(SimplePathResolver(cls, path=root_b))

    ident = cls(name='thing')
    resolved = asyncio.run(ident.resolve_path())
    assert resolved == root_a / 'thing'


def test_first_resolver_without_asset_falls_through(tmp_path):
    cls = _make_identifier_cls()
    root_empty = tmp_path / 'empty'
    root_b = tmp_path / 'b'
    root_empty.mkdir()
    (root_b / 'thing').mkdir(parents=True)
    cls.use(SimplePathResolver(cls, path=root_empty))
    cls.use(SimplePathResolver(cls, path=root_b))

    ident = cls(name='thing')
    resolved = asyncio.run(ident.resolve_path())
    assert resolved == root_b / 'thing'


# ---------------------------------------------------------------------------
# probe() verdicts
# ---------------------------------------------------------------------------


def test_probe_reports_hit_miss_shadowed_in_order(tmp_path):
    cls = _make_identifier_cls()
    root_hit = tmp_path / 'hit'
    root_miss = tmp_path / 'miss'
    root_shadow = tmp_path / 'shadow'
    (root_hit / 'thing').mkdir(parents=True)
    root_miss.mkdir()
    (root_shadow / 'thing').mkdir(parents=True)

    r_hit = SimplePathResolver(cls, path=root_hit)
    r_miss = SimplePathResolver(cls, path=root_miss)
    r_shadow = SimplePathResolver(cls, path=root_shadow)
    cls.use(r_hit, r_miss, r_shadow)

    ident = cls(name='thing')
    verdicts = ident.probe_sync()

    assert [v.resolver for v in verdicts] == [r_hit, r_miss, r_shadow]
    assert [v.verdict for v in verdicts] == [Verdict.HIT, Verdict.MISS, Verdict.SHADOWED]
    assert verdicts[0].path == root_hit / 'thing'
    assert verdicts[1].path is None
    assert verdicts[2].path == root_shadow / 'thing'


def test_probe_all_miss_when_absent_everywhere(tmp_path):
    cls = _make_identifier_cls()
    root_a = tmp_path / 'a'
    root_b = tmp_path / 'b'
    root_a.mkdir()
    root_b.mkdir()
    cls.use(SimplePathResolver(cls, path=root_a), SimplePathResolver(cls, path=root_b))

    ident = cls(name='ghost')
    verdicts = ident.probe_sync()
    assert [v.verdict for v in verdicts] == [Verdict.MISS, Verdict.MISS]


# ---------------------------------------------------------------------------
# destination() / write-path selection
# ---------------------------------------------------------------------------


def test_write_path_uses_fallback_over_simple_resolver(tmp_path):
    cls = _make_identifier_cls()
    root_simple = tmp_path / 'simple'
    root_fallback = tmp_path / 'fallback'
    (root_simple / 'thing').mkdir(parents=True)

    simple = SimplePathResolver(cls, path=root_simple)
    fallback = FallbackResolver(cls, path=root_fallback)
    cls.use(simple, fallback)

    ident = cls(name='thing')
    write_path = asyncio.run(ident.resolve_write_path())
    assert write_path == root_fallback / 'thing'


def test_fallback_resolver_never_a_probe_hit(tmp_path):
    cls = _make_identifier_cls()
    root_simple = tmp_path / 'simple'
    root_fallback = tmp_path / 'fallback'
    (root_simple / 'thing').mkdir(parents=True)

    simple = SimplePathResolver(cls, path=root_simple)
    fallback = FallbackResolver(cls, path=root_fallback)
    cls.use(simple, fallback)

    ident = cls(name='thing')
    verdicts = ident.probe_sync()
    assert [v.verdict for v in verdicts] == [Verdict.HIT, Verdict.MISS]
    assert verdicts[1].resolver is fallback


def test_fallback_resolver_never_a_probe_hit_even_when_only_resolver(tmp_path):
    cls = _make_identifier_cls()
    fallback = FallbackResolver(cls, path=tmp_path)
    cls.use(fallback)

    ident = cls(name='thing')
    verdicts = ident.probe_sync()
    assert [v.verdict for v in verdicts] == [Verdict.MISS]


def test_write_path_raises_without_fallback_resolver(tmp_path):
    cls = _make_identifier_cls()
    cls.use(SimplePathResolver(cls, path=tmp_path))

    ident = cls(name='thing')
    with pytest.raises(FileNotFoundError, match='no write destination registered'):
        asyncio.run(ident.resolve_write_path())


def test_write_path_raises_with_no_resolvers_at_all():
    cls = _make_identifier_cls()

    ident = cls(name='thing')
    with pytest.raises(FileNotFoundError, match='no write destination registered'):
        asyncio.run(ident.resolve_write_path())


# ---------------------------------------------------------------------------
# providers()
# ---------------------------------------------------------------------------


def test_providers_uses_default_when_env_unset():
    var = '_ARENA_TEST_PROVIDERS_UNSET'
    os.environ.pop(var, None)
    assert providers(var, 'default-bucket') == ['default-bucket']


def test_providers_splits_comma_separated_env():
    var = '_ARENA_TEST_PROVIDERS_SET'
    os.environ[var] = 'one,two,three'
    try:
        assert providers(var, 'default') == ['one', 'two', 'three']
    finally:
        del os.environ[var]


def test_providers_drops_empty_segments():
    var = '_ARENA_TEST_PROVIDERS_EMPTY_SEGMENTS'
    os.environ[var] = 'one,,two,'
    try:
        assert providers(var, 'default') == ['one', 'two']
    finally:
        del os.environ[var]


def test_providers_blank_env_value_falls_back_to_default():
    """The shell exports the override slot blank, which must not read as 'no buckets'."""
    var = '_ARENA_TEST_PROVIDERS_BLANK'
    os.environ[var] = ''
    try:
        assert providers(var, 'default') == ['default']
    finally:
        del os.environ[var]


def _registered_identifier_types() -> list[type[Identifier]]:
    """Every shipped Identifier subclass that registers its own resolvers."""
    import arena_simulation_setup.tree.assets.Human  # noqa: F401
    import arena_simulation_setup.tree.assets.Material  # noqa: F401
    import arena_simulation_setup.tree.assets.Object  # noqa: F401
    import arena_simulation_setup.tree.Gesture  # noqa: F401
    import arena_simulation_setup.tree.Wall  # noqa: F401
    import arena_simulation_setup.tree.World  # noqa: F401

    found: list[type[Identifier]] = []
    stack = list(Identifier.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if not cls.__module__.startswith('arena_simulation_setup.tree'):
            continue
        if '_resolvers' in cls.__dict__ and cls._resolvers:
            found.append(cls)
    return found


def test_every_identifier_has_a_read_resolver():
    """FallbackResolver answers no reads, so a type registering only one resolves nothing."""
    types = _registered_identifier_types()
    assert types, 'no identifier types discovered'
    write_only = [cls.__name__ for cls in types if all(isinstance(r, FallbackResolver) for r in cls._resolvers)]
    assert not write_only, f'unresolvable, only a write target registered: {write_only}'


def test_net_listing_args_scans_sentinels_for_annotated_payloads():
    from arena_simulation_setup.tree import NetResolver
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier

    argv = NetResolver(ObjectIdentifier, provider='b')._listing_args()
    assert argv[-3:] == ['net', 'b', 'list']
    assert '--children' not in argv


def test_net_listing_args_uses_children_and_prefix_for_unannotated_payloads():
    from arena_simulation_setup.tree import NetResolver
    from arena_simulation_setup.tree.World import WorldIdentifier

    worlds = NetResolver(WorldIdentifier, provider='b', annotated=False)._listing_args()
    assert worlds[-2:] == ['list', '--children']

    prefixed = NetResolver(WorldIdentifier, provider='b', annotated=False, list_prefix='suites')._listing_args()
    assert prefixed[-3:] == ['list', '--children', 'suites']
