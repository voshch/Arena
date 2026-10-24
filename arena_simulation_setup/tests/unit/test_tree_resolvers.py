"""Offline coverage for resolver ordering, probe verdicts, and write-path selection."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import attrs
import pytest

from arena_simulation_setup.tree import (
    DynamicPaths,
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


def test_expired_net_entry_is_refetched_but_still_served_stale(tmp_path):
    entry = tmp_path / 'b' / 'w'
    entry.mkdir(parents=True)
    (entry / '.ttl').write_text(str(int(time.time())))
    script = textwrap.dedent(
        """
        from arena_simulation_setup.tree import NetResolver
        from arena_simulation_setup.tree.World import WorldIdentifier
        resolver = NetResolver(WorldIdentifier, provider='b', annotated=False)
        candidate = resolver.path / 'w'
        print(resolver._fresh(candidate), resolver.expire(WorldIdentifier('w')), resolver._fresh(candidate))
        print(resolver._serve_stale(WorldIdentifier('w'), candidate) == candidate, resolver.is_stale(WorldIdentifier('w')))
        """
    )
    env = {**os.environ, 'ARENA_ASSETS_DIR': str(tmp_path)}
    done = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, check=True)
    assert done.stdout.splitlines()[-2:] == ['True 1 False', 'True True']
    assert (entry / '.ttl').read_text() == '1'


def test_arena_dynamic_path_reads_the_local_root_at_every_use_until_assigned(tmp_path: Path) -> None:
    previous_env = os.environ.get("ARENA_ASSETS_DIR_LOCAL")
    previous = DynamicPaths.ARENA._assigned
    DynamicPaths.ARENA._assigned = None
    try:
        os.environ["ARENA_ASSETS_DIR_LOCAL"] = str(tmp_path / "first")
        assert DynamicPaths.ARENA.path == tmp_path / "first"
        os.environ["ARENA_ASSETS_DIR_LOCAL"] = str(tmp_path / "second")
        assert DynamicPaths.ARENA.path == tmp_path / "second"
        DynamicPaths.ARENA.path = tmp_path / "pinned"
        assert DynamicPaths.ARENA.path == tmp_path / "pinned"
    finally:
        DynamicPaths.ARENA._assigned = previous
        if previous_env is None:
            del os.environ["ARENA_ASSETS_DIR_LOCAL"]
        else:
            os.environ["ARENA_ASSETS_DIR_LOCAL"] = previous_env


@pytest.fixture
def asset_env(tmp_path):
    saved = {key: os.environ.get(key) for key in ("ARENA_ASSETS_DIR", "ARENA_ASSETS_DIR_IMPORTED")}
    os.environ["ARENA_ASSETS_DIR"] = str(tmp_path / "assets")
    try:
        yield tmp_path
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_imported_root_is_read_at_every_lookup_with_an_empty_value_as_unset(asset_env) -> None:
    from arena_simulation_setup.tree import RootPath, imported_root

    os.environ["ARENA_ASSETS_DIR_IMPORTED"] = str(asset_env / "imported")
    resolvers = list(DynamicPaths.as_resolvers(_make_identifier_cls()))
    assert len(resolvers) == 4
    assert imported_root() == DynamicPaths.IMPORTED.path == resolvers[-1].path == asset_env / "imported"
    os.environ["ARENA_ASSETS_DIR_IMPORTED"] = ""
    assert imported_root() == DynamicPaths.IMPORTED.path == resolvers[-1].path == asset_env / "assets" / "_imported"
    del os.environ["ARENA_ASSETS_DIR_IMPORTED"]
    assert DynamicPaths.IMPORTED.path == asset_env / "assets" / "_imported"
    assigned = RootPath(imported_root)
    assigned.path = asset_env / "assigned"
    os.environ["ARENA_ASSETS_DIR_IMPORTED"] = str(asset_env / "imported")
    assert assigned.path == asset_env / "assigned"


def test_imported_root_is_where_the_importers_write(asset_env) -> None:
    importers = pytest.importorskip("arena_assets.importers")
    from arena_simulation_setup.tree import imported_root

    for value in (str(asset_env / "imported"), ""):
        os.environ["ARENA_ASSETS_DIR_IMPORTED"] = value
        assert imported_root() == importers.import_root()
