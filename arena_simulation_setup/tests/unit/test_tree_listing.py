"""The child-process listing must agree with the in-process walk, pruning included."""

from __future__ import annotations

import asyncio
from pathlib import Path

import attrs
import pytest
from arena_simulation_setup.tree import SimplePathResolver


@attrs.define(frozen=True)
class _Ident:
    parts: tuple[str, ...]

    @classmethod
    def from_relpath(cls, relpath: Path) -> _Ident:
        if len(relpath.parts) != 2:
            raise ValueError(relpath)
        return cls(relpath.parts)

    @classmethod
    def label(cls) -> str:
        return "test"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "pkg" / "asset" / "deeper").mkdir(parents=True)
    (tmp_path / "pkg" / "asset" / "deeper" / "ignored.yaml").touch()
    (tmp_path / "loose").mkdir()
    (tmp_path / "loose" / "thing.yaml").touch()
    (tmp_path / "loose" / "README.md").touch()
    (tmp_path / "toplevel.yaml").touch()
    return tmp_path


def test_async_listing_matches_the_walk(tree: Path) -> None:
    resolver = SimplePathResolver(_Ident, path=tree)
    assert set(resolver.listall()) == set(asyncio.run(resolver.listall_async()))


def test_prunes_below_an_asset_and_skips_readme(tree: Path) -> None:
    resolver = SimplePathResolver(_Ident, path=tree)
    found = {ident.parts for ident in asyncio.run(resolver.listall_async())}
    assert ("pkg", "asset") in found
    assert ("loose", "thing.yaml") in found
    assert not any("deeper" in parts for parts in found)
    assert not any("README.md" in parts for parts in found)


def test_listing_is_cached_until_invalidated(tree: Path) -> None:
    resolver = SimplePathResolver(_Ident, path=tree)
    asyncio.run(resolver.listall_async())
    (tree / "fresh").mkdir()
    (tree / "fresh" / "thing.yaml").touch()
    assert not any("fresh" in ident.parts for ident in asyncio.run(resolver.listall_async()))
    resolver.invalidate()
    assert any("fresh" in ident.parts for ident in asyncio.run(resolver.listall_async()))


def test_missing_directory_lists_empty(tmp_path: Path) -> None:
    resolver = SimplePathResolver(_Ident, path=tmp_path / "nope")
    assert asyncio.run(resolver.listall_async()) == []
