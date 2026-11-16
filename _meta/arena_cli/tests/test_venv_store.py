"""Tests for the venv store garbage collection against real workspace trees."""

import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from arena_cli import venv_store

OLD = time.time() - venv_store.GRACE_S - 60


@pytest.fixture(autouse=True)
def dev_container() -> Iterator[None]:
    saved = os.environ.pop("ARENA_FORK", None)
    yield
    if saved is not None:
        os.environ["ARENA_FORK"] = saved


def entry(ws: Path, name: str) -> Path:
    path = ws / ".uv" / "venvs" / name
    path.mkdir(parents=True)
    (path / ".complete").touch()
    return path


def link(build: Path, pkg: str, name: str) -> None:
    (build / pkg).mkdir(parents=True, exist_ok=True)
    (build / pkg / "venv").symlink_to(f"/opt/arena_ws/.venvs/{name}")


def age(path: Path) -> None:
    os.utime(path, (OLD, OLD))


def test_references_come_from_dev_releases_and_fork_uppers(tmp_path: Path) -> None:
    link(tmp_path / "build", "dev_pkg", "a")
    link(tmp_path / "build" / ".forks" / "releases" / "20261114-000000" / "build", "rel_pkg", "b")
    link(tmp_path / "build" / ".forks" / "casper" / "upper" / "build", "fork_pkg", "c")
    (tmp_path / "build" / "real_pkg" / "venv").mkdir(parents=True)
    assert venv_store.references(tmp_path) == {"a", "b", "c"}


def test_gc_stamps_an_unreferenced_entry_and_keeps_it(tmp_path: Path) -> None:
    kept = entry(tmp_path, "unused")
    assert venv_store.gc(tmp_path) == []
    assert kept.is_dir()
    assert (tmp_path / ".uv" / "venvs" / ".gc" / "unused").is_file()


def test_gc_deletes_an_entry_stamped_over_a_day_ago(tmp_path: Path) -> None:
    gone = entry(tmp_path, "unused")
    venv_store.gc(tmp_path)
    age(tmp_path / ".uv" / "venvs" / ".gc" / "unused")
    assert venv_store.gc(tmp_path) == ["unused"]
    assert not gone.exists()
    assert not (tmp_path / ".uv" / "venvs" / ".gc" / "unused").exists()


def test_gc_drops_the_stamp_of_an_entry_referenced_again(tmp_path: Path) -> None:
    kept = entry(tmp_path, "back")
    venv_store.gc(tmp_path)
    age(tmp_path / ".uv" / "venvs" / ".gc" / "back")
    link(tmp_path / "build" / ".forks" / "p1" / "upper" / "build", "pkg", "back")
    assert venv_store.gc(tmp_path) == []
    assert kept.is_dir()
    assert not (tmp_path / ".uv" / "venvs" / ".gc" / "back").exists()


def test_gc_never_stamps_a_referenced_entry(tmp_path: Path) -> None:
    entry(tmp_path, "used")
    link(tmp_path / "build", "pkg", "used")
    venv_store.gc(tmp_path)
    assert not (tmp_path / ".uv" / "venvs" / ".gc" / "used").exists()


def test_gc_removes_abandoned_builds_after_a_day(tmp_path: Path) -> None:
    stale = tmp_path / ".uv" / "venvs" / ".tmp-x-1"
    fresh = tmp_path / ".uv" / "venvs" / ".tmp-y-2"
    stale.mkdir(parents=True)
    fresh.mkdir()
    age(stale)
    venv_store.gc(tmp_path)
    assert not stale.exists()
    assert fresh.is_dir()


def test_gc_does_nothing_inside_a_fork(tmp_path: Path) -> None:
    entry(tmp_path, "unused")
    os.environ["ARENA_FORK"] = "p1"
    try:
        assert venv_store.gc(tmp_path) == []
    finally:
        del os.environ["ARENA_FORK"]
    assert not (tmp_path / ".uv" / "venvs" / ".gc").exists()
