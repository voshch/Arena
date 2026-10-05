"""Tests for the mount check that picks the uv link mode in arena build."""

from pathlib import Path

from arena_cli.build import _mount_point


def test_mount_point_matches_within_one_mount(tmp_path: Path) -> None:
    cache = tmp_path / "build" / ".uv-cache"
    cache.mkdir(parents=True)
    assert _mount_point(str(cache)) == _mount_point(str(tmp_path / "build"))


def test_mount_point_resolves_missing_build_base(tmp_path: Path) -> None:
    assert _mount_point(str(tmp_path / "build")) == _mount_point(str(tmp_path))


def test_mount_point_differs_across_mounts() -> None:
    assert _mount_point("/proc/self") != _mount_point("/")
