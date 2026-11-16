"""Tests for the staleness test and the package selection of arena build and rebuild."""

import os
from pathlib import Path

from arena_cli.build import Workspace, _changed_build_type, _selects_packages, _stale
from arena_cli.cli import _select_args


def _touch_tree(root: Path, mtime: float) -> None:
    for path in [root, *root.rglob("*")]:
        os.utime(path, (mtime, mtime))


def _package(tmp_path: Path, name: str, rc: str = "0") -> tuple[Workspace, Path]:
    src = tmp_path / "src" / name
    src.mkdir(parents=True)
    (src / "package.xml").write_text("")
    build = tmp_path / "build" / name
    build.mkdir(parents=True)
    (build / "colcon_build.rc").write_text(rc)
    share = tmp_path / "install" / name / "share" / name
    share.mkdir(parents=True)
    (share / "package.xml").write_text("")
    return Workspace([], [], str(tmp_path / "build"), str(tmp_path / "install")), src


def test_stale_skips_a_package_installed_after_its_source(tmp_path: Path) -> None:
    ws, src = _package(tmp_path, "fresh")
    _touch_tree(src, 1_000)
    _touch_tree(tmp_path / "install" / "fresh", 2_000)
    assert _stale(f"fresh\t{src}\t(ros.ament_python)\n", ws) == []


def test_stale_flags_a_source_edited_after_install(tmp_path: Path) -> None:
    ws, src = _package(tmp_path, "edited")
    _touch_tree(tmp_path / "install" / "edited", 1_000)
    _touch_tree(src, 2_000)
    assert _stale(f"edited\t{src}\t(ros.ament_cmake)\n", ws) == [("edited", "ros.ament_cmake")]


def test_stale_flags_a_failed_last_build(tmp_path: Path) -> None:
    ws, src = _package(tmp_path, "failed", rc="1")
    _touch_tree(src, 1_000)
    _touch_tree(tmp_path / "install" / "failed", 2_000)
    assert _stale(f"failed\t{src}\t(ros.ament_python)\n", ws) == [("failed", "ros.ament_python")]


def test_stale_ignores_a_pycache_written_into_the_source_after_install(tmp_path: Path) -> None:
    ws, src = _package(tmp_path, "imported")
    _touch_tree(src, 1_000)
    _touch_tree(tmp_path / "install" / "imported", 2_000)
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "mod.cpython-312.pyc").write_text("")
    assert _stale(f"imported\t{src}\t(ros.ament_python)\n", ws) == []


def test_stale_flags_a_source_file_deleted_after_its_pycache(tmp_path: Path) -> None:
    ws, src = _package(tmp_path, "pruned")
    (src / "gone.py").write_text("")
    (src / "__pycache__").mkdir()
    _touch_tree(src, 1_000)
    _touch_tree(tmp_path / "install" / "pruned", 2_000)
    (src / "gone.py").unlink()
    assert _stale(f"pruned\t{src}\t(ros.ament_python)\n", ws) == [("pruned", "ros.ament_python")]


def test_stale_flags_a_never_built_package(tmp_path: Path) -> None:
    ws = Workspace([], [], str(tmp_path / "build"), str(tmp_path / "install"))
    assert _stale(f"new\t{tmp_path}\t(ros.ament_python)\n\n", ws) == [("new", "ros.ament_python")]


def test_select_args_cleans_named_packages_and_builds_dependents() -> None:
    assert _select_args(["foo", "bar"]) == ["--packages-select", "foo", "bar"]
    assert _select_args(["foo", "bar"], above=True) == ["--packages-above", "foo", "bar"]


def test_select_args_keeps_explicit_flags() -> None:
    assert _select_args(["--packages-select-regex", "arena_.*"], above=True) == ["--packages-select-regex", "arena_.*"]


def test_skip_and_ignore_filters_keep_the_stale_package_selection() -> None:
    assert not _selects_packages(["--packages-skip-regex", "arena_planners_.*", "--packages-ignore", "foo", "--cmake-force-configure"])
    assert _selects_packages(["--packages-skip", "foo", "--packages-select", "bar"])
    assert _selects_packages(["--packages-up-to", "bar"])


def test_changed_build_type_flags_cmake_leftovers_of_a_python_package(tmp_path: Path) -> None:
    (tmp_path / "CMakeCache.txt").write_text("")
    assert _changed_build_type(str(tmp_path), "ros.ament_python")
    assert not _changed_build_type(str(tmp_path), "ros.ament_cmake")


def test_changed_build_type_flags_python_leftovers_of_a_cmake_package(tmp_path: Path) -> None:
    (tmp_path / "colcon_command_prefix_setup_py.sh").write_text("")
    assert _changed_build_type(str(tmp_path), "ros.ament_cmake")
    assert not _changed_build_type(str(tmp_path), "ros.ament_python")
