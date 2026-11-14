"""Tests for the package selection of arena build and rebuild."""

from pathlib import Path

from arena_cli.build import _changed_build_type
from arena_cli.cli import _select_args


def test_select_args_cleans_named_packages_and_builds_dependents() -> None:
    assert _select_args(["foo", "bar"]) == ["--packages-select", "foo", "bar"]
    assert _select_args(["foo", "bar"], above=True) == ["--packages-above", "foo", "bar"]


def test_select_args_keeps_explicit_flags() -> None:
    assert _select_args(["--packages-select-regex", "arena_.*"], above=True) == ["--packages-select-regex", "arena_.*"]


def test_changed_build_type_flags_cmake_leftovers_of_a_python_package(tmp_path: Path) -> None:
    (tmp_path / "CMakeCache.txt").write_text("")
    assert _changed_build_type(str(tmp_path), "ros.ament_python")
    assert not _changed_build_type(str(tmp_path), "ros.ament_cmake")


def test_changed_build_type_flags_python_leftovers_of_a_cmake_package(tmp_path: Path) -> None:
    (tmp_path / "colcon_command_prefix_setup_py.sh").write_text("")
    assert _changed_build_type(str(tmp_path), "ros.ament_cmake")
    assert not _changed_build_type(str(tmp_path), "ros.ament_python")
