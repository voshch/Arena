"""Tests for the staleness test and the package selection of arena build and rebuild."""

import os
import shutil
from pathlib import Path

from arena_cli.build import Workspace, _changed_build_type, _selects_packages, _stale, stale_interfaces
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


def _manifest(src: Path, name: str, depends: list[str], interfaces: bool) -> None:
    group = "<member_of_group>rosidl_interface_packages</member_of_group>" if interfaces else ""
    (src / "package.xml").write_text(f"<package><name>{name}</name>{''.join(f'<exec_depend>{d}</exec_depend>' for d in depends)}{group}</package>")


def _runtime(tmp_path: Path) -> tuple[Workspace, str, Path]:
    ws, runtime = _package(tmp_path, "runtime")
    _, msgs = _package(tmp_path, "demo_msgs")
    _, other = _package(tmp_path, "other_msgs")
    _manifest(runtime, "runtime", ["demo_msgs", "rclpy"], interfaces=False)
    _manifest(msgs, "demo_msgs", [], interfaces=True)
    _manifest(other, "other_msgs", [], interfaces=True)
    (msgs / "msg").mkdir()
    (msgs / "msg" / "Lane.msg").write_text("float64 width\n")
    _touch_tree(tmp_path / "src", 1_000)
    _touch_tree(tmp_path / "install", 2_000)
    return ws, "".join(f"{d.name}\t{d}\t(ros.ament_cmake)\n" for d in (runtime, msgs, other)), msgs


def test_stale_interfaces_passes_a_built_workspace(tmp_path: Path) -> None:
    ws, listing, _ = _runtime(tmp_path)
    assert stale_interfaces(ws, listing, ["runtime"]) == []


def test_stale_interfaces_flags_an_edited_field(tmp_path: Path) -> None:
    ws, listing, msgs = _runtime(tmp_path)
    (msgs / "msg" / "Lane.msg").write_text("float64 width\nfloat64 length\n")
    assert stale_interfaces(ws, listing, ["runtime"]) == ["demo_msgs"]


def test_stale_interfaces_flags_a_removed_message(tmp_path: Path) -> None:
    ws, listing, msgs = _runtime(tmp_path)
    (msgs / "msg" / "Lane.msg").unlink()
    assert stale_interfaces(ws, listing, ["runtime"]) == ["demo_msgs"]


def test_stale_interfaces_flags_a_never_installed_package(tmp_path: Path) -> None:
    ws, listing, _ = _runtime(tmp_path)
    shutil.rmtree(tmp_path / "install" / "demo_msgs")
    assert stale_interfaces(ws, listing, ["runtime"]) == ["demo_msgs"]


def test_stale_interfaces_ignores_python_edits_and_packages_outside_the_roots(tmp_path: Path) -> None:
    ws, listing, _ = _runtime(tmp_path)
    _touch_tree(tmp_path / "src" / "runtime", 3_000)
    _touch_tree(tmp_path / "src" / "other_msgs", 3_000)
    assert stale_interfaces(ws, listing, ["runtime", "not_checked_out"]) == []


def test_stale_interfaces_counts_an_optional_root_once_it_is_installed(tmp_path: Path) -> None:
    ws, listing, _ = _runtime(tmp_path)
    _, flatsim = _package(tmp_path, "flatsim")
    _manifest(flatsim, "flatsim", ["other_msgs"], interfaces=False)
    _touch_tree(tmp_path / "src" / "other_msgs", 3_000)
    assert stale_interfaces(ws, listing + f"flatsim\t{flatsim}\t(ros.ament_python)\n", ["runtime"]) == ["other_msgs"]
