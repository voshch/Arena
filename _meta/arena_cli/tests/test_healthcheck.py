"""Tests for the install checks, the simulator selection and the argument parsing of arena healthcheck."""

from pathlib import Path

import pytest

from arena_cli.common import CLIError
from arena_cli.healthcheck import _parse, boot, build, shell, sims, venv, verdict

ARENA = Path(__file__).resolve().parents[3]


def test_sims_takes_every_installed_simulator_in_launch_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DIR", str(ARENA))
    assert sims(["mujoco", "planners", "gazebo"], ["robot:=jackal"]) == (["gazebo", "mujoco"], ["robot:=jackal"])


def test_sims_launches_only_a_named_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DIR", str(ARENA))
    assert sims(["gazebo", "mujoco"], ["sim:=flatsim", "headless:=true"]) == (["flatsim"], ["headless:=true"])


def test_sims_is_empty_without_an_installed_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DIR", str(ARENA))
    assert sims(["planners", "robots"], []) == ([], [])


def test_verdict_names_the_runner_exit_codes() -> None:
    assert verdict(0) == "ok"
    assert verdict(3) == "wedged or weak"
    assert verdict(4) == "failed (exit 4)"


def test_shell_passes_inside_the_container() -> None:
    assert shell({"ARENA_CONTAINER": "1"}) == (True, "ok")


def test_shell_outside_the_container_names_source_arena() -> None:
    ok, detail = shell({})
    assert not ok
    assert "source arena" in detail


def test_boot_passes_once_the_marker_exists(tmp_path: Path) -> None:
    marker = tmp_path / ".built"
    marker.write_text("")
    assert boot(marker) == (True, "ok")


def test_boot_fails_while_first_boot_runs(tmp_path: Path) -> None:
    ok, detail = boot(tmp_path / ".built")
    assert not ok
    assert "Initial setup complete" in detail


def test_venv_passes_when_provisioned(tmp_path: Path) -> None:
    (tmp_path / ".installed").write_text("")
    assert venv(tmp_path) == (True, "ok")


def test_venv_unprovisioned_names_arena_repair(tmp_path: Path) -> None:
    ok, detail = venv(tmp_path)
    assert not ok
    assert "arena repair" in detail


def test_build_passes_with_nothing_stale() -> None:
    assert build([]) == (True, "ok, nothing stale")


def test_build_lists_stale_packages_and_names_arena_build() -> None:
    assert build(["a", "b"]) == (False, "2 stale package(s): a b, run 'arena build'")


def test_build_truncates_a_long_stale_list() -> None:
    ok, detail = build([f"p{i}" for i in range(11)])
    assert not ok
    assert detail == "11 stale package(s): p0 p1 p2 p3 p4 p5 p6 p7 and 3 more, run 'arena build'"


def test_parse_splits_the_launch_flag_from_launch_arguments() -> None:
    assert _parse([]) == (True, [])
    assert _parse(["--no-launch", "sim:=gazebo"]) == (False, ["sim:=gazebo"])


def test_parse_rejects_unknown_arguments_naming_the_accepted_ones() -> None:
    with pytest.raises(CLIError, match="--no-launch or KEY:=VALUE"):
        _parse(["--fast"])
