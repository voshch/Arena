"""Tests for the tasks feature against real git repos: a task layer repo checked out as a submodule under task_layers/."""

import os
import subprocess
from pathlib import Path

import pytest

from arena_cli.features import tasks

GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], env={**os.environ, **GIT_ENV}, capture_output=True, text=True, check=True)


@pytest.fixture
def arena(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A superproject with task_layers/casper as an uninitialized submodule and a non-layer submodule beside it."""
    for key, value in GIT_ENV.items():
        monkeypatch.setenv(key, value)
    layer = tmp_path / "arena-casper"
    layer.mkdir()
    _git(layer, "init", "-q", "-b", "jazzy")
    (layer / "README.md").write_text("casper\n")
    _git(layer, "add", "README.md")
    _git(layer, "commit", "-q", "-m", "init")
    other = tmp_path / "flatsim"
    other.mkdir()
    _git(other, "init", "-q", "-b", "jazzy")
    _git(other, "commit", "-q", "--allow-empty", "-m", "init")
    root = tmp_path / "Arena"
    root.mkdir()
    _git(root, "init", "-q", "-b", "jazzy")
    _git(root, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(layer), "task_layers/casper")
    _git(root, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(other), "flatsim")
    _git(root, "commit", "-q", "-m", "submodules")
    _git(root, "submodule", "deinit", "-q", "-f", "task_layers/casper")
    return root


def test_layers_are_the_submodules_under_task_layers(arena: Path) -> None:
    assert tasks.layers(arena) == {"casper": "task_layers/casper"}
    assert not tasks.installed(arena, "task_layers/casper")


def test_ls_shows_the_built_in_layer_and_the_available_ones(arena: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert tasks.cmd_ls(arena) == 0
    assert capsys.readouterr().out.splitlines() == ["[x] task_generator (built in)", "[ ] casper"]


def test_add_checks_the_layer_out_and_rm_deinits_it(arena: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert tasks.cmd_add(arena, ["casper"], every=False) == 0
    assert (arena / "task_layers" / "casper" / "README.md").read_text() == "casper\n"
    assert "select it with task_layer:=casper" in capsys.readouterr().out
    assert tasks.cmd_check(arena, []) == 0
    assert tasks.cmd_rm(arena, ["casper"], every=False) == 0
    assert not tasks.installed(arena, "task_layers/casper")
    assert tasks.cmd_check(arena, ["casper"]) == 1
    assert "run 'arena feature tasks add casper'" in capsys.readouterr().out


def test_add_all_installs_every_layer(arena: Path) -> None:
    assert tasks.cmd_add(arena, [], every=True) == 0
    assert tasks.installed(arena, "task_layers/casper")


def test_unknown_layer_names_the_available_ones(arena: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert tasks.cmd_add(arena, ["flatsim"], every=False) == 2
    assert "no task layer flatsim, available: casper" in capsys.readouterr().err


def test_bare_add_says_what_to_name(arena: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert tasks.cmd_add(arena, [], every=False) == 2
    assert "name the task layer(s), available: casper" in capsys.readouterr().err


def test_the_built_in_layer_needs_no_install(arena: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert tasks.cmd_add(arena, ["task_generator"], every=False) == 2
    assert "task_generator is built in" in capsys.readouterr().err


def test_update_without_installed_layers_is_a_noop(arena: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert tasks.cmd_update(arena) == 0
    assert "no task layer installed" in capsys.readouterr().out


def test_tasks_is_a_host_feature() -> None:
    from arena_cli.features import HOST_FEATURES

    assert "tasks" in HOST_FEATURES
