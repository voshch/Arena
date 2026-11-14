"""Tests for the checks behind the installed features' health rows."""

from pathlib import Path

import pytest

from arena_cli.features import fetched
from arena_cli.features.docker import _reserves_gpu


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    arena = tmp_path / "src" / "Arena"
    (arena / "_meta" / "repos").mkdir(parents=True)
    monkeypatch.setenv("ARENA_DIR", str(arena))
    monkeypatch.setenv("ARENA_WS_DIR", str(tmp_path))
    return arena, tmp_path / "src"


def test_fetched_passes_with_checked_out_submodules(workspace: tuple[Path, Path]) -> None:
    arena, _ = workspace
    (arena / "arena_hearing").mkdir()
    (arena / "arena_hearing" / ".git").write_text("gitdir: ../.git/modules/arena_hearing\n")
    assert fetched("hearing", "arena_hearing") == (True, "ok")


def test_fetched_names_an_uninitialized_submodule_and_the_update(workspace: tuple[Path, Path]) -> None:
    arena, _ = workspace
    (arena / "arena_training").mkdir()
    (arena / "arena_training" / ".git").write_text("gitdir: ../.git/modules/arena_training\n")
    assert fetched("training", "arena_training", "arena_training/deps/rosnav_rl") == (False, "missing arena_training/deps/rosnav_rl, run 'arena feature training update'")


def test_fetched_names_repos_of_the_repos_file_that_were_not_pulled(workspace: tuple[Path, Path]) -> None:
    arena, src = workspace
    (arena / "_meta" / "repos" / "meshnav.repos").write_text("repositories:\n  meshnav/lvr2:\n    type: git\n  meshnav/mesh_tools:\n    type: git\n")
    (src / "meshnav" / "lvr2").mkdir(parents=True)
    assert fetched("meshnav") == (False, "missing meshnav/mesh_tools, run 'arena feature meshnav update'")
    (src / "meshnav" / "mesh_tools").mkdir()
    assert fetched("meshnav") == (True, "ok")


DOCKER_GPU = {"DeviceRequests": [{"Driver": "nvidia", "Count": -1}], "Devices": []}
PODMAN_GPU = {"DeviceRequests": None, "Devices": [{"PathOnHost": "/dev/nvidia0", "PathInContainer": "/dev/nvidia0", "CgroupPermissions": ""}]}
PODMAN_PLAIN = {"DeviceRequests": None, "Devices": [{"PathOnHost": "/dev/dri/card0", "PathInContainer": "/dev/dri/card0", "CgroupPermissions": ""}]}


@pytest.mark.parametrize(("host_config", "expected"), [(DOCKER_GPU, True), (PODMAN_GPU, True), (PODMAN_PLAIN, False), ({}, False)])
def test_reserves_gpu_reads_docker_device_requests_and_podman_devices(host_config: dict, expected: bool) -> None:
    assert _reserves_gpu(host_config) is expected
