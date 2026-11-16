"""Tests for arena_containers in _meta/docker/lib under every shell that sources it."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[2] / "docker" / "lib"

SHELLS = [pytest.param(shell, marks=pytest.mark.skipif(shutil.which(shell) is None, reason=f"{shell} not installed")) for shell in ("bash", "zsh")]


def _docker_ps_argv(tmp_path: Path, shell: str, *args: str) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text('#!/bin/sh\n[ "$1" = ps ] && printf \'%s\\n\' "$@"\nexit 0\n')
    docker.chmod(0o755)
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "ARENA_WS_DIR": str(tmp_path / "arena_ws")}
    return subprocess.run([shell, "-c", 'source "$0" && arena_containers "$@"', str(LIB), *args], env=env, capture_output=True, text=True, check=False)


@pytest.mark.parametrize("shell", SHELLS)
def test_running_filter_by_default(tmp_path: Path, shell: str) -> None:
    result = _docker_ps_argv(tmp_path, shell, "arena")
    assert (result.returncode, result.stderr) == (0, "")
    assert result.stdout.splitlines() == [
        "ps",
        "-a",
        "-q",
        "--filter=status=running",
        "--filter",
        "label=com.docker.compose.project=arena-arena_ws",
        "--filter",
        "label=com.docker.compose.service=arena",
    ]


@pytest.mark.parametrize("shell", SHELLS)
def test_all_flag_drops_running_filter(tmp_path: Path, shell: str) -> None:
    result = _docker_ps_argv(tmp_path, shell, "-a", "arena")
    assert (result.returncode, result.stderr) == (0, "")
    assert result.stdout.splitlines() == [
        "ps",
        "-a",
        "-q",
        "--filter",
        "label=com.docker.compose.project=arena-arena_ws",
        "--filter",
        "label=com.docker.compose.service=arena",
    ]
