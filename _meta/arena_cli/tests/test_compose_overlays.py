"""Tests for the compose overlays arena_docker_compose in _meta/docker/lib picks from the host."""

import os
import subprocess
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[3]
LIB = ARENA_DIR / "_meta" / "docker" / "lib"
GCLOUD_OVERLAY = str(ARENA_DIR / "_meta" / "docker" / "docker-compose.gcloud.yaml")


def _compose_files(tmp_path: Path, **host: str) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text('#!/bin/sh\n[ "$1" = compose ] && printf \'%s\\n\' "$COMPOSE_FILE"\nexit 0\n')
    docker.chmod(0o755)
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "ARENA_WS_DIR": str(tmp_path), "ARENA_DIR": str(ARENA_DIR), **host}
    result = subprocess.run(["bash", "-c", 'source "$0" && arena_docker_compose config', str(LIB)], env=env, capture_output=True, text=True, check=True)
    return result.stdout.strip().split(":")


def test_gcloud_config_on_the_host_mounts_it(tmp_path: Path) -> None:
    (tmp_path / "gcloud").mkdir()
    assert GCLOUD_OVERLAY in _compose_files(tmp_path, HOST_GCLOUD_CONFIG=str(tmp_path / "gcloud"))


def test_no_gcloud_config_leaves_the_overlay_out(tmp_path: Path) -> None:
    assert GCLOUD_OVERLAY not in _compose_files(tmp_path)
