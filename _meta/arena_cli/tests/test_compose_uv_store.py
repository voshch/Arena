"""Tests that every Arena container mounts the uv store read-write and its venvs read-only."""

from pathlib import Path

import pytest

DOCKER = Path(__file__).resolve().parents[2] / "docker"
STORE_MOUNT = '"$HOST_ARENA_WS_DIR/.uv:/opt/arena_ws/.uv"'
VIEW_MOUNT = '"$HOST_ARENA_WS_DIR/.uv/venvs:/opt/arena_ws/.venvs:ro"'
STORE_ENV = (
    "UV_CACHE_DIR=/opt/arena_ws/.uv/cache",
    "AMENT_CMAKE_VENV_UV_STORE=/opt/arena_ws/.uv/venvs",
    "AMENT_CMAKE_VENV_UV_STORE_VIEW=/opt/arena_ws/.venvs",
)


@pytest.mark.parametrize("compose", ["docker-compose.yaml", "docker-compose.fork.yaml", "features/isaac/docker-compose.yaml", "features/isaac/docker-compose.fork.yaml"])
def test_container_mounts_the_store_and_a_read_only_view(compose: str) -> None:
    text = (DOCKER / compose).read_text()
    assert STORE_MOUNT in text
    assert VIEW_MOUNT in text
    assert ".uv-cache" not in text


@pytest.mark.parametrize("compose", ["docker-compose.yaml", "features/isaac/docker-compose.yaml"])
def test_container_env_points_uv_and_the_store_at_the_mounts(compose: str) -> None:
    lines = {line.strip().removeprefix("- ") for line in (DOCKER / compose).read_text().splitlines()}
    assert set(STORE_ENV) <= lines
