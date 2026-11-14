"""Tests for the shell init code the gazebo feature adds to `source arena`."""

import subprocess

import pytest

from arena_cli.features.gazebo import _shell_source

OGRE_NEXT = "/opt/ros/jazzy/opt/gz_ogre_next_vendor/lib/pkgconfig"


def _sourced(monkeypatch: pytest.MonkeyPatch, times: int, pkg_config_path: str) -> str:
    monkeypatch.setenv("ARENA_CONTAINER", "1")
    monkeypatch.setenv("ARENA_ROS_DISTRO", "jazzy")
    script = "\n".join([_shell_source()] * times) + '\nprintf %s "$PKG_CONFIG_PATH"'
    return subprocess.run(["bash", "-c", script], env={"PKG_CONFIG_PATH": pkg_config_path}, capture_output=True, text=True, check=True).stdout


def test_container_source_puts_ogre_next_on_the_pkg_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _sourced(monkeypatch, 1, "/usr/lib/pkgconfig") == f"{OGRE_NEXT}:/usr/lib/pkgconfig"
    assert _sourced(monkeypatch, 1, "") == OGRE_NEXT


def test_sourcing_twice_adds_ogre_next_once(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _sourced(monkeypatch, 2, "/usr/lib/pkgconfig") == f"{OGRE_NEXT}:/usr/lib/pkgconfig"
