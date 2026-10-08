"""hearing feature: robot hearing front-ends, belief grid and speed-filter policy."""

import os
import sys

from arena_cli import common
from arena_cli.common import Verb
from arena_cli.features import lifecycle_verbs

NAME = "hearing"

DESCRIPTION = "arena_hearing for robot hearing.\n\nThis enables:\n\n\b\n- robot.hearing:=bus|srp|seld (srp and SELDnet front-ends, belief grid, Nav2 speed-filter mask)\n- the audio comes from acoustics:=arena, installed by the auditory feature"


def _update() -> int:
    """Pull the arena_hearing submodule, install its rosdep and python deps, and rebuild."""
    import subprocess

    arena_dir = common._env("ARENA_DIR")

    rc = subprocess.run(
        ["git", "submodule", "update", "--init", "--rebase", "--depth", "1", "arena_hearing"],
        cwd=arena_dir,
        check=False,
    ).returncode
    if rc:
        return rc

    rc = subprocess.run(["sudo", "apt-get", "update"], check=False).returncode
    if rc:
        return rc
    rc = subprocess.run(
        ["rosdep", "install", "--ignore-src", "-r", "-y", "--rosdistro", common._env("ARENA_ROS_DISTRO"), "--from-paths", os.path.join(arena_dir, "arena_hearing")],
        check=False,
    ).returncode
    if rc:
        return rc

    rc = subprocess.run([sys.executable, os.path.join(arena_dir, "_meta", "tools", "uv_workspace.py"), "compose"], check=False).returncode
    if rc:
        return rc
    rc = subprocess.run(
        ["uv", "sync", "--inexact", "--all-packages", "--project", os.path.join(arena_dir, ".uv-workspace")],
        env={**os.environ, "UV_PROJECT_ENVIRONMENT": common._env("ARENA_VENV_DIR")},
        check=False,
    ).returncode
    if rc:
        return rc

    return common._resourced("arena build")


COMMANDS: dict[str, Verb] = {v.name: v for v in lifecycle_verbs(NAME, _update, deinit="arena_hearing")}
