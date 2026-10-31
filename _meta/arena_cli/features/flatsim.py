"""flatsim feature: the flatsim 2D simulator behind sim:=flatsim."""

import os
import sys

from arena_cli import common
from arena_cli.common import Verb, make_verb
from arena_cli.complete import LaunchArgs
from arena_cli.features import lifecycle_verbs

NAME = "flatsim"

DESCRIPTION = "flatsim, a 2D simulator on Box2D or Warp with no renderer.\n\nThis enables:\n\n\b\n- sim:=flatsim (one server for every env: odom, tf, laserscans, point clouds, IMU and contacts from geometry, no camera images)\n- sim.flatsim.dynamics:=rigid|kinematic\n- sim.flatsim.physics:=auto|box2d|warp"


def _update() -> int:
    """Pull the flatsim submodule, install its rosdep and python deps, and rebuild."""
    import subprocess

    arena_dir = common._env("ARENA_DIR")

    rc = subprocess.run(
        ["git", "submodule", "update", "--init", "--rebase", "--depth", "1", "flatsim"],
        cwd=arena_dir,
        check=False,
    ).returncode
    if rc:
        return rc

    rc = subprocess.run(["sudo", "apt-get", "update"], check=False).returncode
    if rc:
        return rc
    rc = subprocess.run(
        ["rosdep", "install", "--ignore-src", "-r", "-y", "--rosdistro", common._env("ARENA_ROS_DISTRO"), "--from-paths", os.path.join(arena_dir, "flatsim")],
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


def launch(argv: list[str]) -> None:
    """Launch the flatsim server."""
    common._reg_require(NAME)
    common._exec("ros2", "launch", "flatsim", "flatsim.launch.py", *argv)


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        *lifecycle_verbs(NAME, _update, deinit="flatsim"),
        make_verb("launch", launch, passthrough=True, complete=LaunchArgs("flatsim", "flatsim.launch.py")),
    ]
}
