"""mujoco feature: MuJoCo simulator backend."""

import os
import sys

from arena_cli import common
from arena_cli.common import Verb, make_verb
from arena_cli.complete import LaunchArgs
from arena_cli.features import lifecycle_verbs, source_verb

NAME = "mujoco"

DESCRIPTION = "MuJoCo simulator backend (arena_mujoco).\n\nThis enables:\n\n\b\n- sim:=mujoco (CPU physics, ray-cast lidar, offscreen cameras, kinematic pedestrians)"

_VIRTUALGL_VERSION = "3.1.5"

_FORMATS_SOURCE = 'for f in sdf obj; do case "$ARENA_MODELS_FORMATS" in *$f*) ;; *) export ARENA_MODELS_FORMATS="${ARENA_MODELS_FORMATS},$f" ;; esac; done'


def _install_virtualgl() -> int:
    """Install VirtualGL from its release .deb unless present."""
    import subprocess
    import tempfile
    import urllib.request

    if os.path.exists("/opt/VirtualGL/bin/vglrun"):
        return 0
    arch = subprocess.run(["dpkg", "--print-architecture"], capture_output=True, text=True, check=True).stdout.strip()
    deb = f"virtualgl_{_VIRTUALGL_VERSION}_{arch}.deb"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, deb)
        urllib.request.urlretrieve(f"https://github.com/VirtualGL/virtualgl/releases/download/{_VIRTUALGL_VERSION}/{deb}", path)
        os.chmod(tmp, 0o755)
        return subprocess.run(["sudo", "apt-get", "install", "-y", path], check=False).returncode


def _update() -> int:
    """Install VirtualGL and the arena_mujoco python deps into the venv, then rebuild."""
    import subprocess

    arena_dir = common._env("ARENA_DIR")

    rc = _install_virtualgl()
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
    """Launch the MuJoCo server (run_mujoco.launch.py args: headless:=, log_level:=)."""
    common._reg_require(NAME)
    common._exec("ros2", "launch", "arena_mujoco", "run_mujoco.launch.py", *argv)


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        *lifecycle_verbs(NAME, _update),
        make_verb("launch", launch, passthrough=True, complete=LaunchArgs("arena_mujoco", "run_mujoco.launch.py")),
        source_verb(lambda: _FORMATS_SOURCE),
    ]
}
