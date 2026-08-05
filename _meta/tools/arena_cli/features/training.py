"""training feature: arena_training + rosnav_rl for DRL navigation."""

import os
import sys

import common
from common import CLIError, Verb, make_verb

SCRIPT_SHA256 = "f4c6f25f6956145715fa3f2a5914fa41eb12b445d83c6e9416767720616d9581"

NAME = "training"

DESCRIPTION = (
    "arena_training + rosnav_rl for DRL-based navigation.\n\n"
    "This enables:\n\n"
    "\b\n"
    "- Training RL agents with Stable Baselines 3 and DreamerV3\n"
    "- Deploying trained agents as nav2 local planners\n"
    "- Action server for real-time model inference"
)


def _require() -> None:
    if not common._reg_has(NAME):
        raise CLIError(f"{NAME} is not installed; run 'arena feature {NAME} install' first.")


def _update() -> int:
    """Pull arena_training and rosnav_rl submodules, install, and rebuild."""
    import shlex
    import subprocess

    arena_dir = common._env("ARENA_DIR")
    arena_ws_dir = common._env("ARENA_WS_DIR")
    arena_training_dir = os.path.join(arena_dir, "arena_training")

    rc = subprocess.run(
        ["git", "submodule", "update", "--init", "--rebase", "--depth", "1", "arena_training"],
        cwd=arena_dir,
        check=False,
    ).returncode
    if rc:
        return rc

    rc = subprocess.run(
        ["git", "submodule", "update", "--init", "--rebase", "--depth", "1", "deps/rosnav_rl"],
        cwd=arena_training_dir,
        check=False,
    ).returncode
    if rc:
        return rc

    python_bin = os.path.join(arena_dir, ".venv", "bin", "python")
    rc = subprocess.run(
        ["uv", "pip", "install", "--python", python_bin, "-e", ".", "-e", "./deps/rosnav_rl/rosnav_rl"],
        cwd=arena_training_dir,
        check=False,
    ).returncode
    if rc:
        return rc

    src = common._env("SOURCE_FILE")
    cmd = f"source {shlex.quote(src)} > /dev/null 2>&1 && arena build"
    return subprocess.run(
        [os.environ.get("SHELL", "/bin/bash"), "-c", cmd],
        cwd=arena_ws_dir,
        check=False,
    ).returncode


def install(argv: list[str]) -> None:
    """register feature, pull arena_training + rosnav_rl submodules and rebuild"""
    if argv:
        raise CLIError("unexpected arguments")
    from features import default_install

    sys.exit(default_install(NAME, _update))


def update(argv: list[str]) -> None:
    """pull arena_training + rosnav_rl submodules and rebuild"""
    if argv:
        raise CLIError("unexpected arguments")
    if not common._reg_has(NAME):
        raise CLIError(f"{NAME} is not installed, run 'arena feature {NAME} install' first")
    sys.exit(_update())


def uninstall(argv: list[str]) -> None:
    """deinit arena_training and remove from registry"""
    if argv:
        raise CLIError("unexpected arguments")
    import subprocess

    arena_dir = common._env("ARENA_DIR")
    subprocess.run(["git", "submodule", "deinit", "-f", "arena_training"], cwd=arena_dir, check=False)
    common._reg_remove(NAME)


def launch(argv: list[str]) -> None:
    """launch training (ros2 launch arena_training training.launch.py)"""
    _require()
    common._exec("ros2", "launch", "arena_training", "training.launch.py", *argv)


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        make_verb("install", install),
        make_verb("update", update),
        make_verb("uninstall", uninstall),
        make_verb("launch", launch, passthrough=True),
    ]
}
