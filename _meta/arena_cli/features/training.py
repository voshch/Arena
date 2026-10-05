"""training feature: arena_training + rosnav_rl for DRL navigation."""

import os
import sys

from arena_cli import common
from arena_cli.common import Verb, make_verb
from arena_cli.complete import Files, LaunchArgs
from arena_cli.features import lifecycle_verbs

NAME = "training"

DESCRIPTION = "arena_training + rosnav_rl for DRL-based navigation.\n\nThis enables:\n\n\b\n- Training RL agents with Stable Baselines 3 and DreamerV3\n- Deploying trained agents as nav2 local planners\n- Action server for real-time model inference"


def _update() -> int:
    """Pull arena_training and rosnav_rl submodules, sync python deps, and rebuild."""
    import subprocess

    arena_dir = common._env("ARENA_DIR")
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
    """launch training (ros2 launch arena_training training.launch.py)"""
    common._reg_require(NAME)
    common._exec("ros2", "launch", "arena_training", "training.launch.py", *argv)


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        *lifecycle_verbs(NAME, _update, deinit="arena_training"),
        make_verb("launch", launch, passthrough=True, complete=LaunchArgs("arena_training", "training.launch.py", {"train_config": Files()})),
    ]
}
