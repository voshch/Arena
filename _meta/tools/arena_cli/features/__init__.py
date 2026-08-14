"""Python feature groups, one module per feature."""

import importlib
import os
import sys
from collections.abc import Callable
from types import ModuleType

from common import CLIError, Verb, _env, _reg_add, _reg_has, _reg_pull, _reg_remove, _reg_require, make_verb

HOST_FEATURES = ("evaluation", "gazebo", "isaac", "planners", "robots", "training")
CONTAINER_FEATURES = (*HOST_FEATURES, "docker", "vllm")


def in_container() -> bool:
    """True when ARENA_FEATURES_DIR points at the in-container feature tree."""
    return os.path.realpath(_env("ARENA_FEATURES_DIR")) == os.path.realpath(os.path.join(_env("ARENA_DIR"), "_meta", "docker", "features"))


def available() -> tuple[str, ...]:
    """Feature names that exist in the current context."""
    return CONTAINER_FEATURES if in_container() else HOST_FEATURES


def load(name: str) -> ModuleType | None:
    """Return the feature's python module (COMMANDS + DESCRIPTION), or None if no such feature."""
    if name not in available():
        return None
    try:
        return importlib.import_module(f"features.{name}")
    except ImportError:
        return None


def assets_dir(name: str) -> str:
    """The feature's asset directory under ARENA_FEATURES_DIR."""
    return os.path.join(_env("ARENA_FEATURES_DIR"), name)


def compose(args: list[str], env: dict[str, str] | None = None) -> int:
    """Run the arena_docker_compose bash function exported by _meta/docker/lib."""
    import subprocess

    return subprocess.run(["bash", "-c", 'arena_docker_compose "$@"', "arena_docker_compose", *args], env=env, check=False).returncode


def compose_output(args: list[str]) -> str:
    """Like compose, but capture and return stdout."""
    import subprocess

    return subprocess.run(["bash", "-c", 'arena_docker_compose "$@"', "arena_docker_compose", *args], capture_output=True, text=True, check=False).stdout


def source_verb(emit: Callable[[], str]) -> Verb:
    """Hidden verb printing shell init code, eval'd at `source arena` time."""

    def run(argv: list[str]) -> None:
        """Print shell init code for `source arena`."""
        if argv:
            raise CLIError("unexpected arguments")
        text = emit()
        if text:
            print(text)

    return make_verb("source", run, hidden=True)


def default_install(name: str, update: Callable[[], int]) -> int:
    """Pull repos, register, run update, rolling back the registration on failure."""
    if _reg_has(name):
        print(f"{name} is already installed.")
        return 0
    _reg_pull(name)
    _reg_add(name)
    if update() == 0:
        print(f"Installed {name} successfully.")
        return 0
    _reg_remove(name)
    print(f"Install of {name} failed during update.", file=sys.stderr)
    return 1


def lifecycle_verbs(name: str, update_fn: Callable[[], int], deinit: str | None = None) -> list[Verb]:
    """Shared install/update/uninstall verbs for simple features."""

    def install(argv: list[str]) -> None:
        """Install the feature (pull repos, register, run its update)."""
        if argv:
            raise CLIError("unexpected arguments")
        sys.exit(default_install(name, update_fn))

    def update(argv: list[str]) -> None:
        """Update the feature to the latest state."""
        if argv:
            raise CLIError("unexpected arguments")
        _reg_require(name)
        sys.exit(update_fn())

    def uninstall(argv: list[str]) -> None:
        """Uninstall and unregister the feature."""
        if argv:
            raise CLIError("unexpected arguments")
        if deinit is not None:
            import subprocess

            subprocess.run(["git", "submodule", "deinit", "-f", deinit], cwd=_env("ARENA_DIR"), check=False)
        _reg_remove(name)

    return [make_verb("install", install), make_verb("update", update), make_verb("uninstall", uninstall)]
