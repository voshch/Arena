"""planners feature: per-planner submodule management."""

import os
import sys

import common
from common import CLIError, Verb, make_verb

SCRIPT_SHA256: str = "9ec049d7d266aa4c448747ccb4a141174e8b11c59b2766c27e7757e52b5dc3f8"

_NAME = "planners"

DESCRIPTION = (
    "Per-planner submodule management.\n\n"
    "\b\n"
    "  add <name...>        clone planner's submodules (alias: install)\n"
    "  rm <name...|--all>   deinit planner's submodules (alias: uninstall <name...>)\n"
    "  ls                   list planners, [x] ready, [ ] pending\n"
    "  check [--all]        verify planner submodules are initialized\n"
    "  update               refresh initialized planner submodules\n"
    "  uninstall            deinit all planner submodules"
)


def _payload() -> str:
    path = common._reg_resolve(_NAME)
    if path is None:
        raise CLIError(f"unknown feature '{_NAME}'")
    return os.path.join(os.path.dirname(path), f"{_NAME}.py")


def _deps_build() -> int:
    import shlex

    src = common._env("SOURCE_FILE")
    return common._run(os.environ.get("SHELL", "/bin/bash"), "-c", f"source {shlex.quote(src)} > /dev/null 2>&1 && arena deps && arena build --executor sequential")


def _forward(verb: str, args: list[str]) -> int:
    import subprocess

    return subprocess.run(["python3", _payload(), verb, *args], check=False).returncode


def add(argv: list[str]) -> None:
    rc = _forward("add", argv)
    sys.exit(rc if rc else _deps_build())


def update(argv: list[str]) -> None:
    if not common._reg_has(_NAME):
        raise CLIError(f"{_NAME} is not installed, run 'arena feature {_NAME} install' first")
    rc = _forward("update", argv)
    sys.exit(rc if rc else _deps_build())


def rm(argv: list[str]) -> None:
    sys.exit(_forward("rm", argv))


def ls(argv: list[str]) -> None:
    sys.exit(_forward("ls", argv))


def check(argv: list[str]) -> None:
    sys.exit(_forward("check", argv))


def install(argv: list[str]) -> None:
    rc = _forward("add", argv)
    sys.exit(rc if rc else _deps_build())


def uninstall(argv: list[str]) -> None:
    if argv:
        sys.exit(_forward("rm", argv))
    _forward("uninstall", [])
    common._reg_remove(_NAME)
    sys.exit(0)


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        make_verb("add", add, passthrough=True, help_text="clone planner's submodules (alias: install)"),
        make_verb("update", update, passthrough=True, help_text="Update the feature to the latest state."),
        make_verb("rm", rm, passthrough=True, help_text="deinit planner's submodules (alias: uninstall <name...>)"),
        make_verb("ls", ls, passthrough=True, help_text="list planners, [x] ready, [ ] pending"),
        make_verb("check", check, passthrough=True, help_text="verify planner submodules are initialized"),
        make_verb("install", install, passthrough=True, help_text="clone planner's submodules (alias for add)"),
        make_verb("uninstall", uninstall, passthrough=True, help_text="Uninstall and unregister the feature."),
    ]
}
