"""robots feature: per-robot submodule management."""

import os
import sys

import common
from common import CLIError, Verb, make_verb

SCRIPT_SHA256: str = "2e04ebf682574ff4d851917fb2fdacff3bc5ca270174c55714309e0b4191e4d7"

_NAME = "robots"

DESCRIPTION = (
    "Per-robot submodule management (mesh assets + upstream deps).\n\n"
    "\b\n"
    "  add <name...>        clone robot's submodules (alias: install)\n"
    "  rm <name...|--all>   deinit robot's submodules (alias: uninstall <name...>)\n"
    "  ls                   list robots, [x] ready, [ ] pending\n"
    "  check [--all]        verify all package://arena_robots/... URIs resolve\n"
    "  update               refresh initialized robot submodules\n"
    "  uninstall            deinit all robot submodules\n"
    "  drive [opts]         run a random-goal episode per ready robot, report success/total"
)


def _payload() -> str:
    path = common._reg_resolve(_NAME)
    if path is None:
        raise CLIError(f"unknown feature '{_NAME}'")
    return os.path.join(os.path.dirname(path), f"{_NAME}.py")


def _deps_build() -> int:
    import shlex

    src = common._env("SOURCE_FILE")
    return common._run(os.environ.get("SHELL", "/bin/bash"), "-c", f"source {shlex.quote(src)} > /dev/null 2>&1 && arena deps && arena build")


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


def drive(argv: list[str]) -> None:
    sys.exit(_forward("drive", argv))


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
        make_verb("add", add, passthrough=True, help_text="clone robot's submodules (alias: install)"),
        make_verb("update", update, passthrough=True, help_text="Update the feature to the latest state."),
        make_verb("rm", rm, passthrough=True, help_text="deinit robot's submodules (alias: uninstall <name...>)"),
        make_verb("ls", ls, passthrough=True, help_text="list robots, [x] ready, [ ] pending"),
        make_verb("check", check, passthrough=True, help_text="verify all package://arena_robots/... URIs resolve"),
        make_verb("drive", drive, passthrough=True, help_text="run a random-goal episode per ready robot, report success/total"),
        make_verb("install", install, passthrough=True, help_text="clone robot's submodules (alias for add)"),
        make_verb("uninstall", uninstall, passthrough=True, help_text="Uninstall and unregister the feature."),
    ]
}
