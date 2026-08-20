"""planners feature: per-planner submodule management."""

import os
import sys

import common
from common import Verb, make_verb
import complete
from complete import Flags, Static, Union

import features

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
    return os.path.join(features.assets_dir(_NAME), f"{_NAME}.py")


def _deps_build() -> int:
    return common._resourced("arena deps && arena build --executor sequential")


def _names() -> list[str]:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("_planners_payload", _payload())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return sorted(mod.planner_submodules(Path(common._env("ARENA_DIR"))))


_ALL = Flags({"--all": "every planner"})
_SELECT = Union(Static(_names), _ALL)


def _forward(verb: str, args: list[str]) -> int:
    import subprocess

    os.environ.setdefault("GIT_SSH_COMMAND", common._git_ssh_command())
    rc = subprocess.run(["python3", _payload(), verb, *args], check=False).returncode
    if verb in ("add", "rm", "update", "uninstall"):
        complete.invalidate(common._env("ARENA_WS_DIR"))
    return rc


def add(argv: list[str]) -> None:
    rc = _forward("add", argv)
    sys.exit(_deps_build() or rc)


def update(argv: list[str]) -> None:
    common._reg_require(_NAME)
    rc = _forward("update", argv)
    sys.exit(_deps_build() or rc)


def rm(argv: list[str]) -> None:
    sys.exit(_forward("rm", argv))


def ls(argv: list[str]) -> None:
    sys.exit(_forward("ls", argv))


def check(argv: list[str]) -> None:
    sys.exit(_forward("check", argv))


def install(argv: list[str]) -> None:
    rc = _forward("add", argv)
    sys.exit(_deps_build() or rc)


def uninstall(argv: list[str]) -> None:
    if argv:
        sys.exit(_forward("rm", argv))
    _forward("uninstall", [])
    common._reg_remove(_NAME)
    sys.exit(0)


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        make_verb("add", add, passthrough=True, help_text="clone planner's submodules (alias: install)", complete=_SELECT),
        make_verb("update", update, passthrough=True, help_text="Update the feature to the latest state."),
        make_verb("rm", rm, passthrough=True, help_text="deinit planner's submodules (alias: uninstall <name...>)", complete=Union(_SELECT, Flags({"-f": "deinit shared paths too"}))),
        make_verb("ls", ls, passthrough=True, help_text="list planners, [x] ready, [ ] pending"),
        make_verb("check", check, passthrough=True, help_text="verify planner submodules are initialized", complete=Union(_ALL, Flags({"-q": "quiet"}))),
        make_verb("install", install, passthrough=True, help_text="clone planner's submodules (alias for add)", complete=_SELECT),
        make_verb("uninstall", uninstall, passthrough=True, help_text="Uninstall and unregister the feature.", complete=Static(_names)),
    ]
}
