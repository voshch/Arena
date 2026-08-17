"""robots feature: per-robot submodule management."""

import os
import sys

import common
from common import Verb, make_verb
import complete
from complete import Flags, Static, Union

import features

_NAME = "robots"

DESCRIPTION = (
    "Per-robot/component submodule management (mesh assets + upstream deps).\n\n"
    "\b\n"
    "  add <name...>        clone robot/component (<type>/<name>) submodules (alias: install)\n"
    "  rm <name...|--all>   deinit robot/component submodules (alias: uninstall <name...>)\n"
    "  ls                   list robots and components, [x] ready, [ ] pending\n"
    "  check [--all]        verify all package://arena_robots/... URIs resolve\n"
    "  update               refresh initialized robot submodules\n"
    "  uninstall            deinit all robot submodules\n"
    "  drive [opts]         run a random-goal episode per ready robot, report success/total"
)


def _payload() -> str:
    return os.path.join(features.assets_dir(_NAME), f"{_NAME}.py")


def _deps_build() -> int:
    return common._resourced("arena deps && arena build")


def _names() -> list[str]:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("_robots_payload", _payload())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    arena = Path(common._env("ARENA_DIR"))
    return sorted({*mod.all_robots(arena), *mod.component_families(arena)})


_ALL = Flags({"--all": "every robot and component"})
_SELECT = Union(Static(_names), _ALL)


def _forward(verb: str, args: list[str]) -> int:
    import subprocess

    rc = subprocess.run(["python3", _payload(), verb, *args], check=False).returncode
    if verb in ("add", "rm", "update", "uninstall"):
        complete.invalidate(common._env("ARENA_WS_DIR"))
    return rc


def add(argv: list[str]) -> None:
    rc = _forward("add", argv)
    sys.exit(rc if rc else _deps_build())


def update(argv: list[str]) -> None:
    common._reg_require(_NAME)
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
        make_verb("add", add, passthrough=True, help_text="clone robot/component submodules (alias: install)", complete=_SELECT),
        make_verb("update", update, passthrough=True, help_text="Update the feature to the latest state."),
        make_verb("rm", rm, passthrough=True, help_text="deinit robot/component submodules (alias: uninstall <name...>)", complete=Union(_SELECT, Flags({"-f": "deinit shared paths too"}))),
        make_verb("ls", ls, passthrough=True, help_text="list robots and components, [x] ready, [ ] pending"),
        make_verb("check", check, passthrough=True, help_text="verify all package://arena_robots/... URIs resolve", complete=Union(_ALL, Flags({"-q": "quiet"}))),
        make_verb("drive", drive, passthrough=True, help_text="run a random-goal episode per ready robot, report success/total"),
        make_verb("install", install, passthrough=True, help_text="clone robot/component submodules (alias for add)", complete=_SELECT),
        make_verb("uninstall", uninstall, passthrough=True, help_text="Uninstall and unregister the feature.", complete=Static(_names)),
    ]
}
