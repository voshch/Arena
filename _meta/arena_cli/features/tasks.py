"""tasks feature: optional task layers, one submodule each under task_layers/<name>."""

import argparse
import configparser
import os
import subprocess
import sys
from pathlib import Path

from arena_cli import common, complete
from arena_cli.common import Verb, make_verb
from arena_cli.complete import Flags, Static, Union

LAYERS_DIR = "task_layers"
BUILTIN = "task_generator"

DESCRIPTION = (
    "Optional task layers, selected per env with task_layer:=<name>. task_generator is built in.\n\n"
    "\b\n"
    "  add <name...|--all>  clone the task layer's submodule, install its deps and build (alias: install)\n"
    "  rm <name...|--all>   deinit the task layer's submodule (alias: uninstall)\n"
    "  ls                   list task layers, [x] installed, [ ] available\n"
    "  check [name...]      exit 1 unless the named (default: all) task layers are installed\n"
    "  update               refresh the installed task layers and rebuild"
)


def layers(arena: Path) -> dict[str, str]:
    """{name: path} of the .gitmodules entries under task_layers/, the name being the directory and the task_layer:= value."""
    cfg = configparser.ConfigParser()
    cfg.read(arena / ".gitmodules")
    out: dict[str, str] = {}
    for section in cfg.sections():
        path = cfg[section].get("path", "").strip()
        if section.startswith("submodule ") and Path(path).parent.as_posix() == LAYERS_DIR:
            out[Path(path).name] = path
    return dict(sorted(out.items()))


def installed(arena: Path, path: str) -> bool:
    return (arena / path / ".git").exists()


def _git(args: list[str], arena: Path) -> int:
    return subprocess.run(["git", *args], cwd=arena, check=False).returncode


def _available(subs: dict[str, str]) -> str:
    return ", ".join(subs) or "none"


def _select(arena: Path, names: list[str], every: bool, verb: str) -> list[str] | None:
    """The names a verb acts on, None after printing why the selection is invalid."""
    subs = layers(arena)
    if every and names:
        print(f"tasks {verb}: --all takes no names", file=sys.stderr)
        return None
    if every:
        return list(subs)
    if not names:
        print(f"tasks {verb}: name the task layer(s), available: {_available(subs)}, see 'arena feature tasks ls'", file=sys.stderr)
        return None
    if BUILTIN in names:
        print(f"tasks {verb}: {BUILTIN} is built in, nothing to {verb}", file=sys.stderr)
        return None
    if unknown := [name for name in names if name not in subs]:
        print(f"tasks {verb}: no task layer {', '.join(unknown)}, available: {_available(subs)}", file=sys.stderr)
        return None
    return names


def cmd_ls(arena: Path) -> int:
    print(f"[x] {BUILTIN} (built in)")
    for name, path in layers(arena).items():
        print(f"{'[x]' if installed(arena, path) else '[ ]'} {name}")
    return 0


def cmd_add(arena: Path, names: list[str], every: bool) -> int:
    selected = _select(arena, names, every, "add")
    if selected is None:
        return 2
    subs = layers(arena)
    rc = 0
    for name in selected:
        if _git(["-c", "protocol.file.allow=always", "submodule", "update", "--init", "--recursive", "--checkout", subs[name]], arena):
            print(f"tasks add: cloning {name} failed, check the url of {subs[name]} in .gitmodules", file=sys.stderr)
            rc = 1
            continue
        print(f"tasks add: {name} installed, select it with task_layer:={name}")
    return rc


def cmd_rm(arena: Path, names: list[str], every: bool) -> int:
    selected = _select(arena, names, every, "rm")
    if selected is None:
        return 2
    subs = layers(arena)
    rc = 0
    for name in selected:
        if not installed(arena, subs[name]):
            print(f"tasks rm: {name} is not installed")
            continue
        if _git(["submodule", "deinit", "-f", subs[name]], arena):
            rc = 1
            continue
        print(f"tasks rm: {name} removed, rebuild with 'arena build' to drop it from the install space")
    return rc


def cmd_update(arena: Path) -> int:
    paths = [path for path in layers(arena).values() if installed(arena, path)]
    if not paths:
        print("tasks update: no task layer installed, see 'arena feature tasks ls'")
        return 0
    return _git(["submodule", "update", "--recursive", "--checkout", *paths], arena)


def cmd_check(arena: Path, names: list[str]) -> int:
    subs = layers(arena)
    if unknown := [name for name in names if name not in subs]:
        print(f"tasks check: no task layer {', '.join(unknown)}, available: {_available(subs)}", file=sys.stderr)
        return 2
    missing = [name for name in names or subs if not installed(arena, subs[name])]
    for name in missing:
        print(f"[ ] {name}: not installed, run 'arena feature tasks add {name}'")
    return 1 if missing else 0


def _deps_build() -> int:
    """Install the layers' system and python deps, then rebuild."""
    arena = common._env("ARENA_DIR")
    rc = common._resourced("arena deps")
    if rc:
        return rc
    rc = subprocess.run([sys.executable, os.path.join(arena, "_meta", "tools", "uv_workspace.py"), "compose"], check=False).returncode
    if rc:
        return rc
    rc = subprocess.run(
        ["uv", "sync", "--inexact", "--all-packages", "--project", os.path.join(arena, ".uv-workspace")],
        env={**os.environ, "UV_PROJECT_ENVIRONMENT": common._env("ARENA_VENV_DIR")},
        check=False,
    ).returncode
    if rc:
        return rc
    return common._resourced("arena build")


def _arena() -> Path:
    return Path(common._env("ARENA_DIR"))


def _names() -> list[str]:
    return list(layers(_arena()))


def _parse(verb: str, argv: list[str], *, every: bool = True) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog=f"arena feature tasks {verb}")
    ap.add_argument("names", nargs="*", help=f"task layers, see 'arena feature tasks ls' ({BUILTIN} is built in)")
    if every:
        ap.add_argument("--all", action="store_true", help="every task layer")
    return ap.parse_args(argv)


def add(argv: list[str]) -> None:
    args = _parse("add", argv)
    os.environ.setdefault("GIT_SSH_COMMAND", common._git_ssh_command())
    rc = cmd_add(_arena(), args.names, args.all)
    complete.invalidate(common._env("ARENA_WS_DIR"))
    sys.exit(rc or _deps_build())


def rm(argv: list[str]) -> None:
    args = _parse("rm", argv)
    rc = cmd_rm(_arena(), args.names, args.all)
    complete.invalidate(common._env("ARENA_WS_DIR"))
    sys.exit(rc)


def uninstall(argv: list[str]) -> None:
    args = _parse("uninstall", argv)
    rc = cmd_rm(_arena(), args.names, args.all or not args.names)
    complete.invalidate(common._env("ARENA_WS_DIR"))
    sys.exit(rc)


def ls(argv: list[str]) -> None:
    if argv:
        raise common.CLIError("tasks ls: takes no arguments, it lists every task layer")
    sys.exit(cmd_ls(_arena()))


def check(argv: list[str]) -> None:
    args = _parse("check", argv, every=False)
    sys.exit(cmd_check(_arena(), args.names))


def update(argv: list[str]) -> None:
    if argv:
        raise common.CLIError("tasks update: takes no arguments, it refreshes every installed task layer")
    os.environ.setdefault("GIT_SSH_COMMAND", common._git_ssh_command())
    sys.exit(cmd_update(_arena()) or _deps_build())


_SELECT = Union(Static(_names), Flags({"--all": "every task layer"}))

COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        make_verb("add", add, passthrough=True, help_text="clone a task layer, install its deps and build (alias: install)", complete=_SELECT),
        make_verb("install", add, passthrough=True, help_text="alias for add", complete=_SELECT),
        make_verb("rm", rm, passthrough=True, help_text="deinit a task layer's submodule (alias: uninstall)", complete=_SELECT),
        make_verb("uninstall", uninstall, passthrough=True, help_text="deinit the named task layers, all without names", complete=_SELECT),
        make_verb("ls", ls, passthrough=True, help_text="list task layers, [x] installed, [ ] available"),
        make_verb("check", check, passthrough=True, help_text="exit 1 unless the task layers are installed", complete=Static(_names)),
        make_verb("update", update, passthrough=True, help_text="refresh the installed task layers and rebuild"),
    ]
}
