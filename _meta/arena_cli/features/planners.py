"""planners feature: per-planner submodule management."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from arena_cli import common, complete
from arena_cli.common import Verb, make_verb
from arena_cli.complete import Flags, Static, Union

_NAME = "planners"
_SDK_SUBDIR = "arena_planners"

DESCRIPTION = (
    "Per-planner submodule management.\n\n"
    "\b\n"
    "  add <name...>        clone planner's submodules (alias: install)\n"
    "  rm <name...|--all>   deinit planner's submodules (alias: uninstall <name...>)\n"
    "  ls                   list planners, [x] ready, [ ] pending\n"
    "  check [--all]        verify planner submodules are initialized\n"
    "  update               refresh initialized planner submodules\n"
    "  uninstall            deinit all planner submodules\n"
    "  test [--preflight] <name...|--all>  lockstep soak, or a short start-and-drive check, via the benchmark runner (needs the evaluation feature)"
)


# one short crowded stage per contestant with the scheduler stepping the sim, so a
# planner that cannot keep its beat shows up as a stall in the run report
_SOAK_SUITE = {
    "launch": {"lockstep": True, "lockstep.paused": False, "headless": True},
    "references": False,
    "stages": [
        {
            "name": "soak",
            "map": "map_empty",
            "robot": "auto",
            "episodes": 2,
            "tm_robots": "random",
            "tm_obstacles": "random",
            "config": {
                "random": {
                    "dynamic": {"min": 4, "max": 6, "models": ["arenian"]},
                    "static": {"min": 3, "max": 6, "models": ["shelf"]},
                    "interactive": {"min": 0, "max": 0},
                }
            },
            "timeout": "60s",
        }
    ],
}


_PREFLIGHT_STAGE = {
    "name": "preflight",
    "map": "map_empty",
    "robot": "auto",
    "episodes": 2,
    "tm_robots": "scenario",
    "tm_obstacles": "scenario",
    "config": {"scenario": {"file": "preflight"}},
    "timeout": "20s",
}


def _registry() -> ModuleType:
    try:
        from arena_planners import registry
    except ImportError as e:
        raise common.CLIError("arena_planners is not importable, run 'arena pull' then 'arena repair'") from e
    return registry


def _arena() -> Path:
    return Path(common._env("ARENA_DIR"))


def _deps_build() -> int:
    return common._resourced("arena deps && arena build --executor sequential")


def _set_git_ssh() -> None:
    os.environ.setdefault("GIT_SSH_COMMAND", common._git_ssh_command())


def _git(args: list[str], arena: Path, *, check: bool = True) -> int:
    return subprocess.run(["git", *args], cwd=arena, check=check).returncode


def _fetch_weights(arena: Path, planner: str) -> None:
    subprocess.run([sys.executable, "-m", "arena_planners", "fetch", planner], cwd=arena, check=False)


def _path_planners(subs: dict[str, list[str]]) -> dict[str, set[str]]:
    """{path: {planners tagging it}} (reverse of submodule_paths for sharing checks)."""
    out: dict[str, set[str]] = {}
    for planner, paths in subs.items():
        for p in paths:
            out.setdefault(p, set()).add(planner)
    return out


def _is_local_only(planner: str, subs: dict[str, list[str]]) -> bool:
    return planner not in subs


def cmd_ls(arena: Path, _args: argparse.Namespace) -> int:
    return subprocess.run([sys.executable, "-m", "arena_planners", "ls"], cwd=arena, check=False).returncode


def cmd_add(arena: Path, args: argparse.Namespace) -> int:
    reg = _registry()
    subs = reg.submodule_paths(arena)
    kinds = reg.kinds(arena)
    if args.all:
        if args.names:
            print("planners: --all is mutually exclusive with planner names", file=sys.stderr)
            return 2
        names = sorted(subs) + reg.local_planners(arena)
    else:
        if not args.names:
            print("planners: specify planner name(s) or --all", file=sys.stderr)
            return 2
        names = args.names
    rc = 0
    for planner in names:
        if _is_local_only(planner, subs):
            local = arena / reg.PLANNERS_SUBDIR / planner
            if (local / "planner.py").is_file():
                _fetch_weights(arena, planner)
            else:
                available = sorted(subs)
                avail_str = ", ".join(available) if available else "(none)"
                print(
                    f"planners: planner '{planner}' not found. Available: [{avail_str}].",
                    file=sys.stderr,
                )
                rc = 1
            continue
        paths = subs.get(planner)
        if paths is None:
            available = sorted(subs)
            avail_str = ", ".join(available) if available else "(none)"
            msg = f"planner '{planner}' not found. Available: [{avail_str}]."
            if available:
                msg += f" To install: arena feature planners add {available[0]}"
            print(f"planners: {msg}", file=sys.stderr)
            rc = 1
            continue
        _git(["-c", "protocol.file.allow=always", "submodule", "update", "--init", "--checkout", _SDK_SUBDIR], arena)
        sdk = arena / _SDK_SUBDIR
        for p in paths:
            sub_path = Path(p).relative_to(_SDK_SUBDIR).as_posix()
            _git(["-c", "protocol.file.allow=always", "submodule", "update", "--init", "--checkout", sub_path], sdk)
        if kinds.get(planner) == "nav2":
            print(f"planners: '{planner}' is a native Nav2 controller")
        else:
            _fetch_weights(arena, planner)
    return rc


def cmd_rm(arena: Path, args: argparse.Namespace) -> int:
    reg = _registry()
    subs = reg.submodule_paths(arena)
    if args.all:
        if args.names:
            print("planners: --all is mutually exclusive with planner names", file=sys.stderr)
            return 2
        names = sorted(subs)
    else:
        if not args.names:
            print("planners: specify planner name(s) or --all", file=sys.stderr)
            return 2
        names = args.names
    remove_set = set(names)
    shared = _path_planners(subs)
    sdk = arena / _SDK_SUBDIR
    done: set[str] = set()
    rc = 0
    for planner in names:
        if _is_local_only(planner, subs):
            local = arena / reg.PLANNERS_SUBDIR / planner
            if (local / "planner.py").is_file():
                print(
                    f"planners: '{planner}' is a local directory; remove it manually",
                    file=sys.stderr,
                )
            else:
                available = sorted(subs)
                avail_str = ", ".join(available) if available else "(none)"
                print(
                    f"planners: planner '{planner}' not found. Available: [{avail_str}].",
                    file=sys.stderr,
                )
            rc = 1
            continue
        paths = subs.get(planner)
        if paths is None:
            available = sorted(subs)
            avail_str = ", ".join(available) if available else "(none)"
            print(
                f"planners: planner '{planner}' not found. Available: [{avail_str}].",
                file=sys.stderr,
            )
            rc = 1
            continue
        for p in paths:
            if p in done:
                continue
            # keep a shared path only if some planner outside this removal batch still tags it
            others = shared.get(p, set()) - remove_set
            if others and not args.force:
                print(f"planners: keeping '{p}' (still tagged by: {', '.join(sorted(others))})")
                continue
            if others:
                print(f"planners: force-removing '{p}' (also tagged by: {', '.join(sorted(others))})")
            sub_path = Path(p).relative_to(_SDK_SUBDIR).as_posix()
            _git(["submodule", "deinit", "-f", sub_path], sdk, check=False)
            done.add(p)
    return rc


def cmd_update(arena: Path, _args: argparse.Namespace) -> int:
    sdk = arena / _SDK_SUBDIR
    if not (sdk / ".gitmodules").is_file():
        return 0
    return _git(["submodule", "update", "--recursive"], sdk, check=False)


def cmd_uninstall(arena: Path, _args: argparse.Namespace) -> int:
    reg = _registry()
    subs = reg.submodule_paths(arena)
    status = reg.submodule_status(arena)
    sdk = arena / _SDK_SUBDIR
    for p in (p for ps in subs.values() for p in ps):
        if status.get(p) == "init":
            sub_path = Path(p).relative_to(_SDK_SUBDIR).as_posix()
            _git(["submodule", "deinit", "-f", sub_path], sdk, check=False)
    return 0


def cmd_check(arena: Path, args: argparse.Namespace) -> int:
    reg = _registry()
    subs = reg.submodule_paths(arena)
    status = reg.submodule_status(arena)
    if not subs:
        if not args.quiet:
            print("planners: no planners registered")
        return 0
    rc = 0
    for planner in sorted(subs):
        paths = subs[planner]
        pending = [p for p in paths if status.get(p) != "init"]
        if pending:
            rc = 1
            if not args.quiet:
                for p in pending:
                    print(f"[ ] {planner}: {p} not initialized")
        elif not args.quiet:
            for p in paths:
                print(f"[x] {planner}: {p}")
    return rc


def _names() -> list[str]:
    return sorted(_registry().submodule_paths(_arena()))


def _ready_names() -> list[str]:
    """Registered planners whose submodules are all initialized."""
    reg = _registry()
    arena = _arena()
    status = reg.submodule_status(arena)
    return sorted(name for name, paths in reg.submodule_paths(arena).items() if all(status.get(p) == "init" for p in paths))


def _contestants(names: list[str]) -> list[dict]:
    """Bridge planners run under the drl adapter, anything else is a nav2 local planner."""
    kinds = _registry().kinds(_arena())
    out = []
    for name in names:
        if kinds.get(name, "nav2") == "bridge":
            out.append({"name": name, "mobile": {"driver": "drl", "planner": name}})
        else:
            out.append({"name": name, "mobile": {"driver": "nav2", "local_planner": name}})
    return out


_ALL = Flags({"--all": "every planner"})
_PREFLIGHT = Flags({"--preflight": "short start-and-drive check instead of the soak"})
_SELECT = Union(Static(_names), _ALL)


def _run_add(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="arena planners add")
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true", help="fetch every planner")
    args = ap.parse_args(argv)
    _set_git_ssh()
    rc = cmd_add(_arena(), args)
    complete.invalidate(common._env("ARENA_WS_DIR"))
    return rc


def _run_rm(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="arena planners rm")
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true", help="remove every planner")
    ap.add_argument("-f", "--force", action="store_true", help="deinit shared paths too, co-tagged planners become pending")
    args = ap.parse_args(argv)
    _set_git_ssh()
    rc = cmd_rm(_arena(), args)
    complete.invalidate(common._env("ARENA_WS_DIR"))
    return rc


def _run_ls(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="arena planners ls")
    args = ap.parse_args(argv)
    _set_git_ssh()
    return cmd_ls(_arena(), args)


def _run_update(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="arena planners update")
    args = ap.parse_args(argv)
    _set_git_ssh()
    rc = cmd_update(_arena(), args)
    complete.invalidate(common._env("ARENA_WS_DIR"))
    return rc


def _run_check(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="arena planners check")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)
    _set_git_ssh()
    return cmd_check(_arena(), args)


def _run_uninstall_all(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="arena planners uninstall")
    args = ap.parse_args(argv)
    _set_git_ssh()
    rc = cmd_uninstall(_arena(), args)
    complete.invalidate(common._env("ARENA_WS_DIR"))
    return rc


def add(argv: list[str]) -> None:
    rc = _run_add(argv)
    sys.exit(_deps_build() or rc)


def update(argv: list[str]) -> None:
    rc = _run_update(argv)
    sys.exit(_deps_build() or rc)


def rm(argv: list[str]) -> None:
    sys.exit(_run_rm(argv))


def ls(argv: list[str]) -> None:
    sys.exit(_run_ls(argv))


def check(argv: list[str]) -> None:
    sys.exit(_run_check(argv))


def install(argv: list[str]) -> None:
    rc = _run_add(argv)
    sys.exit(_deps_build() or rc)


def test(argv: list[str]) -> None:
    """Lockstep soak of planners via the benchmark runner.

    `arena planners test [--preflight] <name...|--all> [sim:=gazebo] [KEY:=VALUE ...]`
    Soak: one crowded lockstep stage per planner, stall/rtf/beat table, exit 3 on a stall, 4 when the runner hung.
    `--preflight`: two episodes of map_empty's `preflight` scenario, verdict wedged/weak/ok per planner, exit 3 on either.
    `--all`: every initialized planner. Other tokens forward to the runner. Needs the evaluation feature.
    """
    import json

    common._reg_require("evaluation")
    preflight = "--preflight" in argv
    names = [a for a in argv if ":=" not in a and not a.startswith("-")]
    rest = [a for a in argv if (":=" in a or a.startswith("-")) and a not in ("--all", "--preflight")]
    if "--all" in argv:
        if names:
            raise common.CLIError("planners test: --all is mutually exclusive with planner names")
        names = _ready_names()
        if not names:
            raise common.CLIError("planners test: no initialized planners, see 'arena planners check'")
    elif not names:
        raise common.CLIError("planners test: specify planner name(s) or --all, see 'arena planners ls'")
    contest = json.dumps(_contestants(names))
    suite = _SOAK_SUITE
    verdict = ["--lockstep-verdict"]
    if preflight:
        suite = {**_SOAK_SUITE, "launch": {**_SOAK_SUITE["launch"], "env.bootstrap_timeout": 90, "robot.mobile.deadline": 30}, "stages": [_PREFLIGHT_STAGE]}
        verdict = ["--retries", "0", "--strict", "--efficacy", "0.5", "--spawn-budget", "120"]
    common._exec("ros2", "run", "arena_evaluation", "benchmark", "--suite", json.dumps(suite), "--contest", contest, *verdict, *rest)


def uninstall(argv: list[str]) -> None:
    if argv:
        sys.exit(_run_rm(argv))
    _run_uninstall_all([])
    sys.exit(0)


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        make_verb("add", add, passthrough=True, help_text="clone planner's submodules (alias: install)", complete=_SELECT),
        make_verb("update", update, passthrough=True, help_text="refresh initialized submodules"),
        make_verb("rm", rm, passthrough=True, help_text="deinit planner's submodules (alias: uninstall <name...>)", complete=Union(_SELECT, Flags({"-f": "deinit shared paths too"}))),
        make_verb("ls", ls, passthrough=True, help_text="list planners, [x] ready, [ ] pending"),
        make_verb("check", check, passthrough=True, help_text="verify planner submodules are initialized", complete=Union(_ALL, Flags({"-q": "quiet"}))),
        make_verb("install", install, passthrough=True, help_text="clone planner's submodules (alias for add)", complete=_SELECT),
        make_verb("uninstall", uninstall, passthrough=True, help_text="deinit all submodules (alias: rm --all)", complete=Static(_names)),
        make_verb("test", test, passthrough=True, complete=Union(_SELECT, _PREFLIGHT)),
    ]
}
