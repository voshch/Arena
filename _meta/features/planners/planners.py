#!/usr/bin/env python3
"""arena_planners feature backend: per-planner submodule ops."""

from __future__ import annotations

import argparse
import configparser
import os
import subprocess
import sys
from pathlib import Path

_SDK_SUBDIR = "arena_planners"
_PLANNERS_SUBDIR = "arena_planners/planners"


def arena_dir() -> Path:
    if "ARENA_DIR" in os.environ:
        return Path(os.environ["ARENA_DIR"])
    for p in Path(__file__).resolve().parents:
        if (p / ".gitmodules").is_file() and (p / "arena_planners").is_dir():
            return p
    sys.exit("error: set ARENA_DIR or run inside an Arena checkout")


def submodule_status(arena: Path) -> dict[str, str]:
    """{path: 'init'|'uninit'} for each submodule (recursive)."""
    out = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=arena,
        text=True,
        capture_output=True,
        check=False,
    ).stdout
    status: dict[str, str] = {}
    for line in out.splitlines():
        if not line:
            continue
        parts = line[1:].split()
        if len(parts) < 2:
            continue
        status[parts[1]] = "uninit" if line[0] == "-" else "init"
    return status


def planner_submodules(arena: Path) -> dict[str, list[str]]:
    """{planner: [paths-relative-to-arena]} from the SDK submodule's .gitmodules `planner = <name>` tags."""
    sdk_gitmodules = arena / _SDK_SUBDIR / ".gitmodules"
    if not sdk_gitmodules.is_file():
        return {}
    cfg = configparser.ConfigParser()
    cfg.read(sdk_gitmodules)
    out: dict[str, list[str]] = {}
    for section in cfg.sections():
        if not section.startswith("submodule "):
            continue
        sub_path = cfg[section].get("path", "").strip()
        if not sub_path:
            continue
        path = f"{_SDK_SUBDIR}/{sub_path}"
        planners = cfg[section].get("planner", "").split()
        for planner in planners:
            out.setdefault(planner, []).append(path)
    return out


def planner_kinds(arena: Path) -> dict[str, str]:
    """{planner: kind} from the SDK .gitmodules `kind` tag, defaulting to 'bridge'."""
    sdk_gitmodules = arena / _SDK_SUBDIR / ".gitmodules"
    if not sdk_gitmodules.is_file():
        return {}
    cfg = configparser.ConfigParser()
    cfg.read(sdk_gitmodules)
    out: dict[str, str] = {}
    for section in cfg.sections():
        if not section.startswith("submodule "):
            continue
        kind = cfg[section].get("kind", "bridge").strip()
        for planner in cfg[section].get("planner", "").split():
            out[planner] = kind
    return out


def _path_planners(arena: Path) -> dict[str, set[str]]:
    """{path: {planners tagging it}} (reverse of planner_submodules for sharing checks)."""
    out: dict[str, set[str]] = {}
    for planner, paths in planner_submodules(arena).items():
        for p in paths:
            out.setdefault(p, set()).add(planner)
    return out


def _is_local_only(planner: str, subs: dict[str, list[str]]) -> bool:
    return planner not in subs


def _local_planners(arena: Path, subs: dict[str, list[str]]) -> list[str]:
    """Local planner dirs (a planner.py) under the planners subdir, excluding registered submodules."""
    root = arena / _PLANNERS_SUBDIR
    if not root.is_dir():
        return []
    names: list[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith(".") or entry.name.endswith("_wrap"):
            continue
        if entry.name not in subs and (entry / "planner.py").is_file():
            names.append(entry.name)
    return names


def _git(args: list[str], arena: Path, *, check: bool = True) -> int:
    return subprocess.run(["git", *args], cwd=arena, check=check).returncode


def _cli(arena: Path, *args: str) -> subprocess.CompletedProcess:
    """Run the arena_planners CLI (the registry + weights SSOT) from source."""
    pkg_parent = arena / _SDK_SUBDIR / "arena_planners"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(pkg_parent), env.get("PYTHONPATH", "")) if p)
    return subprocess.run([sys.executable, "-m", "arena_planners", *args], cwd=arena, env=env, check=False)


def cmd_ls(arena: Path, _args: argparse.Namespace) -> int:
    return _cli(arena, "ls").returncode


def cmd_add(arena: Path, args: argparse.Namespace) -> int:
    subs = planner_submodules(arena)
    kinds = planner_kinds(arena)
    if args.all:
        if args.names:
            print("planners: --all is mutually exclusive with planner names", file=sys.stderr)
            return 2
        names = sorted(subs) + _local_planners(arena, subs)
    else:
        if not args.names:
            print("planners: specify planner name(s) or --all", file=sys.stderr)
            return 2
        names = args.names
    rc = 0
    for planner in names:
        if _is_local_only(planner, subs):
            local = arena / _PLANNERS_SUBDIR / planner
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


def _fetch_weights(arena: Path, planner: str) -> None:
    _cli(arena, "fetch", planner)


def cmd_rm(arena: Path, args: argparse.Namespace) -> int:
    subs = planner_submodules(arena)
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
    shared = _path_planners(arena)
    sdk = arena / _SDK_SUBDIR
    done: set[str] = set()
    rc = 0
    for planner in names:
        if _is_local_only(planner, subs):
            local = arena / _PLANNERS_SUBDIR / planner
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
    subs = planner_submodules(arena)
    status = submodule_status(arena)
    sdk = arena / _SDK_SUBDIR
    for p in (p for ps in subs.values() for p in ps):
        if status.get(p) == "init":
            sub_path = Path(p).relative_to(_SDK_SUBDIR).as_posix()
            _git(["submodule", "deinit", "-f", sub_path], sdk, check=False)
    return 0


def cmd_check(arena: Path, args: argparse.Namespace) -> int:
    subs = planner_submodules(arena)
    status = submodule_status(arena)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ls")
    sub.add_parser("update")
    sub.add_parser("uninstall")
    p_add = sub.add_parser("add")
    p_add.add_argument("names", nargs="*")
    p_add.add_argument("--all", action="store_true", help="fetch every planner")
    p_rm = sub.add_parser("rm")
    p_rm.add_argument("names", nargs="*")
    p_rm.add_argument("--all", action="store_true", help="remove every planner")
    p_rm.add_argument("-f", "--force", action="store_true", help="deinit shared paths too; co-tagged planners become pending")
    p_check = sub.add_parser("check")
    p_check.add_argument("--all", action="store_true")
    p_check.add_argument("-q", "--quiet", action="store_true")

    args = ap.parse_args()
    handlers = {
        "ls": cmd_ls,
        "add": cmd_add,
        "rm": cmd_rm,
        "update": cmd_update,
        "uninstall": cmd_uninstall,
        "check": cmd_check,
    }
    return handlers[args.cmd](arena_dir(), args)


if __name__ == "__main__":
    raise SystemExit(main())
