#!/usr/bin/env python3
"""arena_robots feature backend: per-robot/component submodule ops + mesh URI check."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_SDK_SUBDIR = "arena_robots"
ROBOTS_PREFIX = "arena_robots/arena_robots/robots"
COMPONENTS_PREFIX = "arena_robots/arena_robots/components"

VARIANTS_RE = re.compile(r"^variants:\s*\[([^\]]*)\]", re.MULTILINE)


MESH_URI_RE = re.compile(r'package://arena_robots/([^\s"\'<>\)\}\$]+)')
FIND_URI_RE = re.compile(r'\$\(find\s+arena_robots\)/([^\s"\'<>\)\}\$]+)')
DYN_RE = re.compile(r'\$\{|\$\(')
SCAN_EXTS = {".urdf", ".xacro", ".yaml", ".xml", ".launch", ".py", ".sdf", ".world"}


def arena_dir() -> Path:
    if "ARENA_DIR" in os.environ:
        return Path(os.environ["ARENA_DIR"])
    for p in Path(__file__).resolve().parents:
        if (p / ".gitmodules").is_file() and (p / "arena_robots").is_dir():
            return p
    sys.exit("error: set ARENA_DIR or run inside an Arena checkout")


def submodule_status(arena: Path) -> dict[str, str]:
    """{path: 'init'|'uninit'} for each submodule (recursive, paths relative to Arena)."""
    out = subprocess.run(
        ["git", "submodule", "status", "--recursive"], cwd=arena,
        text=True, capture_output=True, check=False,
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


def robot_submodules(arena: Path) -> dict[str, list[str]]:
    """{name: [paths-relative-to-arena]} from the SDK submodule's .gitmodules `robot = <name>`
    and `component = <type>/<name>` tags (component refs keep the slash, so the two
    namespaces cannot collide). Paths in the SDK's gitmodules are relative to the SDK;
    we prepend `arena_robots/` for callers. Both tags may list multiple
    whitespace-separated names (shared upstream dep)."""
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
        names = cfg[section].get("robot", "").split() + cfg[section].get("component", "").split()
        if not names:
            rel = path.removeprefix(f"{ROBOTS_PREFIX}/")
            if rel == path:
                continue
            names = [rel.split("/", 1)[0]]
        for name in names:
            out.setdefault(name, []).append(path)
    return out


def submodule_sparse(arena: Path) -> dict[str, str]:
    """{path-relative-to-arena: sparse-checkout dirs} from `sparse = <dir...>` tags, for
    submodules whose repo carries packages we must keep out of the colcon workspace.
    Applied after every init/update since sparse config does not propagate via clone."""
    sdk_gitmodules = arena / _SDK_SUBDIR / ".gitmodules"
    if not sdk_gitmodules.is_file():
        return {}
    cfg = configparser.ConfigParser()
    cfg.read(sdk_gitmodules)
    return {
        f"{_SDK_SUBDIR}/{cfg[section].get('path', '').strip()}": sparse
        for section in cfg.sections()
        if section.startswith("submodule ") and (sparse := cfg[section].get("sparse", "").strip())
    }


def _apply_sparse(arena: Path, path: str) -> None:
    sparse = submodule_sparse(arena).get(path)
    if sparse and (arena / path / ".git").exists():
        _git(["sparse-checkout", "set", *sparse.split()], arena / path, check=False)


def component_families(arena: Path) -> set[str]:
    """{'<type>/<name>'} for every catalog component (dirs holding a component.yaml)."""
    root = arena / COMPONENTS_PREFIX
    if not root.is_dir():
        return set()
    return {
        f"{type_dir.name}/{fam.name}"
        for type_dir in root.iterdir() if type_dir.is_dir()
        for fam in type_dir.iterdir() if (fam / "component.yaml").is_file()
    }


def resolve_component_ref(arena: Path, ref: str) -> str | None:
    """Resolve a slashed ref to its family: exact family passes through, a variant ref
    (e.g. arm/ur5e) maps via the family's `variants:` list. None if unknown."""
    families = component_families(arena)
    if ref in families:
        return ref
    ctype, _, variant = ref.partition("/")
    for fam in families:
        if not fam.startswith(f"{ctype}/"):
            continue
        match = VARIANTS_RE.search((arena / COMPONENTS_PREFIX / fam / "component.yaml").read_text(errors="ignore"))
        if match and variant in {v.strip() for v in match.group(1).split(",")}:
            return fam
    return None


def _path_robots(arena: Path) -> dict[str, set[str]]:
    """{path: {robots tagging it}}: reverse of robot_submodules for sharing checks."""
    out: dict[str, set[str]] = {}
    for robot, paths in robot_submodules(arena).items():
        for p in paths:
            out.setdefault(p, set()).add(robot)
    return out


def all_robots(arena: Path) -> list[str]:
    robots = {r for r in robot_submodules(arena) if "/" not in r}
    root = arena / ROBOTS_PREFIX
    if root.is_dir():
        robots |= {p.name for p in root.iterdir() if p.is_dir()}
    return sorted(robots)


def _git(args: list[str], arena: Path, *, check: bool = True) -> int:
    return subprocess.run(["git", *args], cwd=arena, check=check).returncode


def cmd_ls(arena: Path, _args) -> int:
    subs = robot_submodules(arena)
    status = submodule_status(arena)
    components = component_families(arena) | {n for n in subs if "/" in n}
    for name in all_robots(arena) + sorted(components):
        pending = any(status.get(p) == "uninit" for p in subs.get(name, []))
        print(f"{'[ ]' if pending else '[x]'} {name}")
    return 0


def _normalize_names(arena: Path, names: list[str]) -> list[str] | None:
    """Map slashed component refs to their family, keeping order. None on unknown ref."""
    out: list[str] = []
    for name in names:
        if "/" not in name:
            out.append(name)
            continue
        family = resolve_component_ref(arena, name)
        if family is None:
            print(f"robots: unknown component '{name}'", file=sys.stderr)
            return None
        if family != name:
            print(f"robots: resolving '{name}' to component family '{family}'")
        out.append(family)
    return out


def cmd_add(arena: Path, args) -> int:
    subs = robot_submodules(arena)
    if args.all:
        if args.names:
            print("robots: --all is mutually exclusive with robot names", file=sys.stderr)
            return 2
        names = sorted(subs)
    else:
        if not args.names:
            print("robots: specify robot/component name(s) or --all", file=sys.stderr)
            return 2
        names = _normalize_names(arena, args.names)
        if names is None:
            return 2
    for robot in names:
        paths = subs.get(robot)
        if not paths:
            print(f"robots: '{robot}' has no submodules (nothing to install)")
            continue
        _git(["-c", "protocol.file.allow=always",
              "submodule", "update", "--init", "--checkout", _SDK_SUBDIR], arena)
        sdk = arena / _SDK_SUBDIR
        for p in paths:
            sub_path = Path(p).relative_to(_SDK_SUBDIR).as_posix()
            _git(["-c", "protocol.file.allow=always",
                  "submodule", "update", "--init", "--recursive", "--checkout", sub_path], sdk)
            _apply_sparse(arena, p)
    return 0


def cmd_rm(arena: Path, args) -> int:
    subs = robot_submodules(arena)
    if args.all:
        if args.names:
            print("robots: --all is mutually exclusive with robot names", file=sys.stderr)
            return 2
        names = sorted(subs)
    else:
        if not args.names:
            print("robots: specify robot/component name(s) or --all", file=sys.stderr)
            return 2
        names = _normalize_names(arena, args.names)
        if names is None:
            return 2
    remove_set = set(names)
    shared = _path_robots(arena)
    sdk = arena / _SDK_SUBDIR
    done: set[str] = set()
    for robot in names:
        paths = subs.get(robot)
        if not paths:
            print(f"robots: '{robot}' has no submodules (nothing to remove)")
            continue
        for p in paths:
            if p in done:
                continue
            # keep a shared path only if some robot outside this removal batch still tags it
            others = shared.get(p, set()) - remove_set
            if others and not args.force:
                print(f"robots: keeping '{p}' (still tagged by: {', '.join(sorted(others))})")
                continue
            if others:
                print(f"robots: force-removing '{p}' (also tagged by: {', '.join(sorted(others))})")
            sub_path = Path(p).relative_to(_SDK_SUBDIR).as_posix()
            _git(["submodule", "deinit", "-f", sub_path], sdk, check=False)
            done.add(p)
    return 0


def cmd_update(arena: Path, _args) -> int:
    sdk = arena / _SDK_SUBDIR
    if not (sdk / ".gitmodules").is_file():
        return 0
    subs = robot_submodules(arena)
    status = submodule_status(arena)
    installed = {r for r, paths in subs.items() if any(status.get(p) == "init" for p in paths)}
    rc = 0
    # --checkout overrides update=none
    for p in sorted({p for r in installed for p in subs[r]}):
        sub_path = Path(p).relative_to(_SDK_SUBDIR).as_posix()
        code = _git(["-c", "protocol.file.allow=always",
                     "submodule", "update", "--init", "--recursive", "--checkout", sub_path],
                    sdk, check=False)
        _apply_sparse(arena, p)
        rc = rc or code
    code = _git(["submodule", "update", "--recursive"], sdk, check=False)
    return rc or code


def cmd_uninstall(arena: Path, _args) -> int:
    subs = robot_submodules(arena)
    status = submodule_status(arena)
    sdk = arena / _SDK_SUBDIR
    for p in (p for ps in subs.values() for p in ps):
        if status.get(p) == "init":
            sub_path = Path(p).relative_to(_SDK_SUBDIR).as_posix()
            _git(["submodule", "deinit", "-f", sub_path], sdk, check=False)
    return 0


def _uninit_mesh_paths(arena: Path) -> dict[str, set[Path]]:
    """Robots whose mesh submodule(s) under ROBOTS_PREFIX are uninitialized,
    with paths rewritten relative to arena_robots/arena_robots (matching URI layout)."""
    subs = robot_submodules(arena)
    status = submodule_status(arena)
    out: dict[str, set[Path]] = {}
    for robot, paths in subs.items():
        for p in paths:
            if status.get(p) != "uninit" or not p.startswith(f"{ROBOTS_PREFIX}/"):
                continue
            out.setdefault(robot, set()).add(Path(p).relative_to("arena_robots/arena_robots"))
    return out


def cmd_check(arena: Path, args) -> int:
    root = arena / "arena_robots" / "arena_robots" / "robots"
    if not root.is_dir():
        print(f"error: {root} not found", file=sys.stderr)
        return 1
    uninit = {} if args.all else _uninit_mesh_paths(arena)

    misses: dict[str, dict[str, set[str]]] = {}
    dynamic: set[str] = set()
    files = [p for p in root.rglob("*")
             if p.is_file() and (p.suffix.lower() in SCAN_EXTS or ".urdf." in p.name)]
    for f in files:
        try:
            txt = f.read_text(errors="ignore")
        except OSError:
            continue
        for line in txt.splitlines():
            hits = list(MESH_URI_RE.finditer(line)) + list(FIND_URI_RE.finditer(line))
            if not hits:
                continue
            line_dynamic = bool(DYN_RE.search(line))
            for m in hits:
                rel = m.group(1).rstrip('-"\'')
                if line_dynamic:
                    dynamic.add(rel)
                    continue
                mesh_rest = rel.removeprefix("robots/")
                robot = (mesh_rest.split("/", 1)[0]
                         if mesh_rest != rel and "/" in mesh_rest else None)
                rel_path = Path(rel)
                if robot and any(rel_path.is_relative_to(sm) for sm in uninit.get(robot, ())):
                    continue
                if not (root.parent / rel).is_file():
                    owner = robot or "<other>"
                    src = str(f.relative_to(root.parent.parent))
                    misses.setdefault(owner, {}).setdefault(rel, set()).add(src)

    if not args.quiet:
        if uninit:
            print(f"skipping uninitialized robots: {', '.join(sorted(uninit))}")
        print(f"dynamic refs: {len(dynamic)}, not statically checkable")

    total = sum(len(v) for v in misses.values())
    if total == 0:
        if not args.quiet:
            print("OK: every concrete package://arena_robots/... URI resolves")
        return 0

    print(f"\nFAIL: {total} unresolved path(s) in {len(misses)} robot(s):")
    for robot in sorted(misses):
        print(f"\n  [{robot}]")
        for rel in sorted(misses[robot]):
            print(f"    {rel}")
            for src in sorted(misses[robot][rel]):
                print(f"      ← {src}")
    return 1


DRIVE_EMPTY_RANDOM = {
    "static":      {"min": 0, "max": 0},
    "dynamic":     {"min": 0, "max": 0},
    "interactive": {"min": 0, "max": 0},
}
DRIVE_OWN_KEYS = frozenset({"map", "episodes", "timeout"})
KV_RE = re.compile(r"^(\w+):=(.*)$")


def _ready_robots(arena: Path) -> list[str]:
    subs = robot_submodules(arena)
    status = submodule_status(arena)
    ready: list[str] = []
    for robot in all_robots(arena):
        if any(status.get(p) == "uninit" for p in subs.get(robot, [])):
            continue
        ready.append(robot)
    return ready


def _drive_suite(robots: list[str], *, map_name: str, episodes: int, timeout: str) -> dict:
    return {
        "stages": [
            {
                "name": f"drive-{robot}",
                "robot": robot,
                "map": map_name,
                "episodes": episodes,
                "tm_robots": "random",
                "tm_obstacles": "random",
                "config": {"random": dict(DRIVE_EMPTY_RANDOM)},
                "timeout": timeout,
            }
            for robot in robots
        ]
    }


def _benchmark_data_root() -> Path:
    env = os.environ.get("ARENA_DATA_DIR")
    if env:
        return Path(env) / "benchmarks"
    out = subprocess.run(
        ["ros2", "pkg", "prefix", "arena_evaluation"],
        text=True, capture_output=True, check=False,
    )
    if out.returncode != 0:
        sys.exit("error: ARENA_DATA_DIR is unset and `ros2 pkg prefix arena_evaluation` failed")
    return Path(out.stdout.strip()) / "share" / "arena_evaluation" / "data"


def _newest_run(data_root: Path) -> Path | None:
    if not data_root.is_dir():
        return None
    runs = [p for p in data_root.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def _summarize(run_dir: Path) -> tuple[dict[str, tuple[int, int]], bool]:
    csv_path = run_dir / "progress.csv"
    if not csv_path.is_file():
        sys.exit(f"error: no progress.csv at {csv_path}")
    totals: dict[str, int] = {}
    successes: dict[str, int] = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            robot = row["robots"]
            totals[robot] = totals.get(robot, 0) + 1
            if row["outcome_state"] == "2":
                successes[robot] = successes.get(robot, 0) + 1
    summary = {r: (successes.get(r, 0), totals[r]) for r in totals}
    all_drove = bool(summary) and all(s > 0 for s, _ in summary.values())
    return summary, all_drove


def cmd_drive(arena: Path, args) -> int:
    own: dict[str, str] = {}
    passthrough: list[str] = []
    selected: list[str] = []
    for tok in args.kv:
        match = KV_RE.match(tok)
        if match:
            key, value = match.group(1), match.group(2)
            if key in DRIVE_OWN_KEYS:
                own[key] = value
            else:
                passthrough.append(tok)
        else:
            selected.append(tok)

    ready = _ready_robots(arena)
    if not ready:
        print("robots: no ready robots, run `arena feature robots add ...` first", file=sys.stderr)
        return 2
    if selected:
        ready_set = set(ready)
        unknown = [r for r in selected if r not in ready_set]
        if unknown:
            print(
                f"robots: not ready: {', '.join(unknown)}. ready: {', '.join(ready)}",
                file=sys.stderr,
            )
            return 2
        seen: dict[str, None] = {}
        for r in selected:
            seen.setdefault(r, None)
        robots = list(seen)
    else:
        robots = ready
    suite = _drive_suite(
        robots,
        map_name=own.get("map", "map_empty"),
        episodes=int(own.get("episodes", "3")),
        timeout=own.get("timeout", "30s"),
    )
    cmd = [
        "ros2", "run", "arena_evaluation", "benchmark",
        "--suite", json.dumps(suite),
        "--contest", '[{"name":"default"}]',
        *passthrough,
    ]
    print(
        f"robots: drive audit for {len(robots)} robot(s): {', '.join(robots)}",
        file=sys.stderr,
    )
    rc = subprocess.run(cmd, check=False).returncode
    run_dir = _newest_run(_benchmark_data_root())
    if run_dir is None:
        print("robots: drive completed but no run directory found", file=sys.stderr)
        return rc or 1
    summary, all_drove = _summarize(run_dir)
    width = max(len("ROBOT"), *(len(r) for r in summary)) if summary else len("ROBOT")
    print()
    print(f"  {'ROBOT':<{width}}  SUCCESS  TOTAL")
    for robot in sorted(summary):
        s, t = summary[robot]
        print(f"  {robot:<{width}}  {s:>7}  {t:>5}")
    print()
    print(f"run: {run_dir}")
    if rc != 0:
        return rc
    return 0 if all_drove else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ls")
    sub.add_parser("update")
    sub.add_parser("uninstall")
    p_add = sub.add_parser("add")
    p_add.add_argument("names", nargs="*", help="robot names and/or component refs (<type>/<name>)")
    p_add.add_argument("--all", action="store_true", help="fetch every robot and component")
    p_rm = sub.add_parser("rm")
    p_rm.add_argument("names", nargs="*", help="robot names and/or component refs (<type>/<name>)")
    p_rm.add_argument("--all", action="store_true", help="remove every robot and component")
    p_rm.add_argument("-f", "--force", action="store_true",
                      help="deinit shared paths too; co-tagged robots become pending")
    p = sub.add_parser("check")
    p.add_argument("--all", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    p_drive = sub.add_parser("drive")
    p_drive.add_argument("kv", nargs="*",
                         help="bare tokens select robots (default: all ready); "
                              "key:=value pairs: map/episodes/timeout consumed locally, "
                              "rest forwarded to the benchmark runner")

    args = ap.parse_args()
    handlers = {
        "ls": cmd_ls, "add": cmd_add, "rm": cmd_rm,
        "update": cmd_update, "uninstall": cmd_uninstall, "check": cmd_check,
        "drive": cmd_drive,
    }
    return handlers[args.cmd](arena_dir(), args)


if __name__ == "__main__":
    raise SystemExit(main())
