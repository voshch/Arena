"""Tier C completion manifest: imports the runtime registries and writes JSON. The only ROS-touching CLI module."""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import complete
from complete import LaunchArgs

_ARG_RE = re.compile(r"^\s+'([^']+)':\*?\s*$")
_DEFAULT_RE = re.compile(r"^\s+\(default: '(.*)'\)\s*$")


def _launch_targets() -> set[tuple[str, str]]:
    import cli

    found: set[tuple[str, str]] = set()
    for v in cli._VERBS.values():
        for spec in complete.walk(v.complete):
            if isinstance(spec, LaunchArgs):
                found.add((spec.package, spec.file))
    return found


def _share_file(package: str, file: str) -> str | None:
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

    try:
        share = get_package_share_directory(package)
    except PackageNotFoundError:
        return None
    for dirpath, _dirnames, filenames in os.walk(share):
        if file in filenames:
            return os.path.realpath(os.path.join(dirpath, file))
    return None


def _show_args(package: str, file: str) -> dict[str, dict[str, str]]:
    proc = subprocess.run(["ros2", "launch", package, file, "--show-args"], capture_output=True, text=True, check=False)
    if proc.returncode:
        return {}
    args: dict[str, dict[str, str]] = {}
    name = None
    for line in proc.stdout.splitlines():
        m = _ARG_RE.match(line)
        if m:
            name = m.group(1)
            args[name] = {"desc": "", "default": ""}
            continue
        if name is None or not line.strip():
            continue
        m = _DEFAULT_RE.match(line)
        if m:
            args[name]["default"] = m.group(1)
        elif not args[name]["desc"]:
            args[name]["desc"] = line.strip()
    return args


def _enum_names(keys: object) -> list[str]:
    return sorted(k.value for k in list(keys))


def _asset_names() -> tuple[dict[str, list[str]], list[str]]:
    """Per-kind asset names for completion: local names plus bucket listings already cached on disk."""
    import asset
    from arena_simulation_setup.tree import NetResolver

    values: dict[str, list[str]] = {}
    deps: list[str] = []
    for kind in asset.KINDS:
        identifier_t = asset._identifier_type(kind)
        names = {identifier.shortname for identifier in identifier_t.listall()}
        for resolver in identifier_t._resolvers:
            if isinstance(resolver, NetResolver):
                names.update(identifier.shortname for identifier in resolver._parse_listing(resolver._cached_listing() or []))
                deps.append(str(resolver._listing_path))
        values[f"asset.{kind}"] = sorted(names)
    return values, deps


def _values() -> tuple[dict[str, list[str]], list[str]]:
    import task_generator.tasks.modules
    import task_generator.tasks.obstacles
    import task_generator.tasks.robots
    from arena_robots.Robot import RobotIdentifier
    from arena_runtime import sim as runtime_sim
    from arena_simulation_setup.tree.World.World import WorldIdentifier
    from task_generator.simulators import human as human_sim
    from task_generator.tasks import registry

    arena = Path(os.environ["ARENA_DIR"])
    values = {
        "sim": _enum_names(runtime_sim.SimulatorRegistry.keys()),
        "human": _enum_names(human_sim.HumanSimulatorRegistry.keys()),
        "task.robots": _enum_names(registry.ROBOTS_MODES.keys()),
        "task.obstacles": _enum_names(registry.OBSTACLES_MODES.keys()),
        "task.modules": _enum_names(registry.MODULE_MODES.keys()),
        "world": sorted(registry.identifier_to_available(WorldIdentifier)),
        "robot": sorted(set(registry.identifier_to_available(RobotIdentifier)) - _uninstalled_robots(arena)),
        "robot.planner": _installed_planners(arena),
    }
    deps = [
        os.path.realpath(m.__file__)
        for m in (runtime_sim, human_sim, registry, task_generator.tasks.robots, task_generator.tasks.obstacles, task_generator.tasks.modules)
        if m.__file__
    ]
    deps += [str(arena / "arena_robots" / ".gitmodules"), str(arena / "arena_planners" / ".gitmodules")]
    asset_values, asset_deps = _asset_names()
    values.update(asset_values)
    deps += [*asset_deps, os.environ["ARENA_ASSETS_DIR_LOCAL"], str(arena / "arena_simulation_setup" / "worlds")]
    return values, deps


def _uninstalled_robots(arena: Path) -> set[str]:
    import importlib.util

    payload = arena / "_meta" / "features" / "robots" / "robots.py"
    spec = importlib.util.spec_from_file_location("_robots_payload", payload)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    status = mod.submodule_status(arena)
    return {name for name, paths in mod.robot_submodules(arena).items() if any(status.get(p) == "uninit" for p in paths)}


def _installed_planners(arena: Path) -> list[str]:
    from arena_planners import registry

    paths = registry.submodule_paths(arena)
    status = registry.submodule_status(arena)
    return [n for n in registry.all_planners(arena) if not any(status.get(p) == "uninit" for p in paths.get(n, []))]


def build(ws: str) -> dict:
    values, deps = _values()
    launch: dict[str, dict[str, dict[str, str]]] = {}
    for package, file in sorted(_launch_targets()):
        launch[f"{package}/{file}"] = _show_args(package, file)
        path = _share_file(package, file)
        if path is not None:
            deps.append(path)
    return {"schema": complete.SCHEMA, "deps": deps, "stamp": complete.stamp(ws, deps), "values": values, "launch": launch}


def main(argv: list[str]) -> int:
    flag, *rest = argv or [""]
    if flag != "--out" or len(rest) != 1:
        print("usage: manifest.py --out <path>", file=sys.stderr)
        return 2
    out = rest[0]
    ws = os.environ.get("ARENA_WS_DIR", "")
    try:
        data = build(ws)
        rc = 0
    except BaseException as e:
        data = {"schema": complete.SCHEMA, "deps": [], "stamp": complete.stamp(ws, []), "error": f"{type(e).__name__}: {e}"}
        rc = 1
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = f"{out}.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, out)
    with contextlib.suppress(OSError):
        os.remove(out + ".lock")
    if rc:
        print(f"arena complete: manifest generation failed: {data['error']}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
