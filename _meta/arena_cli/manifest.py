"""Tier C completion manifest: imports the runtime registries and writes JSON. The only ROS-touching CLI module."""

from __future__ import annotations

import contextlib
import enum
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from arena_cli import complete
from arena_cli.complete import LaunchArgs

_ARG_RE = re.compile(r"^\s+'([^']+)':\*?\s*$")
_DEFAULT_RE = re.compile(r"^\s+\(default: '(.*)'\)\s*$")


def _launch_targets() -> set[tuple[str, str]]:
    from arena_cli import cli

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


def _enum_names(keys: Iterable[enum.Enum]) -> list[str]:
    return sorted(k.value for k in list(keys))


def _asset_names() -> tuple[dict[str, list[str]], list[str]]:
    """Per-kind asset names for completion: local names plus bucket listings already cached on disk."""
    from arena_simulation_setup.tree import NetResolver

    from arena_cli import asset

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


def _import_sources() -> dict[str, dict[str, str]]:
    """Registered importer names with descriptions per kind, empty when arena_assets is unavailable."""
    import json
    import subprocess

    proc = subprocess.run(["ros2", "run", "arena_assets", "arena_assets", "-s", "import", "--json"], capture_output=True, text=True, check=False)
    if proc.returncode:
        return {}
    sources: dict[str, dict[str, str]] = {}
    for row in json.loads(proc.stdout):
        sources.setdefault(row["kind"], {})[row["name"]] = row["description"]
    return sources


def _task_layers() -> tuple[dict[str, list[str]], list[str]]:
    """The registered task layers, and per axis the mode names of every layer under `task.<axis>`, with the layer modules as deps."""
    from arena_runtime.env.task_layer import TASK_LAYER_GROUP, mode_param, task_layer

    entries = sorted(importlib.metadata.entry_points(group=TASK_LAYER_GROUP), key=lambda entry: entry.name)
    axes: dict[str, set[str]] = {}
    names: list[str] = []
    deps: list[str] = []
    for entry in entries:
        try:
            namespaces = task_layer(entry.name).mode_namespaces()
        except Exception as exc:
            print(f"arena: skipping task layer {entry.name!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        names.append(entry.name)
        for axis, modes in namespaces.items():
            axes.setdefault(axis, set()).update(modes)
        file = sys.modules[entry.module].__file__
        if file:
            deps.append(os.path.realpath(file))
    return {"task_layer": names, **{mode_param(axis): sorted(modes) for axis, modes in sorted(axes.items())}}, deps


def _mcp_tools() -> tuple[dict[str, dict[str, str]], list[str]]:
    """Tool names and their `key=` candidates, empty when arena_mcp is unavailable."""
    if importlib.util.find_spec("arena_mcp") is None:
        return {}, []
    from arena_mcp.call import completion_values

    return completion_values()


def _task_axis_args(values: dict) -> dict[str, dict[str, str]]:
    from arena_runtime.env.task_layer import TASK_PARAM_PREFIX

    return {key: {"desc": f"{key.removeprefix(TASK_PARAM_PREFIX)} mode of the task layer", "default": ""} for key in values if key.startswith(TASK_PARAM_PREFIX)}


def _values() -> tuple[dict[str, list[str] | dict[str, str]], list[str]]:
    from arena_robots.Robot import RobotIdentifier
    from arena_runtime import humans as human_sim
    from arena_runtime import sim as runtime_sim
    from arena_simulation_setup.tree.World import WorldIdentifier

    arena = Path(os.environ["ARENA_DIR"])
    layer_values, layer_deps = _task_layers()
    values: dict[str, list[str] | dict[str, str]] = {
        "sim": _enum_names(runtime_sim.SimulatorRegistry.keys()),
        "human": _enum_names(human_sim.HumanSimulatorRegistry.keys()),
        **layer_values,
        "world": sorted(identifier.shortname for identifier in WorldIdentifier.listall()),
        "robot": sorted({identifier.shortname for identifier in RobotIdentifier.listall()} - _uninstalled_robots(arena)),
        "robot.planner": _installed_planners(arena),
    }
    deps = [os.path.realpath(m.__file__) for m in (runtime_sim, human_sim) if m.__file__] + layer_deps
    deps += [str(arena / "arena_robots" / ".gitmodules"), str(arena / "arena_planners" / ".gitmodules")]
    asset_values, asset_deps = _asset_names()
    values.update(asset_values)
    values.update({f"asset.import.{kind}": names for kind, names in _import_sources().items()})
    mcp_values, mcp_deps = _mcp_tools()
    values.update(mcp_values)
    deps += mcp_deps
    deps += [*asset_deps, os.environ["ARENA_ASSETS_DIR_LOCAL"], os.environ["ARENA_ASSETS_DIR_IMPORTED"], str(arena / "arena_simulation_setup" / "worlds")]
    return values, deps


def _uninstalled_robots(arena: Path) -> set[str]:
    from arena_cli.features import robots

    status = robots.submodule_status(arena)
    return {name for name, paths in robots.robot_submodules(arena).items() if not all(status.get(p) == "init" for p in paths)}


def _installed_planners(arena: Path) -> list[str]:
    from arena_planners import registry

    paths = registry.submodule_paths(arena)
    status = registry.submodule_status(arena)
    return [n for n in registry.all_planners(arena) if all(status.get(p) == "init" for p in paths.get(n, []))]


def build(ws: str) -> dict:
    values, deps = _values()
    axis_args = _task_axis_args(values)
    launch: dict[str, dict[str, dict[str, str]]] = {}
    for package, file in sorted(_launch_targets()):
        args = _show_args(package, file)
        launch[f"{package}/{file}"] = {**axis_args, **args} if "task_layer" in args else args
        path = _share_file(package, file)
        if path is not None:
            deps.append(path)
    return {"schema": complete.SCHEMA, "deps": deps, "stamp": complete.stamp(ws, deps), "values": values, "launch": launch}


def main(argv: list[str]) -> int:
    flag, *rest = argv or [""]
    if flag != "--out" or len(rest) != 1:
        print("usage: python -m arena_cli.manifest --out <path>", file=sys.stderr)
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
