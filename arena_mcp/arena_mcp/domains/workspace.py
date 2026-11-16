from __future__ import annotations

import concurrent.futures
import os
import re
import subprocess
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Annotated, Any, cast

from pydantic import Field

from arena_mcp.core import Domain, ToolError, cli

if TYPE_CHECKING:
    from arena_cli.complete import Static

domain = Domain(
    "workspace",
    """The workspace and its containers through the arena CLI: git state of every repo, arena update, features, net and gpu settings, forks, robot submodules.
    Every tool that changes the tree, the .env or a container is marked destructive, long ones run as process jobs.
    Settings changes apply to new shells, the running container keeps its values until `source arena` recreates it on the host.""",
    requires=("arena_cli",),
)

FEATURE_ACTIONS = ("install", "update", "stop", "uninstall")
ROBOTS_ACTIONS = ("ls", "add", "rm", "update")
RECREATE = "applies once `source arena` on the host recreates the container, new shells in it pick it up at once"
ROW_RE = re.compile(r"^([a-z]+):\s+(.*)$")
GIT_ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}


def _git(path: Path, *args: str) -> str:
    try:
        done = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, env=GIT_ENV, stdin=subprocess.DEVNULL, check=False, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"git {' '.join(args)} did not finish within 60s in {path}") from exc
    if done.returncode != 0:
        raise ToolError(f"git {' '.join(args)} failed in {path}: {done.stderr.strip()}")
    return done.stdout


def repo_status(path: Path) -> dict[str, Any]:
    """Branch, upstream, ahead/behind and change counts of one repo, read without taking the index lock."""
    out: dict[str, Any] = {"changed": 0, "untracked": 0}
    for line in _git(path, "status", "--porcelain=v2", "--branch", "--untracked-files=normal", "--ignore-submodules=all").splitlines():
        if line.startswith("# branch.oid "):
            out["commit"] = line.split()[2][:12]
        elif line.startswith("# branch.head "):
            out["branch"] = line.split(maxsplit=2)[2]
        elif line.startswith("# branch.upstream "):
            out["upstream"] = line.split(maxsplit=2)[2]
        elif line.startswith("# branch.ab "):
            ahead, behind = line.split()[2:4]
            out["ahead"], out["behind"] = int(ahead), -int(behind)
        elif line.startswith("? "):
            out["untracked"] += 1
        elif line[:2] in ("1 ", "2 ", "u "):
            out["changed"] += 1
    out["dirty"] = bool(out["changed"] or out["untracked"])
    return out


SUBMODULE_STATES = {" ": "pinned", "+": "off_pin", "-": "uninitialized", "U": "conflict"}


def submodules(root: Path) -> list[tuple[str, str]]:
    """(path, state) of every submodule, recursively: pinned, off_pin (checked out away from the gitlink), uninitialized or conflict."""
    if not (root / ".gitmodules").is_file():
        return []
    out = []
    for line in _git(root, "submodule", "status", "--recursive").splitlines():
        if len(line) > 42:
            out.append((line[42:].split(" (", 1)[0].strip(), SUBMODULE_STATES.get(line[0], line[0])))
    return out


def tree_status(root: Path) -> dict[str, Any]:
    """The metarepo and every initialized submodule, each with repo_status."""
    subs = submodules(root)
    live = [path for path, state in subs if state != "uninitialized"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        statuses = dict(zip(live, pool.map(lambda rel: repo_status(root / rel), live), strict=True))
    repos = [{"path": ".", **repo_status(root)}]
    repos += [{"path": rel, "pin": state, **statuses[rel]} for rel, state in subs if rel in statuses]
    return {"root": str(root), "repos": repos, "uninitialized": [rel for rel, state in subs if state == "uninitialized"]}


def rows(text: str) -> dict[str, Any]:
    """The label: value rows an arena settings verb prints, warnings and other lines kept apart."""
    out: dict[str, Any] = {}
    warnings: list[str] = []
    notes: list[str] = []
    for line in text.splitlines():
        match = ROW_RE.match(line)
        if match and match.group(1) == "warning":
            warnings.append(match.group(2))
        elif match:
            out[match.group(1)] = match.group(2)
        elif line.strip():
            notes.append(line.strip())
    if warnings:
        out["warnings"] = warnings
    if notes:
        out["notes"] = notes
    return out


def _installed() -> list[str]:
    from arena_cli import common

    return common._reg_list()


def _feature(name: str) -> ModuleType:
    from arena_cli import features

    available = features.available()
    module = features.load(name) if name in available else None
    if module is None:
        raise ToolError(f"no feature {name!r} here", hint=f"features: {', '.join(available)}")
    return module


def _commands(module: ModuleType) -> list[str]:
    return [name for name, verb in module.COMMANDS.items() if not verb.hidden]


@domain.tool(read_only=True)
def workspace_status() -> dict[str, Any]:
    """Branch, upstream, ahead/behind and dirty state of the metarepo and every initialized submodule, installed features, and the net and gpu settings with their sources."""
    out = tree_status(cli.arena_dir())
    out["features"] = _installed()
    out["settings"] = {"net": _read_setting("settings", "net"), "gpu": _read_setting("feature", "docker", "gpu")}
    return out


def _read_setting(*args: str) -> dict[str, Any]:
    done = cli.run(*args, timeout=60, check=False)
    if done["exit_code"] != 0:
        return {"unavailable": (done["stderr"] or done["stdout"]).strip().splitlines()[-1:]}
    return rows(done["stdout"])


@domain.tool(destructive=True, open_world=True)
async def workspace_update() -> dict[str, Any]:
    """Run `arena update` as a job: pull every Arena repo, apply pending patches, refresh the python env."""
    return (await cli.job("update", "arena update", "update")).as_dict()


@domain.tool(read_only=True)
def workspace_features() -> list[dict[str, Any]]:
    """Every feature this context offers, whether it is installed, its summary and its commands."""
    from arena_cli import features

    installed = set(_installed())
    out = []
    for name in features.available():
        module = features.load(name)
        if module is None:
            continue
        summary = module.DESCRIPTION.replace("\b", "").strip().splitlines()[0]
        out.append({"name": name, "installed": name in installed, "summary": summary, "commands": _commands(module)})
    return out


@domain.tool(destructive=True, open_world=True)
async def workspace_feature(name: str, action: Annotated[str, Field(description="install, update, stop or uninstall, where the feature has that command")]) -> dict[str, Any]:
    """Run `arena feature <name> <action>` as a job."""
    module = _feature(name)
    offered = [command for command in _commands(module) if command in FEATURE_ACTIONS]
    if action not in offered:
        raise ToolError(f"feature {name} has no {action!r} action here", hint=f"actions: {', '.join(offered) or 'none'}")
    return (await cli.job("feature", f"arena feature {name} {action}", "feature", name, action)).as_dict()


def _setting(args: tuple[str, ...], value: str | None, values: list[str]) -> dict[str, Any]:
    if value is not None and value not in values:
        raise ToolError(f"unknown value {value!r}", hint=f"values: {', '.join(values)}")
    done = cli.run(*args, *([value] if value is not None else []), timeout=60)
    out = {**rows(done["stdout"]), "values": values}
    if value is not None:
        out["applies"] = RECREATE
    return out


@domain.tool(destructive=True, idempotent=True)
def workspace_settings_net(value: Annotated[str | None, Field(description="local (DDS on loopback) or lan (every interface), None to read")] = None) -> dict[str, Any]:
    """Read or set the ROS 2 network scope through `arena settings net`, persisted as ARENA_NET in the workspace .env."""
    from arena_cli import settings

    return _setting(("settings", "net"), value, list(settings.NET_MODES))


@domain.tool(destructive=True, idempotent=True)
def workspace_settings_gpu(value: Annotated[str | None, Field(description="on or off, None to read")] = None) -> dict[str, Any]:
    """Read or set NVIDIA GPU passthrough through `arena feature docker gpu`, persisted as ARENA_GPU in the workspace .env."""
    from arena_cli.features import docker

    return _setting(("feature", "docker", "gpu"), value, list(cast("Static", docker.COMMANDS["gpu"].complete).table()))


def fork_rows(text: str) -> list[dict[str, str]]:
    """The fork table `arena fork ls` prints."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith("FORK"):
        return []
    keys = [key.lower() for key in lines[0].split()]
    out = []
    for line in lines[1:]:
        if not line.strip():
            break
        out.append(dict(zip(keys, line.split(), strict=False)))
    return out


@domain.tool(read_only=True)
def workspace_forks() -> dict[str, Any]:
    """Forks with kind, release, ROS domain and container state, plus the releases table, from `arena fork ls`."""
    text = cli.run("fork", "ls", timeout=120)["stdout"]
    return {"forks": fork_rows(text), "text": text}


@domain.tool(destructive=True)
async def workspace_fork_new(
    name: Annotated[str | None, Field(description="fork name, the lowest free pN when None, an existing stopped fork is started")] = None,
    from_fork: Annotated[str | None, Field(description="fork to snapshot instead of the dev tree, with its robots, features, builds, commits and uncommitted edits; only creates")] = None,
) -> dict[str, Any]:
    """Run `arena fork new [<name>] [--from <from_fork>]` as a job."""
    args = [*([name] if name else []), *(["--from", _fork_name(from_fork)] if from_fork else [])]
    return (await cli.job("fork", " ".join(["arena fork new", *args]), "fork", "new", *args)).as_dict()


def _fork_name(name: str) -> str:
    if not name or name.startswith("-"):
        raise ToolError(f"{name!r} is not a fork name", hint="workspace_forks lists them")
    return name


@domain.tool(destructive=True)
async def workspace_fork_down(name: str) -> dict[str, Any]:
    """Run `arena fork down <name>` as a job, deleting the fork's container and its edits."""
    return (await cli.job("fork", f"arena fork down {name}", "fork", "down", _fork_name(name))).as_dict()


@domain.tool(destructive=True)
async def workspace_fork_exec(name: str, command: Annotated[str, Field(description="a bash command line run in the fork's sourced arena environment on its own ROS domain")]) -> dict[str, Any]:
    """Run a command in a fork through `arena fork exec`, starting the fork if it is stopped, as a job."""
    return (await cli.job("fork", f"arena fork exec {name}: {command}", "fork", "exec", _fork_name(name), "bash", "-c", command)).as_dict()


def robot_rows(text: str) -> dict[str, list[str]]:
    """Ready and pending robots and components from `arena feature robots ls`."""
    out: dict[str, list[str]] = {"ready": [], "pending": []}
    for line in text.splitlines():
        if line.startswith("[x] "):
            out["ready"].append(line[4:].strip())
        elif line.startswith("[ ] "):
            out["pending"].append(line[4:].strip())
    return out


@domain.tool(destructive=True, open_world=True)
async def workspace_robots(
    action: Annotated[str, Field(description="ls (ready and pending robots), add (clone submodules, then deps and build), rm (deinit submodules) or update (refresh initialized ones, then deps and build)")],
    names: Annotated[list[str] | None, Field(description="robot names or component refs for add and rm, or ['--all'], rm also takes '-f'")] = None,
) -> dict[str, Any]:
    """Manage robot submodules through `arena feature robots <action>`, ls answers at once, the others run as jobs."""
    module = _feature("robots")
    offered = [command for command in _commands(module) if command in ROBOTS_ACTIONS]
    if action not in offered:
        raise ToolError(f"unknown action {action!r}", hint=f"actions: {', '.join(offered)}")
    if action == "ls":
        return robot_rows(cli.run("feature", "robots", "ls", timeout=120)["stdout"])
    if action in ("add", "rm") and not names:
        raise ToolError(f"{action} needs robot names", hint="names lists robots or components, ['--all'] takes every one, workspace_robots('ls') shows them")
    if action == "update" and names:
        raise ToolError("update takes no names, it refreshes every initialized robot submodule")
    args = list(names or [])
    return (await cli.job("robots", f"arena feature robots {action} {' '.join(args)}".strip(), "feature", "robots", action, *args)).as_dict()
