#!/usr/bin/env python3
"""Compute, populate and validate the uv workspace made of per-package pyproject.toml files."""

from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Iterator

EXCLUDED = (
    "arena_planners/planners",
    "arena_isaac",
    "arena_tools",
    "arena_robots/deps",
)
SKIP_DIRS = {".git", "build", "install", "log", ".venv", "node_modules", "__pycache__", ".uv-workspace"}
SKIP_PATHS = {"_meta/repos"}
IGNORE_MARKERS = {"COLCON_IGNORE", "AMENT_IGNORE"}
EXTRA_MEMBERS = ("_meta/arena_cli",)
REAL_MEMBERS = ("arena_planners/arena_planners", "_meta/arena_cli", "arena_training", "arena_training/deps/rosnav_rl/rosnav_rl")
SHARED_MEMBERS = ("arena_planners/arena_planners",)
FORBIDDEN_DYNAMIC = {"version", "dependencies", "optional-dependencies", "requires-python"}
REMOVED_SETUP_KWARGS = {"version", "install_requires", "extras_require", "tests_require"}
SETUP_KWARG_FIELDS = {
    "description": "description",
    "long_description": "readme",
    "license": "license",
    "maintainer": "maintainers",
    "maintainer_email": "maintainers",
    "author": "authors",
    "author_email": "authors",
    "keywords": "keywords",
    "classifiers": "classifiers",
    "project_urls": "urls",
}
COMPOSE_PROJECT_KEYS = ("name", "version", "requires-python", "dependencies")
ENTRY_POINT_FIELDS = ("scripts", "gui-scripts", "entry-points")
SUPERPROJECT = "."
REPO_UV_DROPPED = ("package", "default-groups", "workspace", "sources")
LOCK_HINT = "run python3 _meta/tools/uv_workspace.py lock"
HOOKS_DIR = "_meta/tools/githooks"
MANIFEST_NAMES = {"pyproject.toml", "setup.py"}
GITMODULES_PATH = re.compile(r"^\s*path\s*=\s*(.+?)\s*$", re.MULTILINE)
REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?\s*([^;]*?)\s*(?:;.*)?$")


class WorkspaceError(Exception):
    """A condition that stops the tool with exit code 2."""


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_toml(path: pathlib.Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise WorkspaceError(f"{path}: invalid TOML ({exc})") from exc


def has_project(path: pathlib.Path) -> bool:
    data = load_toml(path)
    return data is not None and isinstance(data.get("project"), dict)


def is_excluded(rel: str) -> bool:
    return any(rel == entry or rel.startswith(entry + "/") for entry in EXCLUDED)


def join_rel(rel: str, name: str) -> str:
    return name if rel == "." else f"{rel}/{name}"


def rel_posix(path: pathlib.Path, root: pathlib.Path) -> str:
    return path.relative_to(root).as_posix()


def walk(root: pathlib.Path, *, strict: bool) -> Iterator[tuple[pathlib.Path, list[str]]]:
    """Yield (dir, filenames) under root, pruning skip dirs and, when strict, excluded and ignored subtrees."""
    for dirpath, dirnames, filenames in os.walk(root):
        here = pathlib.Path(dirpath)
        if strict and IGNORE_MARKERS.intersection(filenames):
            dirnames[:] = []
            continue
        rel = rel_posix(here, root) if here != root else "."
        keep = []
        for name in sorted(dirnames):
            child = join_rel(rel, name)
            if name in SKIP_DIRS or child in SKIP_PATHS or (strict and is_excluded(child)):
                continue
            keep.append(name)
        dirnames[:] = keep
        yield here, filenames


def member_for_package(pkg_dir: pathlib.Path, root: pathlib.Path) -> str | None:
    for candidate in (pkg_dir, pkg_dir / "src"):
        if has_project(candidate / "pyproject.toml"):
            return rel_posix(candidate, root)
    return None


def compute_members(root: pathlib.Path) -> list[str]:
    found = set()
    for here, files in walk(root, strict=True):
        if "package.xml" in files:
            member = member_for_package(here, root)
            if member is not None:
                found.add(member)
    for extra in EXTRA_MEMBERS:
        if has_project(root / extra / "pyproject.toml"):
            found.add(extra)
    return sorted(found)


def setup_call(tree: ast.AST) -> ast.Call | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Name) and func.id == "setup") or (isinstance(func, ast.Attribute) and func.attr == "setup"):
                return node
    return None


def entry_point_fields(value: ast.expr) -> set[str]:
    if not isinstance(value, ast.Dict):
        return set(ENTRY_POINT_FIELDS)
    keys = [key.value if isinstance(key, ast.Constant) else None for key in value.keys]
    if not all(isinstance(key, str) for key in keys):
        return set(ENTRY_POINT_FIELDS)
    needed = set()
    for key in keys:
        if key == "console_scripts":
            needed.add("scripts")
        elif key == "gui_scripts":
            needed.add("gui-scripts")
        else:
            needed.add("entry-points")
    return needed


def check_setup_py(member: str, project: dict, setup_path: pathlib.Path) -> list[str]:
    try:
        tree = ast.parse(setup_path.read_bytes())
    except (SyntaxError, ValueError):
        return [f"{member}: setup.py does not parse"]
    call = setup_call(tree)
    if call is None:
        return []
    dynamic = set(project.get("dynamic", []))
    problems = []
    for keyword in call.keywords:
        kwarg = keyword.arg
        if kwarg is None:
            continue
        if kwarg in REMOVED_SETUP_KWARGS:
            problems.append(f"{member}: setup.py passes {kwarg}, declare it in pyproject.toml only")
            continue
        if kwarg == "entry_points":
            fields = entry_point_fields(keyword.value)
        elif kwarg in SETUP_KWARG_FIELDS:
            fields = {SETUP_KWARG_FIELDS[kwarg]}
        else:
            continue
        for field in sorted(fields):
            if field not in project and field not in dynamic:
                problems.append(f"{member}: setup.py {kwarg} needs {field} in [project].dynamic or a static [project].{field}")
    return problems


def check_member(member: str, root: pathlib.Path) -> list[str]:
    data = load_toml(root / member / "pyproject.toml")
    project = data.get("project") if data else None
    if not isinstance(project, dict):
        return [f"{member}: pyproject.toml has no [project] table"]
    problems = []
    for key in ("name", "version"):
        if not isinstance(project.get(key), str):
            problems.append(f"{member}: [project].{key} must be a static string")
    if not isinstance(project.get("dependencies"), list):
        problems.append(f"{member}: [project].dependencies must be a list")
    if member in SHARED_MEMBERS and "requires-python" in project:
        problems.append(f"{member}: [project].requires-python is not allowed, other Python environments install this member")
    forbidden = sorted(FORBIDDEN_DYNAMIC.intersection(project.get("dynamic", [])))
    if forbidden:
        problems.append(f"{member}: [project].dynamic must not contain {', '.join(forbidden)}")
    package = data.get("tool", {}).get("uv", {}).get("package")
    if member in REAL_MEMBERS:
        if package is False:
            problems.append(f"{member}: [tool.uv] package = false is not allowed for an installed member")
    elif package is not False:
        problems.append(f"{member}: [tool.uv] package = false is required")
    package_dir = root / member
    if (package_dir / "package.xml").is_file() and (package_dir / "setup.py").is_file():
        problems += check_setup_py(member, project, package_dir / "setup.py")
    return problems


def toml_scalar(key: str, value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise WorkspaceError(f"unsupported TOML value for {key}")


def toml_value(key: str, value: object) -> str:
    if isinstance(value, list):
        if any(isinstance(item, list | dict) for item in value):
            raise WorkspaceError(f"unsupported TOML value for {key}")
        if not value:
            return "[]"
        return "[\n" + "".join(f"    {toml_scalar(key, item)},\n" for item in value) + "]"
    return toml_scalar(key, value)


def toml_key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key, ensure_ascii=False)


def emit_table(header: str, table: dict, *, nested: bool = True) -> list[str]:
    """Render a TOML table with one level of dotted-header subtables."""
    blocks = ["[" + header + "]\n" + "".join(f"{toml_key(key)} = {toml_value(f'{header}.{key}', value)}\n" for key, value in table.items() if not isinstance(value, dict))]
    for key, value in table.items():
        if isinstance(value, dict):
            if not nested:
                raise WorkspaceError(f"unsupported TOML value for {header}.{key}")
            blocks += emit_table(f"{header}.{toml_key(key)}", value, nested=False)
    return blocks


def load_root(root: pathlib.Path) -> dict:
    data = load_toml(root / "pyproject.toml")
    if data is None:
        raise WorkspaceError(f"{root / 'pyproject.toml'} not found")
    return data


def repo_of(member: str, root: pathlib.Path) -> str:
    """Nearest git checkout enclosing member, relative to root."""
    path = root / member
    while path != root:
        if (path / ".git").exists():
            return rel_posix(path, root)
        path = path.parent
    return SUPERPROJECT


def repo_members(root: pathlib.Path) -> dict[str, list[str]]:
    repos: dict[str, list[str]] = {}
    for member in compute_members(root):
        repos.setdefault(repo_of(member, root), []).append(member)
    return repos


def relative_members(members: list[str], base: pathlib.Path, root: pathlib.Path) -> list[str]:
    return sorted(pathlib.PurePath(os.path.relpath(root / member, base)).as_posix() for member in members)


def single_member(repo: str, members: list[str]) -> bool:
    return repo != SUPERPROJECT and len(members) == 1 and members[0] not in SHARED_MEMBERS


def lock_home(repo: str, members: list[str]) -> str:
    """Directory whose pyproject.toml owns the repo's uv.lock: the lone unshared member of a submodule, else the repo root."""
    return members[0] if single_member(repo, members) else repo


def lock_label(home: str) -> str:
    return "uv.lock" if home == SUPERPROJECT else f"{home}/uv.lock"


def manifest_label(home: str) -> str:
    return "pyproject.toml" if home == SUPERPROJECT else f"{home}/pyproject.toml"


def generated_root_name(repo: str) -> str:
    return f"{canonical(pathlib.PurePosixPath(repo).name)}-workspace"


def repo_manifest_text(repo: str, members: list[str], root: pathlib.Path, data: dict) -> str:
    """Generated workspace root pyproject.toml of a submodule repo, depending on every member."""
    project: dict[str, object] = {"name": generated_root_name(repo), "version": "0.0.0"}
    requires_python = data.get("project", {}).get("requires-python")
    if requires_python is not None:
        project["requires-python"] = requires_python
    names = sorted(member_names(members, root))
    project["dependencies"] = names
    tool_uv = {"package": False} | mirrored_uv_settings(data)
    blocks = emit_table("project", project, nested=False) + emit_table("tool.uv", tool_uv)
    blocks.append("[tool.uv.sources]\n" + "".join(f"{toml_key(name)} = {{ workspace = true }}\n" for name in names))
    blocks += emit_table("tool.uv.workspace", {"members": relative_members(members, root / repo, root)}, nested=False)
    return "\n".join(blocks)


def member_names(members: list[str], root: pathlib.Path) -> set[str]:
    names = set()
    for member in members:
        data = load_toml(root / member / "pyproject.toml")
        name = data.get("project", {}).get("name") if data else None
        if isinstance(name, str):
            names.add(canonical(name))
    return names


def with_root_members(text: str, members: list[str]) -> str:
    """Root pyproject.toml text with its [tool.uv.workspace] members replaced in place."""
    rendered = f"members = {toml_value('members', members)}"
    header = re.search(r"^\[tool\.uv\.workspace\][ \t]*$", text, re.MULTILINE)
    if header is None:
        return text.rstrip("\n") + f"\n\n[tool.uv.workspace]\n{rendered}\n"
    following = re.compile(r"^\[", re.MULTILINE).search(text, header.end())
    stop = following.start() if following else len(text)
    key = re.compile(r"^members[ \t]*=[ \t]*\[[^\]]*\]", re.MULTILINE).search(text, header.end(), stop)
    if key is None:
        return text[: header.end()] + "\n" + rendered + text[header.end() :]
    return text[: key.start()] + rendered + text[key.end() :]


def mirrored_uv_settings(data: dict) -> dict:
    return {key: value for key, value in data.get("tool", {}).get("uv", {}).items() if key not in REPO_UV_DROPPED}


def with_table_value(text: str, table: str, key: str, value: str) -> str:
    """Text with key = value in table, replacing a one-line value in place or appending it as the table's last key."""
    rendered = f"{toml_key(key)} = {value}"
    header = re.search(rf"^\[{re.escape(table)}\][ \t]*$", text, re.MULTILINE)
    if header is None:
        return text.rstrip("\n") + f"\n\n[{table}]\n{rendered}\n"
    following = re.compile(r"^\[", re.MULTILINE).search(text, header.end())
    stop = following.start() if following else len(text)
    existing = re.compile(rf"^{re.escape(toml_key(key))}[ \t]*=.*$", re.MULTILINE).search(text, header.end(), stop)
    if existing is not None:
        return text[: existing.start()] + rendered + text[existing.end() :]
    body = text[header.end() : stop].split("\n")
    while body and (not body[-1].strip() or body[-1].lstrip().startswith("#")):
        body.pop()
    end = header.end() + len("\n".join(body))
    return text[:end] + "\n" + rendered + text[end:]


def single_member_text(text: str, data: dict) -> str:
    """Member pyproject.toml text carrying the root's requires-python and mirrored [tool.uv] settings."""
    requires_python = data.get("project", {}).get("requires-python")
    if requires_python is not None:
        text = with_table_value(text, "project", "requires-python", toml_scalar("project.requires-python", requires_python))
    for key, value in mirrored_uv_settings(data).items():
        text = with_table_value(text, "tool.uv", key, toml_scalar(f"tool.uv.{key}", value))
    return text


def expected_manifests(root: pathlib.Path, repos: dict[str, list[str]], data: dict) -> dict[str, str]:
    """Expected text of the pyproject.toml owning each repo's uv.lock, keyed by its directory."""
    root_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    expected = {SUPERPROJECT: with_root_members(root_text, relative_members(repos.get(SUPERPROJECT, []), root, root))}
    for repo, members in repos.items():
        if repo == SUPERPROJECT:
            continue
        home = lock_home(repo, members)
        if single_member(repo, members):
            expected[home] = single_member_text((root / home / "pyproject.toml").read_text(encoding="utf-8"), data)
        else:
            expected[home] = repo_manifest_text(repo, members, root, data)
    return expected


def stray_files(root: pathlib.Path, repo: str, members: list[str]) -> list[pathlib.Path]:
    """Locks outside the repo's lock home and a generated workspace root a single-member repo no longer uses."""
    home = lock_home(repo, members)
    stray = [root / place / "uv.lock" for place in sorted({repo, *members} - {home}) if (root / place / "uv.lock").is_file()]
    if home != repo:
        data = load_toml(root / repo / "pyproject.toml")
        if data is not None and data.get("project", {}).get("name") == generated_root_name(repo):
            stray.append(root / repo / "pyproject.toml")
    return stray


def write_if_changed(path: pathlib.Path, text: str) -> bool:
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def lock_versions(path: pathlib.Path) -> dict[str, set[str]]:
    """Registry package versions recorded in a uv.lock."""
    data = load_toml(path)
    if data is None:
        raise WorkspaceError(f"{path} not found, {LOCK_HINT}")
    versions: dict[str, set[str]] = {}
    for package in data.get("package", []):
        if "registry" in package.get("source", {}):
            versions.setdefault(package["name"], set()).add(package["version"])
    return versions


def collect_pins(root: pathlib.Path, repos: dict[str, list[str]], *, allow_missing: bool) -> dict[str, str]:
    """Single-version registry pins from the locks of the given repos, refusing cross-repo conflicts."""
    owners: dict[str, dict[str, list[str]]] = {}
    for repo, members in sorted(repos.items()):
        home = lock_home(repo, members)
        path = root / home / "uv.lock"
        if allow_missing and not path.is_file():
            continue
        for name, versions in sorted(lock_versions(path).items()):
            if len(versions) > 1:
                print(f"warning: {lock_label(home)} locks several versions of {name}, left unpinned", file=sys.stderr)
                continue
            owners.setdefault(name, {}).setdefault(next(iter(versions)), []).append(lock_label(home))
    conflicts = [
        f"  {name}: " + ", ".join(f"{version} in {' '.join(labels)}" for version, labels in sorted(by_version.items()))
        for name, by_version in sorted(owners.items())
        if len(by_version) > 1
    ]
    if conflicts:
        raise WorkspaceError("repo locks pin different versions, " + LOCK_HINT + " on a full tree:\n" + "\n".join(conflicts))
    return {name: next(iter(by_version)) for name, by_version in owners.items()}


def compose_text(root: pathlib.Path, out: pathlib.Path, pins: dict[str, str]) -> tuple[str, int]:
    """Composed pyproject text for out and the number of members in it."""
    data = load_root(root)
    tool_uv = {key: value for key, value in data.get("tool", {}).get("uv", {}).items() if key != "workspace"}
    if "sources" in tool_uv:
        raise WorkspaceError("tracked root pyproject.toml must not contain [tool.uv.sources]")
    if pins:
        tool_uv["constraint-dependencies"] = [*tool_uv.get("constraint-dependencies", []), *(f"{name}=={version}" for name, version in sorted(pins.items()))]
    project = {key: value for key, value in data.get("project", {}).items() if key in COMPOSE_PROJECT_KEYS}
    members = relative_members(compute_members(root), out, root)
    blocks = []
    if project:
        blocks += emit_table("project", project, nested=False)
    if data.get("dependency-groups"):
        blocks += emit_table("dependency-groups", data["dependency-groups"], nested=False)
    if tool_uv:
        blocks += emit_table("tool.uv", tool_uv)
    blocks += emit_table("tool.uv.workspace", {"members": members}, nested=False)
    return "\n".join(blocks), len(members)


def write_composed(root: pathlib.Path, out: pathlib.Path, pins: dict[str, str]) -> None:
    text, count = compose_text(root, out, pins)
    target = out / "pyproject.toml"
    out.mkdir(parents=True, exist_ok=True)
    if write_if_changed(target, text):
        print(f"composed {count} members and {len(pins)} pins into {target}")
    else:
        print(f"{target} up to date")


def run_compose(args: argparse.Namespace) -> int:
    root = args.root
    repos = repo_members(root)
    write_composed(root, (root / args.out).resolve(), collect_pins(root, repos, allow_missing=False))
    install_repo_hooks(root, repos)
    return 0


def git(args: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, env=without_git_env(), capture_output=True, text=True, check=False)


def without_git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def install_repo_hooks(root: pathlib.Path, repos: dict[str, list[str]]) -> None:
    """Point core.hooksPath of every submodule hosting members at the relock hook, relative so host and container checkouts both resolve it."""
    if shutil.which("git") is None:
        return
    for repo in sorted(repos):
        if repo == SUPERPROJECT:
            continue
        expected = pathlib.PurePath(os.path.relpath(root / HOOKS_DIR, root / repo)).as_posix()
        current = git(["config", "--get", "core.hooksPath"], root / repo).stdout.strip()
        if current == expected:
            continue
        if current and not current.endswith(HOOKS_DIR):
            print(f"warning: {repo} keeps core.hooksPath={current}, dependency edits there will not relock on commit", file=sys.stderr)
            continue
        if git(["config", "core.hooksPath", expected], root / repo).returncode == 0:
            print(f"installed relock hook in {repo}")


def uninitialized_submodules(root: pathlib.Path) -> list[str]:
    gitmodules = root / ".gitmodules"
    if not gitmodules.is_file():
        return []
    paths = GITMODULES_PATH.findall(gitmodules.read_text(encoding="utf-8"))
    return sorted(path for path in paths if not is_excluded(path) and not (root / path / ".git").exists())


def run_uv(command: list[str], failure: str) -> None:
    if subprocess.run(command, env=without_git_env(), check=False).returncode:
        raise WorkspaceError(failure)


def run_lock(args: argparse.Namespace) -> int:
    root = args.root
    missing = uninitialized_submodules(root)
    if missing:
        raise WorkspaceError(f"lock needs a full tree, initialize {', '.join(missing)}")
    uv = shutil.which("uv")
    if uv is None:
        raise WorkspaceError("uv not found on PATH")
    repos = repo_members(root)
    for home, text in expected_manifests(root, repos, load_root(root)).items():
        if write_if_changed(root / home / "pyproject.toml", text):
            print(f"wrote {manifest_label(home)}")
    for repo, members in sorted(repos.items()):
        for path in stray_files(root, repo, members):
            path.unlink()
            print(f"removed {rel_posix(path, root)}")
    out = (root / args.out).resolve()
    released = {canonical(name) for name in args.upgrade_package}
    pins = {} if args.upgrade else {name: version for name, version in collect_pins(root, repos, allow_missing=True).items() if name not in released}
    if args.upgrade:
        (out / "uv.lock").unlink(missing_ok=True)
    write_composed(root, out, pins)
    upgrades = [flag for name in args.upgrade_package for flag in ("--upgrade-package", name)]
    run_uv([uv, "lock", "--project", str(out), *upgrades], "uv lock failed on the composed workspace, release pins with --upgrade-package NAME or re-resolve with --upgrade")
    composed = lock_versions(out / "uv.lock")
    for repo, members in sorted(repos.items()):
        home = lock_home(repo, members)
        target = root / home / "uv.lock"
        shutil.copyfile(out / "uv.lock", target)
        run_uv([uv, "lock", "--offline", "--project", str(root / home)], f"uv lock failed deriving {lock_label(home)}")
        drift = sorted(name for name, versions in lock_versions(target).items() if composed.get(name) != versions)
        if drift:
            raise WorkspaceError(f"{lock_label(home)} differs from the composed lock for {', '.join(drift)}")
        print(f"derived {lock_label(home)}")
    write_composed(root, out, collect_pins(root, repos, allow_missing=False))
    run_uv([uv, "lock", "--offline", "--project", str(out)], "uv lock failed re-pinning the composed workspace")
    return 0


def lock_outputs(root: pathlib.Path, repos: dict[str, list[str]]) -> list[pathlib.Path]:
    """Every file lock may write or remove."""
    places = {SUPERPROJECT}
    for repo, members in repos.items():
        places |= {repo, *members}
    return sorted(root / place / name for place in places for name in ("pyproject.toml", "uv.lock"))


def snapshot(paths: list[pathlib.Path]) -> dict[pathlib.Path, bytes | None]:
    return {path: path.read_bytes() if path.is_file() else None for path in paths}


def run_hook(args: argparse.Namespace) -> int:
    """Relock when the commit stages a manifest, then stage what changed in the committing repo."""
    top = pathlib.Path.cwd().resolve()
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=top, capture_output=True, text=True, check=False).stdout.split()
    if not any(pathlib.PurePosixPath(path).name in MANIFEST_NAMES for path in staged):
        return 0
    root = args.root
    missing = uninitialized_submodules(root)
    if missing or shutil.which("uv") is None:
        reason = f"initialize {', '.join(missing)}" if missing else "install uv"
        print(f"uv-workspace: skipped relock ({reason}), check will flag stale locks", file=sys.stderr)
        return 0
    outputs = lock_outputs(root, repo_members(root))
    before = snapshot(outputs)
    run_lock(argparse.Namespace(root=root, out=pathlib.Path(".uv-workspace"), upgrade=False, upgrade_package=[]))
    after = snapshot(lock_outputs(root, repo_members(root)) + outputs)
    changed = sorted(path for path in after if before.get(path) != after[path])
    mine = [path for path in changed if path.is_relative_to(top) and repo_top(path, root) == top]
    others = [path for path in changed if path not in mine]
    if mine:
        subprocess.run(["git", "add", "-A", "--", *(str(path.relative_to(top)) for path in mine)], cwd=top, check=True)
        print("uv-workspace: staged " + ", ".join(str(path.relative_to(top)) for path in mine))
    if others:
        print("uv-workspace: also updated, commit in their repos: " + ", ".join(rel_posix(path, root) for path in others), file=sys.stderr)
    return 0


def repo_top(path: pathlib.Path, root: pathlib.Path) -> pathlib.Path:
    """Checkout that owns path: the nearest ancestor holding .git, at most root."""
    for parent in path.parents:
        if (parent / ".git").exists() or parent == root:
            return parent
    return root


def run_members(args: argparse.Namespace) -> int:
    for member in compute_members(args.root):
        print(member)
    return 0


def requirement_key(text: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    match = REQUIREMENT.match(text)
    if match is None:
        return text, (), ()
    name, extras, specifier = match.groups()
    return canonical(name), split_sorted(extras or ""), split_sorted(specifier)


def locked_key(entry: dict) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    return entry["name"], tuple(sorted(canonical(extra) for extra in entry.get("extras", []))), split_sorted(entry.get("specifier", ""))


def split_sorted(text: str) -> tuple[str, ...]:
    return tuple(sorted(part.replace(" ", "") for part in text.split(",") if part.strip()))


def declared_requirements(project_data: dict) -> set[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    project = project_data.get("project", {})
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements += extra
    return {requirement_key(text) for text in requirements}


def declared_groups(project_data: dict) -> dict[str, set[tuple[str, tuple[str, ...], tuple[str, ...]]]]:
    groups = project_data.get("dependency-groups", {})
    return {canonical(group): {requirement_key(text) for text in entries} for group, entries in groups.items() if entries}


def lock_problems(root: pathlib.Path, home: str, members: list[str], manifest: dict) -> list[str]:
    """Ways the uv.lock in home no longer matches its manifests."""
    label = lock_label(home)
    lock = load_toml(root / home / "uv.lock")
    if lock is None:
        return [f"{label} is missing, {LOCK_HINT}"]
    projects = {canonical(manifest["project"]["name"]): manifest}
    for member in members:
        data = load_toml(root / member / "pyproject.toml")
        if data is not None and isinstance(data.get("project", {}).get("name"), str):
            projects[canonical(data["project"]["name"])] = data
    stale = []
    if set(lock.get("manifest", {}).get("members", [])) != (set(projects) if len(projects) > 1 else set()):
        stale.append("members")
    locked = {package["name"]: package for package in lock.get("package", []) if "registry" not in package.get("source", {})}
    for name, data in sorted(projects.items()):
        metadata = locked.get(name, {}).get("metadata", {})
        if declared_requirements(data) != {locked_key(entry) for entry in metadata.get("requires-dist", [])}:
            stale.append(f"{name} dependencies")
        groups = {group: {locked_key(entry) for entry in entries} for group, entries in metadata.get("requires-dev", {}).items() if entries}
        if declared_groups(data) != groups:
            stale.append(f"{name} dependency-groups")
    return [f"{label} is stale ({', '.join(stale)}), {LOCK_HINT}"] if stale else []


def check(root: pathlib.Path) -> list[str]:
    data = load_root(root)
    problems = []
    if "sources" in data.get("tool", {}).get("uv", {}):
        problems.append("pyproject.toml: tracked root must not contain [tool.uv.sources]")
    repos = repo_members(root)
    expected_root_members = relative_members(repos.get(SUPERPROJECT, []), root, root)
    if data.get("tool", {}).get("uv", {}).get("workspace", {}).get("members") != expected_root_members:
        problems.append(f"pyproject.toml: [tool.uv.workspace] members are out of date, {LOCK_HINT}")
    expected = expected_manifests(root, repos, data)
    for repo, members in sorted(repos.items()):
        problems += [f"{rel_posix(path, root)} is stale, {LOCK_HINT}" for path in stray_files(root, repo, members)]
        home = lock_home(repo, members)
        manifest = data
        if repo != SUPERPROJECT:
            path = root / home / "pyproject.toml"
            if not path.is_file() or path.read_text(encoding="utf-8") != expected[home]:
                problems.append(f"{manifest_label(home)} is out of date, {LOCK_HINT}")
                continue
            manifest = load_root(root / home)
        problems += lock_problems(root, home, [] if home in members else members, manifest)
    for member in compute_members(root):
        problems += check_member(member, root)
    return problems


def run_check(args: argparse.Namespace) -> int:
    problems = check(args.root)
    for problem in problems:
        print(problem)
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=pathlib.Path, default=argparse.SUPPRESS, help="repo root (default: parent of _meta)")
    parser = argparse.ArgumentParser(description=__doc__, parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)
    members = sub.add_parser("members", parents=[common], help="print computed workspace members")
    members.set_defaults(run=run_members)
    compose = sub.add_parser("compose", parents=[common], help="write the composed uv workspace pyproject.toml, pinned by the present repos' uv.lock files")
    add_out(compose)
    compose.set_defaults(run=run_compose)
    lock = sub.add_parser("lock", parents=[common], help="on a full tree, write repo manifests, lock the composed workspace and derive every repo's uv.lock from it")
    add_out(lock)
    lock.add_argument("--upgrade", action="store_true", help="drop all repo pins and the composed lock, re-resolve to the latest versions")
    lock.add_argument("--upgrade-package", action="append", default=[], metavar="NAME", help="drop the repo pin of NAME and let uv upgrade it (repeatable)")
    lock.set_defaults(run=run_lock)
    hook = sub.add_parser("hook", parents=[common], help="pre-commit: relock when the commit stages a pyproject.toml or setup.py, staging the committing repo's regenerated files")
    hook.set_defaults(run=run_hook)
    verify = sub.add_parser("check", parents=[common], help="validate repo manifests, repo locks and every member manifest, exit 1 on problems")
    verify.set_defaults(run=run_check)
    return parser


def add_out(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".uv-workspace"), metavar="DIR", help="composed workspace directory, relative to the root (default: .uv-workspace)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not root_given(args):
        args.root = pathlib.Path(__file__).resolve().parents[2]
    args.root = args.root.resolve()
    try:
        return args.run(args)
    except WorkspaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def root_given(args: argparse.Namespace) -> bool:
    return "root" in vars(args)


if __name__ == "__main__":
    sys.exit(main())
