"""colcon build wrapper: package-skip and cmake-args logic for the workspace."""

import glob
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from typing import NamedTuple


class Workspace(NamedTuple):
    """Resolved colcon build layout."""

    base_paths: list[str]
    paths: list[str]
    build_base: str
    install_base: str


def workspace() -> Workspace:
    """Build layout from the environment. Assumes cwd is ARENA_WS_DIR."""
    from arena_cli.common import _env

    arena_ws_dir = _env("ARENA_WS_DIR")

    base_paths_env = os.environ.get("BASE_PATHS")
    base_paths = base_paths_env.split(";") if base_paths_env else [os.path.join(arena_ws_dir, "src")]

    paths_env = os.environ.get("PATHS")
    if paths_env:
        paths = paths_env.split(";")
    elif base_paths_env:
        # index what the build will see, or --packages-above rejects unknown names
        paths = list(base_paths)
    elif os.environ.get("BUILD_ALL") == "1":
        paths = [os.path.join(arena_ws_dir, "src", "*")]
    else:
        paths = []
        for entry in sorted(glob.glob(os.path.join(arena_ws_dir, "src", "*") + "/")):
            name = os.path.basename(entry.rstrip("/"))
            if name in ("ros2", "tools"):
                continue
            paths.append(entry + "*")

    return Workspace(
        base_paths,
        paths,
        os.environ.get("BUILD_BASE") or os.path.join(arena_ws_dir, "build"),
        os.environ.get("INSTALL_BASE") or os.path.join(arena_ws_dir, "install"),
    )


def build_python() -> str:
    """The interpreter colcon builds Python extensions for, empty when uv finds none."""
    import subprocess

    from arena_cli.common import _env

    return subprocess.run(["uv", "python", "find"], cwd=_env("ARENA_DIR"), capture_output=True, text=True, check=False).stdout.strip()


def resolve_packages(argv: list[str]) -> list[str]:
    """Package names a colcon selection argv resolves to."""
    import subprocess

    from arena_cli.common import CLIError

    listing = subprocess.run(
        ["colcon", "list", "--names-only", "--base-paths", *workspace().base_paths, *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode:
        raise CLIError("colcon list rejected the arguments")
    return [line.strip() for line in listing.stdout.splitlines() if line.strip()]


def _recursive_mtime(path: str) -> int:
    if not os.path.exists(path):
        return -1
    entries = [path]
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
        for name in dirs + files:
            entries.append(os.path.join(root, name))
    best = -1.0
    for entry in entries:
        if os.path.islink(entry):
            continue
        try:
            mtime = os.stat(entry).st_mtime
        except OSError:
            continue
        if os.path.isdir(entry) and _pycache_mtime(entry) >= mtime:
            continue
        if mtime > best:
            best = mtime
    return int(best)


def _pycache_mtime(directory: str) -> float:
    try:
        return os.stat(os.path.join(directory, "__pycache__")).st_mtime
    except OSError:
        return -1.0


def _changed_build_type(build_dir: str, build_type: str) -> bool:
    """Whether build_dir holds the leftovers of the other build type, whose installed symlinks block a symlink-install."""
    if build_type.rsplit(".", 1)[-1] in ("ament_python", "python"):
        return os.path.isfile(os.path.join(build_dir, "CMakeCache.txt"))
    return os.path.isfile(os.path.join(build_dir, "colcon_command_prefix_setup_py.sh"))


def _stale(listing: str, ws: Workspace) -> list[tuple[str, str]]:
    """The (package, build type) rows of a `colcon list` output whose last build failed or whose install tree is older than their source."""
    stale: list[tuple[str, str]] = []
    for line in listing.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        package = fields[0] if fields else ""
        src_path = fields[1] if len(fields) > 1 else ""

        rc_file = os.path.join(ws.build_base, package, "colcon_build.rc")
        package_xml = os.path.join(ws.install_base, package, "share", package, "package.xml")

        up_to_date = False
        if package and os.path.isfile(rc_file) and os.path.isfile(package_xml):
            try:
                with open(rc_file) as f:
                    rc_value = f.read().strip()
            except OSError:
                rc_value = ""
            if rc_value == "0":
                up_to_date = _recursive_mtime(os.path.join(ws.install_base, package)) >= _recursive_mtime(src_path)

        if not up_to_date:
            stale.append((package, fields[2].strip("()") if len(fields) > 2 else ""))
    return stale


def _list_packages(ws: Workspace) -> str:
    import subprocess

    return subprocess.run(["colcon", "list", "--base-paths", *ws.paths], capture_output=True, text=True, check=False).stdout


def stale_packages(ws: Workspace) -> list[tuple[str, str]]:
    """Packages of the workspace that `arena build` would rebuild, with their build type."""
    return _stale(_list_packages(ws), ws)


LAUNCH_ROOTS = ("arena_bringup", "arena_runtime", "task_generator")
OPTIONAL_LAUNCH_ROOTS = ("flatsim", "arena_auditory", "arena_hearing")
DEPENDENCY_TAGS = ("depend", "build_depend", "build_export_depend", "buildtool_depend", "exec_depend")
ALLOW_STALE_ENV = "ARENA_ALLOW_STALE"


def _interface_rows(listing: str, roots: Iterable[str]) -> str:
    """The rows of a `colcon list` output that generate interfaces the roots depend on, themselves included."""
    rows: dict[str, str] = {}
    manifests: dict[str, ET.Element] = {}
    for line in listing.splitlines():
        package, path = (line.split("\t") + [""])[:2]
        if not path:
            continue
        try:
            manifests[package] = ET.parse(os.path.join(path, "package.xml")).getroot()
        except (OSError, ET.ParseError):
            continue
        rows[package] = line
    needed: set[str] = set()
    queue = list(roots)
    while queue:
        package = queue.pop()
        if package in needed or package not in manifests:
            continue
        needed.add(package)
        queue += [(e.text or "").strip() for tag in DEPENDENCY_TAGS for e in manifests[package].iter(tag)]
    generates = {package for package in needed if any((e.text or "").strip() == "rosidl_interface_packages" for e in manifests[package].iter("member_of_group"))}
    return "\n".join(rows[package] for package in sorted(generates))


def stale_interfaces(ws: Workspace, listing: str, roots: Iterable[str]) -> list[str]:
    """The interface packages the roots and the installed optional roots depend on that `arena build` would rebuild."""
    optional = [root for root in OPTIONAL_LAUNCH_ROOTS if os.path.isdir(os.path.join(ws.install_base, root))]
    return [package for package, _ in _stale(_interface_rows(listing, [*roots, *optional]), ws)]


def require_built_interfaces(*roots: str) -> None:
    """Refuse to start while an interface package the launch or the extra roots need is stale."""
    from arena_cli.common import CLIError

    ws = workspace()
    try:
        listing = _list_packages(ws)
    except OSError:
        listing = ""
    if not listing:
        print("warning: colcon lists no packages, launching without the stale interface check", file=sys.stderr)
        return
    stale = stale_interfaces(ws, listing, [*LAUNCH_ROOTS, *roots])
    if not stale:
        return
    names = " ".join(stale)
    if os.environ.get(ALLOW_STALE_ENV) == "1":
        print(f"warning: launching with stale interface packages: {names}", file=sys.stderr)
        return
    raise CLIError(f"stale interface packages: {names}. Their built message code is missing or older than their sources, so nodes can fail on import or exchange wrong messages.\nStop any running launch and run 'arena build {names}' to rebuild them and their dependents, or set {ALLOW_STALE_ENV}=1 to launch anyway.")


def _selects_packages(args: list[str]) -> bool:
    """Whether a colcon argv picks its own packages instead of the stale ones, skip and ignore filters pick none."""
    return any(a.startswith("--packages-") and not a.startswith(("--packages-skip", "--packages-ignore")) for a in args)


def build_main(argv: list[str]) -> int:
    """Run colcon build with the workspace package-skip and cmake-args logic, then collect the venv store. Assumes cwd is ARENA_WS_DIR."""
    import subprocess
    from pathlib import Path

    from arena_cli import venv_store
    from arena_cli.common import _env

    args = list(argv)

    skip_old_env = os.environ.get("SKIP_OLD")
    if skip_old_env:
        skip_old = skip_old_env
    else:
        skip_old = "" if _selects_packages(args) else "1"

    ws = workspace()

    python = build_python()
    cmake_args = [f"-DPython3_EXECUTABLE={python}"] if python else []

    display_args = ["--symlink-install", "--continue-on-error", "--cmake-args", *cmake_args]
    exec_args = ["--symlink-install", "--continue-on-error", "--cmake-args", *cmake_args]

    print(f"Using base paths: {' '.join(ws.base_paths)}")
    display_args += ["--base-paths", *ws.base_paths]
    exec_args += ["--base-paths", *ws.base_paths]

    print(f"Using build base: {ws.build_base}")
    display_args += ["--build-base", ws.build_base]
    exec_args += ["--build-base", ws.build_base]

    print(f"Using install base: {ws.install_base}")
    display_args += ["--install-base", ws.install_base]
    exec_args += ["--install-base", ws.install_base]

    print(f"Building paths: {' '.join(ws.paths)}")

    build_packages: list[str] = []

    if skip_old == "1":
        print(f"INDEXING: colcon list --base-paths {' '.join(ws.paths)}")
        for package, build_type in stale_packages(ws):
            build_packages.append(package)
            if package and _changed_build_type(os.path.join(ws.build_base, package), build_type):
                for tree in (os.path.join(ws.build_base, package), os.path.join(ws.install_base, package)):
                    print(f"CLEANING: {package} changed its build type to {build_type}, rm -rf {tree}", flush=True)
                    shutil.rmtree(tree, ignore_errors=True)

        if not build_packages:
            print("BUILDING: nothing stale, every package is up to date")
            return 0
        display_args += ["--packages-above", *build_packages]
        exec_args += ["--packages-above", *build_packages]

    print(f"BUILDING: colcon build {' '.join(display_args)} {' '.join(args)}")

    result = subprocess.run(["colcon", "build", *exec_args, *args], check=False)
    for name in venv_store.gc(Path(_env("ARENA_WS_DIR"))):
        print(f"venv store: {name[:12]} deleted, no build uses it", flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(build_main(sys.argv[1:]))
