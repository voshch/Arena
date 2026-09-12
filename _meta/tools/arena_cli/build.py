"""colcon build wrapper: package-skip and cmake-args logic for the workspace."""

import os
import sys
from typing import NamedTuple


class Workspace(NamedTuple):
    """Resolved colcon build layout."""

    base_paths: list[str]
    paths: list[str]
    build_base: str
    install_base: str


def workspace() -> Workspace:
    """Build layout from the environment. Assumes cwd is ARENA_WS_DIR."""
    import glob

    from common import _env

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


def resolve_packages(argv: list[str]) -> list[str]:
    """Package names a colcon selection argv resolves to."""
    import subprocess

    from common import CLIError

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
        if mtime > best:
            best = mtime
    return int(best)


def build_main(argv: list[str]) -> int:
    """Run colcon build with the workspace package-skip and cmake-args logic. Assumes cwd is ARENA_WS_DIR."""
    import subprocess

    from common import _env

    args = list(argv)

    skip_old_env = os.environ.get("SKIP_OLD")
    if skip_old_env:
        skip_old = skip_old_env
    else:
        skip_old = "" if any("--packages-" in a for a in args) else "1"

    ws = workspace()

    python_root = subprocess.run(["uv", "python", "find"], cwd=_env("ARENA_DIR"), capture_output=True, text=True, check=False).stdout.strip()
    cmake_arg_value = f"-DPython3_ROOT_DIR={python_root} -DBUILD_TESTING=OFF"

    display_args = ["--symlink-install", "--continue-on-error", f"--cmake-args '{cmake_arg_value}'"]
    exec_args = ["--symlink-install", "--continue-on-error", "--cmake-args", cmake_arg_value]

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
        listing = subprocess.run(["colcon", "list", "--base-paths", *ws.paths], capture_output=True, text=True, check=False)
        for line in listing.stdout.splitlines():
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
                build_packages.append(package)

        # an empty --packages-above is falsy to colcon and drops the filter entirely
        selection = ["--packages-above", *build_packages] if build_packages else ["--packages-select"]
        display_args += selection
        exec_args += selection

    print(f"BUILDING: colcon build {' '.join(display_args)} {' '.join(args)}")

    result = subprocess.run(["colcon", "build", *exec_args, *args], check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(build_main(sys.argv[1:]))
