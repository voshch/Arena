"""colcon build wrapper: package-skip and cmake-args logic for the workspace."""

import os
import sys


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
    import glob
    import subprocess

    from common import _env

    args = argv[1:]

    skip_old_env = os.environ.get("SKIP_OLD")
    if skip_old_env:
        skip_old = skip_old_env
    else:
        skip_old = "" if any("--packages-" in a for a in args) else "1"

    arena_ws_dir = _env("ARENA_WS_DIR")
    arena_dir = _env("ARENA_DIR")

    build_base = os.environ.get("BUILD_BASE") or os.path.join(arena_ws_dir, "build")
    install_base = os.environ.get("INSTALL_BASE") or os.path.join(arena_ws_dir, "install")

    base_paths_env = os.environ.get("BASE_PATHS")
    base_paths = base_paths_env.split(";") if base_paths_env else [os.path.join(arena_ws_dir, "src")]

    paths_env = os.environ.get("PATHS")
    if paths_env:
        paths = paths_env.split(";")
    elif os.environ.get("BUILD_ALL") == "1":
        paths = [os.path.join(arena_ws_dir, "src", "*")]
    else:
        paths = []
        for entry in sorted(glob.glob(os.path.join(arena_ws_dir, "src", "*") + "/")):
            name = os.path.basename(entry.rstrip("/"))
            if name in ("ros2", "tools"):
                continue
            paths.append(entry + "*")

    python_root = subprocess.run(["uv", "python", "find"], cwd=arena_dir, capture_output=True, text=True, check=False).stdout.strip()
    cmake_arg_value = f"-DPython3_ROOT_DIR={python_root} -DBUILD_TESTING=OFF"

    display_args = ["--symlink-install", "--continue-on-error", f"--cmake-args '{cmake_arg_value}'"]
    exec_args = ["--symlink-install", "--continue-on-error", "--cmake-args", cmake_arg_value]

    print(f"Using base paths: {' '.join(base_paths)}")
    display_args += ["--base-paths", *base_paths]
    exec_args += ["--base-paths", *base_paths]

    print(f"Using build base: {build_base}")
    display_args += ["--build-base", build_base]
    exec_args += ["--build-base", build_base]

    print(f"Using install base: {install_base}")
    display_args += ["--install-base", install_base]
    exec_args += ["--install-base", install_base]

    print(f"Building paths: {' '.join(paths)}")

    build_packages: list[str] = []

    if skip_old == "1":
        print(f"INDEXING: colcon list --base-paths {' '.join(paths)}")
        listing = subprocess.run(["colcon", "list", "--base-paths", *paths], capture_output=True, text=True, check=False)
        for line in listing.stdout.splitlines():
            if not line.strip():
                continue
            fields = line.split("\t")
            package = fields[0] if fields else ""
            src_path = fields[1] if len(fields) > 1 else ""

            rc_file = os.path.join(build_base, package, "colcon_build.rc")
            package_xml = os.path.join(install_base, package, "share", package, "package.xml")

            up_to_date = False
            if package and os.path.isfile(rc_file) and os.path.isfile(package_xml):
                try:
                    with open(rc_file) as f:
                        rc_value = f.read().strip()
                except OSError:
                    rc_value = ""
                if rc_value == "0":
                    up_to_date = _recursive_mtime(os.path.join(install_base, package)) >= _recursive_mtime(src_path)

            if not up_to_date:
                build_packages.append(package)

        display_args += ["--packages-select", *build_packages]
        exec_args += ["--packages-select", *build_packages]

    print(f"BUILDING: colcon build {' '.join(display_args)} {' '.join(args)}")

    result = subprocess.run(["colcon", "build", *exec_args, *args], check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(build_main(sys.argv))
