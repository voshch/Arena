"""Workspace updater: pull repos/submodules/features, refresh rosdep and python deps."""

import os
import sys


def pull_main(argv: list[str]) -> int:
    """Pull Arena repos/submodules/features and refresh rosdep and python deps. Chdirs to ARENA_DIR for the duration."""
    import shutil
    import subprocess

    import features
    from common import _cli, _env, _reg_list, _reg_pull

    arena_dir = _env("ARENA_DIR")
    arena_ws_dir = _env("ARENA_WS_DIR")

    do_python = os.environ.get("PYTHON", "1") == "1"
    do_git = os.environ.get("GIT", "1") == "1"
    do_rosdep = os.environ.get("ROSDEP", "1") == "1"

    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["ROSDEP_EXCLUDES"] = "libignition-gazebo6-dev gazebo_dev gazebo_ros gazebo_plugins gazebo_ros2_control flir_ptu_description"

    prev_cwd = os.getcwd()
    os.chdir(arena_dir)
    try:
        rc = subprocess.run(["sudo", "apt", "update"], env=env, check=False).returncode
        if rc:
            return rc

        if do_git:
            print("updating Arena...")
            has_upstream = subprocess.run(["git", "rev-parse", "--verify", "-q", "@{u}"], env=env, check=False, capture_output=True).returncode == 0
            if has_upstream:
                rc = subprocess.run(["git", "pull", "--ff-only", "--autostash"], env=env, check=False).returncode
                if rc:
                    return rc
            else:
                print("no upstream for current branch, skipping Arena pull")

            if subprocess.run(["git", "submodule", "update", "--init", "--checkout", "arena_planners", "arena_robots", "humansim"], env=env, check=False).returncode:
                print("failed to init/update arena_planners/arena_robots/humansim, ignoring")

            if subprocess.run(["git", "submodule", "update", "--checkout", "--recursive"], env=env, check=False).returncode:
                print("submodule checkout had issues, resolve manually")

            foreach_script = (
                'branch=$(git config -f "$toplevel/.gitmodules" "submodule.$name.branch" 2>/dev/null || true)\n'
                'if [ -n "$branch" ]; then\n'
                '    git switch -C "$branch" HEAD || echo "  $name: could not reset branch $branch"\n'
                "fi\n"
            )
            if subprocess.run(["git", "submodule", "foreach", "--recursive", foreach_script], env=env, check=False).returncode:
                print("submodule branch reset had issues, ignoring")

            repos_file = os.path.join(arena_dir, "_meta", "repos", "arena.repos")
            ws_src = os.path.join(arena_ws_dir, "src")
            if subprocess.run(["vcs", "import", "--input", repos_file, "--recursive", "--ff", "--add-existing", ws_src], env=env, check=False).returncode:
                print("failed to pull all arena repos, ignoring")

            deps_dir = os.path.join(ws_src, "deps")
            if not os.path.isdir(deps_dir) or not os.listdir(deps_dir):
                print(f"no repos imported into {deps_dir} (vcs: {shutil.which('vcs')}), the workspace cannot build", file=sys.stderr)
                return 1

            for name in _reg_list():
                if features.load(name) is None:
                    continue
                _reg_pull(name)
                if _cli("feature", name, "update"):
                    print(f"failed to update feature {name}, ignoring", file=sys.stderr)

        if do_rosdep:
            rosdep_ok = subprocess.run(["rosdep", "update", "--rosdistro", _env("ARENA_ROS_DISTRO")], env=env, check=False).returncode == 0
            deps_ok = rosdep_ok and subprocess.run(
                ["rosdep", "install", "--ignore-src", "-r", "-y", "--rosdistro", _env("ARENA_ROS_DISTRO"), "--from-paths", "src", "--skip-keys", env["ROSDEP_EXCLUDES"]],
                env=env,
                cwd=arena_ws_dir,
                check=False,
            ).returncode == 0
            if not deps_ok:
                print("rosdep failed to install all dependencies")

        if do_python:
            print("updating python deps...")
            rc = subprocess.run(["uv", "sync", "--inexact", "--active"], env=env, check=False).returncode
            if rc:
                return rc

        print("Updating links...")
        rc = subprocess.run([os.path.join(arena_dir, "_meta", "tools", "create_links")], env=env, check=False).returncode
        if rc:
            return rc

        print("\033[0;31m")
        print("don't forget to rebuild!")
        print("\033[0m")

        return 0
    finally:
        os.chdir(prev_cwd)


if __name__ == "__main__":
    sys.exit(pull_main(sys.argv))
