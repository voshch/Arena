"""Workspace updater: pull repos/submodules/features, apply patches, refresh rosdep and python deps."""

import os
import sys


def restore_branches(root: str, env: dict[str, str]) -> bool:
    """Put every pinned submodule back on its .gitmodules branch, printing each one that keeps unpushed commits instead."""
    import subprocess

    listing = 'printf "%s\\t%s\\t%s\\n" "$displaypath" "$PWD" "$(git config -f "$toplevel/.gitmodules" "submodule.$name.branch")"'
    proc = subprocess.run(["git", "submodule", "foreach", "--quiet", "--recursive", listing], cwd=root, env=env, capture_output=True, text=True, check=False)
    ok = proc.returncode == 0
    for line in proc.stdout.splitlines():
        name, path, branch = line.split("\t")
        if not branch:
            continue
        note, attached = attach_branch(path, branch, env)
        ok = ok and attached
        if note:
            print(f"  {name}: {note}")
    return ok


def attach_branch(repo: str, branch: str, env: dict[str, str]) -> tuple[str, bool]:
    """Move branch onto the checked-out pin, or switch to it when it holds unpushed commits the pin lacks."""
    import subprocess

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-C", repo, *args], env=env, capture_output=True, text=True, check=False)

    if git("rev-parse", "-q", "--verify", f"refs/heads/{branch}").returncode == 0:
        unpushed = int(git("rev-list", "--count", f"refs/heads/{branch}", "--not", "HEAD", "--remotes").stdout)
        if unpushed:
            if git("merge-base", "--is-ancestor", "HEAD", f"refs/heads/{branch}").returncode:
                pin = git("rev-parse", "--short", "HEAD").stdout.strip()
                return f"left detached at the pin {pin}, {branch} has {unpushed} unpushed commits not in it, rebase them onto it", True
            switched = git("switch", branch)
            if switched.returncode:
                return f"could not switch to {branch} ({unpushed} unpushed commits ahead of the pin): {switched.stderr.strip()}", False
            return f"kept {branch}, {unpushed} unpushed commits ahead of the pin", True
    reset = git("switch", "-C", branch, "HEAD")
    if reset.returncode:
        return f"could not reset branch {branch}: {reset.stderr.strip()}", False
    return "", True


def apply_patches(arena_dir: str, env: dict[str, str]) -> list[str]:
    """Run every _meta/patches script that has no .done marker beside it, returning the names that failed."""
    import subprocess
    from pathlib import Path

    patches_dir = os.path.join(arena_dir, "_meta", "patches")
    if not os.path.isdir(patches_dir):
        return []
    try:
        names = sorted(os.listdir(patches_dir))
    except OSError as e:
        print(f"patches: {e}", file=sys.stderr)
        return ["_meta/patches"]
    failed: list[str] = []
    for name in names:
        path = os.path.join(patches_dir, name)
        if name.startswith(".") or name.endswith((".done", ".md")) or not os.path.isfile(path) or os.path.exists(path + ".done"):
            continue
        print(f"applying patch {name}...", flush=True)
        try:
            rc = subprocess.run([path], cwd=arena_dir, env=env, stdin=subprocess.DEVNULL, check=False).returncode
            if rc == 0:
                Path(path + ".done").touch()
        except OSError as e:
            print(f"patch {name}: {e}", file=sys.stderr)
            rc = 1
        if rc:
            failed.append(name)
    return failed


def restart_command(argv: list[str], skipped: list[str]) -> tuple[list[str], dict[str, str]]:
    """Command and environment that rerun this update on the freshly pulled code, carrying the steps skipped so far."""
    return [sys.executable, "-m", "arena_cli", "update", *argv], {**os.environ, "ARENA_UPDATE_RESTARTED": ",".join(skipped)}


def restarted_skips() -> list[str] | None:
    """Steps the run before a restart skipped, None when this run is not a restart."""
    carried = os.environ.pop("ARENA_UPDATE_RESTARTED", None)
    return None if carried is None else [step for step in carried.split(",") if step]


def pull_main(argv: list[str]) -> int:
    """Pull Arena repos/submodules/features, apply patches and refresh rosdep and python deps. Chdirs to ARENA_DIR for the duration."""
    import shutil
    import subprocess

    from arena_cli import features
    from arena_cli.common import _cli, _env, _git_ssh_command, _reg_list, _reg_pull

    arena_dir = _env("ARENA_DIR")
    arena_ws_dir = _env("ARENA_WS_DIR")

    do_python = os.environ.get("PYTHON", "1") == "1"
    do_git = os.environ.get("GIT", "1") == "1"
    do_rosdep = os.environ.get("ROSDEP", "1") == "1"

    carried = restarted_skips()
    os.environ.setdefault("GIT_SSH_COMMAND", _git_ssh_command())
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["ROSDEP_EXCLUDES"] = "libignition-gazebo6-dev gazebo_dev gazebo_ros gazebo_plugins gazebo_ros2_control flir_ptu_description"

    skipped: list[str] = carried or []
    prev_cwd = os.getcwd()
    os.chdir(arena_dir)
    try:
        if carried is None and subprocess.run(["sudo", "apt", "update"], env=env, check=False).returncode:
            print("apt update failed, continuing with stale package lists", file=sys.stderr)
            skipped.append("apt update")

        if do_git:
            print("updating Arena...")
            has_upstream = subprocess.run(["git", "rev-parse", "--verify", "-q", "@{u}"], env=env, check=False, capture_output=True).returncode == 0
            if has_upstream:
                before = subprocess.run(["git", "rev-parse", "HEAD"], env=env, capture_output=True, text=True, check=False).stdout
                if subprocess.run(["git", "pull", "--ff-only", "--autostash"], env=env, check=False).returncode:
                    print("Arena pull failed, continuing with current checkout", file=sys.stderr)
                    skipped.append("Arena pull")
                if carried is None and subprocess.run(["git", "rev-parse", "HEAD"], env=env, capture_output=True, text=True, check=False).stdout != before:
                    print("Arena moved, rerunning the update on the pulled code...", flush=True)
                    sys.stderr.flush()
                    cmd, restart_env = restart_command(argv, skipped)
                    os.execvpe(cmd[0], cmd, restart_env)
            else:
                print("no upstream for current branch, skipping Arena pull")

            if subprocess.run(["git", "submodule", "update", "--init", "--checkout", "arena_assets", "arena_planners", "arena_robots", "humansim"], env=env, check=False).returncode:
                print("failed to init/update arena_assets/arena_planners/arena_robots/humansim, ignoring")
                skipped.append("core submodules")

            if subprocess.run(["git", "submodule", "update", "--checkout", "--recursive"], env=env, check=False).returncode:
                print("submodule checkout had issues, resolve manually")
                skipped.append("recursive submodules")

            if not restore_branches(arena_dir, env):
                print("submodule branch reset had issues, ignoring")

        skipped.extend(f"patch {name}" for name in apply_patches(arena_dir, env))

        if do_git:
            repos_file = os.path.join(arena_dir, "_meta", "repos", "arena.repos")
            ws_src = os.path.join(arena_ws_dir, "src")
            if subprocess.run(["vcs", "import", "--input", repos_file, "--recursive", "--ff", "--add-existing", ws_src], env=env, check=False).returncode:
                print("failed to pull all arena repos, ignoring")
                skipped.append("arena.repos")

            deps_dir = os.path.join(ws_src, "deps")
            if not os.path.isdir(deps_dir) or not os.listdir(deps_dir):
                print(f"no repos imported into {deps_dir} (vcs: {shutil.which('vcs')}), the workspace cannot build", file=sys.stderr)
                return 1

            for name in _reg_list():
                if features.load(name) is None:
                    continue
                if _reg_pull(name):
                    skipped.append(f"{name}.repos")
                if _cli("feature", name, "update"):
                    print(f"failed to update feature {name}, ignoring", file=sys.stderr)
                    skipped.append(f"feature {name}")

        if do_rosdep:
            rosdep_ok = subprocess.run(["rosdep", "update", "--rosdistro", _env("ARENA_ROS_DISTRO")], env=env, check=False).returncode == 0
            deps_ok = (
                rosdep_ok
                and subprocess.run(
                    ["rosdep", "install", "--ignore-src", "-r", "-y", "--rosdistro", _env("ARENA_ROS_DISTRO"), "--from-paths", "src", "--skip-keys", env["ROSDEP_EXCLUDES"]],
                    env=env,
                    cwd=arena_ws_dir,
                    check=False,
                ).returncode
                == 0
            )
            if not deps_ok:
                print("rosdep failed to install all dependencies")
                skipped.append("rosdep")

        if do_python:
            print("updating python deps...")
            rc = subprocess.run([sys.executable, os.path.join(arena_dir, "_meta", "tools", "uv_workspace.py"), "compose"], env=env, check=False).returncode
            if rc:
                return rc
            rc = subprocess.run(["uv", "sync", "--inexact", "--all-packages", "--active", "--project", os.path.join(arena_dir, ".uv-workspace")], env=env, check=False).returncode
            if rc:
                return rc

        print("Updating links...")
        rc = subprocess.run([os.path.join(arena_dir, "_meta", "tools", "create_links")], env=env, check=False).returncode
        if rc:
            return rc

        if skipped:
            print("\033[0;33m")
            print("update incomplete, skipped: " + ", ".join(skipped))
            print("\033[0m")

        print("\033[0;31m")
        print("don't forget to rebuild!")
        print("\033[0m")

        return 0
    finally:
        os.chdir(prev_cwd)


if __name__ == "__main__":
    sys.exit(pull_main(sys.argv))
