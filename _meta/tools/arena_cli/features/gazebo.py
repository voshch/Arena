"""gazebo feature: ros_gz and OpenUSD tooling."""

import os
import sys

import common
from common import CLIError, Verb, make_verb

from features import default_install

SCRIPT_SHA256 = "be2a9c6b386b82e82670c6e6c9ced1b980cbdebc715b2ab0e6043d8c62ee80d2"

NAME = "gazebo"

DESCRIPTION = "Gazebo simulator (ros_gz + OpenUSD tooling)."


def _source_env() -> dict[str, str]:
    """Env overlay for the build subprocess, without mutating os.environ."""
    env = dict(os.environ)
    usd_path = os.path.join(common._env("ARENA_WS_DIR"), "tools", "OpenUSD", "install")
    env["USD_PATH"] = usd_path
    env["PATH"] = usd_path + "/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = usd_path + "/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["CMAKE_PREFIX_PATH"] = usd_path + ":" + env.get("CMAKE_PREFIX_PATH", "")
    formats = env.get("ARENA_MODELS_FORMATS", "")
    if "sdf" not in formats:
        env["ARENA_MODELS_FORMATS"] = f"{formats},sdf"
    return env


def _update() -> int:
    """Install ros_gz packages and build OpenUSD."""
    import subprocess

    env = _source_env()
    ws_dir = common._env("ARENA_WS_DIR")

    rc = subprocess.run(["sudo", "apt-get", "update"], cwd=ws_dir, env=env, check=False).returncode
    if rc:
        return rc

    distro = common._env("ARENA_ROS_DISTRO")
    rc = subprocess.run(
        [
            "sudo",
            "apt-get",
            "install",
            "-y",
            f"ros-{distro}-ros-gz-sim",
            f"ros-{distro}-ros-gz-bridge",
            f"ros-{distro}-ros-gz-image",
        ],
        cwd=ws_dir,
        env=env,
        check=False,
    ).returncode
    if rc:
        return rc

    print(f"Gazebo {env.get('GAZEBO_VERSION', '')}, ros_gz, sdformat_urdf installed successfully!")

    tools_dir = os.path.join(ws_dir, "tools")
    if not os.path.isdir(os.path.join(tools_dir, "OpenUSD")):
        print("Installing OpenUSD")
        os.makedirs(tools_dir, exist_ok=True)
        rc = subprocess.run(
            ["git", "clone", "--depth", "1", "-b", "v24.08", "https://github.com/PixarAnimationStudios/OpenUSD.git"],
            cwd=tools_dir,
            env=env,
            check=False,
        ).returncode
        if rc:
            return rc
        rc = subprocess.run(
            ["sudo", "apt-get", "install", "-y", "libpyside2-dev", "python3-opengl", "cmake", "libglu1-mesa-dev", "freeglut3-dev", "mesa-common-dev"],
            cwd=tools_dir,
            env=env,
            check=False,
        ).returncode
        if rc:
            return rc
        usd_src_dir = os.path.join(tools_dir, "OpenUSD")
        rc = subprocess.run(
            [
                "python3",
                "build_scripts/build_usd.py",
                "--build-variant",
                "release",
                "--no-tests",
                "--no-examples",
                "--no-imaging",
                "--onetbb",
                "--no-tutorials",
                "--no-docs",
                "--no-python",
                env["USD_PATH"],
            ],
            cwd=usd_src_dir,
            env=env,
            check=False,
        ).returncode
        if rc:
            return rc

    src_tools_dir = os.path.join(ws_dir, "src", "tools")
    if not os.path.isdir(os.path.join(src_tools_dir, "gz-usd")):
        print("Installing gz-usd")
        rc = subprocess.run(
            ["sudo", "apt-get", "install", "-y", "libgz-cmake4-dev", "libsdformat15-dev", "libgz-common6-dev"],
            cwd=ws_dir,
            env=env,
            check=False,
        ).returncode
        if rc:
            return rc
        os.makedirs(src_tools_dir, exist_ok=True)
        rc = subprocess.run(
            ["git", "clone", "-b", "main", "https://github.com/gazebosim/gz-usd"],
            cwd=src_tools_dir,
            env=env,
            check=False,
        ).returncode
        if rc:
            return rc
        build_env = dict(env)
        build_env["BASE_PATHS"] = "src/tools/gz-usd"
        rc = subprocess.run(
            [
                sys.executable,
                os.path.join(common._env("TOOLS_DIR"), "arena_cli", "__main__.py"),
                "build",
            ],
            cwd=ws_dir,
            env=build_env,
            check=False,
        ).returncode
        if rc:
            return rc
        print("Successfully installed gz-usd")

    return 0


def _require_installed() -> None:
    if not common._reg_has(NAME):
        raise CLIError(f"{NAME} is not installed, run 'arena feature {NAME} install' first")


def install(argv: list[str]) -> None:
    """Install the feature (pull repos, register, run its update)."""
    if argv:
        raise CLIError("unexpected arguments")
    sys.exit(default_install(NAME, _update))


def update(argv: list[str]) -> None:
    """Update the feature to the latest state."""
    if argv:
        raise CLIError("unexpected arguments")
    if not common._reg_has(NAME):
        raise CLIError(f"{NAME} is not installed, run 'arena feature {NAME} install' first")
    sys.exit(_update())


def uninstall(argv: list[str]) -> None:
    """Uninstall and unregister the feature."""
    if argv:
        raise CLIError("unexpected arguments")
    common._reg_remove(NAME)


def launch(argv: list[str]) -> None:
    """Launch gazebo."""
    _require_installed()
    common._exec("ros2", "launch", "arena_bringup", "gazebo.launch.py", *argv)


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        make_verb("install", install),
        make_verb("update", update),
        make_verb("uninstall", uninstall),
        make_verb("launch", launch, passthrough=True),
    ]
}
