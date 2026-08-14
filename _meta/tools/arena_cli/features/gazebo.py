"""gazebo feature: ros_gz and OpenUSD tooling."""

import os

import common
from common import Verb, make_verb

import features
from features import lifecycle_verbs, source_verb

NAME = "gazebo"

DESCRIPTION = "Gazebo simulator (ros_gz + OpenUSD tooling)."

_FORMATS_SOURCE = 'case "$ARENA_MODELS_FORMATS" in *sdf*) ;; *) export ARENA_MODELS_FORMATS="${ARENA_MODELS_FORMATS},sdf" ;; esac'

_HOST_SOURCE = (
    'export USD_PATH="$ARENA_WS_DIR/tools/OpenUSD/install"\n'
    'export PATH="$USD_PATH/bin:$PATH"\n'
    'export LD_LIBRARY_PATH="$USD_PATH/lib:$LD_LIBRARY_PATH"\n'
    'export CMAKE_PREFIX_PATH="$USD_PATH:$CMAKE_PREFIX_PATH"\n' + _FORMATS_SOURCE
)


def _shell_source() -> str:
    return _FORMATS_SOURCE if features.in_container() else _HOST_SOURCE


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
    """Install ros_gz packages and build OpenUSD (no-op in the container)."""
    import subprocess

    if features.in_container():
        return 0

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
        rc = common._cli("build", env={**env, "BASE_PATHS": "src/tools/gz-usd"})
        if rc:
            return rc
        print("Successfully installed gz-usd")

    return common._cli("build", env={**env, "BASE_PATHS": "src/gazebo"})


def launch(argv: list[str]) -> None:
    """Launch gazebo."""
    common._reg_require(NAME)
    common._exec("ros2", "launch", "arena_bringup", "gazebo.launch.py", *argv)


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        *lifecycle_verbs(NAME, _update),
        make_verb("launch", launch, passthrough=True),
        source_verb(_shell_source),
    ]
}
