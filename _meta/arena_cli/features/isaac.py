"""isaac feature: NVIDIA Isaac Sim."""

import os
import sys

from arena_cli import common, features
from arena_cli.common import CLIError, Verb, make_verb
from arena_cli.complete import LaunchArgs
from arena_cli.features import lifecycle_verbs, source_verb

NAME = "isaac"
ISAAC_VERSION = "4.2.0"

DESCRIPTION = "NVIDIA Isaac Sim simulator."

BRIDGE_PACKAGES = ("isaacsim_msgs", "arena_people_msgs")
BRIDGE_DIR = ".isaac_bridge"

_FORMATS_SOURCE = 'case "$ARENA_MODELS_FORMATS" in *usd*) ;; *) export ARENA_MODELS_FORMATS="${ARENA_MODELS_FORMATS},usdz,usd,usda,usdc" ;; esac'

_HOST_SOURCE = f'if [ -z ${{ISAAC_PATH+x}} ] ; then\n    . "$HOME/isaacsim-{ISAAC_VERSION}/setup.sh"\nfi\n{_FORMATS_SOURCE}'

_SETUP_SH = f"""#!/bin/sh
MY_DIR="$HOME/isaacsim-{ISAAC_VERSION}"
export CARB_APP_PATH="$MY_DIR/kit"
export EXP_PATH="$MY_DIR/apps"
if [ -f "$MY_DIR/setup_python_env.sh" ] ; then
    . "$MY_DIR/setup_python_env.sh"
fi
export ISAAC_PATH="$MY_DIR"
"""


def _shell_source() -> str:
    return _FORMATS_SOURCE if features.in_container() else _HOST_SOURCE


def _update_host() -> int:
    """Download Isaac Sim, install python deps, write config, init arena_isaac."""
    import shutil
    import subprocess

    rc = subprocess.run(["sudo", "apt", "install", "libfuse2"], check=False).returncode
    if rc:
        return rc

    install_dir = os.path.expanduser(f"~/isaacsim-{ISAAC_VERSION}")
    if not os.path.isdir(install_dir):
        url = f"https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%40{ISAAC_VERSION}-rc.18%2Brelease.16044.3b2ed111.gl.linux-x86_64.release.zip"
        rc = subprocess.run(["wget", "--content-disposition", url, "-O", "isaac-sim.zip"], check=False).returncode
        if rc:
            return rc
        os.makedirs(install_dir, exist_ok=True)
        rc = subprocess.run(["unzip", "isaac-sim.zip", "-d", install_dir], check=False).returncode
        if rc:
            return rc
        os.remove("isaac-sim.zip")

    while shutil.which("nvidia-smi") is None:
        print("Warning: nvidia-smi command not found. Please install nvidia driver using")
        print("sudo apt-get install nvidia-open")
        input("Confirm installation by pressing [Enter]")
    print("Successfully detected NVIDIA driver installation")

    print("nvidia-driver was installed")

    rc = subprocess.run(["python", "-m", "pip", "install", "torch==2.4.0"], check=False).returncode
    if rc:
        return rc
    rc = subprocess.run(["python", "-m", "pip", "install", "--upgrade", "pip"], check=False).returncode
    if rc:
        return rc
    rc = subprocess.run(["python", "-m", "pip", "install", "typeguard"], check=False).returncode
    if rc:
        return rc

    virt = subprocess.run(["systemd-detect-virt"], capture_output=True, text=True, check=False).stdout.strip()
    if virt == "wsl":
        rc = subprocess.run(["python", "-m", "pip", "install", "git+https://github.com/cpbotha/xdg-open-wsl.git"], check=False).returncode
        if rc:
            return rc

    setup_file = os.path.expanduser(f"~/isaacsim-{ISAAC_VERSION}/setup.sh")
    with open(setup_file, "w") as f:
        f.write(_SETUP_SH)

    print("Completed Isaac Sim download")

    probe = subprocess.run(
        ["python3", "-c", 'import site, os; print(os.path.join(site.getsitepackages()[0], "omni", "EULA_ACCEPTED"))'],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode:
        return probe.returncode
    eula_target = probe.stdout.strip()
    if eula_target:
        os.makedirs(os.path.dirname(eula_target), exist_ok=True)
        try:
            with open(eula_target, "w") as f:
                f.write("yes\n")
        except OSError:
            print("failed to auto-accept EULA, continuing")

    rc = subprocess.run(["git", "submodule", "update", "--init", "arena_isaac"], cwd=common._env("ARENA_DIR"), check=False).returncode
    if rc:
        return rc

    return 0


def _update_container() -> int:
    """Pull arena_isaac and build the isaac compose service."""
    import subprocess

    print(f"Updating {NAME}...")
    rc = subprocess.run(["git", "submodule", "update", "--init", "--rebase", "arena_isaac"], cwd=common._env("ARENA_DIR"), check=False).returncode
    if rc:
        return rc
    rc = features.build(["isaac"])
    if rc:
        return rc
    return features.compose(["up", "-d", "--remove-orphans", "--no-start", "isaac"])


def launch_host(argv: list[str]) -> None:
    """Launch Isaac Sim."""
    common._reg_require(NAME)
    common._exec("ros2", "launch", "arena_isaac", "run_isaacsim.launch.py", *argv)


def _build_bridge() -> int:
    """Build the bridge message packages without symlinks into `install/.isaac_bridge`, which the isaac container mounts."""
    import subprocess

    from arena_cli import build

    ws = build.workspace()
    python = build.build_python()
    print(f"isaac bridge: building {', '.join(BRIDGE_PACKAGES)} into {ws.install_base}/{BRIDGE_DIR}", flush=True)
    done = subprocess.run(
        [
            "colcon",
            "build",
            "--base-paths",
            *ws.base_paths,
            "--packages-select",
            *BRIDGE_PACKAGES,
            "--build-base",
            os.path.join(ws.build_base, BRIDGE_DIR),
            "--install-base",
            os.path.join(ws.install_base, BRIDGE_DIR),
            *(["--cmake-args", f"-DPython3_EXECUTABLE={python}"] if python else []),
        ],
        cwd=common._env("ARENA_WS_DIR"),
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        print(done.stdout + done.stderr, file=sys.stderr)
        print("isaac bridge: building the bridge messages failed, Isaac Sim is not started", file=sys.stderr)
    return done.returncode


def _arena_ros_versions() -> str:
    """This container's rclpy, rcl and Fast DDS versions as the JSON the isaac side compares against."""
    import subprocess

    done = subprocess.run([sys.executable, "-m", "arena_isaac.bridge_check", "versions"], capture_output=True, text=True, check=False)
    if done.returncode:
        raise CLIError(f"reading the arena ROS versions failed: {done.stderr.strip()}")
    return done.stdout.strip()


def launch_container(argv: list[str]) -> None:
    """Launch Isaac Sim in its container, on bridge messages built from this workspace."""
    import signal

    common._reg_require(NAME)
    rc = _build_bridge()
    if rc:
        sys.exit(rc)
    versions = _arena_ros_versions()
    rc = features.compose(["up", "-d", "--remove-orphans", "isaac"])
    if rc:
        sys.exit(rc)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))
    try:
        rc = features.compose(
            [
                "exec",
                "-T",
                "isaac",
                "bash",
                "-c",
                f'source /opt/arena_ws/install/local_setup.bash && source /opt/arena_ws/install/{BRIDGE_DIR}/local_setup.bash && ARENA_ROS_VERSIONS="$1" python3 /opt/arena_ws/build/arena_isaac/arena_isaac/bridge_check.py check && python3 /opt/arena_ws/build/arena_isaac/arena_isaac/run_isaacsim.py "${{@:2}}"',
                "--",
                versions,
                *argv,
            ]
        )
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        features.compose(["stop", "isaac"])
    sys.exit(rc)


def exec_(argv: list[str]) -> None:
    """Run a command inside the isaac container."""
    sys.exit(features.compose(["exec", "isaac", *argv]))


def uninstall_container(argv: list[str]) -> None:
    """Uninstall and unregister the feature."""
    import subprocess

    if argv:
        raise CLIError("unexpected arguments")
    features.compose(["stop", "isaac"])
    features.remove("isaac")
    subprocess.run(["git", "submodule", "deinit", "-f", "arena_isaac"], cwd=common._env("ARENA_DIR"), check=False)
    common._reg_remove(NAME)


def health() -> features.Health:
    return features.fetched(NAME, "arena_isaac")


if features.in_container():
    COMMANDS: dict[str, Verb] = {
        v.name: v
        for v in [
            *lifecycle_verbs(NAME, _update_container),
            make_verb("uninstall", uninstall_container),
            make_verb("launch", launch_container, passthrough=True, complete=LaunchArgs("arena_isaac", "run_isaacsim.launch.py")),
            make_verb("exec", exec_, passthrough=True),
            source_verb(_shell_source),
        ]
    }
else:
    COMMANDS = {
        v.name: v
        for v in [
            *lifecycle_verbs(NAME, _update_host, deinit="arena_isaac"),
            make_verb("launch", launch_host, passthrough=True, complete=LaunchArgs("arena_isaac", "run_isaacsim.launch.py")),
            source_verb(_shell_source),
        ]
    }
