"""isaac feature: NVIDIA Isaac Sim."""

import os
import sys

import common
from common import CLIError, Verb, make_verb
from complete import LaunchArgs

import features
from features import lifecycle_verbs, source_verb

NAME = "isaac"
ISAAC_VERSION = "4.2.0"

DESCRIPTION = "NVIDIA Isaac Sim simulator."

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
    rc = features.compose(["build", "isaac"])
    if rc:
        return rc
    return features.compose(["up", "-d", "--remove-orphans", "--no-start", "isaac"])


def launch_host(argv: list[str]) -> None:
    """Launch Isaac Sim."""
    common._reg_require(NAME)
    common._exec("ros2", "launch", "arena_isaac", "run_isaacsim.launch.py", *argv)


def launch_container(argv: list[str]) -> None:
    """Launch Isaac Sim in its container."""
    import signal

    common._reg_require(NAME)
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
                'source /opt/arena_ws/install/local_setup.bash && python3-arena /opt/arena_ws/build/arena_isaac/arena_isaac/run_isaacsim.py "$@"',
                "--",
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
    features.compose(["rm", "-fs", "isaac"])
    subprocess.run(["git", "submodule", "deinit", "-f", "arena_isaac"], cwd=common._env("ARENA_DIR"), check=False)
    common._reg_remove(NAME)


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
