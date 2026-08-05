"""isaac feature: NVIDIA Isaac Sim."""

import os
import sys

import common
from common import CLIError, Verb, make_verb

from features import default_install

SCRIPT_SHA256 = "5010ed764da2aced24169dc59e3c2f82b26a1e166e549924bc8f64c9023dfba8"

NAME = "isaac"
ISAAC_VERSION = "4.2.0"

DESCRIPTION = "NVIDIA Isaac Sim simulator."

_FASTDDS_XML = """<?xml version="1.0" encoding="UTF-8" ?>

    <license>Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
    NVIDIA CORPORATION and its licensors retain all intellectual property
    and proprietary rights in and to this software, related documentation
    and any modifications thereto.  Any use, reproduction, disclosure or
    distribution of this software and related documentation without an express
    license agreement from NVIDIA CORPORATION is strictly prohibited.</license>


    <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles" >
        <transport_descriptors>
            <transport_descriptor>
                <transport_id>UdpTransport</transport_id>
                <type>UDPv4</type>
            </transport_descriptor>
        </transport_descriptors>

        <participant profile_name="udp_transport_profile" is_default_profile="true">
            <rtps>
                <userTransports>
                    <transport_id>UdpTransport</transport_id>
                </userTransports>
                <useBuiltinTransports>false</useBuiltinTransports>
            </rtps>
        </participant>
    </profiles>
"""

_SETUP_SH = f"""#!/bin/sh
MY_DIR="$HOME/isaacsim-{ISAAC_VERSION}"
export CARB_APP_PATH="$MY_DIR/kit"
export EXP_PATH="$MY_DIR/apps"
if [ -f "$MY_DIR/setup_python_env.sh" ] ; then
    . "$MY_DIR/setup_python_env.sh"
fi
export ISAAC_PATH="$MY_DIR"
"""


def _update() -> int:
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

    fastdds_path = os.path.expanduser("~/.ros/fastdds.xml")
    rc = subprocess.run(["touch", fastdds_path], check=False).returncode
    if rc:
        return rc
    with open(fastdds_path, "w") as f:
        f.write(_FASTDDS_XML)

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
    import subprocess

    subprocess.run(["git", "submodule", "deinit", "-f", "arena_isaac"], cwd=common._env("ARENA_DIR"), check=False)
    common._reg_remove(NAME)


def launch(argv: list[str]) -> None:
    """Launch Isaac Sim."""
    _require_installed()
    common._exec("ros2", "launch", "arena_isaac", "run_isaacsim.launch.py", *argv)


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        make_verb("install", install),
        make_verb("update", update),
        make_verb("uninstall", uninstall),
        make_verb("launch", launch, passthrough=True),
    ]
}
