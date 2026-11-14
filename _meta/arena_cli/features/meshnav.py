"""meshnav feature: MeshNav, Move Base Flex navigation on a mesh of the world, behind the meshnav mobile adapter."""

import os

from arena_cli import common
from arena_cli.common import Verb
from arena_cli.features import Health, fetched, lifecycle_verbs

NAME = "meshnav"

DESCRIPTION = "MeshNav, Move Base Flex navigation on a triangle mesh of every walkable surface (lvr2, mesh_tools, mesh_navigation, move_base_flex, pinned in _meta/repos/meshnav.repos).\n\nThis enables:\n\n\b\n- mobile.adapter: meshnav in a robot setup or scenario robot entry (robot.mobile_adapter:=meshnav at launch)\n- one plan across levels over stairs and escalators within the robot's caps/mobile.yaml meshnav: limits"


def _update() -> int:
    """Pull the pinned meshnav repos, install their rosdeps, and rebuild."""
    import subprocess

    rc = common._reg_pull(NAME)
    if rc:
        return rc
    rc = subprocess.run(["sudo", "apt-get", "update"], check=False).returncode
    if rc:
        return rc
    rc = subprocess.run(
        ["rosdep", "install", "--ignore-src", "-r", "-y", "--rosdistro", common._env("ARENA_ROS_DISTRO"), "--from-paths", os.path.join(common._env("ARENA_WS_DIR"), "src", NAME)],
        check=False,
    ).returncode
    if rc:
        return rc
    return common._resourced("arena build")


def health() -> Health:
    return fetched(NAME)


COMMANDS: dict[str, Verb] = {v.name: v for v in lifecycle_verbs(NAME, _update)}
