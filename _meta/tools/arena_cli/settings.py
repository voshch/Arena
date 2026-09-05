"""Workspace settings persisted in the .env file."""

import os

from common import CLIError, _env_file, _env_set, _host_path, _row, make_verb
from complete import Static

import features

DESCRIPTION = "Workspace settings, persisted in .env.\n\nEvery `source arena` reads them, on the host and in the container. A bare command shows the current value."

NET_MODES = {
    "local": "ROS 2 stays on loopback (default)",
    "lan": "ROS 2 uses every interface",
}


def _net() -> tuple[str, str]:
    """The ARENA_NET value and where it came from: the .env file, or the shell's default."""
    env_file = _env_file()
    if os.path.exists(env_file):
        with open(env_file) as f:
            lines = [line.split("=", 1)[1].strip() for line in f if line.lstrip().startswith("ARENA_NET=")]
        if lines:
            return lines[-1], env_file
    return os.environ.get("ARENA_NET", ""), "default"


def _container_net() -> str:
    """The ARENA_NET the running arena container was created with: a mode, 'down' or 'unknown'."""
    import json
    import subprocess

    ids = features.compose_output(["ps", "-q", "arena"]).split()
    if not ids:
        return "down"
    sudo = os.environ.get("arena_compose_sudo", "").split()
    p = subprocess.run([*sudo, "docker", "inspect", "--format", "{{json .Config.Env}}", ids[0]], capture_output=True, text=True, check=False)
    if p.returncode:
        return "unknown"
    for entry in json.loads(p.stdout or "[]"):
        key, _, value = entry.partition("=")
        if key == "FASTRTPS_DEFAULT_PROFILES_FILE":
            return value.rsplit(".", 2)[-2] if value.endswith(".xml") else "unknown"
    return "unknown"


def _net_status(hint: bool) -> None:
    value, source = _net()
    mode = value if value in NET_MODES else "local"
    _row("net", mode)
    _row("source", source if source == "default" else _host_path(source))
    if value and value not in NET_MODES:
        _row("warning", f"{value!r} is not a recognized value, so it counts as local")
    container = "down" if features.in_container() else _container_net()
    if container == "unknown":
        _row("warning", "could not inspect the running container")
    elif container != "down" and container != mode:
        _row("warning", "the running container is stale, run 'source arena' to recreate it")
    if hint:
        print(f"set it with 'arena settings net {'|'.join(NET_MODES)}'")


def net(argv: list[str]) -> None:
    """Show or set the network scope of ROS 2.

    `arena settings net [local|lan]`. `local` keeps DDS discovery and data
    on loopback (default), `lan` uses every interface. Gazebo's own
    transport always stays on loopback. New shells pick the value up on
    `source arena`, running nodes keep theirs until restarted.
    """
    if len(argv) > 1:
        raise CLIError(f"expected at most one argument, got {len(argv)}")
    if argv:
        if argv[0] not in NET_MODES:
            raise CLIError(f"unknown argument '{argv[0]}', expected one of: {', '.join(NET_MODES)}")
        _env_set("ARENA_NET", argv[0])
    _net_status(hint=not argv)


COMMANDS = {
    v.name: v
    for v in (
        make_verb("net", net, complete=Static(NET_MODES)),
    )
}
