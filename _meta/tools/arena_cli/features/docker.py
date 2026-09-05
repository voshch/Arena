"""docker feature: container lifecycle via docker compose."""

import os
import sys

from common import CLIError, Verb, _env, _env_set, _host_path, _row, make_verb
from complete import Static

import features

NAME = "docker"

DESCRIPTION = "Container lifecycle and gpu passthrough for the arena container."


def build(argv: list[str]) -> None:
    """Build the arena container image."""
    if argv:
        raise CLIError("unexpected arguments")
    sys.exit(features.compose(["build", "arena"]))


def commit(argv: list[str]) -> None:
    """Commit the running arena container back to its image."""
    import subprocess

    if argv:
        raise CLIError("unexpected arguments")
    out = features.compose_output(["ps", "-q", "-a", "arena"])
    container_id = out.splitlines()[0].strip() if out.splitlines() else ""
    image = os.environ.get("ARENA_IMAGE", "")
    print(f"committing container {container_id} to image {image}...")
    rc = subprocess.run(["sudo", "env", *os.environ.get("docker_env", "").split(), "docker", "commit", "--no-pause", container_id, image], check=False).returncode
    if rc:
        sys.exit(rc)
    print("done")


def stop(argv: list[str]) -> None:
    """Stop the project's containers."""
    if argv:
        raise CLIError("unexpected arguments")
    print("Stopping containers...")
    sys.exit(features.compose(["stop", "--timeout", "0"]))


def down(argv: list[str]) -> None:
    """Stop and remove the project's containers."""
    import subprocess

    if argv:
        raise CLIError("unexpected arguments")
    print("Stopping and removing containers...")
    project = os.environ.get("ARENA_PROJECT_NAME", "")
    sys.exit(
        subprocess.run(
            [
                "sudo",
                "docker",
                "run",
                "--rm",
                "-v",
                "/var/run/docker.sock:/var/run/docker.sock",
                "docker:cli",
                "sh",
                "-lc",
                f"docker compose --project-name '{project}' down --timeout 0",
            ],
            check=False,
        ).returncode
    )


def compose(argv: list[str]) -> None:
    """Run docker compose with the project's compose files."""
    sys.exit(features.compose(argv))


def _gpu() -> tuple[str, str, str]:
    """The value of ARENA_GPU, where it came from, and whether it reads as on, off or invalid.

    The rule for what counts as on lives in _meta/docker/lib, which the compose gate also uses.
    """
    import subprocess

    p = subprocess.run(["bash", "-c", "arena_gpu"], capture_output=True, text=True, check=False)
    if p.returncode == 127:
        raise CLIError("the arena shell functions are not loaded, run 'source arena' first")
    fields = p.stdout.rstrip("\n").split("\t")
    if len(fields) != 3:
        raise CLIError("could not resolve ARENA_GPU")
    return fields[0], fields[1], fields[2]


def _container_gpu() -> str:
    """Whether the running arena container reserves a gpu: 'yes', 'no', 'down' or 'unknown'."""
    import json
    import subprocess

    ids = features.compose_output(["ps", "-q", "arena"]).split()
    if not ids:
        return "down"
    sudo = os.environ.get("arena_compose_sudo", "").split()
    p = subprocess.run([*sudo, "docker", "inspect", "--format", "{{json .HostConfig.DeviceRequests}}", ids[0]], capture_output=True, text=True, check=False)
    if p.returncode:
        return "unknown"
    requests = json.loads(p.stdout or "null") or []
    return "yes" if any(r["Driver"] == "nvidia" for r in requests) else "no"


def _gpu_status(hint: bool) -> None:
    """Report the resolved setting and where it came from, warning only when something needs acting on."""
    value, source, state = _gpu()
    on = state == "on"
    env_file = os.environ.get("ARENA_ENV_FILE", "")

    _row("gpu", "on" if on else "off")
    _row("source", _host_path(source) if source not in ("unset", "environment") else source)
    if state == "invalid":
        _row("warning", f"{value!r} is not a recognized value, so it counts as off")
    if os.path.exists(env_file):
        with open(env_file) as f:
            declared = sum(1 for line in f if line.lstrip().startswith("ARENA_GPU="))
        if declared > 1:
            _row("warning", f"{declared} ARENA_GPU lines in {_host_path(env_file)}, the last one wins")

    container = _container_gpu()
    if container == "unknown":
        _row("warning", "could not inspect the running container")
    elif container in ("yes", "no") and (container == "yes") != on:
        _row("warning", "the running container is stale, run 'source arena' to recreate it")

    if hint and not on:
        print("run 'arena feature docker gpu on' to enable it")


def gpu(argv: list[str]) -> None:
    """Show or set NVIDIA GPU passthrough for the arena container."""
    if len(argv) > 1:
        raise CLIError(f"expected at most one argument, got {len(argv)}")
    if not argv:
        _gpu_status(hint=True)
        return
    action = argv[0]
    if action not in ("on", "off"):
        raise CLIError(f"unknown argument '{action}', expected one of: on, off")
    _env_set("ARENA_GPU", "1" if action == "on" else "0")
    _gpu_status(hint=False)


def update(argv: list[str]) -> None:
    """No-op."""
    if argv:
        raise CLIError("unexpected arguments")


def uninstall(argv: list[str]) -> None:
    """No-op."""
    if argv:
        raise CLIError("unexpected arguments")


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        make_verb("build", build),
        make_verb("commit", commit),
        make_verb("stop", stop),
        make_verb("down", down),
        make_verb("compose", compose, passthrough=True),
        make_verb("gpu", gpu, complete=Static({"on": "enable gpu passthrough", "off": "disable gpu passthrough"})),
        make_verb("update", update, hidden=True),
        make_verb("uninstall", uninstall, hidden=True),
    ]
}
