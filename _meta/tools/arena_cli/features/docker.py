"""docker feature: container lifecycle via docker compose."""

import os
import sys

from common import CLIError, Verb, make_verb

import features

NAME = "docker"

DESCRIPTION = "Container lifecycle (compose build, commit, stop, down)."


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
        make_verb("update", update, hidden=True),
        make_verb("uninstall", uninstall, hidden=True),
    ]
}
