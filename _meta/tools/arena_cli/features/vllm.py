"""vllm feature: local LLM inference containers."""

import os
import sys

import common
from common import CLIError, Verb, make_verb

import features
from features import source_verb

NAME = "vllm"

DESCRIPTION = "Local LLM inference (vllm + litellm proxy containers)."

_SERVICES = ("vllm", "vllm-proxy")


def _config() -> dict:
    import yaml

    with open(os.path.join(features.assets_dir(NAME), "config.yaml")) as f:
        return yaml.safe_load(f)


def _config_env() -> dict[str, str]:
    env = dict(os.environ)
    for k, v in _config().items():
        env[f"VLLM_{k.upper()}"] = str(v)
    return env


def _update() -> int:
    env = _config_env()
    for args in (["pull", *_SERVICES], ["create", *_SERVICES], ["up", "-d", "--wait", *_SERVICES]):
        rc = features.compose(["--profile", "vllm", *args], env=env)
        if rc:
            return rc
    return 0


def install(argv: list[str]) -> None:
    """Install the feature (pull repos, register, run its update)."""
    if argv:
        raise CLIError("unexpected arguments")
    sys.exit(features.default_install(NAME, _update))


def update(argv: list[str]) -> None:
    """Update the feature to the latest state."""
    if argv:
        raise CLIError("unexpected arguments")
    common._reg_require(NAME)
    sys.exit(_update())


def uninstall(argv: list[str]) -> None:
    """Uninstall and unregister the feature."""
    if argv:
        raise CLIError("unexpected arguments")
    features.compose(["--profile", "vllm", "down", "--volumes", "--remove-orphans", *_SERVICES], env=_config_env())
    common._reg_remove(NAME)


def _shell_source() -> str:
    import shlex

    lines = [f"export VLLM_{k.upper()}={shlex.quote(str(v))}" for k, v in _config().items()]
    lines += [
        'export LLM_API_ENDPOINT="http://localhost:${VLLM_PROXY_PORT}"',
        'if [ "$(arena_docker_compose --profile vllm ps --format json vllm vllm-proxy 2>/dev/null | grep -o \'"Health":"healthy"\' | wc -l)" -ne 2 ] ; then',
        "    arena_docker_compose --profile vllm up -d --wait vllm vllm-proxy",
        "fi",
    ]
    return "\n".join(lines)


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        make_verb("install", install),
        make_verb("update", update),
        make_verb("uninstall", uninstall),
        source_verb(_shell_source),
    ]
}
