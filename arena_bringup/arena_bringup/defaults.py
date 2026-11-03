"""Per-sim launch defaults shared by the launch files, the supervisor and the benchmark runner."""

import os
import sys
from collections.abc import Iterable

from arena_bringup.constants import SimSimulator

SIM_ORDER = (SimSimulator.ISAAC, SimSimulator.GAZEBO, SimSimulator.MUJOCO, SimSimulator.FLATSIM)

DEFAULT_HUMAN: dict[str, str] = {
    "dummy": "dummy",
    "flatsim": "arena",
    "gazebo": "arena",
    "isaac": "arena",
    "mujoco": "arena",
}


GZ_DISCOVERY_BASE_PORT = 5500


def gz_discovery_env(domain_id: int) -> dict[str, str]:
    """gz-transport discovery ports of one ROS domain: a private pair per domain, below the DDS port blocks."""
    msg_port = GZ_DISCOVERY_BASE_PORT + 2 * domain_id
    return {"GZ_DISCOVERY_MSG_PORT": str(msg_port), "GZ_DISCOVERY_SRV_PORT": str(msg_port + 1)}


def default_human(sim: str) -> str:
    """Human-backend default for a sim, unknown sims fall back to dummy."""
    return DEFAULT_HUMAN.get(sim, "dummy")


def available_sims(installed: Iterable[str]) -> list[str]:
    """The installed simulators in SIM_ORDER."""
    names = set(installed)
    return [sim.value for sim in SIM_ORDER if sim.value in names]


def resolve_sim(setting: str, installed: Iterable[str]) -> str:
    """The sim a launch without sim:= runs: the setting if usable, else the first installed simulator, else dummy."""
    available = available_sims(installed)
    if setting in (SimSimulator.DUMMY.value, *available):
        return setting
    return available[0] if available else SimSimulator.DUMMY.value


def installed_features(path: str) -> list[str]:
    """Feature names in the installed-features file, empty when it is missing."""
    try:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    except OSError:
        return []


def env_setting(path: str, key: str) -> str:
    """The last value of key in a .env file, empty when absent."""
    try:
        with open(path) as f:
            values = [line.split("=", 1)[1].strip() for line in f if line.lstrip().startswith(f"{key}=")]
    except OSError:
        return ""
    return values[-1] if values else ""


def default_sim() -> str:
    """The sim default of the sourced workspace, read from its installed features and .env at call time."""
    ws = os.environ.get("ARENA_WS_DIR", "")
    env_file = os.environ.get("ARENA_ENV_FILE") or (os.path.join(ws, ".env") if ws else "")
    setting = env_setting(env_file, "ARENA_SIM")
    installed = installed_features(os.environ.get("INSTALLED", ""))
    sim = resolve_sim(setting, installed)
    if setting not in ("", "auto", sim):
        sys.stderr.write(f"ARENA_SIM={setting} is not an installed simulator, ignoring it (installed: {', '.join(available_sims(installed)) or 'none'})\n")
    return sim
