"""Sim default resolution from installed features and the ARENA_SIM setting."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from arena_bringup.defaults import SIM_ORDER, available_sims, env_setting, installed_features, resolve_sim
from arena_bringup.constants import SimSimulator

DUMMY = SimSimulator.DUMMY.value
FIRST, SECOND, THIRD, LAST = (s.value for s in SIM_ORDER)


def test_nothing_installed_resolves_to_dummy():
    assert resolve_sim("", []) == DUMMY
    assert resolve_sim("auto", ["evaluation", "planners"]) == DUMMY


def test_auto_takes_the_installed_simulator_earliest_in_sim_order():
    assert resolve_sim("", [LAST, THIRD]) == THIRD
    assert resolve_sim("", [THIRD, SECOND, FIRST]) == FIRST
    assert resolve_sim("auto", [LAST, THIRD, SECOND]) == SECOND


def test_usable_setting_wins():
    assert resolve_sim(LAST, [FIRST, LAST]) == LAST
    assert resolve_sim(DUMMY, [FIRST]) == DUMMY


def test_unusable_setting_falls_back_to_auto():
    assert resolve_sim(FIRST, [THIRD]) == THIRD
    assert resolve_sim(SimSimulator.UNITY.value, []) == DUMMY


def test_simulators_outside_sim_order_are_never_picked():
    assert available_sims([SimSimulator.UNITY.value, SimSimulator.FLATLAND.value, LAST]) == [LAST]


def test_env_setting_takes_the_last_line(tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"ARENA_NET=lan\nARENA_SIM={FIRST}\n  ARENA_SIM={THIRD} \nARENA_SIMX={SECOND}\n")
    assert env_setting(str(env), "ARENA_SIM") == THIRD
    assert env_setting(str(env), "ARENA_GPU") == ""
    assert env_setting(str(tmp_path / "missing"), "ARENA_SIM") == ""


def test_installed_features_skips_blank_lines(tmp_path):
    installed = tmp_path / ".installed"
    installed.write_text(f"{THIRD}\n\n{LAST}\n")
    assert installed_features(str(installed)) == [THIRD, LAST]
    assert installed_features(str(tmp_path / "missing")) == []


def _default_sim(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    package = str(Path(__file__).parents[2])
    full = {**os.environ, **env, "PYTHONPATH": os.pathsep.join([package, os.environ.get("PYTHONPATH", "")])}
    return subprocess.run([sys.executable, "-c", "from arena_bringup.defaults import default_sim; print(default_sim())"], env=full, capture_output=True, text=True, check=True)


def test_default_sim_reads_the_workspace_files(tmp_path):
    (tmp_path / ".installed").write_text(f"{LAST}\n{THIRD}\n")
    env = {"ARENA_WS_DIR": str(tmp_path), "ARENA_ENV_FILE": "", "INSTALLED": str(tmp_path / ".installed"), "ARENA_SIM": LAST}
    assert _default_sim(env).stdout.strip() == THIRD
    (tmp_path / ".env").write_text(f"ARENA_SIM={LAST}\n")
    assert _default_sim(env).stdout.strip() == LAST
    (tmp_path / "other.env").write_text(f"ARENA_SIM={DUMMY}\n")
    assert _default_sim({**env, "ARENA_ENV_FILE": str(tmp_path / "other.env")}).stdout.strip() == DUMMY


def test_default_sim_warns_about_an_uninstalled_setting(tmp_path):
    (tmp_path / ".installed").write_text(f"{LAST}\n")
    (tmp_path / ".env").write_text(f"ARENA_SIM={FIRST}\n")
    done = _default_sim({"ARENA_WS_DIR": str(tmp_path), "ARENA_ENV_FILE": "", "INSTALLED": str(tmp_path / ".installed")})
    assert done.stdout.strip() == LAST
    assert f"ARENA_SIM={FIRST} is not an installed simulator" in done.stderr
