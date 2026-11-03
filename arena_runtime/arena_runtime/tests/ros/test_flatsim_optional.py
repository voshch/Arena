"""sim:=flatsim without the flatsim packages: the rest of arena_runtime imports and the error names the feature."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_STRIPPED = ("flatsim", "flatsim_msgs")


def _env() -> dict[str, str]:
    if "AMENT_PREFIX_PATH" not in os.environ:
        pytest.skip("AMENT_PREFIX_PATH unset: source install/setup.bash")
    prefixes = [p for p in os.environ["AMENT_PREFIX_PATH"].split(os.pathsep) if os.path.basename(p) not in _STRIPPED]
    paths = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if not any(package in p.split(os.sep) for package in _STRIPPED)]
    return {**os.environ, "AMENT_PREFIX_PATH": os.pathsep.join(prefixes), "PYTHONPATH": os.pathsep.join(paths)}


def _python(code: str, cwd: os.PathLike[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], env=_env(), cwd=cwd, capture_output=True, text=True, timeout=120, check=False)


def test_runtime_imports_without_flatsim(tmp_path) -> None:
    code = "import importlib.util\nimport arena_runtime.arena_runtime_node, arena_runtime.sim\nassert importlib.util.find_spec('flatsim_msgs') is None\nassert importlib.util.find_spec('flatsim') is None"
    proc = _python(code, tmp_path)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("registry", ["LifecycleRegistry", "SimulatorRegistry"])
def test_flatsim_without_its_packages_names_the_feature(tmp_path, registry: str) -> None:
    code = f"import asyncio\nfrom arena_bringup.constants import SimSimulator\nfrom arena_runtime.sim import {registry}\nasyncio.run({registry}.get(SimSimulator.FLATSIM, node=None))"
    proc = _python(code, tmp_path)
    assert proc.returncode != 0
    assert "arena feature flatsim install" in proc.stderr
