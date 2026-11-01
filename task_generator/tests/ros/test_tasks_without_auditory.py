from __future__ import annotations

import os
import subprocess
import sys

_STRIPPED = ("arena_auditory", "arena_hearing")


def _without_feature(variable: str) -> str:
    return os.pathsep.join(path for path in os.environ.get(variable, "").split(os.pathsep) if not any(package in path for package in _STRIPPED))


def test_the_task_layer_imports_without_arena_auditory_or_arena_hearing(tmp_path) -> None:
    code = "import importlib, importlib.util\nimportlib.import_module('task_generator.tasks.modules.sounds.impl')\nassert importlib.util.find_spec('arena_auditory') is None\nassert importlib.util.find_spec('arena_hearing') is None"
    env = {**os.environ, "AMENT_PREFIX_PATH": _without_feature("AMENT_PREFIX_PATH"), "PYTHONPATH": _without_feature("PYTHONPATH")}
    proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
