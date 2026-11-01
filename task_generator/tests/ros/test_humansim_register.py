from __future__ import annotations

import subprocess
import sys

import pytest

_PROBE = """
from {module} import {cls}
from task_generator.constants import Constants
from task_generator.tasks.registry import OBSTACLES_MODES

{cls}._register_task_modes()
print(OBSTACLES_MODES.get(Constants.TaskMode.TM_Obstacles.PROMPT).__module__)
"""


@pytest.mark.parametrize(
    ("msgs", "module", "cls", "prompt"),
    [
        ("arena_humansim_msgs.msg", "task_generator.simulators.human.arena_humansim.arena_humansim", "ArenaHumanSimulator", "task_generator.tasks.obstacles.prompt.arena"),
        pytest.param(
            "hunav_msgs.msg",
            "task_generator.simulators.human.hunav.hunav",
            "HunavHumanSimulator",
            "task_generator.tasks.obstacles.prompt.hunav",
            marks=pytest.mark.xfail(strict=True, reason="arena_text_crowd imports the removed WorldDescription"),
        ),
    ],
)
def test_register_task_modes_binds_the_prompt_mode_of_the_simulator(msgs: str, module: str, cls: str, prompt: str) -> None:
    pytest.importorskip(msgs)
    result = subprocess.run([sys.executable, "-c", _PROBE.format(module=module, cls=cls)], capture_output=True, text=True, check=False, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == prompt
