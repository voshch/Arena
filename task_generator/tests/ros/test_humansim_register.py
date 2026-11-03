from __future__ import annotations

import subprocess
import sys

import pytest

_PROBE = """
from arena_runtime.env.constants import Constants
from task_generator.modes import Modes
from task_generator.tasks.obstacles.prompt import register_prompt_mode
from task_generator.tasks.registry import OBSTACLES_MODES

register_prompt_mode(Constants.HumanSimulator({human!r}))
register_prompt_mode(Constants.HumanSimulator({human!r}))
prompt = Modes.TM_Obstacles.PROMPT
print(OBSTACLES_MODES.get(prompt).__module__ if prompt in OBSTACLES_MODES else None)
"""


@pytest.mark.parametrize(
    ("msgs", "human", "prompt"),
    [
        ("arena_humansim_msgs.msg", "arena", "task_generator.tasks.obstacles.prompt.arena"),
        ("arena_people_msgs.msg", "none", "None"),
        pytest.param(
            "hunav_msgs.msg",
            "hunav",
            "task_generator.tasks.obstacles.prompt.hunav",
            marks=pytest.mark.xfail(strict=True, reason="arena_text_crowd imports the removed WorldDescription"),
        ),
    ],
)
def test_register_prompt_mode_binds_the_prompt_mode_of_the_simulator(msgs: str, human: str, prompt: str) -> None:
    pytest.importorskip(msgs)
    result = subprocess.run([sys.executable, "-c", _PROBE.format(human=human)], capture_output=True, text=True, check=False, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == prompt
