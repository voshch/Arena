from __future__ import annotations

import re
import subprocess

import launch.actions

_ARG_LINE = re.compile(r"^\s{4}'(?P<name>[^']+)':\s*$")


def _undescribed_text() -> str:
    return launch.actions.DeclareLaunchArgument("probe").description


def _show_args() -> str:
    result = subprocess.run(
        ["ros2", "launch", "arena_bringup", "arena_runtime.launch.py", "--show-args"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return result.stdout


def _undescribed(output: str, marker: str) -> list[str]:
    offenders: list[str] = []
    current = ""
    for line in output.splitlines():
        match = _ARG_LINE.match(line)
        if match:
            current = match["name"]
        elif marker in line and current:
            offenders.append(current)
    return offenders


def test_every_argument_of_arena_runtime_launch_has_a_description():
    marker = _undescribed_text()
    output = _show_args()
    assert "'sim'" in output
    assert _undescribed(output, marker) == []
