"""A robot's launched ROS nodes end when the robot is deleted or spawned again."""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import types

import launch
from arena_rclpy_mixins.Async import AsyncLaunchManager, LaunchHandle
from launch.actions import ExecuteProcess

from arena_runtime.sim import BaseSim


class _Host:
    _track_robot_stack = BaseSim._track_robot_stack
    _stop_robot_stacks = BaseSim._stop_robot_stacks

    def __init__(self) -> None:
        self.node = types.SimpleNamespace(do_launch_tracked=AsyncLaunchManager().launch_description)
        self._robot_stacks: dict[str, LaunchHandle] = {}
        self._logger = logging.getLogger("test_robot_stack_lifecycle")


def _stack(pid_file: pathlib.Path) -> launch.LaunchDescription:
    return launch.LaunchDescription([ExecuteProcess(cmd=["sh", "-c", f"echo $$ > {pid_file}; exec sleep 60"])])


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as stat:
        return stat.read().split()[2] != "Z"


async def _pid(pid_file: pathlib.Path) -> int:
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text().strip():
            return int(pid_file.read_text())
        await asyncio.sleep(0.05)
    raise TimeoutError(f"{pid_file} never written")


async def _gone(pid: int) -> bool:
    for _ in range(100):
        if not _alive(pid):
            return True
        await asyncio.sleep(0.05)
    return False


def test_spawning_a_robot_again_stops_the_nodes_of_its_previous_spawn(tmp_path: pathlib.Path) -> None:
    async def main() -> tuple[bool, bool]:
        host = _Host()
        await host._track_robot_stack("/robots/fr3_0", _stack(tmp_path / "first"))
        first = await _pid(tmp_path / "first")
        await host._track_robot_stack("/robots/fr3_0", _stack(tmp_path / "second"))
        second = await _pid(tmp_path / "second")
        result = (await _gone(first), _alive(second))
        await host._stop_robot_stacks(["/robots/fr3_0"])
        return result

    assert asyncio.run(main()) == (True, True)


def test_deleting_robots_stops_only_their_nodes(tmp_path: pathlib.Path) -> None:
    async def main() -> tuple[bool, bool, bool]:
        host = _Host()
        await host._track_robot_stack("/robots/fr3_0", _stack(tmp_path / "fr3_0"))
        await host._track_robot_stack("/robots/fr3_1", _stack(tmp_path / "fr3_1"))
        await host._track_robot_stack("/robots/tiago", _stack(tmp_path / "tiago"))
        pids = {name: await _pid(tmp_path / name) for name in ("fr3_0", "fr3_1", "tiago")}
        await host._stop_robot_stacks(["/robots/fr3_0", "/robots/fr3_1", "/robots/never_spawned"])
        result = (await _gone(pids["fr3_0"]), await _gone(pids["fr3_1"]), _alive(pids["tiago"]))
        await host._stop_robot_stacks(["/robots/tiago"])
        return result

    assert asyncio.run(main()) == (True, True, True)
