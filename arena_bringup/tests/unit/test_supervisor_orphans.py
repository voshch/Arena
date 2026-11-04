from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_GRANDCHILD = """
import os, signal, sys, time
def interrupted(*_):
    open(sys.argv[1], "w").write("INT")
    sys.exit(0)
signal.signal(signal.SIGINT, interrupted)
open(sys.argv[2], "w").write(str(os.getpid()))
time.sleep(120)
"""

_SUPERVISOR = """
import sys, time
import rclpy
from arena_bringup.supervisor import Supervisor
rclpy.init(args=[])
sup = Supervisor(rclpy.create_node("orphan_test_supervisor"))
sup.spawn("runtime", ["bash", "-c", sys.executable + ' -c "$0" "$1" "$2"; true', *sys.argv[1:]])
time.sleep(120)
"""


def _wait_for(path: Path, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text():
            return path.read_text()
        time.sleep(0.1)
    pytest.fail(f"{path.name} not written within {timeout} s")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_grandchild_is_interrupted_when_the_supervisor_is_killed(tmp_path: Path) -> None:
    marker, ready = tmp_path / "marker", tmp_path / "ready"
    supervisor = subprocess.Popen([sys.executable, "-c", _SUPERVISOR, _GRANDCHILD, str(marker), str(ready)])
    grandchild = 0
    try:
        grandchild = int(_wait_for(ready, 30.0))
        supervisor.kill()
        supervisor.wait(timeout=30)

        assert _wait_for(marker, 60.0) == "INT"
    finally:
        supervisor.kill()
        if grandchild:
            with contextlib.suppress(ProcessLookupError):
                os.kill(grandchild, signal.SIGKILL)


def test_released_reaper_leaves_the_groups_alone() -> None:
    from arena_bringup.reaper import Reaper

    sleeper = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        reaper = Reaper()
        reaper.watch(sleeper.pid)
        reaper.release()
        time.sleep(0.5)

        assert _alive(sleeper.pid)
    finally:
        sleeper.kill()
        sleeper.wait()


_EXITING_CHILD_SUPERVISOR = """
import sys, time
import rclpy
from arena_bringup.supervisor import Supervisor
rclpy.init(args=[])
sup = Supervisor(rclpy.create_node("orphan_test_supervisor"))
sup.spawn("env_0", ["bash", "-c", sys.executable + ' -c "$0" "$1" "$2" & exit 0', *sys.argv[2:]])
time.sleep(2.0)
if sys.argv[1] == "sweep":
    sup.release()
    for _ in range(1200):
        sup.sweep_exited()
        time.sleep(0.1)
else:
    sup.cleanup()
    sup.release()
    time.sleep(120)
"""


@pytest.mark.parametrize("stage", ["sweep", "cleanup"])
def test_orphan_of_an_exited_child_is_interrupted_while_the_supervisor_lives(tmp_path: Path, stage: str) -> None:
    marker, ready = tmp_path / "marker", tmp_path / "ready"
    supervisor = subprocess.Popen([sys.executable, "-c", _EXITING_CHILD_SUPERVISOR, stage, _GRANDCHILD, str(marker), str(ready)])
    grandchild = 0
    try:
        grandchild = int(_wait_for(ready, 30.0))

        assert _wait_for(marker, 60.0) == "INT"
        assert supervisor.poll() is None
    finally:
        supervisor.kill()
        supervisor.wait(timeout=30)
        if grandchild:
            with contextlib.suppress(ProcessLookupError):
                os.kill(grandchild, signal.SIGKILL)
