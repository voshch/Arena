from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("arena_runtime.arena_runtime_node")

from arena_runtime.arena_runtime_node import ArenaRuntimeNode  # noqa: E402


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)


class _Node:
    def __init__(self) -> None:
        self.logger = _Logger()

    def get_logger(self) -> _Logger:
        return self.logger


def _running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


def _wrapper(tmp_path: Path, member: str, *, exit_wrapper: bool) -> tuple[subprocess.Popen, int]:
    pid_file = tmp_path / "member.pid"
    script = f"import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', {member!r}]); open({str(pid_file)!r}, 'w').write(str(p.pid)); time.sleep(0 if {exit_wrapper} else 60)"
    popen = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
    deadline = time.monotonic() + 20.0
    while not pid_file.is_file() or not pid_file.read_text():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    ready = tmp_path / "member.ready"
    while not ready.is_file():
        assert time.monotonic() < deadline, "the member never signalled ready"
        time.sleep(0.05)
    return popen, int(pid_file.read_text())


def test_a_member_outliving_its_exited_wrapper_is_reaped(tmp_path):
    popen, member = _wrapper(tmp_path, f"import time; open({str(tmp_path / 'member.ready')!r}, 'w').write('1'); time.sleep(60)", exit_wrapper=True)
    popen.wait(timeout=20.0)
    assert _running(member)
    asyncio.run(ArenaRuntimeNode._shutdown_pg(_Node(), 1, popen, grace_seconds=0.0))  # pyright: ignore[reportArgumentType]
    assert not _running(member)


def test_a_member_ignoring_sigterm_is_killed(tmp_path):
    popen, member = _wrapper(tmp_path, f"import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); open({str(tmp_path / 'member.ready')!r}, 'w').write('1'); time.sleep(60)", exit_wrapper=False)
    node = _Node()
    asyncio.run(ArenaRuntimeNode._shutdown_pg(node, 1, popen, grace_seconds=0.5))  # pyright: ignore[reportArgumentType]
    deadline = time.monotonic() + 20.0
    while _running(member) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _running(member)
    assert node.logger.warnings == ["env_1 did not exit within 0.5s, sending SIGTERM", "env_1 did not exit after SIGTERM, sending SIGKILL"]
    popen.wait(timeout=20.0)
