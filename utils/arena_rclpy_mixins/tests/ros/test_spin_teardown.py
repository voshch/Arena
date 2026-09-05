"""Teardown is quiet: no traceback, no invalid-context error and exit 0 on SIGINT, for async_main and spin_node alike."""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import time

_CHILD = pathlib.Path(__file__).with_name("_teardown_child.py")
_NOISE = ("Traceback", "context is not valid", "context is invalid", "never retrieved")


def _env() -> dict[str, str]:
    return {**os.environ, "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID", "93")}


def _run(mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(_CHILD), mode], env=_env(), capture_output=True, text=True, timeout=30, check=False)


def test_async_main_sigint_is_quiet():
    proc = subprocess.Popen([sys.executable, str(_CHILD), "async_storm"], env=_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    deadline = time.monotonic() + 20.0
    while proc.stdout.readline().strip() != "READY":
        assert time.monotonic() < deadline, "child never became ready"
    time.sleep(0.5)
    proc.send_signal(signal.SIGINT)
    _, stderr = proc.communicate(timeout=20)
    assert proc.returncode == 0, stderr
    assert not any(marker in stderr for marker in _NOISE), stderr


def test_async_main_reports_loop_stall():
    proc = subprocess.Popen([sys.executable, str(_CHILD), "async_stall"], env=_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    deadline = time.monotonic() + 40.0
    while proc.stdout.readline().strip() != "RESUMED":
        assert time.monotonic() < deadline, "child never resumed"
    proc.send_signal(signal.SIGINT)
    _, stderr = proc.communicate(timeout=20)
    assert proc.returncode == 0, stderr
    assert "event loop stalled" in stderr, stderr
    assert "Thread 0x" in stderr, stderr


def test_watchdog_deadline_exits_process():
    proc = _run("watchdog_deadline")
    assert proc.returncode == 7, proc.stderr


def test_spin_context_swallows_after_shutdown():
    proc = _run("sync_late")
    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr


def test_spin_context_raises_while_running():
    proc = _run("sync_real")
    assert proc.returncode != 0
    assert "real failure" in proc.stderr
