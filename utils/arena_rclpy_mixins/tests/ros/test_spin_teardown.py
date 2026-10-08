"""Teardown is quiet: no traceback, no invalid-context error and exit 0 on SIGINT, for async_main and spin_node alike."""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import time

_CHILD = pathlib.Path(__file__).with_name("_teardown_child.py")
_NOISE = ("Traceback", "context is not valid", "context is invalid", "never retrieved", "terminate called", "Fatal Python error")


def _env() -> dict[str, str]:
    return {**os.environ, "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID", "93")}


def _run(mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(_CHILD), mode], env=_env(), capture_output=True, text=True, timeout=30, check=False)


def _exits_quietly(mode: str, sig: signal.Signals = signal.SIGINT, *, again_after: str | None = None) -> None:
    proc = subprocess.Popen([sys.executable, str(_CHILD), mode], env=_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    try:
        deadline = time.monotonic() + 20.0
        while proc.stdout.readline().strip() != "READY":
            assert time.monotonic() < deadline, "child never became ready"
        time.sleep(0.5)
        proc.send_signal(sig)
        if again_after is not None:
            while proc.stdout.readline().strip() != again_after:
                assert time.monotonic() < deadline + 20.0, f"child never printed {again_after}"
            proc.send_signal(sig)
        _, stderr = proc.communicate(timeout=20)
    finally:
        proc.kill()
    assert proc.returncode == 0, stderr
    assert not any(marker in stderr for marker in _NOISE), stderr


def test_async_main_sigint_is_quiet():
    _exits_quietly("async_storm")


def test_async_main_sigterm_is_quiet():
    _exits_quietly("async_storm", signal.SIGTERM)


def test_async_main_absorbs_a_repeated_sigint_during_teardown():
    _exits_quietly("async_slow_teardown", again_after="TEARDOWN")


def test_async_main_absorbs_a_repeated_sigint_after_returning():
    _exits_quietly("async_linger", again_after="RETURNED")


def test_async_main_absorbs_a_repeated_sigint_during_interpreter_finalization():
    _exits_quietly("async_finalize", again_after="FINALIZING")


def test_async_main_request_shutdown_tears_down_quietly():
    proc = _run("async_request")
    assert proc.returncode == 0, proc.stderr
    assert "TEARDOWN" in proc.stdout
    assert not any(marker in proc.stderr for marker in _NOISE), proc.stderr


def test_spin_node_on_events_executor_sigint_is_quiet():
    _exits_quietly("sync_events")


def test_spin_node_absorbs_a_repeated_sigint_after_returning():
    _exits_quietly("sync_events_linger", again_after="RETURNED")


def test_spin_node_absorbs_a_repeated_sigint_during_interpreter_finalization():
    _exits_quietly("sync_global_finalize", again_after="FINALIZING")


def test_spin_node_on_events_executor_sigint_mid_callback_is_quiet():
    _exits_quietly("sync_events_slow")


def test_spin_node_on_events_executor_sigterm_mid_callback_is_quiet():
    _exits_quietly("sync_events_slow", signal.SIGTERM)


def test_spin_node_on_global_executor_sigint_mid_callback_is_quiet():
    _exits_quietly("sync_global_slow")


def test_spin_node_on_global_executor_sigterm_mid_callback_is_quiet():
    _exits_quietly("sync_global_slow", signal.SIGTERM)


def test_spin_node_on_idle_events_executor_sigint_is_quiet():
    _exits_quietly("sync_events_idle")


def test_spin_node_on_idle_global_executor_sigterm_is_quiet():
    _exits_quietly("sync_global_idle", signal.SIGTERM)


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


def test_async_main_exits_on_failing_async_callback():
    proc = _run("callback_failure")
    assert proc.returncode == 1, proc.stderr
    assert "callback failed" in proc.stderr


def test_spin_context_swallows_after_shutdown():
    proc = _run("sync_late")
    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr


def test_spin_context_raises_while_running():
    proc = _run("sync_real")
    assert proc.returncode != 0
    assert "real failure" in proc.stderr
