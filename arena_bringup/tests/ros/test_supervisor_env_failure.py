from __future__ import annotations

import time
import uuid

import pytest


@pytest.fixture
def sup(rclpy_context):
    if rclpy_context is None:
        pytest.skip("ROS2 not available")
    from arena_bringup.supervisor import Supervisor

    node = rclpy_context.create_node(f"supervisor_env_failure_{uuid.uuid4().hex[:8]}")
    supervisor = Supervisor(node)
    try:
        yield supervisor
    finally:
        supervisor.cleanup()
        supervisor.release()
        node.destroy_node()


def test_wait_ends_with_the_exit_code_of_a_failed_env(sup) -> None:
    from arena_bringup.supervisor import wait_until

    runtime = sup.spawn("runtime", ["sleep", "60"])
    sup.spawn("env_0", ["bash", "-c", "exit 3"])
    started = time.monotonic()

    assert wait_until(lambda: False, sup=sup, runtime_proc=runtime, label="1 env(s)", period_s=0.1) == 3
    assert time.monotonic() - started < 10.0


def test_wait_ignores_an_env_that_exits_cleanly(sup) -> None:
    from arena_bringup.supervisor import wait_until

    runtime = sup.spawn("runtime", ["sleep", "60"])
    env = sup.spawn("env_0", ["bash", "-c", "exit 0"])
    env.wait(timeout=10)
    deadline = time.monotonic() + 1.0

    assert wait_until(lambda: time.monotonic() > deadline, sup=sup, runtime_proc=runtime, label="1 env(s)", period_s=0.1) is None
    assert sup.env_failure() is None


def test_wait_ignores_a_failed_child_that_is_not_an_env(sup) -> None:
    from arena_bringup.supervisor import wait_until

    sup.spawn("viz_0", ["bash", "-c", "exit 1"])
    deadline = time.monotonic() + 1.0

    assert wait_until(lambda: time.monotonic() > deadline, sup=sup, runtime_proc=None, label="1 env(s)", period_s=0.1) is None


def test_wait_fails_when_the_runtime_exits(sup) -> None:
    from arena_bringup.supervisor import wait_until

    runtime = sup.spawn("runtime", ["bash", "-c", "exit 0"])

    assert wait_until(lambda: False, sup=sup, runtime_proc=runtime, label="/arena/register_env", period_s=0.1) == 1
