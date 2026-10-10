from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

rclpy = pytest.importorskip("rclpy")

import rclpy.qos  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402

RATE_HZ = 1000.0
SECONDS = 3.0


def _cpu_seconds(pid: int) -> float:
    with open(f"/proc/{pid}/stat") as stat:
        fields = stat.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def test_server_stays_idle_under_a_fast_clock() -> None:
    server = subprocess.Popen([sys.executable, "-m", "task_generator_mcp.server"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init()
    node = rclpy.create_node("mcp_server_process_test_driver")
    try:
        publisher = node.create_publisher(Clock, "/clock", rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT))
        deadline = time.monotonic() + 30.0
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert publisher.get_subscription_count() > 0
        time.sleep(1.0)
        before = _cpu_seconds(server.pid)
        start = time.monotonic()
        tick = 0
        while time.monotonic() - start < SECONDS:
            tick += 1
            stamp_ns = tick * 1_000_000
            message = Clock()
            message.clock.sec = stamp_ns // 1_000_000_000
            message.clock.nanosec = stamp_ns % 1_000_000_000
            publisher.publish(message)
            time.sleep(max(0.0, start + tick / RATE_HZ - time.monotonic()))
        used = _cpu_seconds(server.pid) - before
        assert server.poll() is None
    finally:
        node.destroy_node()
        if owns_context:
            rclpy.shutdown()
        server.terminate()
        try:
            server.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

    assert used < 0.2 * SECONDS, f"server used {used:.2f}s of CPU in {SECONDS}s"
