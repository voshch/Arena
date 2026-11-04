from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest

rclpy = pytest.importorskip("rclpy")

import rclpy.qos
from rosgraph_msgs.msg import Clock

from mcp import ClientSession, StdioServerParameters, stdio_client

RATE_HZ = 1000.0
SECONDS = 3.0


def _cpu_seconds(pid: int) -> float:
    with open(f"/proc/{pid}/stat") as stat:
        fields = stat.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


async def _wait_cpu_settled(pid: int) -> None:
    deadline = time.monotonic() + 30.0
    previous = None
    last = _cpu_seconds(pid)
    while time.monotonic() < deadline:
        await asyncio.sleep(1.0)
        now = _cpu_seconds(pid)
        rate = now - last
        last = now
        if previous is not None and abs(rate - previous) < 0.05:
            return
        previous = rate
    raise AssertionError("server cpu never settled")


def _server_pid(parent: int) -> int:
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/stat") as stat:
                    fields = stat.read().rsplit(")", 1)[1].split()
                with open(f"/proc/{entry}/cmdline", "rb") as cmdline:
                    args = cmdline.read().split(b"\0")
            except OSError:
                continue
            if int(fields[1]) == parent and b"arena_mcp" in args:
                return int(entry)
    raise AssertionError("server process not found")


def _drive_clock() -> None:
    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init()
    node = rclpy.create_node("arena_mcp_idle_test_clock")
    try:
        publisher = node.create_publisher(Clock, "/clock", rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT))
        deadline = time.monotonic() + 30.0
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert publisher.get_subscription_count() > 0
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
    finally:
        node.destroy_node()
        if owns_context:
            rclpy.shutdown()


def test_server_stays_idle_under_a_fast_clock(tmp_path):
    params = StdioServerParameters(command=sys.executable, args=["-m", "arena_mcp", "serve", "ros", "jobs"], env={**os.environ, "ARENA_DATA_DIR": str(tmp_path)})

    async def scenario() -> float:
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            started = await session.call_tool("ros_graph", {})
            assert not started.is_error, started.content
            pid = _server_pid(os.getpid())
            await _wait_cpu_settled(pid)
            before = _cpu_seconds(pid)
            await asyncio.to_thread(_drive_clock)
            return _cpu_seconds(pid) - before

    used = asyncio.run(scenario())
    assert used < 0.5 * SECONDS, f"server used {used:.2f}s of CPU in {SECONDS}s"
