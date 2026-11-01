from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


@pytest.fixture()
def ros_node():
    rclpy = pytest.importorskip("rclpy")
    owned = not rclpy.ok()
    if owned:
        rclpy.init()
    node = rclpy.create_node("node_paths_probe")
    yield node
    node.destroy_node()
    if owned:
        rclpy.shutdown()


def test_provide_node_paths_lets_a_cancelled_launch_propagate_and_stops_its_poller(ros_node):
    from arena_runtime.env.robot_manager.robots_manager import RobotsManager

    paths: set[str] = set()
    pollers: list[asyncio.Task] = []

    async def launch() -> None:
        with RobotsManager.provide_node_paths(SimpleNamespace(node=ros_node), paths) as poller:
            pollers.append(poller)
            await asyncio.sleep(60)

    async def scenario() -> bool:
        task = asyncio.create_task(launch())
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        return pollers[0].cancelled()

    assert asyncio.run(scenario())
    assert "/node_paths_probe" in paths
