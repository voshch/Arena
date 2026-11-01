from __future__ import annotations

import asyncio

import pytest
from arena_runtime.testing.env_peer import ENV, PLAIN_ENV, Peers, TaskModes, fqn


def _task_modes() -> TaskModes:
    import importlib

    from task_generator.tasks.registry import MODULE_MODES, OBSTACLES_MODES, ROBOTS_MODES, walk_schemas

    for package in ("robots", "obstacles", "modules"):
        importlib.import_module(f"task_generator.tasks.{package}")
    return TaskModes(walk_schemas, *(tuple(key.value for key in registry.keys()) for registry in (ROBOTS_MODES, OBSTACLES_MODES, MODULE_MODES)))


@pytest.fixture(scope="module")
def loop(rclpy_context: object) -> asyncio.AbstractEventLoop:
    """One event loop for every test in the module, with the shared ROS node started on it."""
    from arena_mcp.core import ROS

    event_loop = asyncio.new_event_loop()
    event_loop.run_until_complete(ROS.node())
    yield event_loop
    event_loop.run_until_complete(ROS.shutdown())
    event_loop.close()


@pytest.fixture(scope="module")
def peers(loop: asyncio.AbstractEventLoop) -> Peers:
    from arena_mcp.core import ROS
    from task_generator_msgs.srv import QueryWorlds

    started = Peers(_task_modes())
    for env_id in (ENV, PLAIN_ENV):
        assert loop.run_until_complete(ROS.ready(QueryWorlds, f"{fqn(env_id)}/query/worlds", timeout=20.0))
    yield started
    started.close()


@pytest.fixture()
def run(loop: asyncio.AbstractEventLoop, peers: Peers) -> object:
    def runner(awaitable: object) -> object:
        return loop.run_until_complete(awaitable)

    return runner
