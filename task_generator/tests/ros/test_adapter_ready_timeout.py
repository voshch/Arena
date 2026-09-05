from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def _stuck_adapter():
    from task_generator.tasks.robots.adapters import Adapter

    class _StuckClient:
        def action_endpoint(self) -> str:
            return "/env_0/jackal_0/goto_pose"

        async def wait_ready(self) -> None:
            await asyncio.Event().wait()

    class _StuckAdapter(Adapter):
        kind = "stuck"

        def __init__(self) -> None:
            self._clients = {"goto_pose": _StuckClient()}

        async def dispatch_phase(self, phase, robot) -> None: ...

    return _StuckAdapter()


def test_await_ready_raises_on_expiry():
    adapter = _stuck_adapter()
    robot = SimpleNamespace(name="jackal_0")

    with pytest.raises(TimeoutError) as excinfo:
        asyncio.run(adapter.await_ready(robot, set(), 0.2))

    msg = str(excinfo.value)
    assert "'stuck'" in msg
    assert "'jackal_0'" in msg
    assert "/env_0/jackal_0/goto_pose" in msg


def test_await_ready_infinite_timeout_stays_unbounded():
    adapter = _stuck_adapter()
    robot = SimpleNamespace(name="jackal_0")

    async def _race() -> None:
        task = asyncio.ensure_future(adapter.await_ready(robot, set(), float("inf")))
        await asyncio.sleep(0.3)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_race())


def test_await_ready_passes_through_inner_timeout():
    from task_generator.tasks.robots.adapters import Adapter

    class _InnerTimeoutAdapter(Adapter):
        kind = "inner"

        def __init__(self) -> None:
            self._clients = {}

        async def wait_until_ready(self, robot, node_paths) -> None:
            raise TimeoutError("planner did not send init_ack within 60s")

        async def dispatch_phase(self, phase, robot) -> None: ...

    robot = SimpleNamespace(name="jackal_0")
    for timeout in (5.0, float("inf")):
        with pytest.raises(TimeoutError) as excinfo:
            asyncio.run(_InnerTimeoutAdapter().await_ready(robot, set(), timeout))
        assert "init_ack" in str(excinfo.value)
