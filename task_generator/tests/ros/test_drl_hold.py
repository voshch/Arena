"""DrlAdapter parks the planner while a phase holds at its goal and resumes it when the robot leaves."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace


def _adapter():
    from task_generator.tasks.robots.adapters.mobile.drl import DrlAdapter

    class _Edge:
        def __init__(self) -> None:
            self.cancels = 0

        async def request_cancel(self) -> None:
            self.cancels += 1

    class _Adapter(DrlAdapter):
        def __init__(self) -> None:
            self._edge_node = _Edge()
            self._parked = False
            self.dispatched: list[object] = []

        async def dispatch_phase(self, phase, robot) -> None:
            self._parked = False
            self.dispatched.append(phase)

    return _Adapter()


def test_hold_parks_once_and_leaving_the_tolerance_redispatches():
    async def run() -> None:
        adapter = _adapter()
        robot = SimpleNamespace(waiting=False, node=SimpleNamespace(event_loop=asyncio.get_running_loop()))
        phase = object()

        adapter.on_phase_tick(phase, robot)
        await asyncio.sleep(0)
        assert adapter._edge_node.cancels == 0
        assert adapter.dispatched == []

        robot.waiting = True
        for _ in range(3):
            adapter.on_phase_tick(phase, robot)
            await asyncio.sleep(0)
        assert adapter._edge_node.cancels == 1
        assert adapter.dispatched == []

        robot.waiting = False
        for _ in range(3):
            adapter.on_phase_tick(phase, robot)
            await asyncio.sleep(0)
        assert adapter._edge_node.cancels == 1
        assert adapter.dispatched == [phase]

    asyncio.run(run())


def test_tick_before_the_edge_node_exists_does_nothing():
    adapter = _adapter()
    adapter._edge_node = None
    adapter.on_phase_tick(object(), SimpleNamespace(waiting=True))
    assert adapter._parked is False
