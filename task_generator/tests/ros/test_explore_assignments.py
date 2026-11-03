from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

TIMEOUT = 60


@pytest.fixture(autouse=True)
def _ros_gate() -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_people_msgs.msg")


class _Manager:
    safe_distance = 0.5

    def __init__(self, name: str, done: bool) -> None:
        self.name = name
        self.done = done
        self.submitted: list[object] = []
        self.moved: list[object] = []

    @property
    async def is_done(self) -> bool:
        return self.done

    async def submit_task(self, request: object) -> None:
        self.submitted.append(request)
        self.done = False

    async def move(self, pose: object) -> None:
        self.moved.append(pose)


class _World:
    def get_positions_on_map(self, n: int, safe_dist: float, forbidden_zones: object = None) -> list[object]:
        from arena_runtime.shared import Position

        return [Position(float(i), 0.0, 0.0) for i in range(n)]

    def get_position_on_map(self, safe_dist: float, forbid: bool) -> object:
        from arena_runtime.shared import Position

        return Position(7.0, 7.0, 0.0)


def _explore(managers: list[_Manager]) -> tuple[object, SimpleNamespace]:
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.env.constants.rng import EpisodeRng

    from task_generator.tasks.robots.explore.impl import TM_Explore

    rng = EpisodeRng()
    rng.reseed(3)
    node = SimpleNamespace(sim_time=SimpleNamespace(sec=0))
    ctx = SimpleNamespace(
        robots={manager.name: manager for manager in managers},
        world_manager=_World(),
        conf=SimpleNamespace(RNG=rng, TIMEOUT=SimpleNamespace(value=TIMEOUT), GOAL_TOLERANCE_RADIUS=SimpleNamespace(value=0.3)),
    )
    mode = TM_Explore(ctx=ctx, namespace=Namespace("task.explore"), node=node)
    asyncio.run(mode.reset())
    return mode, node


def _ended(outcome: str, reason: str = "") -> object:
    from arena_runtime.env.task_api import PhaseState

    return PhaseState(index=0, phase=None, monitor=None, outcome=outcome, reason=reason)


def test_explore_reclaims_a_robot_after_its_external_task_finishes() -> None:
    manager = _Manager("r0", done=False)
    mode, _ = _explore([manager])
    asyncio.run(mode.tick({}, ["r0"]))
    assert manager.submitted == []
    manager.done = True
    asyncio.run(mode.tick({"r0": _ended("met")}, []))
    assert len(manager.submitted) == 1


def test_explore_keeps_a_robot_whose_external_task_still_has_phases() -> None:
    manager = _Manager("r0", done=False)
    mode, node = _explore([manager])
    asyncio.run(mode.tick({}, ["r0"]))
    node.sim_time = SimpleNamespace(sec=TIMEOUT)
    asyncio.run(mode.tick({"r0": _ended("met")}, []))
    assert manager.submitted == []
    assert manager.moved == []


def test_explore_reclaims_a_robot_after_its_external_task_was_stopped() -> None:
    manager = _Manager("r0", done=False)
    mode, _ = _explore([manager])
    asyncio.run(mode.tick({}, ["r0"]))
    assert manager.submitted == []
    manager.done = True
    asyncio.run(mode.tick({}, []))
    assert len(manager.submitted) == 1


def test_explore_reclaims_a_robot_assigned_while_already_idle() -> None:
    manager = _Manager("r0", done=True)
    mode, _ = _explore([manager])
    asyncio.run(mode.tick({}, ["r0"]))
    assert len(manager.submitted) == 1
