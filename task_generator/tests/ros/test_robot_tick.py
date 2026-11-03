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


def _explore(managers: list[_Manager], *, reset: bool = True) -> tuple[object, SimpleNamespace]:
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
    if reset:
        asyncio.run(mode.reset())
    return mode, node


def test_explore_gives_a_robot_whose_task_finished_a_fresh_goal() -> None:
    from arena_runtime.env.task_api import GoToPhase

    finished, running = _Manager("r0", done=False), _Manager("r1", done=False)
    mode, _ = _explore([finished, running])
    finished.done = True
    asyncio.run(mode.tick({"r0": object()}, []))
    (request,) = finished.submitted
    assert [type(phase) for phase in request.phases] == [GoToPhase]
    assert (request.phases[0].pose.position.x, request.phases[0].pose.position.y) == (7.0, 7.0)
    assert running.submitted == []
    assert running.moved == []


def test_explore_also_restarts_a_robot_stopped_without_a_phase_ending() -> None:
    stopped = _Manager("r0", done=False)
    mode, _ = _explore([stopped])
    stopped.done = True
    asyncio.run(mode.tick({}, []))
    assert len(stopped.submitted) == 1
    assert stopped.moved == []


def test_explore_teleports_a_robot_that_ran_into_the_timeout_and_restarts_its_clock() -> None:
    stuck, fresh = _Manager("r0", done=False), _Manager("r1", done=False)
    mode, node = _explore([stuck, fresh])
    node.sim_time = SimpleNamespace(sec=TIMEOUT - 1)
    asyncio.run(mode.tick({}, []))
    assert stuck.moved == []
    node.sim_time = SimpleNamespace(sec=TIMEOUT)
    asyncio.run(mode.tick({}, []))
    assert len(stuck.moved) == len(fresh.moved) == 1
    asyncio.run(mode.tick({}, []))
    assert len(stuck.moved) == 1


def test_explore_holds_a_robot_assigned_from_outside_while_its_task_runs_until_the_next_reset() -> None:
    assigned, free = _Manager("r0", done=False), _Manager("r1", done=True)
    mode, node = _explore([assigned, free])
    asyncio.run(mode.tick({}, ["r0"]))
    assert assigned.submitted == []
    assert len(free.submitted) == 1
    node.sim_time = SimpleNamespace(sec=TIMEOUT)
    asyncio.run(mode.tick({}, []))
    assert assigned.submitted == []
    assert assigned.moved == []
    asyncio.run(mode.reset())
    node.sim_time = SimpleNamespace(sec=2 * TIMEOUT)
    asyncio.run(mode.tick({}, []))
    assert len(assigned.moved) == 1


def test_explore_does_not_teleport_a_timed_out_robot_assigned_from_outside() -> None:
    assigned = _Manager("r0", done=False)
    mode, node = _explore([assigned])
    node.sim_time = SimpleNamespace(sec=TIMEOUT)
    asyncio.run(mode.tick({}, ["r0"]))
    assert assigned.moved == []
    assert assigned.submitted == []


def test_explore_ticks_before_its_first_reset() -> None:
    manager = _Manager("r0", done=True)
    mode, _ = _explore([manager], reset=False)
    asyncio.run(mode.tick({}, []))
    assert len(manager.submitted) == 1


def test_explore_is_never_done() -> None:
    mode, _ = _explore([_Manager("r0", done=True)])
    assert asyncio.run(mode.done) is False


def _composite(*, failing: frozenset[str] = frozenset()) -> tuple[object, dict[str, tuple[dict, list]]]:
    from arena_rclpy_mixins.shared import Namespace

    from task_generator.tasks.robots import TM_Robots
    from task_generator.tasks.robots.composite import TM_Composite

    seen: dict[str, tuple[dict, list]] = {}

    class _Recording(TM_Robots):
        async def tick(self, ended: dict, assigned: list) -> None:
            key = ",".join(self._ctx.robots)
            seen[key] = (ended, list(assigned))
            if key in failing:
                raise RuntimeError(key)

    node = SimpleNamespace()
    first = _Recording(ctx=SimpleNamespace(robots={"a": 1, "b": 2}), namespace=Namespace("task.first"), node=node)
    second = _Recording(ctx=SimpleNamespace(robots={"c": 3}), namespace=Namespace("task.second"), node=node)
    composite = TM_Composite(ctx=SimpleNamespace(robots={"a": 1, "b": 2, "c": 3}), namespace=Namespace("task.composite"), node=node, sub_modes=[first, second])
    return composite, seen


def test_composite_hands_each_sub_mode_only_the_phases_and_assignments_of_its_robots() -> None:
    composite, seen = _composite()
    ended = {"a": object(), "c": object(), "other": object()}
    asyncio.run(composite.tick(ended, ["b", "c", "other"]))
    assert seen == {"a,b": ({"a": ended["a"]}, ["b"]), "c": ({"c": ended["c"]}, ["c"])}


def test_composite_ticks_every_sub_mode_when_one_raises_then_raises_the_first_error() -> None:
    composite, seen = _composite(failing=frozenset({"a,b", "c"}))
    with pytest.raises(RuntimeError, match="a,b"):
        asyncio.run(composite.tick({}, []))
    assert set(seen) == {"a,b", "c"}
