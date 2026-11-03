from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from arena_runtime.env.constants.rng import EpisodeRng
from arena_runtime.env.task_layer import TaskConfiguration
from arena_runtime.shared import Position


class _World:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def get_positions_on_map(self, n: int, safe_dist: float, **kwargs: object) -> list[Position]:
        del kwargs
        self.calls.append((n, safe_dist))
        offset = 10.0 * len(self.calls)
        return [Position(offset + i, offset + i) for i in range(n)]


def _ctx(world: _World, names: list[str]) -> SimpleNamespace:
    rng = EpisodeRng()
    rng.reseed(7)
    value = SimpleNamespace(value=0.5)
    conf = TaskConfiguration(RNG=rng, HUMAN=value, SAFE_DIST=value, GOAL_TOLERANCE_RADIUS=value, TIMEOUT=value, KG_DERIVE=value)
    robots = {name: SimpleNamespace(name=name, safe_distance=0.3) for name in names}
    return SimpleNamespace(robots=robots, world_manager=world, conf=conf)


def test_random_mode_returns_starts_and_tasks_and_drives_no_robot() -> None:
    pytest.importorskip("rclpy")
    from task_generator.tasks.robots.random.impl import TM_Random

    world = _World()
    mode = TM_Random(ctx=_ctx(world, ["r0", "r1"]), namespace=SimpleNamespace(), node=SimpleNamespace())
    plan = asyncio.run(mode.reset())

    assert sorted(plan.robots) == ["r0", "r1"]
    goals, starts = world.calls
    assert goals == (2, 0) and starts == (2, 0.3)
    for index, name in enumerate(["r0", "r1"]):
        start = plan.robots[name].start_pose
        assert (start.position.x, start.position.y) == (20.0 + index, 20.0 + index)
        goto = plan.robots[name].task.phases[0]
        assert (goto.pose.position.x, goto.pose.position.y) == (10.0 + index, 10.0 + index)
    assert plan.forbidden_zones == []


class _Param:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def __getitem__(self, kind: object) -> _Param:
        del kind
        return self

    def __call__(self, name: object, default: object = None, **kwargs: object) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(value=self.values.get(str(name).rsplit(".", 1)[-1], default))


def _node(**values: object) -> SimpleNamespace:
    return SimpleNamespace(ROSParam=_Param(values), sim_time=SimpleNamespace(sec=5.0))


def _namespace() -> object:
    from arena_rclpy_mixins.shared import Namespace

    return Namespace("task.mode").ParamNamespace()


def test_stationary_mode_samples_starts_and_plans_no_task() -> None:
    pytest.importorskip("rclpy")
    from task_generator.tasks.robots.stationary.impl import TM_Stationary

    world = _World()
    mode = TM_Stationary(ctx=_ctx(world, ["r0", "r1"]), namespace=_namespace(), node=_node())
    plan = asyncio.run(mode.reset())

    assert list(plan.robots) == ["r0", "r1"]
    assert [plan.robots[name].start_pose.position.x for name in ("r0", "r1")] == [10.0, 11.0]
    assert all(entry.task is None for entry in plan.robots.values())
    assert plan.forbidden_zones == []


def test_stationary_mode_parks_every_robot_on_the_configured_pose() -> None:
    pytest.importorskip("rclpy")
    from task_generator.tasks.robots.stationary.impl import TM_Stationary

    world = _World()
    mode = TM_Stationary(ctx=_ctx(world, ["r0", "r1"]), namespace=_namespace(), node=_node(pos_x=1.5, pos_y=2.5))
    plan = asyncio.run(mode.reset())

    assert world.calls == []
    assert {(entry.start_pose.position.x, entry.start_pose.position.y) for entry in plan.robots.values()} == {(1.5, 2.5)}
    assert all(entry.task is None for entry in plan.robots.values())


def test_explore_mode_plans_like_random_and_starts_each_robot_timeout() -> None:
    pytest.importorskip("rclpy")
    from task_generator.tasks.robots.explore.impl import TM_Explore

    mode = TM_Explore(ctx=_ctx(_World(), ["r0", "r1"]), namespace=_namespace(), node=_node())
    plan = asyncio.run(mode.reset())

    assert sorted(plan.robots) == ["r0", "r1"]
    assert (plan.robots["r0"].task.phases[0].pose.position.x, plan.robots["r1"].task.phases[0].pose.position.x) == (10.0, 11.0)
    assert sorted(mode._timeouts) == ["r0", "r1"]


class _Markers:
    def erase_prefix(self, prefix: str) -> None:
        del prefix


def test_guided_mode_plans_one_goto_to_each_robots_own_start() -> None:
    pytest.importorskip("rclpy")
    from task_generator.tasks.robots.guided.impl import TM_Guided

    ctx = _ctx(_World(), ["r0", "r1"])
    ctx.interactive = SimpleNamespace(markers=_Markers())
    node = _node()
    node.rosparam = {list[list[float]]: SimpleNamespace(set=lambda name, value: None)}
    mode = TM_Guided(ctx=ctx, namespace=_namespace(), node=node)
    plan = asyncio.run(mode.reset())

    assert sorted(plan.robots) == ["r0", "r1"]
    for entry in plan.robots.values():
        assert [phase.pose for phase in entry.task.phases] == [entry.start_pose]


class _FreeSpots:
    """A world that samples the first free spots, honoring forbidden zones the way the occupancy grid does."""

    SPOTS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)

    def __init__(self) -> None:
        self.zones: list[object] = []

    def forbid(self, zones: list[object]) -> None:
        self.zones.extend(zones)

    def get_positions_on_map(self, n: int, safe_dist: float, forbidden_zones: list[object] | None = None, **kwargs: object) -> list[Position]:
        del safe_dist, kwargs
        blocked = [*self.zones, *(forbidden_zones or [])]
        free = [spot for spot in self.SPOTS if not any(abs(spot - zone.x) < zone.radius for zone in blocked)]
        return [Position(spot, 0.0) for spot in free[:n]]


def test_composite_later_sub_modes_keep_out_of_the_zones_of_earlier_ones() -> None:
    pytest.importorskip("rclpy")
    from arena_runtime.env.task_layer import ResetPlan
    from arena_runtime.shared import PositionRadius

    from task_generator.tasks.robots import TM_Robots
    from task_generator.tasks.robots.composite import TM_Composite
    from task_generator.tasks.robots.random.impl import TM_Random

    zone = PositionRadius(x=0.0, y=0.0, radius=1.5)

    class _Declared(TM_Robots):
        async def reset(self) -> ResetPlan:
            return ResetPlan(forbidden_zones=[zone])

    world = _FreeSpots()
    ctx = _ctx(world, ["r0"])
    kwargs = {"ctx": ctx, "namespace": _namespace(), "node": _node()}
    mode = TM_Composite(sub_modes=[_Declared(**kwargs), TM_Random(**kwargs)], **kwargs)
    plan = asyncio.run(mode.reset())

    assert plan.forbidden_zones == [zone]
    assert plan.robots["r0"].start_pose.position.x >= zone.radius
    assert plan.robots["r0"].task.phases[0].pose.position.x >= zone.radius
