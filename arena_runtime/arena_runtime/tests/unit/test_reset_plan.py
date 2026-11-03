"""ResetPlan merging and what EpisodeNode does with a plan."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


def _plan(tag: str, robot: str) -> object:
    from arena_runtime.env.task_layer import ResetPlan, RobotReset

    return ResetPlan(
        robots={robot: RobotReset(start_pose=f"{tag}_start", task=f"{tag}_task", humans=frozenset({f"{tag}_human"}))},
        forbidden_zones=[f"{tag}_zone"],
        obstacles=[f"{tag}_obstacle"],
        dynamic_obstacles=[f"{tag}_walker"],
        lights=(f"{tag}_light",),
    )


def test_merge_keeps_every_field_of_both_plans() -> None:
    plan = _plan("a", "r0")
    other = _plan("b", "r1")
    plan.merge(other)
    assert plan.robots["r1"] == other.robots["r1"]
    assert plan.robots["r0"].start_pose == "a_start"
    assert plan.forbidden_zones == ["a_zone", "b_zone"]
    assert plan.obstacles == ["a_obstacle", "b_obstacle"]
    assert plan.dynamic_obstacles == ["a_walker", "b_walker"]
    assert plan.lights == ("a_light", "b_light")


def test_merge_is_last_one_wins_over_the_whole_robot_entry() -> None:
    from arena_runtime.env.task_layer import RobotReset

    plan = _plan("a", "r0")
    later = RobotReset(start_pose="b_start")
    other = _plan("b", "r0")
    other.robots["r0"] = later
    plan.merge(other)
    assert plan.robots == {"r0": later}
    assert plan.robots["r0"].task is None
    assert plan.robots["r0"].humans == frozenset()


def test_plans_do_not_share_their_containers() -> None:
    from arena_runtime.env.task_layer import ResetPlan, RobotReset

    first, second = ResetPlan(), ResetPlan()
    first.robots["r"] = RobotReset(start_pose="start")
    first.forbidden_zones.append("zone")
    first.obstacles.append("obstacle")
    first.dynamic_obstacles.append("walker")
    assert second == ResetPlan()


class _Manager:
    def __init__(self, log: list[str], name: str) -> None:
        self.log = log
        self.name = name

    async def submit_task(self, request: object, start: object, pedestrians: frozenset[str]) -> None:
        self.log.append(f"submit {self.name} {request} from {start} seeing {sorted(pedestrians)}")

    async def reset(self, ctx: object) -> dict:
        self.log.append(f"reset {self.name} at {ctx.start_pose}")
        return {}


class _Task:
    def __init__(self, log: list[str], plan: object) -> None:
        self.log = log
        self.plan = plan

    async def reset_robots(self, record: object) -> object:
        self.log.append(f"reset_robots {record.seed}")
        return self.plan

    async def reset_obstacles(self, plan: object, record: object) -> None:
        self.log.append(f"reset_obstacles {record.seed}")
        plan.obstacles = ["crate"]
        plan.dynamic_obstacles = ["walker"]
        plan.lights = ("lamp",)

    def after_reset(self) -> None:
        self.log.append("after_reset")


class _Environment:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    async def before_reset_episode(self) -> None:
        self.log.append("before_reset")

    async def after_reset_episode(self) -> None:
        self.log.append("after_reset_episode")

    async def respawn(self, callback: object) -> None:
        self.log.append("respawn")
        await callback()

    async def spawn_obstacles(self, obstacles: list) -> None:
        self.log.append(f"spawn {obstacles}")

    async def spawn_dynamic_obstacles(self, obstacles: list) -> None:
        self.log.append(f"spawn dynamic {obstacles}")

    async def spawn_mode_lights(self, lights: object) -> None:
        self.log.append(f"lights {lights}")


def test_the_node_applies_the_plan_between_the_robot_and_the_obstacle_reset() -> None:
    pytest.importorskip("rclpy")
    from arena_runtime.env.constants.rng import EpisodeRng
    from arena_runtime.env.episode_node import EpisodeNode
    from arena_runtime.env.task_layer import EpisodeRecord, ResetPlan, RobotReset

    log: list[str] = []
    plan = ResetPlan(robots={"r0": RobotReset(start_pose="start0", task="task0", humans=frozenset({"h0"})), "r1": RobotReset(start_pose="start1", humans=frozenset({"h1"}))}, forbidden_zones=["zone"])

    async def nothing() -> None:
        return None

    node = SimpleNamespace(
        _task=_Task(log, plan),
        _robots_manager=SimpleNamespace(set_up=nothing, launch_pending=nothing, managers={name: _Manager(log, name) for name in ("r0", "r1", "r2")}),
        _world_manager=SimpleNamespace(forbid=lambda zones: log.append(f"forbid {zones}")),
        _environment_manager=_Environment(log),
        _pub_reset_start=SimpleNamespace(publish=lambda message: None),
        _pub_reset_end=SimpleNamespace(publish=lambda message: None),
        _apply_staged_params=lambda: None,
        _apply_staged_human_params=nothing,
        _consecutive_reset_failures=0,
        conf=SimpleNamespace(General=SimpleNamespace(RNG=EpisodeRng())),
        sim_time=SimpleNamespace(sec=0.0),
        episode_id=0,
        get_logger=lambda: SimpleNamespace(error=log.append),
    )

    asyncio.run(EpisodeNode._reset_episode(node, EpisodeRecord(seed=3)))

    assert log[log.index("reset_robots 3") : log.index("respawn") + 1] == ["reset_robots 3", "submit r0 task0 from start0 seeing ['h0']", "forbid ['zone']", "reset_obstacles 3", "respawn"]
    assert {"spawn ['crate']", "spawn dynamic ['walker']", "lights ('lamp',)"} <= set(log)
    assert {entry for entry in log if entry.startswith("reset r")} == {"reset r0 at start0", "reset r1 at start1", "reset r2 at None"}
