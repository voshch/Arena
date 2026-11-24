"""Random task mode: goal and start areas, and the optional hold at each goal, with the fulfilment unchanged."""

from __future__ import annotations

import asyncio
import math

import pytest
from arena_runtime.env.constants.rng import EpisodeRng
from arena_runtime.env.task_api import GoToPhase, ReachPhase
from arena_simulation_setup.shared.conditions import HoldNode, WithinAtom, parse_point, parse_until
from robot_fakes import BAY, GOAL_TOLERANCE, World, ctx, namespace, node

SLIVER = [(0.0, 0.0), (4.0, 0.0), (8.0, 0.0)]  # zero width: no spot fits in it


def _plan(seed: int, zones: dict[str, list[tuple[float, float]]] | None = None, **params: object) -> tuple[World, object]:
    from task_generator.tasks.robots.random.impl import TM_Random

    rng = EpisodeRng()
    rng.reseed(seed)
    world = World(rng, zones or {"bay": BAY, "sliver": SLIVER})
    mode = TM_Random(ctx=ctx(world, ["r0", "r1"], rng), namespace=namespace(), node=node(**params))
    return world, asyncio.run(mode.reset())


def _xy(pose: object) -> tuple[float, float]:
    return pose.position.x, pose.position.y


def _goals(plan: object) -> list[tuple[float, float]]:
    return [_xy(plan.robots[name].task.phases[0].pose) for name in ("r0", "r1")]


def _starts(plan: object) -> list[tuple[float, float]]:
    return [_xy(plan.robots[name].start_pose) for name in ("r0", "r1")]


def test_goals_are_drawn_inside_the_named_goal_zone() -> None:
    pytest.importorskip("rclpy")
    world, plan = _plan(1, goal_zone="bay")
    assert world.polygons[0].bounds == (0.0, 0.0, 4.0, 4.0)
    for x, y in _goals(plan):
        assert 0.0 <= x <= 4.0
        assert 0.0 <= y <= 4.0


def test_goals_are_drawn_inside_the_goal_center_and_radius() -> None:
    pytest.importorskip("rclpy")
    world, plan = _plan(5, goal_center_x=2.0, goal_center_y=2.0, goal_radius=1.0)
    assert world.polygons[0].bounds == pytest.approx((1.0, 1.0, 3.0, 3.0), abs=1e-6)
    for goal in _goals(plan):
        assert math.dist(goal, (2.0, 2.0)) <= 1.0 + 1e-9


def test_starts_are_drawn_inside_the_start_zone() -> None:
    pytest.importorskip("rclpy")
    world, plan = _plan(1, start_zone="bay")
    assert world.polygons[0].bounds == (0.0, 0.0, 4.0, 4.0)
    for x, y in _starts(plan):
        assert 0.0 <= x <= 4.0
        assert 0.0 <= y <= 4.0


def test_the_goal_and_start_areas_are_sampled_separately() -> None:
    pytest.importorskip("rclpy")
    world, plan = _plan(7, goal_center_x=2.0, goal_center_y=2.0, goal_radius=1.0, start_zone="bay")
    assert len(world.polygons) == 2
    assert world.polygons[0].bounds == pytest.approx((1.0, 1.0, 3.0, 3.0), abs=1e-6)
    assert world.polygons[1].bounds == (0.0, 0.0, 4.0, 4.0)
    for goal in _goals(plan):
        assert math.dist(goal, (2.0, 2.0)) <= 1.0 + 1e-9
    for x, y in _starts(plan):
        assert 0.0 <= x <= 4.0
        assert 0.0 <= y <= 4.0


def test_without_an_area_goals_and_starts_come_from_the_whole_map() -> None:
    pytest.importorskip("rclpy")
    world, _ = _plan(1)
    assert world.polygons == []


def test_a_start_zone_too_small_for_every_robot_is_refused() -> None:
    pytest.importorskip("rclpy")
    with pytest.raises(RuntimeError, match="fewer than 2 start positions"):
        _plan(1, start_zone="sliver")


def test_the_fulfilment_stays_a_goto_then_the_arm_phases() -> None:
    pytest.importorskip("rclpy")
    _, plan = _plan(2, goal_zone="bay")
    for name in ("r0", "r1"):
        phases = plan.robots[name].task.phases
        assert isinstance(phases[0], GoToPhase)
        assert phases[0].target is None
        assert phases[0].hold_time is None
        assert phases[0].until is None
        assert [type(later) for later in phases[1:]] == [ReachPhase, ReachPhase, ReachPhase]


def test_a_goal_zone_and_a_goal_center_together_are_refused() -> None:
    pytest.importorskip("rclpy")
    with pytest.raises(ValueError, match="not both"):
        _plan(1, goal_zone="bay", goal_center_x=2.0, goal_center_y=2.0)


def test_a_goal_sampled_from_a_zone_is_held_by_its_target_disc_not_by_the_zone() -> None:
    pytest.importorskip("rclpy")
    _, plan = _plan(4, goal_zone="bay", hold_s=45.0)
    for name in ("r0", "r1"):
        phases = plan.robots[name].task.phases
        phase = phases[0]
        assert isinstance(phase, GoToPhase)
        assert phase.target is None
        assert phase.hold_time == 0.0
        assert phase.tolerance_radius == pytest.approx(GOAL_TOLERANCE)
        assert phase.tolerance_angle == pytest.approx(math.pi)
        hold = parse_until(phase.until)
        assert isinstance(hold, HoldNode)
        assert hold.seconds == pytest.approx(45.0)
        assert isinstance(hold.node, WithinAtom)
        assert hold.node.radius == pytest.approx(GOAL_TOLERANCE)
        assert parse_point(hold.node.other) == pytest.approx(_xy(phase.pose), abs=1e-6)  # the until text keeps micrometres
        assert [type(later) for later in phases[1:]] == [ReachPhase, ReachPhase, ReachPhase]


def test_random_mode_registers_its_area_and_hold_params() -> None:
    pytest.importorskip("rclpy")
    from task_generator.modes import Modes
    from task_generator.tasks.registry import ROBOTS_MODES
    from task_generator.tasks.robots.random import RandomParams

    assert ROBOTS_MODES.meta(Modes.TM_Robots.RANDOM).schema is RandomParams
