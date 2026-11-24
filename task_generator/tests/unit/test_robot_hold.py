"""Hold task mode: where the spots are drawn, and the target disc each robot must stay within for hold_s seconds."""

from __future__ import annotations

import asyncio
import math

import pytest
from arena_runtime.env.constants.rng import EpisodeRng
from arena_runtime.env.task_layer import ResetPlan
from arena_simulation_setup.shared.conditions import HoldNode, WithinAtom, parse_point, parse_until
from robot_fakes import BAY, GOAL_TOLERANCE, World, ctx, namespace, node


def _plan(seed: int, zones: dict[str, list[tuple[float, float]]] | None = None, **params: object) -> tuple[World, ResetPlan]:
    from task_generator.tasks.robots.hold.impl import TM_Hold

    rng = EpisodeRng()
    rng.reseed(seed)
    world = World(rng, zones or {"bay": BAY})
    mode = TM_Hold(ctx=ctx(world, ["r0", "r1"], rng), namespace=namespace(), node=node(**params))
    return world, asyncio.run(mode.reset())


def _spot(plan: ResetPlan, name: str) -> tuple[float, float]:
    pose = plan.robots[name].task.phases[0].pose
    return pose.position.x, pose.position.y


def _spots(plan: ResetPlan) -> list[tuple[float, float]]:
    return [_spot(plan, name) for name in ("r0", "r1")]


def test_hold_spots_are_drawn_inside_the_named_zone() -> None:
    pytest.importorskip("rclpy")
    for seed in (1, 2, 3, 4):
        world, plan = _plan(seed, zone="bay")
        assert world.polygons[0].bounds == (0.0, 0.0, 4.0, 4.0)
        for x, y in _spots(plan):
            assert 0.0 <= x <= 4.0
            assert 0.0 <= y <= 4.0


def test_the_seed_picks_the_hold_spots() -> None:
    pytest.importorskip("rclpy")
    _, first = _plan(3, zone="bay")
    _, again = _plan(3, zone="bay")
    _, other = _plan(4, zone="bay")
    assert _spots(first) == _spots(again)
    assert _spots(first) != _spots(other)


def test_center_and_radius_draw_each_spot_inside_the_disc() -> None:
    pytest.importorskip("rclpy")
    world, plan = _plan(5, center_x=2.0, center_y=2.0, radius=1.0)
    assert world.polygons[0].bounds == pytest.approx((1.0, 1.0, 3.0, 3.0), abs=1e-6)
    for spot in _spots(plan):
        assert math.dist(spot, (2.0, 2.0)) <= 1.0 + 1e-9


def test_a_spot_sampled_from_a_zone_is_held_by_the_target_disc_around_it_not_by_the_zone() -> None:
    pytest.importorskip("rclpy")
    _, plan = _plan(2, zone="bay", hold_s=45.0)
    for name in ("r0", "r1"):
        phases = plan.robots[name].task.phases
        assert len(phases) == 1
        phase = phases[0]
        assert phase.target is None
        assert phase.hold_time == 0.0
        assert phase.tolerance_radius == pytest.approx(GOAL_TOLERANCE)
        assert phase.tolerance_angle == pytest.approx(math.pi)
        hold = parse_until(phase.until)
        assert isinstance(hold, HoldNode)
        assert hold.seconds == pytest.approx(45.0)
        assert isinstance(hold.node, WithinAtom)
        assert hold.node.radius == pytest.approx(GOAL_TOLERANCE)
        assert parse_point(hold.node.other) == pytest.approx(_spot(plan, name), abs=1e-6)  # the until text keeps micrometres


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        ({"zone": "bay", "center_x": 2.0, "center_y": 2.0}, "not both"),
        ({}, "both task.hold.center_x"),
        ({"zone": "gap"}, "not a zone of the loaded world"),
    ],
)
def test_declarations_that_do_not_name_one_region_are_refused(params: dict[str, object], reason: str) -> None:
    pytest.importorskip("rclpy")
    with pytest.raises(ValueError, match=reason):
        _plan(1, **params)


def test_hold_mode_registers_under_its_own_name() -> None:
    pytest.importorskip("rclpy")
    from task_generator.modes import Modes
    from task_generator.tasks.registry import ROBOTS_MODES
    from task_generator.tasks.robots.hold import HoldParams

    assert Modes.TM_Robots.HOLD.value == "hold"
    assert ROBOTS_MODES.meta(Modes.TM_Robots.HOLD).schema is HoldParams
