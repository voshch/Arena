from __future__ import annotations

import asyncio
import math
import uuid
from types import SimpleNamespace

import pytest
import shapely
from arena_runtime.env.constants.rng import EpisodeRng
from arena_runtime.env.task_layer import ResetPlan, TaskConfiguration
from arena_runtime.shared import Position

BAY = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]


class _Level:
    """The compacted world, answering zone lookups the way the real level does."""

    def __init__(self, zones: dict[str, list[tuple[float, float]]]) -> None:
        self._zones = zones

    def lookup_zone_polygon(self, name: str) -> list[Position] | None:
        corners = self._zones.get(name)
        return None if corners is None else [Position(x, y) for x, y in corners]


class _World:
    """Draws spots inside the polygon it is handed, from the episode's seeded world stream."""

    def __init__(self, rng: EpisodeRng, zones: dict[str, list[tuple[float, float]]]) -> None:
        self._rng = rng
        self._level = _Level(zones)
        self.polygons: list[shapely.Polygon] = []

    def world_compacted(self) -> _Level:
        return self._level

    def get_positions_on_map(
        self,
        n: int,
        safe_dist: float,
        forbidden_zones: object = None,
        forbid: bool = True,
        polygon: shapely.Polygon | None = None,
        level_id: str = "",
    ) -> list[Position]:
        del safe_dist, forbidden_zones, forbid, level_id
        if polygon is None:
            return [Position(-5.0 - index, -5.0) for index in range(n)]
        self.polygons.append(polygon)
        stream = self._rng.stream("world", "positions")
        min_x, min_y, max_x, max_y = polygon.bounds
        spots: list[Position] = []
        while len(spots) < n:
            x, y = stream.uniform(min_x, max_x), stream.uniform(min_y, max_y)
            if polygon.contains(shapely.Point(x, y)):
                spots.append(Position(x, y))
        return spots


def _ctx(world: _World, names: list[str], rng: EpisodeRng) -> SimpleNamespace:
    value = SimpleNamespace(value=0.5)
    conf = TaskConfiguration(RNG=rng, HUMAN=value, SAFE_DIST=value, GOAL_TOLERANCE_RADIUS=value, TIMEOUT=value, KG_DERIVE=value)
    robots = {name: SimpleNamespace(name=name, safe_distance=0.3) for name in names}
    return SimpleNamespace(robots=robots, world_manager=world, conf=conf)


def _node(**values: object) -> object:
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from rclpy.parameter import Parameter

    node = ROSParamServer(f"robot_hold_{uuid.uuid4().hex[:8]}", start_parameter_services=False, parameter_overrides=[Parameter(f"task.mode.{name}", value=value) for name, value in values.items()])
    node.sim_time = SimpleNamespace(sec=5.0)
    return node


def _namespace() -> object:
    from arena_rclpy_mixins.shared import Namespace

    return Namespace("task.mode").ParamNamespace()


def _plan(seed: int, zones: dict[str, list[tuple[float, float]]] | None = None, **params: object) -> tuple[_World, ResetPlan]:
    from task_generator.tasks.robots.hold.impl import TM_Hold

    rng = EpisodeRng()
    rng.reseed(seed)
    world = _World(rng, zones or {"bay": BAY})
    mode = TM_Hold(ctx=_ctx(world, ["r0", "r1"], rng), namespace=_namespace(), node=_node(**params))
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


def test_each_robot_holds_its_spot_for_hold_s_seconds() -> None:
    pytest.importorskip("rclpy")
    _, plan = _plan(2, zone="bay", hold_s=45.0)
    for name in ("r0", "r1"):
        phases = plan.robots[name].task.phases
        assert len(phases) == 1
        assert phases[0].hold_time == 45.0


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
