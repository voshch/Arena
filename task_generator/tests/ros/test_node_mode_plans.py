"""Plans of the robot modes that need a real parameter server: demo and scenario."""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

pytest.importorskip("rclpy")
pytest.importorskip("arena_people_msgs.msg")

SCENARIO = {
    "format": "arena-scenario/2",
    "robots": [
        {"start": [1.0, 2.0, 0.0], "level": "0", "goal_level": "0", "phases": [{"goto": [3.0, 4.0, 0.0]}]},
        {"start": [5.0, 6.0, 0.0], "level": "0", "phases": []},
    ],
}


class _World:
    loaded_world = ""

    def __init__(self) -> None:
        self.mechanism_map: tuple[bool, list[str]] | None = None

    def get_positions_on_map(self, n: int, safe_dist: float, **kwargs: object) -> list[object]:
        from arena_runtime.shared import Position

        del safe_dist, kwargs
        return [Position(10.0 + index, 20.0, 0.0) for index in range(n)]

    def world_compacted(self) -> object:
        from arena_simulation_setup.tree.World.Scenario import converter

        return SimpleNamespace(zone_converter=lambda rng: converter)

    def set_mechanism_map(self, enabled: bool, agents: list[str]) -> None:
        self.mechanism_map = (enabled, agents)


def _robot(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, safe_distance=0.5, robot=SimpleNamespace(sim_path=f"sim/{name}"))


def _reset(key: object, registry: object, world: _World, names: tuple[str, ...]) -> object:
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_runtime.env.constants.rng import EpisodeRng
    from arena_runtime.env.realizer import Realizer

    from arena_runtime.env.task_layer import declare_mode_schemas

    from task_generator.tasks import layer

    async def main() -> object:
        node = ArenaMixinNode(f"mode_plans_{uuid.uuid4().hex[:8]}")
        try:
            declare_mode_schemas(layer, node)
            rng = EpisodeRng()
            rng.reseed(3)
            ctx = SimpleNamespace(
                robots={name: _robot(name) for name in names},
                world_manager=world,
                realizer=Realizer(level_origins={"0": (100.0, 0.0)}),
                conf=SimpleNamespace(RNG=rng),
            )
            mode = registry.get(key)(ctx=ctx, namespace=registry.meta(key).namespace, node=node)
            return await mode.reset()
        finally:
            node.destroy_node()

    return asyncio.run(main())


def test_demo_plan_drives_the_polygon_from_its_center_and_forbids_every_corner(rclpy_context: object) -> None:
    from arena_runtime.env.task_api import GoToPhase, PlayGesturePhase

    from task_generator.modes import Modes
    from task_generator.tasks.registry import ROBOTS_MODES

    plan = _reset(Modes.TM_Robots.DEMO, ROBOTS_MODES, _World(), ("r0", "r1"))

    assert sorted(plan.robots) == ["r0", "r1"]
    start = plan.robots["r0"].start_pose
    assert (start.position.x, start.position.y) == (10.0, 20.0)
    phases = plan.robots["r0"].task.phases
    assert [type(phase) for phase in phases] == [GoToPhase, PlayGesturePhase] * 4 + [GoToPhase]
    assert phases[-1].pose == start
    assert len(plan.forbidden_zones) == 2 * (1 + 4)
    assert (plan.forbidden_zones[0].x, plan.forbidden_zones[0].y) == (10.0, 20.0)


def test_scenario_plan_keeps_the_declared_starts_and_gives_an_idle_robot_no_task(rclpy_context: object) -> None:
    from arena_simulation_setup.tree import local_root

    from task_generator.modes import Modes
    from task_generator.tasks.registry import ROBOTS_MODES

    world = _World()
    world.loaded_world = f"plans_{uuid.uuid4().hex[:8]}"
    directory = local_root() / "worlds" / world.loaded_world / "scenarios" / "default"
    directory.mkdir(parents=True)
    (directory / "scenario.yaml").write_text(yaml.safe_dump(SCENARIO))

    plan = _reset(Modes.TM_Robots.SCENARIO, ROBOTS_MODES, world, ("r0", "r1", "r2"))

    assert sorted(plan.robots) == ["r0", "r1"]
    assert plan.robots["r0"].start_pose.position.x == pytest.approx(101.0)
    assert plan.robots["r0"].task.phases[0].pose.position.x == pytest.approx(103.0)
    assert plan.robots["r1"].task is None
    assert all(entry.humans == frozenset() for entry in plan.robots.values())
    assert np.allclose([zone.radius for zone in plan.forbidden_zones], 0.5)
    assert [(zone.x, zone.y) for zone in plan.forbidden_zones] == [(101.0, 2.0), (103.0, 4.0), (105.0, 6.0)]
    assert world.mechanism_map == (False, ["r0", "sim/r0", "r1", "sim/r1"])
