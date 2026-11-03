from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_people_msgs.msg")

from arena_runtime.env.task_layer import EpisodeRecord
from arena_runtime.shared import Position

from task_generator.modes import Modes
from task_generator.tasks import layer
from task_generator.tasks.task import Task


class _World:
    loaded_world_path = Path("")

    def __init__(self) -> None:
        self.forbid_clear_calls = 0
        self.world = SimpleNamespace(levels={"0": SimpleNamespace(zones=[SimpleNamespace(name="kitchen", polygon=[Position(0.0, 0.0, 0.0), Position(1.0, 0.0, 0.0)])])})

    def forbid_clear(self) -> None:
        self.forbid_clear_calls += 1

    def get_positions_on_map(self, *, n: int, safe_dist: float, forbidden_zones: object = None, forbid: bool = True) -> list[Position]:
        del safe_dist, forbidden_zones, forbid
        return [Position(float(i), 0.0, 0.0) for i in range(n)]


class _Markers:
    def __init__(self) -> None:
        self.puts: list[object] = []

    def put(self, marker: object, **kwargs: object) -> None:
        del kwargs
        self.puts.append(marker)


class _Robot:
    name = "r0"
    safe_distance = 0.5


def test_a_module_selected_after_construction_hooks_in_at_the_next_reset(rclpy_context: object) -> None:
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from arena_runtime.env.constants.rng import EpisodeRng

    world = _World()
    markers = _Markers()
    rng = EpisodeRng()
    rng.reseed(3)
    ctx = SimpleNamespace(
        robots={"r0": _Robot()},
        world_manager=world,
        realizer=SimpleNamespace(realize=lambda position, level_id: Position(0.0, 0.0, 0.0)),
        interactive=SimpleNamespace(markers=markers),
        conf=SimpleNamespace(RNG=rng, GOAL_TOLERANCE_RADIUS=SimpleNamespace(value=0.3)),
    )
    node = ROSParamServer(f"task_modules_{uuid.uuid4().hex[:8]}", start_parameter_services=False)
    for name in ("task.random.static.models", "task.random.interactive.models", "task.random.dynamic.models"):
        node.declare_parameter(name, ["shelf"])
    try:
        params = layer.declare_task_params(node)
        params.select({"robots": ["random"], "obstacles": ["random"]})
        task = Task(node=node, ctx=ctx, params=params, modules=[Modes.TM_Module.CLEAR_FORBIDDEN_ZONES])
        asyncio.run(task.set_tm_robots(Modes.TM_Robots.RANDOM))
        asyncio.run(task.set_tm_obstacles(Modes.TM_Obstacles.RANDOM))

        asyncio.run(task.reset_robots(EpisodeRecord(seed=1, modes={"modules": ["clear_forbidden_zones"]})))
        task.after_reset()
        assert world.forbid_clear_calls == 1
        assert markers.puts == []

        asyncio.run(task.reset_robots(EpisodeRecord(seed=2, modes={"modules": ["clear_forbidden_zones", "zone_edit"]})))
        task.after_reset()
        assert world.forbid_clear_calls == 2
        assert len(markers.puts) == 2

        asyncio.run(task.reset_robots(EpisodeRecord(seed=3, modes={"modules": ["clear_forbidden_zones"]})))
        task.after_reset()
        assert world.forbid_clear_calls == 3
        assert len(markers.puts) == 2
    finally:
        node.destroy_node()
