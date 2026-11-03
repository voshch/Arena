"""Slots packed from the world map bounds keep the walls of the training worlds apart in the shared sim."""

from __future__ import annotations

import functools
import itertools
import pathlib

import pytest

pytest.importorskip("arena_runtime_msgs.msg")
pytest.importorskip("arena_runtime.env.world_manager.utils")

import arena_runtime_msgs.msg  # noqa: E402
import builtin_interfaces.msg  # noqa: E402
import rclpy.time  # noqa: E402
from arena_simulation_setup.tree.World import WorldView  # noqa: E402
from arena_runtime.env.world_manager.utils import WorldMap  # noqa: E402

from arena_runtime.registry import EnvRegistry  # noqa: E402

WORLDS = pathlib.Path(__file__).resolve().parents[4] / "arena_simulation_setup" / "worlds"
TRAINING_WORLDS = ("map_empty", "demo", "sbend_corridor", "hospital_2")

Box = tuple[float, float, float, float]


@functools.cache
def _walls_and_bounds(name: str) -> tuple[Box, Box]:
    view = WorldView(WORLDS / name)
    world = view.load()
    origins = view.level_origins() or {level_id: (0.0, 0.0) for level_id in world.level_ids}
    compacted = world.compact_world(origins=origins)
    xs = [x for wall in compacted.all_walls for x in (wall.start.x, wall.end.x)]
    ys = [y for wall in compacted.all_walls for y in (wall.start.y, wall.end.y)]
    bounds = WorldMap.from_world_description(compacted, resolution=0.05, time=rclpy.time.Time()).bounds
    return (min(xs), min(ys), max(xs), max(ys)), bounds


def _extent(bounds: Box) -> arena_runtime_msgs.msg.WorldExtent:
    return arena_runtime_msgs.msg.WorldExtent(x_min=bounds[0], y_min=bounds[1], x_max=bounds[2], y_max=bounds[3])


def _shift(box: Box, reference: tuple[float, float]) -> Box:
    return box[0] + reference[0], box[1] + reference[1], box[2] + reference[0], box[3] + reference[1]


def _overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


@pytest.mark.parametrize("name", TRAINING_WORLDS)
def test_map_bounds_hold_every_wall_of_every_level(name: str):
    walls, bounds = _walls_and_bounds(name)
    assert bounds[0] <= walls[0] and bounds[1] <= walls[1] and walls[2] <= bounds[2] and walls[3] <= bounds[3]


@pytest.mark.parametrize("order", list(itertools.permutations(TRAINING_WORLDS)))
def test_training_worlds_never_share_space_in_any_spawn_order(order: tuple[str, ...]):
    registry = EnvRegistry(slot_buffer=5.0)
    realized: dict[str, Box] = {}
    for name in order:
        walls, bounds = _walls_and_bounds(name)
        env_id, _ = registry.reserve(now=builtin_interfaces.msg.Time())
        realized[name] = _shift(walls, registry.place(env_id, _extent(bounds)).reference)
    for a, b in itertools.combinations(order, 2):
        assert not _overlap(realized[a], realized[b]), f"{a} {realized[a]} overlaps {b} {realized[b]}"
