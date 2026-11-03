"""A random-mode start never lands inside its goal's tolerance."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from arena_rclpy_mixins.Time import Time
    from arena_runtime.env.world_manager.utils import WorldLayers, WorldMap, WorldOccupancy
    from arena_runtime.env.world_manager.world_manager import _occupancy_to_available, _sample_from_candidates
    from arena_runtime.shared import Position
    from task_generator.tasks.robots.random.impl import goal_clearance
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

RESOLUTION = 0.05
ROBOT_SAFE_DIST = 0.3


def open_room(cells: int) -> WorldMap:
    grid = np.full((cells, cells), WorldOccupancy.EMPTY, dtype=np.uint8)
    return WorldMap(occupancy=WorldLayers(walls=WorldOccupancy(grid)), origin=Position(x=0.0, y=0.0), resolution=RESOLUTION, time=Time())


@pytest.mark.parametrize("tolerance", [0.8, 1.0, 1.5])
def test_starts_keep_out_of_the_goal_tolerance(tolerance: float):
    world = open_room(120)
    goal = Position(x=3.0, y=3.0)
    fork = world.occupancy.fork()
    for zone in goal_clearance([goal], tolerance):
        fork.occupy(*world.tf_posr2rect(zone))
    available = _occupancy_to_available(fork.grid, ROBOT_SAFE_DIST / RESOLUTION)
    starts = _sample_from_candidates(available, 400, 0.0, np.random.default_rng(0))

    xs, ys = world.tf_grid2xy(available[:, 0], available[:, 1])
    sx, sy = world.tf_grid2xy(starts[:, 0], starts[:, 1])

    assert np.hypot(xs - goal.x, ys - goal.y).min() > tolerance
    assert np.hypot(sx - goal.x, sy - goal.y).min() > tolerance


def test_without_the_clearance_a_start_can_land_on_the_goal():
    world = open_room(120)
    goal = Position(x=3.0, y=3.0)
    available = _occupancy_to_available(world.occupancy.fork().grid, ROBOT_SAFE_DIST / RESOLUTION)
    xs, ys = world.tf_grid2xy(available[:, 0], available[:, 1])

    assert np.hypot(xs - goal.x, ys - goal.y).min() < RESOLUTION
