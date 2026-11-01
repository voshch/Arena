"""Scenario zone draws are valid only where the occupancy grid is free within the safe distance."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import shapely
    from arena_rclpy_mixins.Time import Time

    from arena_runtime.env.world_manager.utils import WorldLayers, WorldMap, WorldOccupancy
    from arena_runtime.shared import Position
    from task_generator.tasks.obstacles._validation import make_is_valid
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

RESOLUTION = 0.05


def room_with_wall() -> WorldMap:
    """A 6 x 6 m free room with a 0.2 m thick wall along x = 3 m."""
    grid = np.full((120, 120), WorldOccupancy.EMPTY, dtype=np.uint8)
    world = WorldMap(occupancy=WorldLayers(walls=WorldOccupancy(grid)), origin=Position(x=0.0, y=0.0), resolution=RESOLUTION, time=Time())
    world.occupancy.obstacle_occupy(world.tf_poly2mask(shapely.box(2.9, 0.0, 3.1, 6.0)))
    return world


@pytest.mark.parametrize("safe_dist", [0.0, -0.5])
def test_no_safe_distance_disables_the_check(safe_dist: float) -> None:
    assert make_is_valid(room_with_wall(), safe_dist) is None


def test_a_point_clear_of_the_wall_by_the_safe_distance_is_valid() -> None:
    is_valid = make_is_valid(room_with_wall(), 0.5)
    assert is_valid is not None
    assert is_valid(Position(x=1.5, y=3.0))
    assert is_valid(Position(x=4.5, y=3.0))


def test_a_point_closer_to_the_wall_than_the_safe_distance_is_invalid() -> None:
    is_valid = make_is_valid(room_with_wall(), 0.5)
    assert is_valid is not None
    assert not is_valid(Position(x=3.0, y=3.0))
    assert not is_valid(Position(x=2.6, y=3.0))
    assert is_valid(Position(x=2.3, y=3.0))


def test_the_safe_distance_sets_how_close_to_the_wall_a_point_may_be() -> None:
    point = Position(x=2.3, y=3.0)
    near, far = make_is_valid(room_with_wall(), 0.4), make_is_valid(room_with_wall(), 1.0)
    assert near is not None and far is not None
    assert near(point)
    assert not far(point)


def test_a_point_off_the_map_is_invalid() -> None:
    is_valid = make_is_valid(room_with_wall(), 0.3)
    assert is_valid is not None
    assert not is_valid(Position(x=-5.0, y=3.0))
    assert not is_valid(Position(x=3.0, y=20.0))
