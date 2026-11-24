"""The sampling area: a named zone or stair group, a disc, the whole map, and the parameter names its errors give."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import shapely
from arena_runtime.env.constants.rng import EpisodeRng
from robot_fakes import BAY, World

NAN = math.nan
FLIGHT_A = SimpleNamespace(name="stair_a", group="stairs", corners=[(10.0, 0.0), (12.0, 0.0), (12.0, 2.0), (10.0, 2.0)])
FLIGHT_B = SimpleNamespace(name="stair_b", group="stairs", corners=[(12.0, 0.0), (14.0, 0.0), (14.0, 2.0), (12.0, 2.0)])


def test_a_named_zone_is_its_polygon() -> None:
    pytest.importorskip("rclpy")
    from task_generator.tasks.robots.sampling_area import sampling_area

    world = World(EpisodeRng(), {"bay": BAY})
    area = sampling_area(prefix="task.random.start_", world=world, zone="bay", center_x=NAN, center_y=NAN, radius=1.5)
    assert area.bounds == (0.0, 0.0, 4.0, 4.0)


def test_a_stair_group_covers_every_flight() -> None:
    pytest.importorskip("rclpy")
    from task_generator.tasks.robots.sampling_area import sampling_area

    world = World(EpisodeRng(), {}, stairs=[FLIGHT_A, FLIGHT_B])
    area = sampling_area(prefix="task.random.goal_", world=world, zone="stairs", center_x=NAN, center_y=NAN, radius=1.5)
    assert area.contains(shapely.Point(11.0, 1.0))
    assert area.contains(shapely.Point(13.0, 1.0))


def test_a_center_and_radius_is_the_disc_around_it() -> None:
    pytest.importorskip("rclpy")
    from task_generator.tasks.robots.sampling_area import sampling_area

    world = World(EpisodeRng(), {"bay": BAY})
    area = sampling_area(prefix="task.random.goal_", world=world, zone="", center_x=2.0, center_y=2.0, radius=1.0)
    assert area.bounds == pytest.approx((1.0, 1.0, 3.0, 3.0), abs=1e-6)


def test_no_zone_and_no_center_is_the_whole_map() -> None:
    pytest.importorskip("rclpy")
    from task_generator.tasks.robots.sampling_area import sampling_area

    world = World(EpisodeRng(), {"bay": BAY})
    assert sampling_area(prefix="task.random.goal_", world=world, zone="", center_x=NAN, center_y=NAN, radius=1.5) is None


def test_a_zone_and_a_center_together_are_refused_by_their_parameter_names() -> None:
    pytest.importorskip("rclpy")
    from task_generator.tasks.robots.sampling_area import sampling_area

    world = World(EpisodeRng(), {"bay": BAY})
    with pytest.raises(ValueError, match="task.random.start_zone"):
        sampling_area(prefix="task.random.start_", world=world, zone="bay", center_x=2.0, center_y=2.0, radius=1.0)


def test_an_unknown_zone_is_refused_by_its_parameter_name() -> None:
    pytest.importorskip("rclpy")
    from task_generator.tasks.robots.sampling_area import sampling_area

    world = World(EpisodeRng(), {"bay": BAY})
    with pytest.raises(ValueError, match="task.random.start_zone 'gap' is not a zone"):
        sampling_area(prefix="task.random.start_", world=world, zone="gap", center_x=NAN, center_y=NAN, radius=1.5)
