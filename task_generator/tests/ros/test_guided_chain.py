from __future__ import annotations

import math


def _poses(*xy_yaw):
    from task_generator.shared import Orientation, Pose, Position

    return [Pose(position=Position(x, y), orientation=Orientation.from_yaw(yaw)) for x, y, yaw in xy_yaw]


def _xy_yaw(poses):
    return [(round(p.position.x, 6), round(p.position.y, 6), round(p.orientation.to_yaw(), 6)) for p in poses]


def test_scope_prefix_sorts_robot_names():
    from task_generator.tasks.robots.guided import chain

    assert chain.scope_prefix(["b", "a"]) == "guided/a+b/"


def test_delete_removes_only_the_indexed_waypoint():
    from task_generator.tasks.robots.guided import chain

    waypoints = _poses((0, 0, 0), (1, 0, 0), (2, 0, 0))
    assert _xy_yaw(chain.delete(waypoints, 1)) == [(0, 0, 0), (2, 0, 0)]
    assert len(waypoints) == 3


def test_replace_moves_only_the_indexed_waypoint():
    from task_generator.tasks.robots.guided import chain

    waypoints = _poses((0, 0, 0), (1, 0, 0))
    (new,) = _poses((5, 5, 1.0))
    assert _xy_yaw(chain.replace(waypoints, 0, new)) == [(5, 5, 1.0), (1, 0, 0)]


def test_insert_after_in_the_middle_lands_on_the_midpoint():
    from task_generator.tasks.robots.guided import chain

    waypoints = _poses((0, 0, 0.5), (2, 4, 0))
    assert _xy_yaw(chain.insert_after(waypoints, 0)) == [(0, 0, 0.5), (1, 2, 0.5), (2, 4, 0)]


def test_insert_after_the_last_goes_ahead_along_its_heading():
    from task_generator.tasks.robots.guided import chain

    waypoints = _poses((1, 1, math.pi / 2))
    assert _xy_yaw(chain.insert_after(waypoints, 0)) == [(1, 1, round(math.pi / 2, 6)), (1, 2, round(math.pi / 2, 6))]
