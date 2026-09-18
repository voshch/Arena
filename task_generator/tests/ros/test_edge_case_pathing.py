"""Tests for the Level C route planner.

The straight line from the robot's start to its goal is only its route inside one open area.
In a furnished multi-room world it cuts through walls the robot paths around, which made
roughly 52 of 56 aborts in an hour of live episodes. These cover the planner that replaced it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def _map(grid):
    from test_edge_case_validation import _FakeMap

    return _FakeMap(grid)


def _grid(rows=24, cols=24):
    from task_generator.manager.world_manager.utils import WorldOccupancy

    return np.full((rows, cols), WorldOccupancy.EMPTY, dtype=np.uint8)


def _wall_with_doorway():
    """A wall across the middle with a 5-cell doorway at x ~ 16."""
    from task_generator.manager.world_manager.utils import WorldOccupancy

    grid = _grid()
    grid[9:12, :] = WorldOccupancy.FULL
    grid[9:12, 14:19] = WorldOccupancy.EMPTY
    return grid


def _is_valid(grid, safe_dist=1.0):
    from task_generator.tasks.obstacles._validation import make_is_valid

    return make_is_valid(_map(grid), safe_dist)


# plan_route
# ----------


def test_clear_line_is_returned_unchanged():
    """An open-area route must behave exactly as it did before planning existed — same two
    points, so `deviated` and the recorded geometry are untouched."""
    from task_generator.tasks.obstacles.edge_case.pathing import plan_route

    grid = _grid()
    assert plan_route(_map(grid), (2.0, 2.0), (8.0, 2.0), is_valid=_is_valid(grid)) == [(2.0, 2.0), (8.0, 2.0)]


def test_route_bends_through_the_doorway():
    from task_generator.tasks.obstacles.edge_case.pathing import plan_route, route_length

    grid = _wall_with_doorway()
    route = plan_route(_map(grid), (5.0, 5.0), (5.0, 15.0), is_valid=_is_valid(grid))

    assert route[0] == (5.0, 5.0)
    assert route[-1] == (5.0, 15.0)
    assert len(route) > 2
    assert max(p[0] for p in route) > 12.0, "must detour toward the doorway"
    assert route_length(route) > math.dist((5.0, 5.0), (5.0, 15.0))


def test_every_interior_point_of_a_planned_route_is_navigable():
    """The whole promise of the planner. If the route it returns is not itself walkable, the
    encounter designed on it is no better than the straight line it replaced."""
    from task_generator.shared import Position
    from task_generator.tasks.obstacles.edge_case.pathing import iter_points, plan_route

    grid = _wall_with_doorway()
    is_valid = _is_valid(grid)
    route = plan_route(_map(grid), (5.0, 5.0), (5.0, 15.0), is_valid=is_valid)

    # Skip a margin at each end: both endpoints are occupied by construction (the robot
    # stands at the start, and the robots scenario mode forbids start and goal).
    interior = [p for p in iter_points(route, 0.25) if math.dist(p, route[0]) > 1.5 and math.dist(p, route[-1]) > 1.5]
    assert interior
    assert all(is_valid(Position(x=p[0], y=p[1])) for p in interior)


def test_unreachable_goal_raises_rather_than_returning_a_fake_route():
    from task_generator.manager.world_manager.utils import WorldOccupancy
    from task_generator.tasks.obstacles.edge_case.pathing import PathError, plan_route

    grid = _grid()
    grid[9:12, :] = WorldOccupancy.FULL  # a wall with no doorway at all
    with pytest.raises(PathError, match="no free-space route"):
        plan_route(_map(grid), (5.0, 5.0), (5.0, 15.0), is_valid=_is_valid(grid))


def test_off_map_endpoint_is_named():
    from task_generator.tasks.obstacles.edge_case.pathing import PathError, plan_route

    grid = _wall_with_doorway()
    with pytest.raises(PathError, match="outside the occupancy grid"):
        plan_route(_map(grid), (5.0, 5.0), (500.0, 500.0), is_valid=_is_valid(grid))


def test_no_predicate_means_the_straight_line():
    """`safe_dist <= 0` yields no predicate, matching the zone converter's convention."""
    from task_generator.tasks.obstacles.edge_case.pathing import plan_route

    grid = _wall_with_doorway()
    assert plan_route(_map(grid), (5.0, 5.0), (5.0, 15.0), is_valid=None) == [(5.0, 5.0), (5.0, 15.0)]


def test_simplify_collapses_the_staircase():
    """A raw cell path is a staircase of 0.2 m steps. Left alone, "the robot's heading at the
    encounter" would be the angle of one grid step rather than a direction of travel."""
    from task_generator.tasks.obstacles.edge_case.pathing import plan_route

    grid = _wall_with_doorway()
    route = plan_route(_map(grid), (5.0, 5.0), (5.0, 15.0), is_valid=_is_valid(grid))
    assert len(route) <= 8, f"expected a handful of waypoints, got {len(route)}"


# Polyline helpers
# ----------------


def test_route_length_sums_the_segments():
    from task_generator.tasks.obstacles.edge_case.pathing import route_length

    assert route_length([(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]) == pytest.approx(7.0)


def test_point_at_arc_walks_the_polyline_and_reports_local_heading():
    """The approach angle is measured against the heading *at the encounter*. On a route that
    turns a corner, a single global heading would make 'head-on' mean different things at
    different points."""
    from task_generator.tasks.obstacles.edge_case.pathing import point_at_arc

    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]

    pt, heading = point_at_arc(route, 5.0)
    assert pt == pytest.approx((5.0, 0.0))
    assert heading == pytest.approx((1.0, 0.0))

    pt, heading = point_at_arc(route, 15.0)
    assert pt == pytest.approx((10.0, 5.0))
    assert heading == pytest.approx((0.0, 1.0)), "heading must follow the segment it is on"


def test_point_at_arc_clamps_past_the_end():
    from task_generator.tasks.obstacles.edge_case.pathing import point_at_arc

    route = [(0.0, 0.0), (10.0, 0.0)]
    assert point_at_arc(route, 999.0)[0] == pytest.approx((10.0, 0.0))


def test_iter_points_covers_the_whole_route():
    from task_generator.tasks.obstacles.edge_case.pathing import iter_points

    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    pts = list(iter_points(route, 1.0))
    assert pts[0] == pytest.approx((0.0, 0.0))
    assert pts[-1] == pytest.approx((10.0, 10.0))
    assert len(pts) >= 20


# Solving on a planned route
# --------------------------


def test_encounter_on_a_bent_route_lands_on_the_route():
    """End to end: solve against a route that turns a corner, and the encounter point must sit
    on the polyline rather than on the chord between its endpoints."""
    from task_generator.tasks.obstacles.edge_case.geometry import solve_encounter

    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    enc = solve_encounter(route, robot_speed=1.0, ped_speed=1.0, encounter_at=0.75)

    # 75 % of a 20 m route is 15 m: 10 m along the first leg, then 5 m up the second.
    assert enc.encounter == pytest.approx((10.0, 5.0))
    assert enc.t_encounter == pytest.approx(15.0)
    # The chord from (0,0) to (10,10) would put it near (7.5, 7.5) - nowhere the robot goes.
    assert math.dist(enc.encounter, (7.5, 7.5)) > 2.0


def test_head_on_follows_the_local_heading_after_a_corner():
    from task_generator.tasks.obstacles.edge_case.geometry import solve_encounter

    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    enc = solve_encounter(route, robot_speed=1.0, ped_speed=1.0, encounter_at=0.75, approach_angle=180.0)

    # Robot is heading north there, so a head-on pedestrian starts north of the encounter
    # and walks south - not east/west, which a global heading would have produced.
    assert enc.spawn[1] > enc.encounter[1]
    assert enc.spawn[0] == pytest.approx(enc.encounter[0])


def test_forbidden_endpoint_discs_do_not_trap_the_search():
    """The robots scenario mode `forbid`s a disc around the robot's start and goal so the crowd
    cannot spawn on them. Without exempting those, A* cannot take its first step — every
    neighbour of the start reads as blocked — and it reports "no free-space route" for a route
    the robot drives every episode. That is exactly what happened live."""
    from task_generator.manager.world_manager.utils import WorldOccupancy
    from task_generator.tasks.obstacles.edge_case.pathing import plan_route

    grid = _grid()
    grid[4:7, 4:7] = WorldOccupancy.FULL
    grid[16:19, 4:7] = WorldOccupancy.FULL

    route = plan_route(_map(grid), (5.0, 5.0), (5.0, 17.0), is_valid=_is_valid(grid))
    assert route[0] == (5.0, 5.0)
    assert route[-1] == (5.0, 17.0)


def test_exemption_does_not_tunnel_through_a_real_wall():
    """The exemption is a small bubble, not a licence to ignore geometry: a wall with no
    doorway must still be refused rather than routed straight through."""
    from task_generator.manager.world_manager.utils import WorldOccupancy
    from task_generator.tasks.obstacles.edge_case.pathing import PathError, plan_route

    grid = _grid()
    grid[9:12, :] = WorldOccupancy.FULL
    with pytest.raises(PathError, match="no free-space route"):
        plan_route(_map(grid), (5.0, 2.0), (5.0, 20.0), is_valid=_is_valid(grid))
