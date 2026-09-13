"""Tests for the occupancy predicate lifted out of TM_Scenario, and the edge_case guard
built on it.

Placing an agent inside a wall confounds "the perturbed parameter caused the failure" with
"the agent spawned somewhere impossible" - README section 7 measured ~35 % of generated
positions landing outside every navigable zone.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


class _FakeMap:
    """World map with a 1:1 world-to-grid mapping, so grid[y][x] is the cell at (x, y)."""

    def __init__(self, grid: np.ndarray, physical: np.ndarray | None = None) -> None:
        # `physical` is walls + furniture only; `grid` also folds in the forbidden layer.
        # They differ in the real world manager, so the fake keeps them separable.
        self.occupancy = type("_Occ", (), {"grid": grid, "physical": grid if physical is None else physical})()

    def tf_posr2rect(self, posr):
        lo = (int(posr.y - posr.radius), int(posr.x - posr.radius))
        hi = (int(posr.y + posr.radius), int(posr.x + posr.radius))
        return lo, hi

    def tf_pos2grid(self, position):
        return int(position.y), int(position.x)

    def tf_grid2pos(self, grid_pos):
        from task_generator.shared import Position

        return Position(x=float(grid_pos[1]), y=float(grid_pos[0]))


def _grid(rows: int = 20, cols: int = 20) -> np.ndarray:
    from task_generator.manager.world_manager.utils import WorldOccupancy

    return np.full((rows, cols), WorldOccupancy.EMPTY, dtype=np.uint8)


def test_no_predicate_when_safe_dist_is_zero():
    """Matches the convention the zone converter expects: no predicate = no filtering."""
    from task_generator.tasks.obstacles._validation import make_is_valid

    assert make_is_valid(_FakeMap(_grid()), 0.0) is None
    assert make_is_valid(_FakeMap(_grid()), -1.0) is None


def test_free_space_is_valid():
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import make_is_valid

    is_valid = make_is_valid(_FakeMap(_grid()), 1.0)
    assert is_valid(Position(10.0, 10.0)) is True


def test_occupied_cell_is_invalid():
    from task_generator.manager.world_manager.utils import WorldOccupancy
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import make_is_valid

    grid = _grid()
    grid[9:12, 9:12] = WorldOccupancy.FULL  # a wall block around (10, 10)

    is_valid = make_is_valid(_FakeMap(grid), 1.0)
    assert is_valid(Position(10.0, 10.0)) is False
    assert is_valid(Position(3.0, 3.0)) is True


def test_safe_dist_widens_the_footprint():
    """A pose clear of the wall at radius 1 can still be rejected at radius 4."""
    from task_generator.manager.world_manager.utils import WorldOccupancy
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import make_is_valid

    grid = _grid()
    grid[0:3, 0:3] = WorldOccupancy.FULL

    assert make_is_valid(_FakeMap(grid), 1.0)(Position(6.0, 6.0)) is True
    assert make_is_valid(_FakeMap(grid), 4.0)(Position(6.0, 6.0)) is False


def test_out_of_bounds_is_invalid():
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import make_is_valid

    is_valid = make_is_valid(_FakeMap(_grid()), 1.0)
    assert is_valid(Position(500.0, 500.0)) is False


def _walled_grid():
    """A 20x20 grid at 1 m/cell with a wall column at x=10."""
    grid = np.full((20, 20), 255, dtype=np.uint8)
    grid[:, 10] = 0
    return grid


def test_edge_case_rejects_an_injected_spawn_inside_a_wall(tmp_path):
    """The injected agent is the case; a spawn in a wall aborts rather than running."""
    import asyncio

    from edge_case_fixtures import _mode, _Robot, obstacle
    from task_generator.tasks.obstacles.edge_case.scenario_block import parse

    grid = _walled_grid()
    # Route along y=15 with the wall column in the way at x=10: every candidate spawn on the
    # straight line sits in occupied cells once the map says so.
    tm = _mode(tmp_path, robots={"jackal": _Robot((2.0, 15.0), (18.0, 15.0))}, safe_dist=1.0,
               world_map=_FakeMap(grid), population=[obstacle("w", (2.0, 2.0), [(3.0, 2.0)])],
               block=parse({"id": "c", "effects": [{"type": "intercept", "search": 0.0}]}))
    asyncio.run(tm._reset())
    assert tm.aborted, "an intercept through a wall must abort, not relocate"


def test_a_bystander_in_a_wall_is_recorded_not_fatal(tmp_path):
    import asyncio
    import json

    from edge_case_fixtures import _mode, obstacle
    from task_generator.tasks.obstacles.edge_case.scenario_block import parse

    grid = _walled_grid()
    tm = _mode(tmp_path, safe_dist=0.5, world_map=_FakeMap(grid),
               population=[obstacle("in_wall", (10.0, 5.0), [(10.0, 6.0)]), obstacle("fine", (3.0, 3.0), [(4.0, 3.0)])],
               block=parse({"id": "c"}))
    _static, dynamic = asyncio.run(tm._reset())
    assert [o.name for o in dynamic] == ["in_wall", "fine"] and not tm.aborted
    rows = [json.loads(l) for l in (tmp_path / "records" / "cases.jsonl").read_text().splitlines()]
    assert rows[-1]["invalid_spawns"] == ["in_wall"]


def test_in_map_bounds_accepts_a_point_inside_the_grid():
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import in_map_bounds

    assert in_map_bounds(_FakeMap(_grid()), Position(10.0, 10.0)) is True


def test_in_map_bounds_rejects_a_point_off_the_grid():
    """The signal that a caller is holding a coordinate in the wrong frame."""
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import in_map_bounds

    m = _FakeMap(_grid())
    assert in_map_bounds(m, Position(400.0, 400.0)) is False
    assert in_map_bounds(m, Position(-5.0, 10.0)) is False


def test_in_map_bounds_is_not_navigability():
    """A fully occupied map is still a map. Conflating the two would make every blocked goal
    look like a frame error."""
    from task_generator.manager.world_manager.utils import WorldOccupancy
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import in_map_bounds, make_is_valid

    grid = _grid()
    grid[:] = WorldOccupancy.FULL
    m = _FakeMap(grid)

    assert in_map_bounds(m, Position(10.0, 10.0)) is True
    assert make_is_valid(m, 1.0)(Position(10.0, 10.0)) is False


def test_traversability_ignores_the_forbidden_layer():
    """The robot drives through its own start/goal exclusion discs every episode. Planning
    against them made it look sealed into ~115 cells of a 250x410 grid."""
    from task_generator.manager.world_manager.utils import WorldOccupancy
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import make_is_traversable, make_is_valid

    physical = _grid()               # nothing physically in the way
    combined = _grid()
    combined[9:12, 9:12] = WorldOccupancy.FULL   # ... but the space is reserved
    m = _FakeMap(combined, physical=physical)

    assert make_is_valid(m, 1.0)(Position(10.0, 10.0)) is False, "nothing may spawn there"
    assert make_is_traversable(m, 1.0)(Position(10.0, 10.0)) is True, "the robot may still drive there"
