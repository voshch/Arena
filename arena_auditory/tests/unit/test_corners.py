from __future__ import annotations

import math

import numpy as np

from arena_auditory.hearing.corners import find_blind_bend, occupied_cells_on_line

ORIGIN = (0.0, 0.0)
RESOLUTION = 0.1
GRID_SIZE = 200


def _l_corridor_occupancy() -> np.ndarray:
    rows = (np.arange(GRID_SIZE) + 0.5) * RESOLUTION
    cols = (np.arange(GRID_SIZE) + 0.5) * RESOLUTION
    xx, yy = np.meshgrid(cols, rows)
    occupied = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
    leg_a = (xx >= 0.5) & (xx <= 10.0) & (yy >= 0.4) & (yy <= 1.6)
    leg_b = (yy >= 1.0) & (yy <= 10.0) & (xx >= 9.4) & (xx <= 10.6)
    occupied[leg_a | leg_b] = False
    return occupied


def _l_corridor_plan() -> np.ndarray:
    xs = np.round(np.arange(1.0, 10.0 + 1e-9, 0.1), 6)
    leg_a = np.stack([xs, np.full_like(xs, 1.0)], axis=1)
    ys = np.round(np.arange(1.1, 9.5 + 1e-9, 0.1), 6)
    leg_b = np.stack([np.full_like(ys, 10.0), ys], axis=1)
    return np.concatenate([leg_a, leg_b], axis=0)


def test_find_blind_bend_at_the_l_corner() -> None:
    occupied = _l_corridor_occupancy()
    plan = _l_corridor_plan()
    bend = find_blind_bend(
        plan,
        occupied,
        ORIGIN,
        RESOLUTION,
        (2.0, 1.0),
        lookahead_m=4.0,
        approach_m=4.0,
        hold_len_m=1.5,
        hold_offset_m=1.0,
    )
    assert bend is not None
    bx, by = bend.bend_xy
    assert math.hypot(bx - 10.0, by - 1.0) < 1.0
    dx, dy = bend.direction
    assert math.isclose(dx * dx + dy * dy, 1.0, abs_tol=1e-6)
    assert max(abs(dx), abs(dy)) > 0.9
    assert 6.5 <= bend.dist_from(plan, (2.0, 1.0)) <= 8.5
    for x, _y in bend.approach_xy:
        assert bx - x <= 4.1
    for x, _y in bend.hold_xy:
        assert 1.0 <= bx - x <= 2.6


def test_straight_plan_has_no_blind_bend() -> None:
    occupied = _l_corridor_occupancy()
    xs = np.round(np.arange(1.0, 7.0 + 1e-9, 0.1), 6)
    plan = np.stack([xs, np.full_like(xs, 1.0)], axis=1)
    bend = find_blind_bend(
        plan,
        occupied,
        ORIGIN,
        RESOLUTION,
        (2.0, 1.0),
        lookahead_m=4.0,
        approach_m=4.0,
        hold_len_m=1.5,
        hold_offset_m=1.0,
    )
    assert bend is None


def test_robot_already_past_the_bend_has_no_blind_bend() -> None:
    occupied = _l_corridor_occupancy()
    ys = np.round(np.arange(5.0, 9.5 + 1e-9, 0.1), 6)
    plan = np.stack([np.full_like(ys, 10.0), ys], axis=1)
    bend = find_blind_bend(
        plan,
        occupied,
        ORIGIN,
        RESOLUTION,
        (10.0, 5.0),
        lookahead_m=4.0,
        approach_m=4.0,
        hold_len_m=1.5,
        hold_offset_m=1.0,
    )
    assert bend is None


def test_occupied_cells_on_line_counts_a_three_cell_wall() -> None:
    occupied = np.zeros((10, 10), dtype=bool)
    occupied[5, 3:6] = True
    assert occupied_cells_on_line(occupied, 5, 0, 5, 9) == 3


def test_occupied_cells_on_line_free_path_is_zero() -> None:
    occupied = np.zeros((10, 10), dtype=bool)
    assert occupied_cells_on_line(occupied, 0, 0, 9, 9) == 0


def test_goal_at_the_corner_is_a_bend() -> None:
    full = _l_corridor_plan()
    plan = full[full[:, 1] <= 1.0 + 1e-9]
    bend = find_blind_bend(plan, _l_corridor_occupancy(), ORIGIN, RESOLUTION, (2.0, 1.0), lookahead_m=4.0, approach_m=4.0, hold_len_m=1.5, hold_offset_m=1.0)
    assert bend is not None
    assert 0.0 <= 10.0 - bend.bend_xy[0] < 1.5
    assert abs(bend.bend_xy[1] - 1.0) < 0.2
