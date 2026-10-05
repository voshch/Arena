from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from arena_auditory.hearing.belief_grid import SPEED_MASK_NO_LIMIT, BeliefGrid, BeliefParams


def _grid(**params: object) -> BeliefGrid:
    return BeliefGrid(origin_x=-10.0, origin_y=-10.0, resolution=0.1, width=200, height=200, params=BeliefParams(**params))


def test_event_paints_mass_along_the_map_frame_bearing() -> None:
    grid = _grid()
    grid.add_event(0.0, 0.0, 0.0, math.pi / 2, sound_type="footstep", bearing_frame="map")
    x, y, mass = grid.argmax_world()
    assert mass > 0.0
    assert abs(x) < 0.5
    assert y > 0.5


def test_robot_frame_bearing_is_rotated_by_yaw() -> None:
    grid = _grid()
    grid.add_event(0.0, 0.0, math.pi / 2, 0.0, sound_type="footstep", bearing_frame="robot")
    x, y, _ = grid.argmax_world()
    assert abs(x) < 0.5
    assert y > 0.5


def test_decay_and_clear_drain_the_grid() -> None:
    grid = _grid(tau_sec=1.0)
    grid.add_event(0.0, 0.0, 0.0, 0.0, sound_type="footstep", bearing_frame="map")
    before = grid.mass.sum()
    grid.decay(1.0)
    assert grid.mass.sum() == pytest.approx(before * math.exp(-1.0), rel=1e-4)
    grid.clear()
    assert grid.mass.sum() == 0.0
    assert grid.argmax_world()[2] == 0.0


def test_wall_blocks_mass_but_wedge_crosses_into_the_far_corridor() -> None:
    free = np.ones((40, 40), dtype=bool)
    free[:, 20:23] = False
    grid = BeliefGrid(origin_x=0.0, origin_y=0.0, resolution=0.1, width=40, height=40, params=BeliefParams(), free=free)
    grid.add_event(0.5, 2.0, 0.0, 0.0, sound_type="footstep", bearing_frame="map")
    assert grid.mass[:, 20:23].sum() == 0.0
    assert grid.mass[:, :20].sum() > 0.0
    assert grid.mass[:, 23:].sum() > 0.0

    open_grid = BeliefGrid(origin_x=0.0, origin_y=0.0, resolution=0.1, width=40, height=40, params=BeliefParams())
    open_grid.add_event(0.5, 2.0, 0.0, 0.0, sound_type="footstep", bearing_frame="map")
    assert grid.mass.sum() < open_grid.mass.sum()


def test_speed_mask_limits_within_the_reaction_radius_only() -> None:
    grid = _grid(reaction_radius_m=1.0, belief_threshold=0.1, nominal_event_rate_hz=1.0, tau_sec=1.0)
    grid.add_event(0.0, 0.0, 0.0, 0.0, sound_type="footstep", bearing_frame="map", weight=1.0)
    mask = grid.speed_mask_int8()
    px, py, _ = grid.argmax_world()
    row, col = grid.world_to_cell(px, py)
    assert grid.params.speed_min_pct <= mask[row, col] < grid.params.speed_free_pct
    far_row, far_col = grid.world_to_cell(px, py - 5.0)
    assert mask[far_row, far_col] == SPEED_MASK_NO_LIMIT
    assert mask.dtype == np.int8
