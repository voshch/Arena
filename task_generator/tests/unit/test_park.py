from __future__ import annotations

import math

from task_generator.utils.park import PARK_DRIFT_M, PARK_TURN_RAD, ParkTimer


def test_still_robot_accumulates_hold():
    timer = ParkTimer()
    assert timer.held_for(1.0, 2.0, 0.0, 10.0) == 0.0
    assert timer.held_for(1.0, 2.0, 0.0, 12.5) == 2.5


def test_jitter_inside_bounds_keeps_the_anchor():
    timer = ParkTimer()
    timer.held_for(1.0, 2.0, 0.0, 10.0)
    timer.held_for(1.0 + 0.8 * PARK_DRIFT_M, 2.0, 0.8 * PARK_TURN_RAD, 11.0)
    assert timer.held_for(1.0, 2.0, -0.8 * PARK_TURN_RAD, 13.0) == 3.0


def test_creeping_past_the_drift_restarts_the_hold():
    timer = ParkTimer()
    timer.held_for(0.0, 0.0, 0.0, 0.0)
    assert timer.held_for(2.0 * PARK_DRIFT_M, 0.0, 0.0, 5.0) == 0.0
    assert timer.held_for(2.0 * PARK_DRIFT_M, 0.0, 0.0, 6.0) == 1.0


def test_turning_in_place_restarts_the_hold():
    timer = ParkTimer()
    timer.held_for(0.0, 0.0, 0.0, 0.0)
    assert timer.held_for(0.0, 0.0, 2.0 * PARK_TURN_RAD, 5.0) == 0.0


def test_turn_wraps_across_pi():
    timer = ParkTimer()
    timer.held_for(0.0, 0.0, math.pi - 0.01, 0.0)
    assert timer.held_for(0.0, 0.0, -math.pi + 0.01, 4.0) == 4.0


def test_reset_restarts_the_hold():
    timer = ParkTimer()
    timer.held_for(0.0, 0.0, 0.0, 0.0)
    timer.reset()
    assert timer.held_for(0.0, 0.0, 0.0, 9.0) == 0.0
