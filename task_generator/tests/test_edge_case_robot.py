from __future__ import annotations

import math

import pytest

pytest.importorskip("rclpy")


def _phases(traversals: int):
    from task_generator.shared import Pose, Position
    from task_generator.tasks.robots.edge_case.impl import TM_EdgeCase

    return TM_EdgeCase.traversal_phases(Pose(Position(0.0, 0.0)), Pose(Position(10.0, 0.0)), traversals)


# Route expansion
# ---------------


def test_single_traversal_is_one_leg():
    """Must stay byte-for-byte equivalent to what tm_robots:=scenario already does, so
    `traversals: 1` is a safe drop-in."""
    ph = _phases(1)
    assert len(ph) == 1
    assert ph[0].pose.position.x == 10.0


def test_n_traversals_is_2n_minus_1_legs():
    assert [len(_phases(n)) for n in (1, 2, 3, 5)] == [1, 3, 5, 9]


def test_legs_alternate_and_end_at_the_goal():
    xs = [p.pose.position.x for p in _phases(3)]
    assert xs == [10.0, 0.0, 10.0, 0.0, 10.0]


def test_zero_traversals_is_rejected():
    with pytest.raises(ValueError, match="traversals must be >= 1"):
        _phases(0)


# Obstruction watchdog
# --------------------


def _fresh():
    from task_generator.tasks.robots.edge_case.impl import _Progress

    return _Progress()


def _blocked(progress, distance, now, *, min_improvement=0.25, timeout=15.0):
    from task_generator.tasks.robots.edge_case.impl import TM_EdgeCase

    return TM_EdgeCase.is_blocked(progress, distance, now, min_improvement=min_improvement, timeout=timeout)


def test_steady_progress_never_blocks():
    p = _fresh()
    for i in range(30):
        assert _blocked(p, distance=20.0 - i, now=float(i)) is False


def test_stalled_robot_blocks_after_the_timeout():
    p = _fresh()
    assert _blocked(p, 5.0, 0.0) is False          # first sample establishes the baseline
    assert _blocked(p, 5.0, 14.0) is False         # still inside the window
    assert _blocked(p, 5.0, 15.0) is True          # window elapsed with no improvement


def test_a_short_pause_does_not_block():
    """Robots legitimately pause, rotate in place, or back out of a doorway. Only a
    sustained lack of progress is obstruction."""
    p = _fresh()
    _blocked(p, 10.0, 0.0)
    assert _blocked(p, 10.0, 8.0) is False         # paused 8 s
    assert _blocked(p, 9.0, 9.0) is False          # moved again -> timer resets
    assert _blocked(p, 9.0, 20.0) is False         # only 11 s since that progress
    assert _blocked(p, 9.0, 24.1) is True          # now 15.1 s


def test_progress_must_exceed_the_threshold_to_count():
    """Jitter below `min_improvement` is not progress — otherwise a robot vibrating in
    place against a crowd would look like it was advancing forever."""
    p = _fresh()
    _blocked(p, 5.0, 0.0)
    for i in range(1, 16):
        assert _blocked(p, 5.0 - i * 0.01, float(i)) is (i >= 15)


def test_moving_away_from_the_goal_is_not_progress():
    p = _fresh()
    _blocked(p, 5.0, 0.0)
    assert _blocked(p, 8.0, 10.0) is False         # pushed backwards, still inside window
    assert _blocked(p, 8.0, 15.0) is True          # never got closer than 5.0


def test_best_distance_is_a_high_water_mark():
    """Getting close, being pushed back, and returning to the same spot is not progress."""
    p = _fresh()
    _blocked(p, 10.0, 0.0)
    _blocked(p, 4.0, 1.0)                          # progress: best = 4.0
    assert _blocked(p, 9.0, 5.0) is False
    assert _blocked(p, 4.1, 10.0) is False         # back near best, but not better
    assert _blocked(p, 4.1, 16.1) is True          # 15.1 s since the real progress at t=1


def test_reset_clears_the_watchdog():
    p = _fresh()
    _blocked(p, 5.0, 0.0)
    _blocked(p, 5.0, 20.0)
    p.reset(100.0)
    assert p.best_distance == math.inf
    assert _blocked(p, 5.0, 110.0) is False        # new leg, new window


def test_retries_repeat_each_leg_in_place() -> None:
    from task_generator.shared import Pose, Position
    from task_generator.tasks.robots.edge_case.impl import TM_EdgeCase

    phases = TM_EdgeCase.traversal_phases(Pose(Position(0.0, 0.0)), Pose(Position(10.0, 0.0)), 2, retries=3)
    xs = [p.pose.position.x for p in phases]
    assert xs == [10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0]
    with pytest.raises(ValueError, match="retries"):
        TM_EdgeCase.traversal_phases(Pose(Position(0.0, 0.0)), Pose(Position(10.0, 0.0)), 1, retries=0)


def test_hold_counts_sim_seconds_from_the_reset():
    from task_generator.tasks.robots.edge_case.impl import TM_EdgeCase

    assert TM_EdgeCase.holding(105.0, 100.0, 15.0)
    assert not TM_EdgeCase.holding(115.0, 100.0, 15.0)
    assert not TM_EdgeCase.holding(100.0, 100.0, 0.0)


# Near-goal acceptance
# --------------------


def _accepted(p, distance, now, within=0.6, after=5.0):
    from task_generator.tasks.robots.edge_case.impl import TM_EdgeCase

    return TM_EdgeCase.near_goal_accepted(p, distance, now, within=within, min_improvement=0.25, after=after)


def test_a_leg_is_accepted_when_the_robot_stalls_next_to_an_unplannable_goal():
    p = _fresh()
    assert _accepted(p, 0.23, 0.0) is False   # baseline
    assert _accepted(p, 0.23, 4.0) is False   # inside the window
    assert _accepted(p, 0.23, 5.0) is True    # 5 s without getting closer, 0.23 m short: arrived


def test_a_stall_far_from_the_goal_is_not_accepted():
    p = _fresh()
    for t in (0.0, 5.0, 30.0):
        assert _accepted(p, 3.0, t) is False


def test_acceptance_can_be_switched_off():
    p = _fresh()
    assert _accepted(p, 0.1, 0.0, within=0.0) is False
    assert _accepted(p, 0.1, 60.0, within=0.0) is False
