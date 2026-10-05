from __future__ import annotations

from task_generator.utils.goal_progress import GoalProgress, GoalProgressTracker


def test_record_reflects_lowest_progress_robot():
    tracker = GoalProgressTracker()

    tracker.sample("mostly_done", (0.0, 0.0), (10.0, 0.0), 1.0, 0.0)
    tracker.sample("mostly_done", (8.0, 0.0), (10.0, 0.0), 1.0, 0.0)  # closed 80%

    tracker.sample("barely_moved", (0.0, 0.0), (10.0, 0.0), 1.0, 0.0)
    tracker.sample("barely_moved", (2.0, 0.0), (10.0, 0.0), 1.0, 0.0)  # closed 20%

    best = tracker.least_progress()
    assert best is not None
    assert best.closed_fraction == 0.2


def test_start_distance_fixed_at_first_sample():
    progress = GoalProgress()
    progress.sample((0.0, 0.0), (10.0, 0.0), 1.0, 0.0)
    assert progress.start_dist == 10.0

    progress.sample((-2.0, 0.0), (10.0, 0.0), 1.0, 0.0)  # moved further away
    assert progress.start_dist == 10.0
    assert progress.min_dist == 10.0  # min unaffected by the setback

    progress.sample((9.0, 0.0), (10.0, 0.0), 1.0, 0.0)
    assert progress.min_dist == 1.0
    assert progress.start_dist == 10.0


def test_path_length_accumulates():
    progress = GoalProgress()
    progress.sample((0.0, 0.0), (10.0, 0.0), 1.0, 0.0)
    progress.sample((3.0, 0.0), (10.0, 0.0), 1.0, 0.0)
    progress.sample((3.0, 4.0), (10.0, 0.0), 1.0, 0.0)
    assert progress.path_length == 7.0


def test_no_goal_gives_zeros():
    tracker = GoalProgressTracker()
    assert tracker.least_progress() is None

    progress = GoalProgress()
    assert progress.start_dist == 0.0
    assert progress.min_dist == 0.0
    assert progress.path_length == 0.0


def test_stall_is_time_spent_within_the_stall_radius():
    progress = GoalProgress()
    progress.sample((0.0, 0.0), (10.0, 0.0), 1.0, 0.0)
    progress.sample((0.6, 0.0), (10.0, 0.0), 1.0, 5.0)
    assert progress.stalled_for(20.0) == 20.0

    progress.sample((1.5, 0.0), (10.0, 0.0), 1.0, 21.0)
    assert progress.stalled_for(30.0) == 9.0


def test_winding_route_is_not_a_stall():
    progress = GoalProgress()
    for i, t in enumerate(range(0, 60, 5)):
        progress.sample((-2.0 * i, 0.0), (1.0, 0.0), 1.0, float(t))
    assert progress.stalled_for(60.0) == 5.0


def test_longest_stall_ignores_robots_at_the_goal():
    tracker = GoalProgressTracker()
    tracker.sample("arrived", (0.0, 0.0), (10.0, 0.0), 1.0, 0.0)
    tracker.sample("arrived", (9.5, 0.0), (10.0, 0.0), 1.0, 1.0)
    assert tracker.longest_stall(100.0) == 0.0

    tracker.sample("stuck", (0.0, 0.0), (10.0, 0.0), 1.0, 0.0)
    assert tracker.longest_stall(100.0) == 100.0


def test_longest_stall_uses_each_robots_own_tolerance():
    tracker = GoalProgressTracker()
    tracker.sample("wide_goal", (0.0, 0.0), (10.0, 0.0), 3.0, 0.0)
    tracker.sample("wide_goal", (8.0, 0.0), (10.0, 0.0), 3.0, 1.0)
    assert tracker.longest_stall(100.0) == 0.0

    tracker.sample("tight_goal", (0.0, 0.0), (10.0, 0.0), 0.5, 0.0)
    tracker.sample("tight_goal", (8.0, 0.0), (10.0, 0.0), 0.5, 1.0)
    assert tracker.longest_stall(100.0) == 99.0


def test_least_progress_carries_its_goal_tolerance():
    tracker = GoalProgressTracker()
    tracker.sample("near", (0.0, 0.0), (10.0, 0.0), 0.5, 0.0)
    tracker.sample("near", (9.0, 0.0), (10.0, 0.0), 0.5, 1.0)
    tracker.sample("far", (0.0, 0.0), (10.0, 0.0), 3.0, 0.0)
    tracker.sample("far", (1.0, 0.0), (10.0, 0.0), 3.0, 1.0)
    best = tracker.least_progress()
    assert best is not None
    assert best.goal_tolerance == 3.0
