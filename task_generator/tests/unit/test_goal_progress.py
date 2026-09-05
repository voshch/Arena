from __future__ import annotations

from task_generator.utils.goal_progress import GoalProgress, GoalProgressTracker


def test_record_reflects_lowest_progress_robot():
    tracker = GoalProgressTracker()

    tracker.sample("mostly_done", (0.0, 0.0), (10.0, 0.0))
    tracker.sample("mostly_done", (8.0, 0.0), (10.0, 0.0))  # closed 80%

    tracker.sample("barely_moved", (0.0, 0.0), (10.0, 0.0))
    tracker.sample("barely_moved", (2.0, 0.0), (10.0, 0.0))  # closed 20%

    best = tracker.least_progress()
    assert best is not None
    assert best.closed_fraction == 0.2


def test_start_distance_fixed_at_first_sample():
    progress = GoalProgress()
    progress.sample((0.0, 0.0), (10.0, 0.0))
    assert progress.start_dist == 10.0

    progress.sample((-2.0, 0.0), (10.0, 0.0))  # moved further away
    assert progress.start_dist == 10.0
    assert progress.min_dist == 10.0  # min unaffected by the setback

    progress.sample((9.0, 0.0), (10.0, 0.0))
    assert progress.min_dist == 1.0
    assert progress.start_dist == 10.0


def test_path_length_accumulates():
    progress = GoalProgress()
    progress.sample((0.0, 0.0), (10.0, 0.0))
    progress.sample((3.0, 0.0), (10.0, 0.0))
    progress.sample((3.0, 4.0), (10.0, 0.0))
    assert progress.path_length == 7.0


def test_no_goal_gives_zeros():
    tracker = GoalProgressTracker()
    assert tracker.least_progress() is None

    progress = GoalProgress()
    assert progress.start_dist == 0.0
    assert progress.min_dist == 0.0
    assert progress.path_length == 0.0
