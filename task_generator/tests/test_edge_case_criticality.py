"""Tests for the criticality panel — what actually happened, as opposed to what was designed.

The distinction these guard: `designed_ttc` is fixed when the pedestrian is placed and cannot
respond to a tail. These metrics are computed from the trajectory, so a tail that changes
behaviour changes them. Getting that wrong would mean reporting the experiment's input as its
result.
"""

from __future__ import annotations

import json
import math

import pytest

from task_generator.tasks.obstacles.edge_case.criticality import (
    Criticality,
    Sample,
    post_encroachment_time,
    score,
    time_to_collision,
)


# time_to_collision
# -----------------


def test_head_on_closing_gives_the_expected_time():
    """Two discs 10 m apart closing at 2 m/s combined, touching at r=0.6, meet at 4.7 s."""
    ttc = time_to_collision((0.0, 0.0), (1.0, 0.0), 0.3, (10.0, 0.0), (-1.0, 0.0), 0.3)
    assert ttc == pytest.approx((10.0 - 0.6) / 2.0)


def test_parallel_motion_never_collides():
    assert time_to_collision((0.0, 0.0), (1.0, 0.0), 0.3, (0.0, 5.0), (1.0, 0.0), 0.3) is None


def test_receding_pair_never_collides():
    assert time_to_collision((0.0, 0.0), (-1.0, 0.0), 0.3, (10.0, 0.0), (1.0, 0.0), 0.3) is None


def test_already_touching_is_zero_not_none():
    """Overlap is the most critical state there is; returning None would drop it from the min."""
    assert time_to_collision((0.0, 0.0), (1.0, 0.0), 0.3, (0.4, 0.0), (0.0, 0.0), 0.3) == 0.0


def test_no_relative_motion_is_not_a_collision():
    assert time_to_collision((0.0, 0.0), (1.0, 0.0), 0.3, (5.0, 0.0), (1.0, 0.0), 0.3) is None


# clearance
# ---------


def test_clearance_is_surface_to_surface():
    """Centre-to-centre would call a 0.6 m gap between two 0.3 m discs a near miss; it is a
    touch."""
    s = [Sample(t=0.0, robot=(0.0, 0.0), robot_radius=0.3, peds={"p": ((0.6, 0.0), (0.0, 0.0), 0.3)})]
    out = score(s)
    assert out.min_clearance_m == pytest.approx(0.0)
    assert out.collided is True


def test_clearance_tracks_the_closest_moment_and_names_it():
    s = [
        Sample(t=0.0, robot=(0.0, 0.0), peds={"a": ((5.0, 0.0), (0.0, 0.0), 0.3)}),
        Sample(t=1.0, robot=(0.0, 0.0), peds={"a": ((2.0, 0.0), (0.0, 0.0), 0.3)}),
        Sample(t=2.0, robot=(0.0, 0.0), peds={"a": ((4.0, 0.0), (0.0, 0.0), 0.3)}),
    ]
    out = score(s)
    assert out.min_clearance_m == pytest.approx(2.0 - 0.6)
    assert out.min_clearance_with == "a"
    assert out.min_clearance_at == pytest.approx(1.0)


# freeze
# ------


def test_freeze_ignores_the_settling_window():
    """nav2 is still bringing costmaps up at t=0. Counting that would score every episode as
    frozen regardless of what the tail did."""
    s = [Sample(t=float(i), robot=(0.0, 0.0), robot_vel=(0.0, 0.0)) for i in range(5)]
    out = score(s, settle_s=2.0)
    assert out.freeze_duration_s == pytest.approx(2.0), "only t>=2 counts"


def test_a_moving_robot_never_freezes():
    s = [Sample(t=float(i), robot=(float(i), 0.0), robot_vel=(1.0, 0.0)) for i in range(6)]
    assert score(s).freeze_duration_s == 0.0


# intrusion
# ---------


def test_intrusion_time_counts_only_while_inside_personal_space():
    s = [
        Sample(t=0.0, robot=(0.0, 0.0), robot_vel=(1.0, 0.0), peds={"a": ((5.0, 0.0), (0.0, 0.0), 0.3)}),
        Sample(t=1.0, robot=(0.0, 0.0), robot_vel=(1.0, 0.0), peds={"a": ((0.8, 0.0), (0.0, 0.0), 0.3)}),
        Sample(t=2.0, robot=(0.0, 0.0), robot_vel=(1.0, 0.0), peds={"a": ((5.0, 0.0), (0.0, 0.0), 0.3)}),
    ]
    # clearance at t=1 is 0.8-0.6 = 0.2 < 0.5, and covers the interval to the next sample
    assert score(s).intrusion_time_s == pytest.approx(1.0)


# isolating the injected agent
# ---------------------------


def test_only_restricts_scoring_to_the_named_agent():
    """A busy scenario is full of ambient near-misses. Without this the crowd's noise buries
    whatever the injected agent did, which is the entire signal."""
    s = [Sample(t=0.0, robot=(0.0, 0.0), peds={
        "bystander": ((1.0, 0.0), (0.0, 0.0), 0.3),
        "edge_0": ((4.0, 0.0), (0.0, 0.0), 0.3),
    })]
    assert score(s).min_clearance_with == "bystander"
    assert score(s, only=["edge_0"]).min_clearance_with == "edge_0"
    assert score(s, only=["edge_0"]).min_clearance_m == pytest.approx(4.0 - 0.6)


# PET
# ---


def test_pet_is_the_time_gap_through_the_encounter_point():
    E = (5.0, 0.0)
    s = [
        Sample(t=0.0, robot=(0.0, 0.0), peds={"a": ((5.0, 4.0), (0.0, 0.0), 0.3)}),
        Sample(t=2.0, robot=(5.0, 0.0), peds={"a": ((5.0, 2.0), (0.0, 0.0), 0.3)}),   # robot at E
        Sample(t=5.0, robot=(9.0, 0.0), peds={"a": ((5.0, 0.2), (0.0, 0.0), 0.3)}),   # ped at E
    ]
    assert post_encroachment_time(s, E) == pytest.approx(3.0)


def test_pet_is_none_when_the_pedestrian_never_goes_there():
    """Otherwise two unrelated moments get subtracted and a PET is invented."""
    E = (5.0, 0.0)
    s = [
        Sample(t=0.0, robot=(0.0, 0.0), peds={"a": ((50.0, 50.0), (0.0, 0.0), 0.3)}),
        Sample(t=2.0, robot=(5.0, 0.0), peds={"a": ((51.0, 50.0), (0.0, 0.0), 0.3)}),
    ]
    assert post_encroachment_time(s, E) is None


# the empty case
# --------------


def test_an_episode_with_no_pedestrians_is_not_a_safe_episode():
    """A record full of None must never read as 'nothing came close'."""
    out = score([Sample(t=float(i), robot=(float(i), 0.0), robot_vel=(1.0, 0.0)) for i in range(4)])
    assert out.min_clearance_m is None
    assert out.min_ttc_s is None
    assert out.observed is False


def test_no_samples_at_all_is_empty_not_zero():
    out = score([])
    assert out == Criticality()
    assert out.observed is False


def test_measured_ttc_can_differ_from_a_designed_one():
    """The reason this module exists. The encounter is designed for 5 s; the pedestrian stops
    dead, so nothing ever closes and the measured min-TTC is None — a fact no designed number
    can express."""
    s = [
        Sample(t=float(i), robot=(float(i), 0.0), robot_vel=(1.0, 0.0),
               peds={"edge_0": ((10.0, 0.0), (0.0, 0.0), 0.3)})
        for i in range(4)
    ]
    out = score(s, only=["edge_0"])
    assert out.min_ttc_s is not None, "the robot is still closing on a stationary agent"
    # ... but with the robot also stopped, nothing closes at all:
    frozen = [Sample(t=float(i), robot=(0.0, 0.0), robot_vel=(0.0, 0.0),
                     peds={"edge_0": ((10.0, 0.0), (0.0, 0.0), 0.3)}) for i in range(4)]
    assert score(frozen, only=["edge_0"]).min_ttc_s is None


# The scores file
# ---------------


def test_score_row_puts_designed_and_achieved_side_by_side(tmp_path):
    """The comparison the panel exists to make. `designed_ttc` is fixed when the pedestrian is
    placed; `min_ttc_s` is what happened. A row carrying only one of them cannot show a tail
    having any effect."""
    from task_generator.tasks.obstacles.edge_case.scoring import write_score

    case = {
        "run_seed": "abc", "episode_id": 7, "world": "arena_arena_002", "seed": 11,
        "knob": "vision_fov", "level": 0.0, "designed_ttc": 4.0, "designed_pet": 0.0,
    }
    result = score([
        Sample(t=0.0, robot=(0.0, 0.0), robot_vel=(1.0, 0.0), peds={"edge_0": ((6.0, 0.0), (-1.0, 0.0), 0.3)}),
        Sample(t=1.0, robot=(1.0, 0.0), robot_vel=(1.0, 0.0), peds={"edge_0": ((5.0, 0.0), (-1.0, 0.0), 0.3)}),
    ], only=["edge_0"])

    path = write_score(tmp_path, case, result)
    row = json.loads(path.read_text().splitlines()[-1])

    assert row["episode_id"] == 7
    assert row["designed_ttc"] == 4.0
    assert row["min_ttc_s"] is not None
    assert row["observed"] is True
    assert row["min_clearance_with"] == "edge_0"


def test_an_unobserved_episode_is_flagged_not_silently_zero(tmp_path):
    """A row of nulls must be distinguishable from a genuinely safe episode."""
    from task_generator.tasks.obstacles.edge_case.scoring import write_score

    path = write_score(tmp_path, {"run_seed": "abc", "episode_id": 1}, score([]))
    row = json.loads(path.read_text().splitlines()[-1])
    assert row["observed"] is False
    assert row["min_ttc_s"] is None
    assert row["min_clearance_m"] is None


def test_scores_are_a_separate_file_from_cases(tmp_path):
    """A case is written when the episode is BUILT; a score only exists once it has RUN.
    Merging them would mean holding the case back or rewriting it."""
    from task_generator.tasks.obstacles.edge_case.scoring import write_score

    path = write_score(tmp_path, {"run_seed": "a", "episode_id": 1}, score([]))
    assert path.name == "scores.jsonl"
    write_score(tmp_path, {"run_seed": "a", "episode_id": 2}, score([]))
    assert len(path.read_text().strip().splitlines()) == 2, "append-only, one row per episode"


def test_formation_hold_reports_when_the_group_broke() -> None:
    from task_generator.tasks.obstacles.edge_case.criticality import Sample, formation_hold

    def sample(t, a, b, c):
        return Sample(t=t, robot=(0.0, 0.0), robot_vel=(0.0, 0.0), robot_radius=0.3, peds={"/env_0/d_1": (a, (0.0, 0.0), 0.3), "/env_0/d_2": (b, (0.0, 0.0), 0.3), "/env_0/d_3": (c, (0.0, 0.0), 0.3)})

    held = [sample(t, (0.0, 0.0), (1.0, 0.0), (0.5, 0.8)) for t in (0.0, 1.0, 2.0)]
    broke = [*held, sample(3.0, (0.0, 0.0), (1.0, 0.0), (4.0, 4.0)), sample(4.0, (0.0, 0.0), (1.0, 0.0), (5.0, 5.0))]
    out = formation_hold(broke, ["d_1", "d_2", "d_3"])
    assert out["seen"] and out["formed_at"] == 0.0 and out["held_s"] == 2.0 and out["broke_at"] == 3.0
    out = formation_hold(held, ["d_1", "d_2", "d_3"])
    assert out["seen"] and out["held_s"] == 2.0 and out["broke_at"] is None
    assert formation_hold(held, ["nobody"])["seen"] is False
    assert formation_hold([], ["d_1"])["seen"] is False
    # With the generator's centre: not formed until everyone arrives, then held, then broken.
    walking = [sample(0.0, (9.0, 9.0), (1.0, 0.0), (0.5, 0.8)), sample(1.0, (0.2, 0.1), (1.0, 0.0), (0.5, 0.8)), sample(2.0, (0.1, 0.0), (1.0, 0.0), (0.5, 0.8)), sample(3.0, (6.0, 6.0), (1.0, 0.0), (0.5, 0.8))]
    out = formation_hold(walking, ["d_1", "d_2", "d_3"], radius=1.5, centre=(0.5, 0.3))
    assert out["seen"] and out["formed_at"] == 1.0 and out["held_s"] == 1.0 and out["broke_at"] == 3.0
    never = formation_hold(walking[:1], ["d_1", "d_2", "d_3"], radius=1.5, centre=(0.5, 0.3))
    assert never["seen"] and never["formed_at"] is None and never["held_s"] is None
    assert never["arrived"] == 2 and out["arrived"] == 3  # two of three were there; later all three


# Pose-stream differencing
# ------------------------


def test_an_unrefreshed_pose_keeps_the_last_velocity_briefly():
    from task_generator.tasks.obstacles.edge_case.scoring import STALE_POSE_S, differenced_velocity

    vel, moved = differenced_velocity((0.0, 0.0), (0.0, 0.0), (0.1, 0.0), 0.1, 0.0)
    assert vel == pytest.approx((1.0, 0.0)) and moved == 0.1
    vel, moved = differenced_velocity((0.1, 0.0), vel, (0.1, 0.0), 0.2, moved)  # same pose 0.1 s later: TF lag
    assert vel == pytest.approx((1.0, 0.0)) and moved == 0.1
    vel, moved = differenced_velocity((0.1, 0.0), vel, (0.1, 0.0), 0.1 + STALE_POSE_S, moved)
    assert vel == (0.0, 0.0), "unchanged for the stale window: standing"


def test_a_changed_pose_is_differenced_over_the_time_since_it_last_changed():
    from task_generator.tasks.obstacles.edge_case.scoring import differenced_velocity

    vel, moved = differenced_velocity((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 0.3, 0.0)  # three ticks without a TF update
    vel, moved = differenced_velocity((0.0, 0.0), vel, (0.2, 0.0), 0.4, moved)
    assert vel == pytest.approx((0.5, 0.0)) and moved == 0.4
