from __future__ import annotations

import math

import pytest

from arena_humansim.core.interaction_kinds import InteractionType, MembershipRole
from arena_humansim_msgs.msg import InteractionFrame

from arena_simulation_setup.utils.affordance import AffordanceKind, conflicts, placed_slots
from arena_simulation_setup.utils.geometry import Pose

from task_generator.simulators.human.arena_humansim.arena_humansim import conflict_offsets, held_seats, reach_radius, seat_capacity, seat_height, seat_violation, slot_height

SOFA = placed_slots(
    {
        "sit": [
            {"bbox": [[-0.46, 0.48], [0.03, 0.72], [0.0, 0.47]], "yaw": "-y"},
            {"bbox": [[-0.98, -0.48], [-0.50, 0.03], [0.0, 0.47]], "yaw": "+x"},
            {"bbox": [[-1.30, -0.56], [-1.13, -0.26], [0.0, 0.47]], "yaw": ["+x", "-y"]},
        ]
    }
)


def test_capacity_counts_sit_slots_that_can_be_taken_together():
    assert seat_capacity({}, SOFA) == 2


def test_capacity_prefers_the_scenario_capacity_then_its_seats():
    assert seat_capacity({"capacity": 5}, SOFA) == 5
    assert seat_capacity({"seats": [{"x": 0.0, "y": 0.0}] * 3}, SOFA) == 3
    assert seat_capacity({"seats": []}, SOFA) == 1


def test_capacity_is_one_without_slots():
    assert seat_capacity({}, []) == 1


def test_slot_height_is_the_top_of_its_z_range():
    flat = placed_slots({AffordanceKind.SIT: [{"bbox": [[2.0, 2.5], [0.0, 0.5]], "yaw": "-y"}]})
    assert slot_height(SOFA[0]) == 0.47
    assert math.isnan(slot_height(flat[0]))


def test_scenario_seat_height_is_its_z_else_the_slot_it_lies_in():
    assert seat_height(Pose.parse({"x": 0.0, "y": 0.1, "z": 1.0, "yaw": 0.0}), SOFA) == 1.0
    assert seat_height(Pose.parse({"x": 0.0, "y": 0.4}), SOFA) == 0.47
    assert math.isnan(seat_height(Pose.parse({"x": 3.0, "y": 3.0}), SOFA))


def test_scenario_seat_just_off_its_slot_still_takes_its_height():
    slots = placed_slots({AffordanceKind.SIT: [{"bbox": [[0.0, 0.5], [0.0, 0.5], [0.0, 0.45]], "yaw": "-y"}, {"bbox": [[0.4, 1.0], [0.0, 0.5], [0.0, 0.6]], "yaw": "-y"}]})
    assert seat_height(Pose.parse({"x": 0.25, "y": -0.05}), slots) == 0.45  # 5 cm outside the first slot
    assert seat_height(Pose.parse({"x": 0.45, "y": 0.25}), slots) == 0.45  # inside both, nearer the first's center
    assert seat_height(Pose.parse({"x": 0.55, "y": 0.25}), slots) == 0.6
    assert math.isnan(seat_height(Pose.parse({"x": 0.25, "y": -0.5}), slots))


def test_reach_radius_covers_the_footprint_from_beside_it():
    assert reach_radius([[-0.7, 0.7], [-0.3, 0.3], [0.0, 0.9]]) == pytest.approx(math.hypot(0.7, 0.3) + 0.3)
    assert reach_radius([[0.0, 2.0], [-0.5, 0.0], [0.0, 0.5]]) == pytest.approx(math.hypot(2.0, 0.5) + 0.3)


def test_held_seats_lists_the_seats_of_sit_and_lie_participants_only():
    frame = InteractionFrame(
        interaction_id=[1, 2, 3, 4],
        interaction_type=[InteractionType.SIT_ON, InteractionType.LIE_ON, InteractionType.SIT_ON, InteractionType.QUEUE_USE],
        object_id=["env_0/sofa", "env_0/sofa", "env_0/bench", "env_0/desk"],
        holding=[True, False, True, True],
        member_offset=[0, 2, 3, 4, 5],
        member_id=[7, 8, 9, 10, 11],
        member_role=[MembershipRole.PARTICIPANT, MembershipRole.QUEUED, MembershipRole.PARTICIPANT, MembershipRole.PARTICIPANT, MembershipRole.PARTICIPANT],
        member_seat=[0, -1, 2, -1, 3],
    )
    assert held_seats(frame) == {"env_0/sofa": [0, 2], "env_0/bench": []}


def test_conflicts_pair_the_overlapping_slots_both_ways():
    assert conflicts(SOFA) == [[], [2], [1]]
    assert conflict_offsets(SOFA) == ([0, 0, 1, 2], [2, 1])


def test_seat_violation_names_overlapping_held_seats():
    assert seat_violation({"env_0/sofa": SOFA}, {"env_0/sofa": [0, 1]}) is None
    assert "env_0/sofa" in (seat_violation({"env_0/sofa": SOFA}, {"env_0/sofa": [1, 2]}) or "")


def test_seat_violation_names_a_seat_outside_the_slots():
    assert "seat 5 of 3" in (seat_violation({"env_0/sofa": SOFA}, {"env_0/sofa": [5]}) or "")


def test_seat_violation_ignores_objects_without_annotated_slots():
    assert seat_violation({}, {"env_0/bench": [0, 0]}) is None
