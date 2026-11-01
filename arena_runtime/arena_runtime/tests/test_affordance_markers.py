from __future__ import annotations

import math

import pytest
from arena_simulation_setup.utils.affordance import placed_slots
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker

from arena_runtime.env.affordance_markers import _BLOCKED_COLOR, _KIND_COLORS, _OCCUPIED_COLOR, affordance_markers, slot_markers
from arena_runtime.shared import Orientation, Pose, Position

SLOTS = placed_slots(
    {
        "sit": [
            {"bbox": [[-0.46, 0.48], [0.03, 0.72], [0.0, 0.47]], "yaw": "-y"},
            {"bbox": [[0.59, 1.36], [-1.06, -0.38], [0.0, 0.45]], "yaw": ["-x", "+x", "-y"]},
        ]
    }
)
TURNED = Pose(Position(2.0, 1.0, 0.0), Orientation.parse(math.pi / 2))


def test_overlay_clears_then_draws_a_box_and_an_arrow_per_facing():
    markers = affordance_markers([("sofa", TURNED, SLOTS, [])], Time())
    assert (markers[0].action, markers[0].ns) == (Marker.DELETEALL, "sit")
    drawn = markers[1:]
    assert [marker.type for marker in drawn] == [Marker.LINE_LIST, Marker.ARROW, Marker.LINE_LIST, Marker.ARROW, Marker.ARROW, Marker.ARROW]
    assert len(drawn[0].points) == 28
    assert {marker.ns for marker in drawn} == {"sit"}


def test_arrow_starts_on_the_box_edge_and_points_along_the_facing_in_the_map_frame():
    _, arrow = slot_markers("sofa", TURNED, SLOTS[0], Time())
    start, end = arrow.points
    assert (start.x, start.y, start.z) == pytest.approx((2.0 - 0.03, 1.0 + 0.01, 0.47))
    assert (end.x - start.x, end.y - start.y) == pytest.approx((0.25, 0.0))


def test_marker_ids_are_stable_per_object_and_slot():
    first = [marker.id for marker in slot_markers("sofa", TURNED, SLOTS[1], Time())]
    again = [marker.id for marker in slot_markers("sofa", Pose(Position(0.0, 0.0, 0.0)), SLOTS[1], Time())]
    other = [marker.id for marker in slot_markers("couch", TURNED, SLOTS[1], Time())]
    assert first == again
    assert len(set(first)) == len(first)
    assert not set(first) & set(other)


def test_held_seat_marks_its_slot_occupied_and_overlapping_slots_blocked():
    slots = placed_slots(
        {
            "sit": [
                {"bbox": [[0.0, 0.5], [0.0, 0.5], [0.0, 0.45]], "yaw": "-y"},
                {"bbox": [[0.3, 0.9], [0.0, 0.5], [0.0, 0.45]], "yaw": "-y"},
                {"bbox": [[1.0, 1.5], [0.0, 0.5], [0.0, 0.45]], "yaw": "-y"},
            ]
        }
    )
    boxes = [marker for marker in affordance_markers([("bench", TURNED, slots, [0])], Time()) if marker.type == Marker.LINE_LIST]
    assert [box.color for box in boxes] == [_OCCUPIED_COLOR, _BLOCKED_COLOR, _KIND_COLORS["sit"]]


def test_held_seat_outside_the_slots_claims_nothing():
    boxes = [marker for marker in affordance_markers([("sofa", TURNED, SLOTS, [9])], Time()) if marker.type == Marker.LINE_LIST]
    assert all(box.color == _KIND_COLORS["sit"] for box in boxes)
