from __future__ import annotations

import pytest

pytest.importorskip("visualization_msgs")

from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker

from arena_runtime.env.realizer import Realizer
from arena_runtime.env.task_board import AgentEntry
from arena_runtime.env.task_markers import _STATE_COLORS, task_markers
from arena_runtime.shared import Position

SQUARE = [Position(0.0, 0.0), Position(4.0, 0.0), Position(4.0, 2.0), Position(0.0, 2.0)]
ZONES = {"kitchen": SQUARE}


def build(agents, positions, activity=None, realizer=None, frames=None):
    frames = {name: f"{name}/base_link" for name in positions} if frames is None else frames
    return task_markers(agents, frames, positions, ZONES.get, activity or {}, realizer or Realizer(), Time())


def in_ns(markers, ns):
    return [marker for marker in markers if marker.ns == ns and marker.action == Marker.ADD]


def test_overlay_clears_every_namespace_before_drawing():
    markers = build([], {})

    assert [(marker.ns, marker.action) for marker in markers] == [("task_labels", Marker.DELETEALL), ("task_links", Marker.DELETEALL), ("task_zones", Marker.DELETEALL)]


@pytest.mark.parametrize(
    ("entry", "text", "color"),
    [
        (AgentEntry("tb1", "fetch", "working", ("kitchen",)), "tb1: fetch (working)", "working"),
        (AgentEntry("tb1", "fetch", "assigned"), "tb1: fetch (assigned)", "assigned"),
        (AgentEntry("tb1", "fetch", "met"), "tb1: fetch (met)", "met"),
        (AgentEntry("tb1", "fetch", "failed"), "tb1: fetch (failed)", "failed"),
        (AgentEntry("tb1", "fetch", "dropped"), "tb1: fetch (dropped)", "dropped"),
        (AgentEntry("tb1"), "tb1: idle", "idle"),
    ],
)
def test_label_text_and_color_follow_the_agent_state(entry, text, color):
    (label,) = in_ns(build([entry], {"tb1": (1.0, 2.0)}), "task_labels")

    assert label.type == Marker.TEXT_VIEW_FACING
    assert label.text == text
    assert label.color == _STATE_COLORS[color]


def test_a_label_rides_in_its_agent_frame_and_follows_it_between_publishes():
    (label,) = in_ns(build([AgentEntry("tb1", "fetch", "assigned")], {"tb1": (1.0, 2.0)}), "task_labels")

    assert (label.header.frame_id, label.frame_locked) == ("tb1/base_link", True)
    assert (label.header.stamp.sec, label.header.stamp.nanosec) == (0, 0)
    assert (label.pose.position.x, label.pose.position.y, label.pose.position.z) == pytest.approx((0.0, 0.0, 2.0))


def test_activity_replaces_working_in_the_label_only_for_working_agents():
    agents = [AgentEntry("tb1", "fetch", "working", ("kitchen",)), AgentEntry("tb2", "fetch", "assigned")]
    labels = in_ns(build(agents, {"tb1": (0.0, 0.0), "tb2": (1.0, 1.0)}, {"tb1": "holding", "tb2": "navigating"}), "task_labels")

    assert [label.text for label in labels] == ["tb1: fetch (holding)", "tb2: fetch (assigned)"]


def test_working_agent_links_to_the_first_zone_centroid_and_outlines_each_zone():
    markers = build([AgentEntry("tb1", "fetch", "working", ("kitchen", "kitchen"))], {"tb1": (-1.0, 5.0)})

    (link,) = in_ns(markers, "task_links")
    start, end = link.points
    assert (start.x, start.y) == (-1.0, 5.0)
    assert (end.x, end.y) == pytest.approx((2.0, 1.0))
    (outline,) = in_ns(markers, "task_zones")
    assert len(outline.points) == 16
    assert outline.color == _STATE_COLORS["working"]


def test_zone_outline_carries_the_env_offset():
    realizer = Realizer(Realizer._Configuration(x=10.0, y=20.0))
    markers = build([AgentEntry("tb1", "fetch", "working", ("kitchen",))], {"tb1": (0.0, 0.0)}, realizer=realizer)

    (link,) = in_ns(markers, "task_links")
    assert (link.points[1].x, link.points[1].y) == pytest.approx((12.0, 21.0))


def test_only_working_agents_with_known_zones_get_a_link_and_outlines():
    agents = [
        AgentEntry("a", "t", "assigned", ("kitchen",)),
        AgentEntry("b", "t", "met", ("kitchen",)),
        AgentEntry("c", "t", "working", ("nowhere",)),
        AgentEntry("d", "t", "working"),
    ]
    markers = build(agents, {name: (0.0, 0.0) for name in "abcd"})

    assert len(in_ns(markers, "task_labels")) == 4
    assert in_ns(markers, "task_links") == []
    assert in_ns(markers, "task_zones") == []


def test_agents_without_a_position_are_skipped():
    agents = [AgentEntry("tb1", "fetch", "working", ("kitchen",)), AgentEntry("ped", "fetch", "working", ("kitchen",))]
    markers = build(agents, {"ped": (0.0, 0.0)})

    assert [label.text for label in in_ns(markers, "task_labels")] == ["ped: fetch (working)"]
    assert len(in_ns(markers, "task_links")) == 1


def test_agents_without_a_frame_get_no_label():
    agents = [AgentEntry("tb1", "fetch", "assigned"), AgentEntry("ped", "fetch", "assigned")]
    markers = build(agents, {"tb1": (0.0, 0.0), "ped": (1.0, 1.0)}, frames={"tb1": "tb1/base_link"})

    assert [label.text for label in in_ns(markers, "task_labels")] == ["tb1: fetch (assigned)"]
