from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("interactive_markers")


def test_menu_flattening_numbers_entries_and_links_parents():
    from arena_runtime.env.interactive.hub import Menu, _flatten

    async def noop() -> None: ...

    entries = []
    actions = {}
    _flatten([Menu("Edit", None, [Menu("Remove", noop), Menu("Rename", noop)]), Menu("Close", noop)], 0, entries, actions)

    assert [(e.id, e.parent_id, e.title) for e in entries] == [(1, 0, "Edit"), (2, 1, "Remove"), (3, 1, "Rename"), (4, 0, "Close")]
    assert sorted(actions) == [2, 3, 4]


def test_planar_marker_has_drag_and_yaw_controls():
    from geometry_msgs.msg import Pose
    from visualization_msgs.msg import InteractiveMarkerControl

    from arena_runtime.env.interactive.hub import planar_marker

    marker = planar_marker("m", Pose())
    assert marker.header.frame_id == "map"
    assert [c.interaction_mode for c in marker.controls] == [InteractiveMarkerControl.MOVE_PLANE, InteractiveMarkerControl.ROTATE_AXIS]


def test_planar_marker_without_rotation_has_only_drag_control():
    from geometry_msgs.msg import Pose

    from arena_runtime.env.interactive.hub import planar_marker

    assert len(planar_marker("m", Pose(), rotate=False).controls) == 1


def test_planar_marker_frame_defaults_to_map_and_can_follow_a_robot_frame():
    from geometry_msgs.msg import Pose

    from arena_runtime.env.interactive.hub import planar_marker

    assert planar_marker("goal/r1", Pose()).header.frame_id == "map"
    robot = planar_marker("robot/r1", Pose(), frame="env_0/r1/base_link")
    assert robot.header.frame_id == "env_0/r1/base_link"
    assert robot.header.stamp.sec == 0
    assert robot.header.stamp.nanosec == 0
