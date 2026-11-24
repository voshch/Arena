from __future__ import annotations

import math

import pytest

from human_steering import markers

msg = pytest.importorskip("visualization_msgs.msg")
geometry = pytest.importorskip("geometry_msgs.msg")


def _marker(kind: int, points: tuple[tuple[float, float], ...] = (), *, ns: str = "walls", mid: int = 0, at: tuple[float, float, float] = (0.0, 0.0, 0.0), action: int = 0) -> msg.Marker:
    marker = msg.Marker(ns=ns, id=mid, type=kind, action=action)
    marker.pose.position.x, marker.pose.position.y = at[0], at[1]
    marker.pose.orientation.z, marker.pose.orientation.w = math.sin(at[2] / 2.0), math.cos(at[2] / 2.0)
    marker.points = [geometry.Point(x=x, y=y) for x, y in points]
    return marker


def test_line_list_is_one_segment_per_pair_shifted_by_the_marker_pose() -> None:
    walls = _marker(msg.Marker.LINE_LIST, ((0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (0.0, 3.0)), at=(5.0, 5.0, 0.0))
    assert markers.outlines(walls) == [[(5.0, 5.0), (6.0, 5.0)], [(5.0, 7.0), (5.0, 8.0)]]


def test_line_strip_stays_one_open_polyline() -> None:
    strip = _marker(msg.Marker.LINE_STRIP, ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)))
    assert markers.outlines(strip) == [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]]


def test_cube_is_its_rotated_closed_footprint() -> None:
    box = _marker(msg.Marker.CUBE, at=(2.0, 1.0, math.pi / 2.0))
    box.scale.x, box.scale.y = 2.0, 1.0
    (outline,) = markers.outlines(box)
    assert outline[0] == outline[-1]
    assert outline[:4] == [pytest.approx(corner) for corner in [(2.5, 0.0), (2.5, 2.0), (1.5, 2.0), (1.5, 0.0)]]


def test_text_markers_draw_nothing() -> None:
    assert markers.outlines(_marker(msg.Marker.TEXT_VIEW_FACING)) == []


def test_merge_replaces_by_ns_and_id_and_honors_deletes() -> None:
    cache: dict[tuple[str, int], msg.Marker] = {}
    markers.merge(cache, [_marker(msg.Marker.CUBE, mid=1), _marker(msg.Marker.CUBE, mid=2)])
    markers.merge(cache, [_marker(msg.Marker.CUBE, mid=1, action=msg.Marker.DELETE)])
    assert sorted(cache) == [("walls", 2)]
    markers.merge(cache, [_marker(msg.Marker.CUBE, action=msg.Marker.DELETEALL), _marker(msg.Marker.LINE_LIST, mid=3)])
    assert sorted(cache) == [("walls", 3)]
