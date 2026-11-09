"""Geometry of the map and marker renderers: what lands where, before anything is logged."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker

from rerun_utils.entity_paths import frame_name, in_frame
from rerun_utils.tf_mirror import Tie, TFTree
from rerun_utils.renderers.map import FREE, OCCUPIED, UNKNOWN, grid_cells, grid_quad, grid_key, grid_texture, downsample, is_overlay, occupied_cells
from rerun_utils.marker_geometry import convert, posed
from rerun_utils.renderers.marker_array import MarkerLog, MarkerStore


def _grid(width: int, height: int, resolution: float, x: float, y: float, yaw: float = 0.0, data: list[int] | None = None) -> OccupancyGrid:
    g = OccupancyGrid()
    g.info.width, g.info.height, g.info.resolution = width, height, resolution
    g.info.origin.position.x, g.info.origin.position.y = x, y
    g.info.origin.orientation.z, g.info.origin.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    g.data = data if data is not None else [0] * (width * height)
    return g


def test_map_quad_has_the_grid_metric_size_at_its_origin() -> None:
    corners, uv = grid_quad(_grid(200, 100, 0.05, -1.0, 2.0), 0.0)
    assert corners[0] == pytest.approx([-1.0, 2.0, 0.0])
    assert corners[2] == pytest.approx([-1.0 + 10.0, 2.0 + 5.0, 0.0])
    assert uv.tolist() == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def test_map_quad_turns_with_the_origin_yaw() -> None:
    corners, _ = grid_quad(_grid(20, 10, 0.5, 0.0, 0.0, yaw=math.pi / 2), 0.0)
    assert corners[1] == pytest.approx([0.0, 10.0, 0.0])  # the grid's x edge now points along +y


def test_map_texture_rows_start_at_the_origin() -> None:
    tex = grid_texture(grid_cells(_grid(2, 2, 1.0, 0.0, 0.0, data=[100, 0, -1, 50])))
    assert tex.shape == (2, 2, 3)
    assert tex[0, 0, :3].tolist() == OCCUPIED.tolist()
    assert tex[0, 1, :3].tolist() == FREE.tolist()
    assert tex[1, 0, :3].tolist() == UNKNOWN.tolist()
    assert (FREE > tex[1, 1, :3]).all() and (tex[1, 1, :3] > OCCUPIED).all()


def _marker(type_: int, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, points: list[tuple[float, float, float]] | None = None) -> Marker:
    m = Marker()
    m.type = type_
    m.pose.position.x, m.pose.position.y = x, y
    m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    m.scale.x = m.scale.y = m.scale.z = 0.2
    m.color.r = m.color.a = 1.0
    m.points = [Point(x=px, y=py, z=pz) for px, py, pz in points or []]
    return m


def test_list_points_follow_the_marker_pose() -> None:
    m = _marker(Marker.LINE_STRIP, x=1.0, y=2.0, yaw=math.pi / 2, points=[(1.0, 0.0, 0.0)])
    assert posed(m, m.points)[0] == pytest.approx([1.0, 3.0, 0.0])


def test_each_marker_kind_becomes_its_archetype() -> None:
    markers = [
        _marker(Marker.CUBE),
        _marker(Marker.SPHERE),
        _marker(Marker.CYLINDER),
        _marker(Marker.ARROW),
        _marker(Marker.LINE_LIST, points=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]),
        _marker(Marker.POINTS, points=[(0, 0, 0), (1, 1, 0)]),
        _marker(Marker.TRIANGLE_LIST, points=[(0, 0, 0), (1, 0, 0), (0, 1, 0)]),
    ]
    text = _marker(Marker.TEXT_VIEW_FACING)
    text.text = "hello"
    out = convert([*markers, text]).archetypes()
    assert set(out) == {"boxes", "spheres", "cylinders", "arrows", "lines", "points", "triangles", "text"}


def test_line_list_is_one_segment_per_point_pair() -> None:
    geometry = convert([_marker(Marker.LINE_LIST, points=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])])
    assert len(geometry.strips.strips) == 2


def test_frame_names_drop_the_leading_slash_and_empty_means_the_env_root() -> None:
    assert frame_name("/map") == "map"
    assert in_frame("") == []
    assert len(in_frame("env_0/jackal/base_link")) == 1


def test_namespace_parts_stay_apart() -> None:
    from rerun_utils.renderers.marker_array import _safe

    names = ["a/b", "a_b", "a b", "", "~", "~~", "default", "~x", "x"]
    parts = [_safe(n) for n in names]
    assert len(set(parts)) == len(names)
    assert all(part and "/" not in part.replace("\\/", "") for part in parts)


def _lapsing(ns: str, id_: int, lifetime_s: float, action: int = Marker.ADD) -> Marker:
    m = _marker(Marker.CUBE)
    m.ns, m.id, m.action = ns, id_, action
    m.lifetime.sec, m.lifetime.nanosec = int(lifetime_s), int(round((lifetime_s % 1) * 1e9))
    return m


S = 1_000_000_000


def test_a_marker_lapses_its_lifetime_after_it_arrived() -> None:
    store = MarkerStore()
    store.ingest([_lapsing("agents", 1, 1.0), _lapsing("walls", 1, 0.0)], now_ns=10 * S)
    assert store.lapsing()
    assert not store.expire(10 * S + S // 2)
    assert store.expire(11 * S)
    assert [ns for ns, _ in store.groups()] == ["walls"]  # no lifetime: stays until deleted
    assert not store.lapsing()


def test_resending_a_marker_renews_its_lifetime() -> None:
    store = MarkerStore()
    store.ingest([_lapsing("agents", 1, 1.0)], now_ns=0)
    store.ingest([_lapsing("agents", 1, 1.0)], now_ns=S - 1)
    assert not store.expire(S)
    assert store.expire(2 * S - 1)


def test_delete_and_deleteall_still_remove_markers() -> None:
    store = MarkerStore()
    store.ingest([_lapsing("a", 1, 0.0), _lapsing("a", 2, 0.0), _lapsing("b", 1, 0.0)], now_ns=0)
    store.ingest([_lapsing("a", 1, 0.0, Marker.DELETE)], now_ns=0)
    assert sorted(len(v) for v in store.groups().values()) == [1, 1]
    store.ingest([_lapsing("", 0, 0.0, Marker.DELETEALL)], now_ns=0)
    assert not store.groups()


def _at(m: Marker, frame: str) -> Marker:
    m.header.frame_id = frame
    return m


def test_only_the_groups_a_message_touched_are_dirty() -> None:
    store = MarkerStore()
    store.ingest([_at(_lapsing("a", 1, 0.0), "map"), _at(_lapsing("b", 1, 0.0), "map")], now_ns=0)
    assert set(store.take_dirty()) == {("a", "map"), ("b", "map")}
    assert store.take_dirty() == {}
    store.ingest([_at(_lapsing("b", 2, 0.0), "map")], now_ns=0)
    assert set(store.take_dirty()) == {("b", "map")}


def test_a_group_that_lost_its_markers_is_dirty_and_empty() -> None:
    store = MarkerStore()
    store.ingest([_at(_lapsing("a", 1, 1.0), "map"), _at(_lapsing("b", 1, 0.0), "map")], now_ns=0)
    store.take_dirty()
    assert store.expire(S)
    assert store.take_dirty() == {("a", "map"): []}
    store.ingest([_lapsing("b", 1, 0.0, Marker.DELETE)], now_ns=S)
    assert store.take_dirty() == {("b", "map"): []}


def test_a_marker_moving_to_another_frame_dirties_both_groups() -> None:
    store = MarkerStore()
    store.ingest([_at(_lapsing("a", 1, 0.0), "map")], now_ns=0)
    store.take_dirty()
    store.ingest([_at(_lapsing("a", 1, 0.0), "odom")], now_ns=0)
    assert store.take_dirty() == {("a", "map"): [], ("a", "odom"): store.groups()[("a", "odom")]}


def test_the_log_clears_the_kinds_a_group_lost_and_logs_the_kinds_it_has() -> None:
    log = MarkerLog("env_0/hunav")
    cube = _at(_lapsing("agents", 1, 0.0), "map")
    sphere = _at(_marker(Marker.SPHERE), "map")
    sphere.ns = "agents"
    first = log.entries({("agents", "map"): [cube, sphere]})
    assert {e.entity for e in first if e.archetype is not None} == {"env_0/hunav/agents/map/boxes", "env_0/hunav/agents/map/spheres"}
    second = log.entries({("agents", "map"): [cube]})
    assert [(e.entity, e.archetype is None) for e in second] == [("env_0/hunav/agents/map/boxes", False), ("env_0/hunav/agents/map/spheres", True)]
    third = log.entries({("agents", "map"): []})
    assert [(e.entity, e.archetype is None) for e in third] == [("env_0/hunav/agents/map/boxes", True)]
    assert log.entries({("agents", "map"): []}) == []


def test_the_pedestrian_roster_is_read_from_the_env_namespace() -> None:
    from rerun_utils.renderers.pedestrians import peds_topic

    assert peds_topic("/arena/env_3") == "/arena/env_3/arena_peds"
    assert peds_topic("/arena/env_3/") == "/arena/env_3/arena_peds"
    assert peds_topic("") == "/arena_peds"


def test_pedestrians_stand_on_the_z_of_their_pose() -> None:
    from arena_people_msgs.msg import Pedestrian

    from rerun_utils.renderers.pedestrians import HEIGHT, RADIUS, body_centers, heading_origins

    ped = Pedestrian()
    ped.pose.position.x, ped.pose.position.y, ped.pose.position.z = 1.0, 2.0, 3.0
    assert body_centers([ped])[0] == pytest.approx([1.0, 2.0, 3.0 + RADIUS])
    assert heading_origins([ped])[0] == pytest.approx([1.0, 2.0, 3.0 + HEIGHT * 0.6])


def test_a_grid_is_an_overlay_when_the_manifest_styles_it_as_a_costmap() -> None:
    from arena_viz import StyleSpec

    assert is_overlay(StyleSpec(extra={"rviz": {"Color Scheme": "costmap"}}))
    assert not is_overlay(StyleSpec(extra={"rviz": {"Color Scheme": "map"}}))
    assert not is_overlay(StyleSpec())


def test_an_overlay_draws_only_the_cells_with_a_cost_at_their_centers() -> None:
    grid = _grid(2, 2, 1.0, 3.0, 4.0, yaw=math.pi / 2, data=[100, 0, -1, 50])
    centers, colors = occupied_cells(grid, grid_cells(grid), 0.02)
    assert centers == pytest.approx(np.array([[2.5, 4.5, 0.02], [1.5, 5.5, 0.02]]))
    assert colors[0].tolist() == OCCUPIED.tolist()
    assert (FREE > colors[1]).all() and (colors[1] > OCCUPIED).all()


def test_downsampling_keeps_the_darkest_cell_of_each_block_and_small_grids_as_they_are() -> None:
    cells = np.zeros((5, 3), dtype=np.int16)
    cells[4, 2] = 100
    small = downsample(cells, cap=8)
    assert small is cells
    reduced = downsample(cells, cap=3)
    assert reduced.shape == (3, 2)
    assert reduced[2, 1] == 100 and reduced.sum() == 100


def test_a_grid_key_changes_with_the_data_and_not_with_the_stamp() -> None:
    a, b = _grid(2, 1, 1.0, 0.0, 0.0, data=[0, 0]), _grid(2, 1, 1.0, 0.0, 0.0, data=[0, 0])
    b.header.stamp.sec = 5
    assert grid_key(a) == grid_key(b)
    b.data = [0, 100]
    assert grid_key(a) != grid_key(b)


def _roots_after(edges: list[tuple[str, str]]) -> tuple[frozenset[str], set[str]]:
    """The tree's roots and the frames left tied once the ties it asked for are applied in order."""
    tree, tied = TFTree(), set()
    for child, parent in edges:
        for tie in tree.ingest(child, parent):
            if tie.tied:
                tied.add(tie.frame)
            else:
                tied.discard(tie.frame)
    return tree.roots, tied


def test_the_roots_do_not_depend_on_the_order_transforms_arrive_in() -> None:
    chain = [("odom", "map"), ("base_link", "odom"), ("laser", "base_link")]
    for order in (chain, chain[::-1], [chain[1], chain[0], chain[2]]):
        roots, tied = _roots_after(order)
        assert roots == {"map"}
        assert tied == {"map"}


def test_a_frame_seen_first_as_a_parent_is_untied_once_it_gains_a_parent() -> None:
    tree = TFTree()
    assert tree.ingest("odom", "map") == [Tie("map", True)]
    assert tree.ingest("map", "world") == [Tie("map", False), Tie("world", True)]
    assert tree.ingest("base_link", "odom") == []


def test_one_shared_tree_has_one_root_whatever_the_env_frames_are() -> None:
    edges = [("env_0/map", "world"), ("env_1/map", "world"), ("env_0/base", "env_0/map"), ("env_1/base", "env_1/map")]
    for order in (edges, edges[::-1]):
        roots, tied = _roots_after(order)
        assert roots == tied == {"world"}


def test_two_separate_trees_have_two_roots() -> None:
    roots, _ = _roots_after([("a_child", "a"), ("b_child", "b")])
    assert roots == {"a", "b"}


def test_arrow_shaft_width_follows_the_scale_rviz_reads_for_its_form() -> None:
    two_point = _marker(Marker.ARROW, points=[(0, 0, 0), (1, 0, 0)])
    two_point.scale.x, two_point.scale.y = 0.1, 0.3
    posed_arrow = _marker(Marker.ARROW)
    posed_arrow.scale.x, posed_arrow.scale.y, posed_arrow.scale.z = 2.0, 0.08, 0.2
    geometry = convert([two_point, posed_arrow])
    assert geometry.arrows.radii == pytest.approx([0.05, 0.04])
    assert geometry.arrows.vectors[1] == pytest.approx([2.0, 0.0, 0.0])
