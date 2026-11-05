"""Geometry of the map and marker renderers: what lands where, before anything is logged."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("rclpy")

from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker

from rerun_utils.entity_paths import frame_name, in_frame
from rerun_utils.renderers.map import FREE, OCCUPIED, UNKNOWN, grid_quad, grid_texture
from rerun_utils.renderers.marker_array import _archetypes, _posed


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
    tex = grid_texture(_grid(2, 2, 1.0, 0.0, 0.0, data=[100, 0, -1, 50]))
    assert tex.shape == (2, 2, 3)
    assert tex[0, 0].tolist() == OCCUPIED.tolist()
    assert tex[0, 1].tolist() == FREE.tolist()
    assert tex[1, 0].tolist() == UNKNOWN.tolist()
    assert (FREE > tex[1, 1]).all() and (tex[1, 1] > OCCUPIED).all()


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
    assert _posed(m, m.points)[0] == pytest.approx([1.0, 3.0, 0.0])


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
    out = _archetypes([*markers, text])
    assert set(out) == {"boxes", "spheres", "cylinders", "arrows", "lines", "points", "triangles", "text"}


def test_line_list_is_one_segment_per_point_pair() -> None:
    from rerun_utils.renderers import marker_array

    calls: list[list] = []
    real = marker_array.rr.LineStrips3D
    marker_array.rr.LineStrips3D = lambda strips, **_: calls.append(strips)  # type: ignore[assignment]
    try:
        _archetypes([_marker(Marker.LINE_LIST, points=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])])
    finally:
        marker_array.rr.LineStrips3D = real  # type: ignore[assignment]
    assert len(calls[0]) == 2


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
    from rerun_utils.renderers.marker_array import MarkerStore

    store = MarkerStore()
    store.ingest([_lapsing("agents", 1, 1.0), _lapsing("walls", 1, 0.0)], now_ns=10 * S)
    assert store.lapsing()
    assert not store.expire(10 * S + S // 2)
    assert store.expire(11 * S)
    assert [ns for ns, _ in store.groups()] == ["walls"]  # no lifetime: stays until deleted
    assert not store.lapsing()


def test_resending_a_marker_renews_its_lifetime() -> None:
    from rerun_utils.renderers.marker_array import MarkerStore

    store = MarkerStore()
    store.ingest([_lapsing("agents", 1, 1.0)], now_ns=0)
    store.ingest([_lapsing("agents", 1, 1.0)], now_ns=S - 1)
    assert not store.expire(S)
    assert store.expire(2 * S - 1)


def test_delete_and_deleteall_still_remove_markers() -> None:
    from rerun_utils.renderers.marker_array import MarkerStore

    store = MarkerStore()
    store.ingest([_lapsing("a", 1, 0.0), _lapsing("a", 2, 0.0), _lapsing("b", 1, 0.0)], now_ns=0)
    store.ingest([_lapsing("a", 1, 0.0, Marker.DELETE)], now_ns=0)
    assert sorted(len(v) for v in store.groups().values()) == [1, 1]
    store.ingest([_lapsing("", 0, 0.0, Marker.DELETEALL)], now_ns=0)
    assert not store.groups()


class _Clock:
    def __init__(self) -> None:
        self.ns = 0

    def now(self):  # noqa: ANN201 - stands in for rclpy.time.Time
        clock = self

        class _T:
            nanoseconds = clock.ns

        return _T()


class _Node:
    def __init__(self) -> None:
        self.clock, self.subs, self.timers = _Clock(), [], []

    def get_clock(self) -> _Clock:
        return self.clock

    def create_subscription(self, _type, topic, cb, _qos):  # noqa: ANN001, ANN201
        self.subs.append((topic, cb))

    def create_timer(self, period, cb):  # noqa: ANN001, ANN201
        self.timers.append((period, cb))


def test_the_bridge_clears_a_lapsed_marker_between_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_runtime_msgs.msg import AdapterDisplay
    from visualization_msgs.msg import MarkerArray

    from rerun_utils.renderers import marker_array
    from rerun_utils.renderers._registry import RendererCtx

    logged: list[tuple[str, str]] = []
    monkeypatch.setattr(marker_array.rr, "log", lambda entity, archetype, *_: logged.append((entity, type(archetype).__name__)))
    node = _Node()
    marker_array.subscribe_marker_array(RendererCtx(env_id=0, node=node), AdapterDisplay(topic="/hunav/markers", style_json="{}"), "env_0/hunav")
    ((_, cb),) = node.subs
    m = _lapsing("agents", 1, 1.0)
    m.header.frame_id = "map"
    cb(MarkerArray(markers=[m]))
    assert logged == [("env_0/hunav/agents/map/boxes", "Boxes3D")]
    ((_, tick),) = node.timers
    node.clock.ns = S // 2
    tick()
    assert len(logged) == 1
    node.clock.ns = S
    tick()
    assert logged[-1] == ("env_0/hunav/agents/map/boxes", "Clear")


def test_the_pedestrian_roster_is_read_from_the_env_namespace_whatever_the_tf_mode() -> None:
    from rerun_utils.renderers._registry import RendererCtx
    from rerun_utils.renderers.pedestrians import peds_topic
    from rerun_utils.scripts.rerun_bridge import env_namespace

    ns = env_namespace("/arena/env_3/arena_env_node")
    assert ns == "/arena/env_3"
    assert peds_topic(RendererCtx(env_id=3, node=None, env_namespace=ns, tf_namespace="")) == "/arena/env_3/arena_peds"  # type: ignore[arg-type]
    assert peds_topic(RendererCtx(env_id=3, node=None, env_namespace=ns, tf_namespace=ns)) == "/arena/env_3/arena_peds"  # type: ignore[arg-type]
    assert env_namespace("/arena_env_node") == ""
