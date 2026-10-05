"""Node-level tests for hearing_belief and hearing_policy, driven without a fleet binding."""

from __future__ import annotations

import json
import math
import time
import uuid

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("scipy")
    pytest.importorskip("geometry_msgs.msg")
    pytest.importorskip("nav_msgs.msg")
    pytest.importorskip("std_msgs.msg")
    pytest.importorskip("task_generator_msgs.msg")
    pytest.importorskip("visualization_msgs.msg")


def _spin_until(rclpy, nodes, predicate, timeout_sec: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_sec

    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)

        if predicate():
            return

    raise AssertionError("timed out waiting for ROS round-trip")


def _make_heard_sound_event(bearing_rad: float = 0.0):
    from task_generator_msgs.msg import HeardSoundEvent

    event = HeardSoundEvent()
    event.header.frame_id = "map"
    event.sound_type = "footstep"
    event.audible = True
    event.received_volume_db = 50.0
    event.bearing_rad = bearing_rad
    return event


def _static_transform(parent: str, child: str, x: float, y: float):
    from geometry_msgs.msg import TransformStamped

    transform = TransformStamped()
    transform.header.frame_id = parent
    transform.child_frame_id = child
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.rotation.w = 1.0
    return transform


def _l_shaped_occupancy(size: int, resolution: float) -> np.ndarray:
    """True = occupied. Free legs: a 0.5-10.0 x-leg at y in [0.4, 1.6], a 1.0-10.0 y-leg at x in [9.4, 10.6]."""
    rows = (np.arange(size) + 0.5) * resolution
    cols = (np.arange(size) + 0.5) * resolution
    xx, yy = np.meshgrid(cols, rows)
    occupied = np.ones((size, size), dtype=bool)
    leg_a = (xx >= 0.5) & (xx <= 10.0) & (yy >= 0.4) & (yy <= 1.6)
    leg_b = (yy >= 1.0) & (yy <= 10.0) & (xx >= 9.4) & (xx <= 10.6)
    occupied[leg_a | leg_b] = False
    return occupied


def _l_shaped_plan() -> np.ndarray:
    xs = np.round(np.arange(1.0, 10.0 + 1e-9, 0.1), 6)
    leg_a = np.stack([xs, np.full_like(xs, 1.0)], axis=1)
    ys = np.round(np.arange(1.1, 9.5 + 1e-9, 0.1), 6)
    leg_b = np.stack([np.full_like(ys, 10.0), ys], axis=1)
    return np.concatenate([leg_a, leg_b], axis=0)


def test_belief_node_paints_wedge_along_bearing_and_not_behind(rclpy_context):
    import rclpy
    from nav_msgs.msg import OccupancyGrid
    from rclpy.parameter import Parameter

    from arena_auditory.hearing.belief_node import BeliefNode

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    heard_topic = f"/test/{suffix}/heard_sound"
    belief_topic = f"/test/{suffix}/belief_grid"
    base_frame = f"{suffix}/base_link"

    node = BeliefNode(
        parameter_overrides=[
            Parameter("heard_sound_topic", Parameter.Type.STRING, heard_topic),
            Parameter("standalone_grid", Parameter.Type.BOOL, True),
            Parameter("standalone_origin", Parameter.Type.DOUBLE_ARRAY, [-10.0, -10.0]),
            Parameter("standalone_size", Parameter.Type.DOUBLE_ARRAY, [20.0, 20.0]),
            Parameter("belief_topic", Parameter.Type.STRING, belief_topic),
            Parameter("publish_markers", Parameter.Type.BOOL, False),
            Parameter("bearing_frame", Parameter.Type.STRING, "map"),
            Parameter("base_frame", Parameter.Type.STRING, base_frame),
            Parameter("map_frame", Parameter.Type.STRING, "map"),
            Parameter("use_sim_time", Parameter.Type.BOOL, False),
            Parameter("publish_rate_hz", Parameter.Type.DOUBLE, 0.01),
        ],
    )
    node._tf_buffer.set_transform_static(_static_transform("map", base_frame, 0.0, 0.0), "test")

    consumer = rclpy.create_node(f"belief_wedge_consumer_{suffix}")
    received: list[OccupancyGrid] = []
    consumer.create_subscription(OccupancyGrid, belief_topic, received.append, 1)

    try:
        _spin_until(rclpy, [node, consumer], lambda: node._pub_belief.get_subscription_count() > 0)

        node._cb_event(_make_heard_sound_event(bearing_rad=0.0))
        node._on_timer()

        _spin_until(rclpy, [node, consumer], lambda: bool(received))

        grid_msg = received[-1]
        info = grid_msg.info
        grid = np.asarray(grid_msg.data, dtype=np.int8).reshape(info.height, info.width)
        origin_col = int(round((0.0 - info.origin.position.x) / info.resolution))
        assert grid[:, :origin_col].sum() == 0
        assert grid[:, origin_col:].sum() > 0
    finally:
        consumer.destroy_node()
        node.destroy_node()


def test_belief_node_drops_event_with_nan_bearing(rclpy_context):
    import rclpy
    from nav_msgs.msg import OccupancyGrid
    from rclpy.parameter import Parameter

    from arena_auditory.hearing.belief_node import BeliefNode

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    heard_topic = f"/test/{suffix}/heard_sound"
    belief_topic = f"/test/{suffix}/belief_grid"

    node = BeliefNode(
        parameter_overrides=[
            Parameter("heard_sound_topic", Parameter.Type.STRING, heard_topic),
            Parameter("standalone_grid", Parameter.Type.BOOL, True),
            Parameter("standalone_origin", Parameter.Type.DOUBLE_ARRAY, [-5.0, -5.0]),
            Parameter("standalone_size", Parameter.Type.DOUBLE_ARRAY, [10.0, 10.0]),
            Parameter("belief_topic", Parameter.Type.STRING, belief_topic),
            Parameter("publish_markers", Parameter.Type.BOOL, False),
            Parameter("use_sim_time", Parameter.Type.BOOL, False),
            Parameter("publish_rate_hz", Parameter.Type.DOUBLE, 0.01),
        ],
    )

    consumer = rclpy.create_node(f"belief_nan_consumer_{suffix}")
    received: list[OccupancyGrid] = []
    consumer.create_subscription(OccupancyGrid, belief_topic, received.append, 1)

    try:
        _spin_until(rclpy, [node, consumer], lambda: node._pub_belief.get_subscription_count() > 0)

        node._cb_event(_make_heard_sound_event(bearing_rad=float("nan")))
        assert node._dropped_no_bearing == 1

        node._on_timer()
        _spin_until(rclpy, [node, consumer], lambda: bool(received))

        grid = np.asarray(received[-1].data, dtype=np.int8)
        assert grid.sum() == 0
    finally:
        consumer.destroy_node()
        node.destroy_node()


def test_belief_node_wedge_markers_add_then_expire(rclpy_context):
    import rclpy
    from rclpy.parameter import Parameter
    from visualization_msgs.msg import Marker, MarkerArray

    from arena_auditory.hearing.belief_node import BeliefNode

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    heard_topic = f"/test/{suffix}/heard_sound"
    belief_topic = f"/test/{suffix}/belief_grid"
    marker_topic = f"/test/{suffix}/belief_wedges"
    base_frame = f"{suffix}/base_link"

    node = BeliefNode(
        parameter_overrides=[
            Parameter("heard_sound_topic", Parameter.Type.STRING, heard_topic),
            Parameter("standalone_grid", Parameter.Type.BOOL, True),
            Parameter("standalone_origin", Parameter.Type.DOUBLE_ARRAY, [-10.0, -10.0]),
            Parameter("standalone_size", Parameter.Type.DOUBLE_ARRAY, [20.0, 20.0]),
            Parameter("belief_topic", Parameter.Type.STRING, belief_topic),
            Parameter("marker_topic", Parameter.Type.STRING, marker_topic),
            Parameter("publish_markers", Parameter.Type.BOOL, True),
            Parameter("bearing_frame", Parameter.Type.STRING, "map"),
            Parameter("base_frame", Parameter.Type.STRING, base_frame),
            Parameter("map_frame", Parameter.Type.STRING, "map"),
            Parameter("use_sim_time", Parameter.Type.BOOL, False),
            Parameter("publish_rate_hz", Parameter.Type.DOUBLE, 0.01),
        ],
    )
    node._tf_buffer.set_transform_static(_static_transform("map", base_frame, 0.0, 0.0), "test")

    consumer = rclpy.create_node(f"belief_marker_consumer_{suffix}")
    received: list[MarkerArray] = []
    consumer.create_subscription(MarkerArray, marker_topic, received.append, 1)

    try:
        _spin_until(rclpy, [node, consumer], lambda: node._pub_markers.get_subscription_count() > 0)

        node._cb_event(_make_heard_sound_event(bearing_rad=0.0))
        node._cb_event(_make_heard_sound_event(bearing_rad=0.3))
        node._on_timer()
        _spin_until(rclpy, [node, consumer], lambda: len(received) == 1)

        added = [m for m in received[-1].markers if m.action == Marker.ADD]
        assert len(added) == 2
        added_ids = {m.id for m in added}
        assert len(added_ids) == 2

        # set directly: a set_parameters() value is not visible until the next tick
        node._params.tau_sec = 0.001
        time.sleep(0.05)
        node._on_timer()
        _spin_until(rclpy, [node, consumer], lambda: len(received) == 2)

        deleted = [m for m in received[-1].markers if m.action == Marker.DELETE]
        assert {m.id for m in deleted} == added_ids
    finally:
        consumer.destroy_node()
        node.destroy_node()


def test_policy_node_cruise_speed_mask_around_blob(rclpy_context):
    import rclpy
    from nav_msgs.msg import OccupancyGrid
    from rclpy.parameter import Parameter
    from std_msgs.msg import String

    from arena_auditory.hearing.belief_node import latched_qos
    from arena_auditory.hearing.policy_node import PolicyNode

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    mask_topic = f"/test/{suffix}/speed_filter_mask"
    state_topic = f"/test/{suffix}/policy_state"
    base_frame = f"{suffix}/base_link"

    node = PolicyNode(
        parameter_overrides=[
            Parameter("base_frame", Parameter.Type.STRING, base_frame),
            Parameter("max_linear_mps", Parameter.Type.DOUBLE, 1.0),
            Parameter("speed_mask_topic", Parameter.Type.STRING, mask_topic),
            Parameter("state_topic", Parameter.Type.STRING, state_topic),
            Parameter("listen_enabled", Parameter.Type.BOOL, False),
            Parameter("yield_enabled", Parameter.Type.BOOL, False),
            Parameter("use_sim_time", Parameter.Type.BOOL, False),
            Parameter("publish_rate_hz", Parameter.Type.DOUBLE, 0.01),
        ],
    )
    node._tf_buffer.set_transform_static(_static_transform("map", base_frame, 0.0, 0.0), "test")

    size, resolution = 100, 0.1
    origin = -5.0
    grid_map = OccupancyGrid()
    grid_map.header.frame_id = "map"
    grid_map.info.resolution = resolution
    grid_map.info.width = size
    grid_map.info.height = size
    grid_map.info.origin.position.x = origin
    grid_map.info.origin.position.y = origin
    grid_map.info.origin.orientation.w = 1.0
    grid_map.data = [0] * (size * size)

    blob_xy = (2.0, 0.0)
    xs = origin + (np.arange(size) + 0.5) * resolution
    ys = origin + (np.arange(size) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys)
    belief_data = np.where(np.hypot(xx - blob_xy[0], yy - blob_xy[1]) <= 0.3, 100, 0).astype(np.int8)
    belief = OccupancyGrid()
    belief.header.frame_id = "map"
    belief.info.resolution = resolution
    belief.info.width = size
    belief.info.height = size
    belief.info.origin.position.x = origin
    belief.info.origin.position.y = origin
    belief.info.origin.orientation.w = 1.0
    belief.data = belief_data.reshape(-1).tolist()

    consumer = rclpy.create_node(f"policy_cruise_consumer_{suffix}")
    masks: list[OccupancyGrid] = []
    states: list[String] = []
    consumer.create_subscription(OccupancyGrid, mask_topic, masks.append, latched_qos())
    consumer.create_subscription(String, state_topic, states.append, latched_qos())

    try:
        _spin_until(
            rclpy,
            [node, consumer],
            lambda: node._pub_mask.get_subscription_count() > 0 and node._pub_state.get_subscription_count() > 0,
        )

        node._cb_map(grid_map)
        node._cb_belief(belief)
        node._on_timer()
        _spin_until(rclpy, [node, consumer], lambda: bool(masks) and bool(states))

        mask = np.asarray(masks[-1].data, dtype=np.int8).reshape(size, size)

        def _cell(x: float, y: float) -> tuple[int, int]:
            return int((y - origin) / resolution), int((x - origin) / resolution)

        blob_row, blob_col = _cell(*blob_xy)
        far_row, far_col = _cell(-4.0, -4.0)
        assert mask[blob_row, blob_col] == 40
        assert mask[far_row, far_col] == 0

        payload = json.loads(states[-1].data)
        assert payload["state"] == "cruise"
    finally:
        consumer.destroy_node()
        node.destroy_node()


def test_policy_node_listen_lane_on_approach_to_blind_bend(rclpy_context):
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import OccupancyGrid, Path
    from rclpy.parameter import Parameter
    from std_msgs.msg import String

    from arena_auditory.hearing.belief_node import latched_qos
    from arena_auditory.hearing.policy_node import PolicyNode

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    mask_topic = f"/test/{suffix}/speed_filter_mask"
    state_topic = f"/test/{suffix}/policy_state"
    base_frame = f"{suffix}/base_link"

    node = PolicyNode(
        parameter_overrides=[
            Parameter("base_frame", Parameter.Type.STRING, base_frame),
            Parameter("max_linear_mps", Parameter.Type.DOUBLE, 1.0),
            Parameter("speed_mask_topic", Parameter.Type.STRING, mask_topic),
            Parameter("state_topic", Parameter.Type.STRING, state_topic),
            Parameter("listen_enabled", Parameter.Type.BOOL, True),
            Parameter("use_sim_time", Parameter.Type.BOOL, False),
            Parameter("publish_rate_hz", Parameter.Type.DOUBLE, 0.01),
        ],
    )
    robot_xy = (7.0, 1.0)
    node._tf_buffer.set_transform_static(_static_transform("map", base_frame, *robot_xy), "test")

    size, resolution = 200, 0.1
    occupied = _l_shaped_occupancy(size, resolution)
    grid_map = OccupancyGrid()
    grid_map.header.frame_id = "map"
    grid_map.info.resolution = resolution
    grid_map.info.width = size
    grid_map.info.height = size
    grid_map.info.origin.orientation.w = 1.0
    grid_map.data = np.where(occupied, 100, 0).astype(np.int8).reshape(-1).tolist()

    belief = OccupancyGrid()
    belief.header.frame_id = "map"
    belief.info.resolution = resolution
    belief.info.width = size
    belief.info.height = size
    belief.info.origin.orientation.w = 1.0
    belief.data = [0] * (size * size)

    plan_xy = _l_shaped_plan()
    path = Path()
    path.header.frame_id = "map"
    for x, y in plan_xy:
        pose = PoseStamped()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)

    consumer = rclpy.create_node(f"policy_listen_consumer_{suffix}")
    masks: list[OccupancyGrid] = []
    states: list[String] = []
    consumer.create_subscription(OccupancyGrid, mask_topic, masks.append, latched_qos())
    consumer.create_subscription(String, state_topic, states.append, latched_qos())

    try:
        _spin_until(
            rclpy,
            [node, consumer],
            lambda: node._pub_mask.get_subscription_count() > 0 and node._pub_state.get_subscription_count() > 0,
        )

        node._cb_map(grid_map)
        node._cb_belief(belief)
        node._cb_plan(path)
        node._on_timer()
        _spin_until(rclpy, [node, consumer], lambda: bool(masks) and bool(states))

        payload = json.loads(states[-1].data)
        assert payload["state"] == "listen"
        assert payload["dist_to_bend_m"] is not None
        assert math.isfinite(payload["dist_to_bend_m"])

        mask = np.asarray(masks[-1].data, dtype=np.int8).reshape(size, size)
        row, col = int(robot_xy[1] / resolution), int(robot_xy[0] / resolution)
        assert mask[row, col] == 20
    finally:
        consumer.destroy_node()
        node.destroy_node()
