from __future__ import annotations

import math
import time
import uuid
import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_people_msgs.msg")
    pytest.importorskip("geometry_msgs.msg")
    pytest.importorskip("task_generator_msgs.msg")
    pytest.importorskip("nav_msgs.msg")
    pytest.importorskip("visualization_msgs.msg")


def _spin_until(rclpy, nodes, predicate, timeout_sec: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_sec

    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)

        if predicate():
            return

    raise AssertionError("timed out waiting for ROS round-trip")


def _make_pedestrian(ped_id: int, x: float, y: float):
    from arena_people_msgs.msg import Pedestrian

    ped = Pedestrian()
    ped.id = ped_id
    ped.name = f"ped_{ped_id}"
    ped.pose.position.x = x
    ped.pose.position.y = y
    ped.pose.position.z = 0.0
    ped.pose.orientation.w = 1.0
    return ped


def _make_sound_event():
    from task_generator_msgs.msg import SoundEvent

    event = SoundEvent()
    event.header.frame_id = "map"
    event.event_id = "roundtrip_001"
    event.source_agent_id = 1
    event.source_agent_name = "ped_1"
    event.sound_type = "greeting"
    event.label = "greeting"
    event.asset_id = "greeting"
    event.source_position.x = 0.0
    event.source_position.y = 0.0
    event.source_position.z = 0.0
    event.source_yaw = 0.0
    event.source_volume_db = 60.0
    event.duration.sec = 1
    event.loop = False
    return event


def _make_heard_sound_event():
    from task_generator_msgs.msg import HeardSoundEvent

    event = HeardSoundEvent()
    event.header.frame_id = "map"
    event.event_id = "heard_roundtrip_001"
    event.listener_id = "robot:robot1"
    event.source_agent_id = 1
    event.source_agent_name = "ped_1"
    event.sound_type = "greeting"
    event.label = "greeting"
    event.asset_id = "greeting"
    event.source_position.x = 0.0
    event.source_position.y = 0.0
    event.listener_position.x = 1.0
    event.listener_position.y = 0.0
    event.distance = 1.0
    event.source_volume_db = 60.0
    event.received_volume_db = 45.0
    event.hearing_threshold_db = 20.0
    event.direct_delay_sec = 0.0
    event.audible = True
    event.occluded = False
    return event

def _make_robot_fleet(robot_name: str, namespace: str):
    from task_generator_msgs.msg import RobotDescriptor, RobotFleet, RobotState

    robot = RobotDescriptor()
    robot.name = robot_name
    robot.model = "jackal"
    robot.ns = namespace
    robot.frame = robot_name

    fleet = RobotFleet()
    state = RobotState()
    state.descriptor = robot
    fleet.robots.append(state)
    return fleet

def test_sound_event_round_trips_to_heard_sound_event(rclpy_context):
    import rclpy
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy
    from arena_people_msgs.msg import Pedestrians
    from geometry_msgs.msg import Point
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid
    from task_generator.auditory.acoustic_frame import runtime_acoustic_offset
    from task_generator.auditory.qos_profiles import transient_event_qos
    from task_generator.auditory.sound_propagation_node import SoundPropagationNode
    from task_generator_msgs.msg import HeardSoundEvent, SoundEvent

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    sound_topic = f"/test/{suffix}/human_sound_events"
    heard_topic = f"/test/{suffix}/heard_sound_events"
    peds_topic = f"/test/{suffix}/arena_peds"
    map_topic = f"/test/{suffix}/map"
    robot_fleet_topic = f"/test/{suffix}/state/robots"
    world_topic = f"/test/{suffix}/state/world"

    
    propagation = SoundPropagationNode(
        parameter_overrides=[
            Parameter("sound_events_topic", Parameter.Type.STRING, sound_topic),
            Parameter("heard_sound_events_topic", Parameter.Type.STRING, heard_topic),
            Parameter("arena_peds_topic", Parameter.Type.STRING, peds_topic),
            Parameter("map_topic", Parameter.Type.STRING, map_topic),
            Parameter("robot_fleet_topic", Parameter.Type.STRING, robot_fleet_topic),
            Parameter("world_topic", Parameter.Type.STRING, world_topic),
            Parameter("ped_hearing", Parameter.Type.BOOL, True),
        ],
    )
    assert (
        propagation._world_subscription.qos_profile.durability
        == DurabilityPolicy.TRANSIENT_LOCAL
    )

    emitter = rclpy.create_node(f"sound_event_emitter_{suffix}")
    consumer = rclpy.create_node(f"heard_sound_consumer_{suffix}")

    received: list[HeardSoundEvent] = []

    publisher = emitter.create_publisher(
        SoundEvent,
        sound_topic,
        transient_event_qos(),
    )
    consumer.create_subscription(
        HeardSoundEvent,
        heard_topic,
        received.append,
        transient_event_qos(),
    )

    propagation._map = OccupancyGrid()
    propagation._map.header.frame_id = "map"
    propagation._map.info.resolution = 1.0
    propagation._map.info.width = 20
    propagation._map.info.height = 20
    propagation._map.info.origin.orientation.w = 1.0
    pedestrians = Pedestrians()
    pedestrians.pedestrians.append(_make_pedestrian(2, 3.0, 4.0))
    propagation._cb_peds(pedestrians)

    try:
        _spin_until(
            rclpy,
            [emitter, propagation, consumer],
            lambda: propagation.count_subscribers(heard_topic) > 0
            and emitter.count_subscribers(sound_topic) > 0,
        )

        publisher.publish(_make_sound_event())

        _spin_until(
            rclpy,
            [emitter, propagation, consumer],
            lambda: len(received) == 1,
        )

        heard = received[0]
        assert heard.event_id == "roundtrip_001"
        assert heard.listener_id == "agent:2"
        assert heard.sound_type == "greeting"
        assert heard.distance == pytest.approx(5.0)
        assert heard.occluded is False
        assert heard.audible is True
        assert heard.propagation_backend == "legacy_distance_occlusion"
        assert heard.used_backend_fallback is False
        assert heard.backend_fallback_reason == ""
        assert heard.portal_ids == []
        assert heard.traversed_zones == []
        assert heard.portal_hop_count == 0
        assert heard.received_volume_db == pytest.approx(
            60.0 - 20.0 * math.log10(5.0),
            abs=1e-3,
        )

        # Runtime maps are shifted into the environment frame. Auditory
        # positions remain in authored world coordinates.
        shifted_map = OccupancyGrid()
        shifted_map.info.width = 510
        shifted_map.info.height = 695
        shifted_map.info.resolution = 0.05
        shifted_map.info.origin.position.x = 4.75
        shifted_map.info.origin.position.y = 4.70
        shifted_map.info.origin.orientation.w = 1.0
        propagation._map = shifted_map
        propagation._authored_map_origin = (-0.25, -0.25)
        assert runtime_acoustic_offset(
            propagation._map, propagation._authored_map_origin
        ) == pytest.approx((5.0, 4.95))
        assert propagation._world_to_grid(Point(x=5.0, y=4.95)) == (5, 5)
    finally:
        emitter.destroy_node()
        consumer.destroy_node()
        propagation.destroy_node()


def test_robot_only_policy_excludes_pedestrian_listeners(rclpy_context):
    from geometry_msgs.msg import Point
    from nav_msgs.msg import OccupancyGrid
    from rclpy.parameter import Parameter
    from task_generator.auditory.sound_propagation_node import (
        SoundPropagationNode,
    )

    propagation = SoundPropagationNode(
        parameter_overrides=[
            Parameter("ped_hearing", Parameter.Type.BOOL, False),
        ]
    )
    propagation._peds = {
        1: _make_pedestrian(1, 1.0, 1.0),
        2: _make_pedestrian(2, 2.0, 2.0),
    }
    propagation._robots = {
        "robot:jackal": (Point(x=3.0, y=3.0, z=0.0), "map"),
    }
    propagation._map = OccupancyGrid()
    propagation._map.header.frame_id = "map"
    propagation._map.info.resolution = 1.0
    propagation._map.info.width = 20
    propagation._map.info.height = 20
    propagation._map.info.origin.orientation.w = 1.0
    event = _make_sound_event()
    event.source_agent_id = 1

    try:
        assert set(propagation._listeners_for_event(event)) == {
            "robot:jackal"
        }

        propagation.set_parameters([
            Parameter("ped_hearing", Parameter.Type.BOOL, True)
        ])
        assert set(propagation._listeners_for_event(event)) == {
            "agent:2",
            "robot:jackal",
        }
    finally:
        propagation.destroy_node()


def test_spawn_microphone_assigns_zone_and_next_index(rclpy_context):
    from nav_msgs.msg import OccupancyGrid
    from shapely.geometry import Polygon
    from task_generator.auditory.acoustic_scene import (
        AcousticScene,
        AcousticZone,
    )
    from task_generator.auditory.microphone_config import WorldMicrophoneSpec
    from task_generator.auditory.sound_propagation_node import (
        SoundPropagationNode,
    )
    from task_generator_msgs.msg import EpisodeRecord
    from task_generator_msgs.srv import RemoveMicrophone, SpawnMicrophone

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    propagation = SoundPropagationNode(namespace=f"/test/{suffix}")
    propagation._map = OccupancyGrid()
    propagation._map.header.frame_id = "map"
    propagation._map.info.resolution = 1.0
    propagation._map.info.width = 10
    propagation._map.info.height = 10
    propagation._map.info.origin.orientation.w = 1.0
    propagation._scene = AcousticScene(
        zones=(
            AcousticZone(
                name="reception",
                polygon=Polygon(
                    [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
                ),
                floor_material_id="default",
            ),
        ),
        walls=(),
    )
    authored_id = "microphone:zone:reception:placed:1"
    propagation._world_microphones[authored_id] = WorldMicrophoneSpec(
        listener_id=authored_id,
        zone="reception",
        placement="placed",
        frame="map",
        position=(1.0, 1.0, 1.5),
        ceiling_height_m=None,
    )

    request = SpawnMicrophone.Request()
    request.position.header.frame_id = "map"
    request.position.point.x = 2.0
    request.position.point.y = 3.0
    request.position.point.z = 1.5
    request.placement = "placed"

    try:
        response = propagation._spawn_microphone(
            request,
            SpawnMicrophone.Response(),
        )
        assert response.success is True
        assert response.zone == "reception"
        assert response.listener_id == "microphone1"
        position, frame = propagation._spawned_microphones[
            response.listener_id
        ]
        assert (position.x, position.y, position.z) == (2.0, 3.0, 1.5)
        assert frame == "map"

        second = propagation._spawn_microphone(
            request,
            SpawnMicrophone.Response(),
        )
        assert second.success is True
        assert second.listener_id == "microphone2"

        from geometry_msgs.msg import TransformStamped

        transform = TransformStamped()
        transform.header.frame_id = "map"
        transform.child_frame_id = "robot/base_link"
        transform.transform.translation.x = 1.0
        transform.transform.translation.y = 1.0
        transform.transform.rotation.w = 1.0
        propagation._tf_buffer.set_transform_static(transform, "test")
        attached_request = SpawnMicrophone.Request()
        attached_request.position.header.frame_id = "map"
        attached_request.position.point.x = 2.0
        attached_request.position.point.y = 3.0
        attached_request.position.point.z = 1.5
        attached_request.placement = "body"
        attached_request.attached_frame = "robot/base_link"
        attached = propagation._spawn_microphone(
            attached_request,
            SpawnMicrophone.Response(),
        )
        assert attached.success is True
        assert attached.listener_id == "microphone3"
        local_position, attached_frame = propagation._spawned_microphones[
            attached.listener_id
        ]
        assert attached_frame == "robot/base_link"
        assert (local_position.x, local_position.y) == (1.0, 2.0)
        resolved = propagation._all_microphone_positions()[attached.listener_id]
        assert (resolved.x, resolved.y, resolved.z) == (2.0, 3.0, 1.5)

        request.position.point.x = 8.0
        outside_zone = propagation._spawn_microphone(
            request,
            SpawnMicrophone.Response(),
        )
        assert outside_zone.success is True
        assert outside_zone.zone == ""
        assert outside_zone.listener_id == "microphone4"

        request.position.point.x = 12.0
        outside_map = propagation._spawn_microphone(
            request,
            SpawnMicrophone.Response(),
        )
        assert outside_map.success is True
        assert outside_map.listener_id == "microphone5"
        assert len(propagation._spawned_microphones) == 5

        propagation._scene = None
        request.position.point.x = 6.0
        no_zones = propagation._spawn_microphone(
            request,
            SpawnMicrophone.Response(),
        )
        assert no_zones.success is True
        assert no_zones.zone == ""
        assert no_zones.listener_id == "microphone6"

        removed = propagation._remove_microphone(
            RemoveMicrophone.Request(listener_id=no_zones.listener_id),
            RemoveMicrophone.Response(),
        )
        assert removed.success is True
        assert no_zones.listener_id not in propagation._spawned_microphones

        authored_removal = propagation._remove_microphone(
            RemoveMicrophone.Request(listener_id=authored_id),
            RemoveMicrophone.Response(),
        )
        assert authored_removal.success is False
        assert authored_removal.error_msg == (
            "world-authored microphones cannot be removed"
        )

        propagation._map = None
        request.position.header.frame_id = "rviz_map"
        request.position.point.x = 4.0
        without_map = propagation._spawn_microphone(
            request,
            SpawnMicrophone.Response(),
        )
        assert without_map.success is True
        assert without_map.listener_id == "microphone7"
        marker_position, marker_frame = propagation._microphone_marker_poses()[
            without_map.listener_id
        ]
        assert marker_frame == "rviz_map"
        assert marker_position.x == 4.0

        episode = EpisodeRecord()
        episode.episode_id = 7
        propagation._cb_episode_world(episode)
        assert propagation._spawned_microphones == {}
        assert propagation._spawned_microphone_index == 0
    finally:
        propagation.destroy_node()


def test_propagation_runtime_toggle_stops_continuous_outputs(rclpy_context):
    from geometry_msgs.msg import Point
    from nav_msgs.msg import OccupancyGrid
    from rclpy.parameter import Parameter
    from task_generator.auditory.sound_propagation_node import (
        SoundPropagationNode,
    )
    from task_generator_msgs.msg import ContinuousHeardSoundState

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    propagation = SoundPropagationNode(namespace=f"/test/{suffix}")
    previous = ContinuousHeardSoundState()
    previous.source_id = "environment:runtime_music_1:radio"
    previous.listener_id = "microphone1"
    previous.active = True
    previous.audible = True
    key = (previous.source_id, previous.listener_id)
    propagation._last_continuous_outputs[key] = previous
    excluded = ContinuousHeardSoundState()
    excluded.source_id = previous.source_id
    excluded.listener_id = "microphone2"
    excluded.active = True
    excluded.audible = True
    excluded_key = (excluded.source_id, excluded.listener_id)
    propagation._last_continuous_outputs[excluded_key] = excluded
    robot_listener = ContinuousHeardSoundState()
    robot_listener.source_id = previous.source_id
    robot_listener.listener_id = "robot:robot1"
    robot_listener.active = True
    robot_listener.audible = True
    robot_key = (robot_listener.source_id, robot_listener.listener_id)
    propagation._last_continuous_outputs[robot_key] = robot_listener
    propagation._robot_microphones = {
        "microphone1": (Point(), "map"),
        "microphone2": (Point(), "map"),
    }
    propagation._map = OccupancyGrid()
    propagation._map.header.frame_id = "map"
    propagation._map.info.resolution = 1.0
    propagation._map.info.width = 20
    propagation._map.info.height = 20
    propagation._map.info.origin.orientation.w = 1.0

    try:
        selection_results = propagation.set_parameters([
            Parameter(
                "active_microphone_id",
                Parameter.Type.STRING,
                "microphone1",
            ),
        ])
        assert selection_results[0].successful is True
        assert set(propagation._microphone_positions()) == {"microphone1"}
        assert propagation._last_continuous_outputs[key].active is True
        assert propagation._last_continuous_outputs[excluded_key].active is False
        assert propagation._last_continuous_outputs[robot_key].active is True

        results = propagation.set_parameters([
            Parameter("enable_propagation", Parameter.Type.BOOL, False),
        ])
        assert results[0].successful is True
        assert propagation.get_parameter("enable_propagation").value is False
        stopped = propagation._last_continuous_outputs[key]
        assert stopped.active is False
        assert stopped.audible is False
        assert propagation._last_continuous_outputs[robot_key].active is False
    finally:
        propagation.destroy_node()


def test_viewport_camera_registers_selectable_microphones(rclpy_context):
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import OccupancyGrid
    from rclpy.parameter import Parameter
    from task_generator.auditory.sound_propagation_node import (
        SoundPropagationNode,
    )

    propagation = SoundPropagationNode(
        parameter_overrides=[
            Parameter(
                "active_microphone_id",
                Parameter.Type.STRING,
                "microphone:viewport:projective_center",
            ),
            Parameter(
                "viewport_down_projection_height_m",
                Parameter.Type.DOUBLE,
                1.7,
            ),
        ],
    )
    propagation._map = OccupancyGrid()
    propagation._map.header.frame_id = "map"
    propagation._map.info.resolution = 1.0
    propagation._map.info.width = 20
    propagation._map.info.height = 20
    propagation._map.info.origin.orientation.w = 1.0
    camera_pose = PoseStamped()
    camera_pose.header.frame_id = "map"
    camera_pose.pose.position.x = 3.0
    camera_pose.pose.position.y = 4.0
    camera_pose.pose.position.z = 8.0

    try:
        propagation._cb_viewport_camera_pose(camera_pose)

        assert set(propagation._viewport_microphones) == {
            "microphone:viewport:down_projection",
            "microphone:viewport:projective_center",
        }
        projective_center = propagation._microphone_positions()[
            "microphone:viewport:projective_center"
        ]
        assert (
            projective_center.x,
            projective_center.y,
            projective_center.z,
        ) == (3.0, 4.0, 8.0)
        assert propagation._listener_height(
            "microphone:viewport:projective_center",
            projective_center,
        ) == 8.0
        down_projection = propagation._all_microphone_positions()[
            "microphone:viewport:down_projection"
        ]
        assert (
            down_projection.x,
            down_projection.y,
            down_projection.z,
        ) == (3.0, 4.0, 1.7)
    finally:
        propagation.destroy_node()


def test_propagation_reconciles_robot_odom_subscriptions(rclpy_context):
    from geometry_msgs.msg import Point
    from rclpy.parameter import Parameter
    from task_generator.auditory.sound_propagation_node import (
        SoundPropagationNode,
    )
    from task_generator_msgs.msg import RobotFleet

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    propagation = SoundPropagationNode(
        namespace=f"/test/{suffix}",
        parameter_overrides=[
            Parameter(
                "odom_topic_template",
                Parameter.Type.STRING,
                "{namespace}/odom",
            ),
        ],
    )
    fleet = _make_robot_fleet("robot1", f"/test/{suffix}/robot1")

    try:
        propagation._cb_robot_fleet(fleet)
        first = dict(propagation._odom_subs)
        assert ("robot1", f"/test/{suffix}/robot1/odom") in first
        assert "robot1_mic" in propagation._robot_microphones
        robot_mic_position, robot_mic_frame = (
            propagation._robot_microphones["robot1_mic"]
        )
        assert robot_mic_position.z == 0.35
        assert robot_mic_frame == "robot1/base_link"

        propagation._cb_robot_fleet(fleet)
        assert propagation._odom_subs.keys() == first.keys()
        assert all(
            propagation._odom_subs[key] is subscription
            for key, subscription in first.items()
        )

        propagation._robots["robot:robot1"] = (Point(), "map")
        propagation._cb_robot_fleet(RobotFleet())
        assert propagation._odom_subs == {}
        assert "robot:robot1" not in propagation._robots
        assert "robot1_mic" not in propagation._robot_microphones
    finally:
        propagation.destroy_node()


def test_auditory_round_trip_greeting_reaches_robot_marker(rclpy_context):
    import rclpy
    from rclpy.parameter import Parameter
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid, Odometry
    from task_generator.auditory.qos_profiles import acoustic_metadata_qos, transient_event_qos
    from task_generator.auditory.robot_hearing_node import RobotHearingNode
    from task_generator.auditory.sound_propagation_node import SoundPropagationNode
    from task_generator_msgs.msg import HeardSoundEvent, RobotFleet, SoundEvent
    from visualization_msgs.msg import Marker

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    ns = f"/test/{suffix}"

    sound_topic = f"{ns}/human_sound_events"
    heard_topic = f"{ns}/heard_sound_events"
    peds_topic = f"{ns}/arena_peds"
    map_topic = f"{ns}/map"
    robot_fleet_topic = f"{ns}/state/robots"
    world_topic = f"{ns}/state/world"
    robot_ns = f"{ns}/robot1"
    robot_odom_topic = f"{robot_ns}/odom"
    robot_heard_topic = f"{ns}/robot1/heard_sound"
    robot_marker_topic = f"{ns}/robot1/heard_sound_marker"

    propagation = SoundPropagationNode(
        parameter_overrides=[
            Parameter("sound_events_topic", Parameter.Type.STRING, sound_topic),
            Parameter("heard_sound_events_topic", Parameter.Type.STRING, heard_topic),
            Parameter("arena_peds_topic", Parameter.Type.STRING, peds_topic),
            Parameter("map_topic", Parameter.Type.STRING, map_topic),
            Parameter("robot_fleet_topic", Parameter.Type.STRING, robot_fleet_topic),
            Parameter("world_topic", Parameter.Type.STRING, world_topic),
            Parameter("odom_topic_template", Parameter.Type.STRING, "{namespace}/odom"),
            Parameter("default_hearing_threshold_db", Parameter.Type.DOUBLE, 10.0),
            Parameter("publish_inaudible", Parameter.Type.BOOL, True),
        ],
    )
    propagation._map = OccupancyGrid()
    propagation._map.header.frame_id = "map"
    propagation._map.info.resolution = 1.0
    propagation._map.info.width = 20
    propagation._map.info.height = 20
    propagation._map.info.origin.orientation.w = 1.0

    hearing = RobotHearingNode(
        namespace=ns,
        parameter_overrides=[
            Parameter("heard_sound_events_topic", Parameter.Type.STRING, heard_topic),
            Parameter("robot_fleet_topic", Parameter.Type.STRING, robot_fleet_topic),
            Parameter("heard_sound_topic_suffix", Parameter.Type.STRING, "heard_sound"),
            Parameter("marker_topic_suffix", Parameter.Type.STRING, "heard_sound_marker"),
            Parameter("honor_propagation_delay", Parameter.Type.BOOL, False),
            Parameter("min_snr_db", Parameter.Type.DOUBLE, -15.0),
        ],
    )

    emitter = rclpy.create_node(f"auditory_roundtrip_emitter_{suffix}")
    consumer = rclpy.create_node(f"auditory_roundtrip_consumer_{suffix}")

    heard_by_robot: list[HeardSoundEvent] = []
    markers: list[Marker] = []

    fleet_pub = emitter.create_publisher(RobotFleet, robot_fleet_topic, acoustic_metadata_qos())
    odom_pub = emitter.create_publisher(Odometry, robot_odom_topic, 10)
    sound_pub = emitter.create_publisher(SoundEvent, sound_topic, transient_event_qos())

    consumer.create_subscription(
        HeardSoundEvent,
        robot_heard_topic,
        heard_by_robot.append,
        transient_event_qos(),
    )
    consumer.create_subscription(Marker, robot_marker_topic, markers.append, 10)

    odom = Odometry()
    odom.header.frame_id = "map"
    odom.pose.pose.position.x = 1.0
    odom.pose.pose.position.y = 0.0
    odom.pose.pose.orientation.w = 1.0

    try:
        _spin_until(
            rclpy,
            [emitter, propagation, hearing, consumer],
            lambda: fleet_pub.get_subscription_count() >= 2
            and sound_pub.get_subscription_count() >= 1,
            timeout_sec=5.0,
        )

        fleet_pub.publish(_make_robot_fleet("robot1", robot_ns))

        _spin_until(
            rclpy,
            [emitter, propagation, hearing, consumer],
            lambda: "robot1" in hearing._robot_names
            and odom_pub.get_subscription_count() >= 1,
            timeout_sec=5.0,
        )
        assert hearing._robot_frames["robot1"] == "robot1/base_link"

        inaudible = _make_heard_sound_event()
        inaudible.listener_id = "robot:robot1"
        inaudible.audible = False
        hearing._cb_heard_sound(inaudible)
        assert hearing._pending == []

        odom_pub.publish(odom)

        _spin_until(
            rclpy,
            [emitter, propagation, hearing, consumer],
            lambda: "robot:robot1" in propagation._robots,
            timeout_sec=5.0,
        )
        # Listener poses resolve through TF, not odom coordinates.
        transform = TransformStamped()
        transform.header.frame_id = "map"
        transform.child_frame_id = "robot1/base_link"
        transform.transform.translation.x = 1.0
        transform.transform.rotation.w = 1.0
        propagation._tf_buffer.set_transform_static(transform, "test")

        event = _make_sound_event()
        event.event_id = "full_auditory_roundtrip_001"
        event.sound_type = "greeting"
        event.label = "greeting"
        event.asset_id = "greeting"
        sound_pub.publish(event)

        _spin_until(
            rclpy,
            [emitter, propagation, hearing, consumer],
            lambda: len(heard_by_robot) == 1 and len(markers) == 1,
            timeout_sec=5.0,
        )

        heard = heard_by_robot[0]
        assert heard.event_id == "full_auditory_roundtrip_001"
        assert heard.listener_id == "robot:robot1"
        assert heard.sound_type == "greeting"
        assert heard.audible is True

        marker = markers[0]
        assert marker.ns == "robot1_heard_sound"
        assert marker.header.frame_id == "robot1/base_link"
        assert "GREETING" in marker.text
        assert "dB" in marker.text
        assert marker.type == Marker.TEXT_VIEW_FACING
        assert marker.action == Marker.ADD
        assert marker.pose.position.z == pytest.approx(1.2)
        assert marker.scale.z == pytest.approx(0.35)
        assert marker.lifetime.sec == 1
        assert marker.lifetime.nanosec == 500_000_000
    finally:
        emitter.destroy_node()
        consumer.destroy_node()
        hearing.destroy_node()
        propagation.destroy_node()


def test_continuous_source_on_moving_frame_relocalizes_per_update(rclpy_context):
    from geometry_msgs.msg import Point, TransformStamped
    from nav_msgs.msg import OccupancyGrid
    from task_generator.auditory.sound_propagation_node import SoundPropagationNode
    from task_generator_msgs.msg import ContinuousAudioSourceState

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    propagation = SoundPropagationNode(namespace=f"/test/{suffix}")
    propagation._map = OccupancyGrid()
    propagation._map.header.frame_id = "map"
    propagation._map.info.resolution = 1.0
    propagation._map.info.width = 40
    propagation._map.info.height = 40
    propagation._map.info.origin.orientation.w = 1.0
    propagation._robots = {"robot:listener": (Point(), "map")}

    def _place_rider(x: float) -> None:
        transform = TransformStamped()
        transform.header.frame_id = "map"
        transform.child_frame_id = "rider/base_link"
        transform.transform.translation.x = x
        transform.transform.rotation.w = 1.0
        propagation._tf_buffer.set_transform_static(transform, "test")

    state = ContinuousAudioSourceState()
    state.header.frame_id = "rider/base_link"
    state.source_id = "environment:horn"
    state.sound_type = "alarm"
    state.source_backend = "wav_loop"
    state.source_position = Point(x=1.0, y=0.0, z=0.5)
    state.source_volume_db = 80.0
    state.active = True
    key = (state.source_id, "robot:listener")

    try:
        _place_rider(10.0)
        propagation._cb_continuous_source(state)
        assert propagation._last_continuous_outputs[key].source_position.x == pytest.approx(11.0)

        _place_rider(20.0)
        propagation._cb_continuous_source(state)
        assert propagation._last_continuous_outputs[key].source_position.x == pytest.approx(21.0)
    finally:
        propagation.destroy_node()


def test_motor_sound_publishes_cone_and_clears_it(rclpy_context):
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.parameter import Parameter
    from task_generator.auditory.qos_profiles import transient_event_qos
    from task_generator.auditory.robot_sound_node import RobotSoundNode
    from visualization_msgs.msg import Marker, MarkerArray

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    namespace = f"/test/{suffix}"
    marker_topic = f"{namespace}/robot1/motor_sound_markers"
    motor = RobotSoundNode(
        namespace=namespace,
        parameter_overrides=[
            Parameter(
                "sound_events_topic",
                Parameter.Type.STRING,
                f"/test/{suffix}/human_sound_events",
            ),
            Parameter(
                "robot_fleet_topic",
                Parameter.Type.STRING,
                f"/test/{suffix}/state/robots",
            ),
            Parameter("only_when_moving", Parameter.Type.BOOL, True),
            Parameter("publish_period_sec", Parameter.Type.DOUBLE, 10.0),
            Parameter("audio_device", Parameter.Type.STRING, "none"),
        ]
    )
    consumer = rclpy.create_node(f"motor_marker_consumer_{suffix}")
    received: list[MarkerArray] = []
    consumer.create_subscription(
        MarkerArray,
        marker_topic,
        received.append,
        transient_event_qos(depth=10),
    )
    motor._cb_robot_fleet(
        _make_robot_fleet("robot1", f"{namespace}/robot1")
    )
    odom = Odometry()
    odom.pose.pose.position.x = 2.0
    odom.pose.pose.position.y = 3.0
    odom.pose.pose.orientation.w = 1.0
    odom.twist.twist.linear.x = 0.2
    motor._cb_odom("robot1", odom)

    try:
        assert motor._marker_pub is None
        assert motor._marker_pubs["robot1"].topic_name == marker_topic
        assert motor._robot_base_frame("jackal", "robot1") == "robot1/base_link"
        assert f"{namespace}/robot1/odom" in motor._robot_odom_topics(
            model_name="jackal",
            namespace=f"{namespace}/robot1",
            robot_name="robot1",
        )
        _spin_until(
            rclpy,
            [motor, consumer],
            lambda: (
                motor._marker_pubs["robot1"].get_subscription_count() > 0
            ),
        )
        motor._publish_robot_sounds()
        _spin_until(
            rclpy,
            [motor, consumer],
            lambda: any(
                marker.action == Marker.ADD
                for message in received
                for marker in message.markers
            ),
        )
        added = [
            marker
            for message in received
            for marker in message.markers
            if marker.action == Marker.ADD
        ]
        assert len(added) == 2
        assert {marker.type for marker in added} == {
            Marker.TRIANGLE_LIST,
            Marker.LINE_STRIP,
        }
        assert all(marker.header.frame_id == "robot1/base_link" for marker in added)
        assert all(len(marker.points) > 12 for marker in added)
        assert max(
            abs(point.x)
            for marker in added
            for point in marker.points
        ) <= 1.25
        assert all(
            (marker.color.r, marker.color.g, marker.color.b)
            == pytest.approx((1.0, 0.55, 0.05))
            for marker in added
        )

        received.clear()
        motor._last_speed["robot1"] = 0.0
        motor._publish_robot_sounds()
        _spin_until(
            rclpy,
            [motor, consumer],
            lambda: any(
                marker.action == Marker.DELETE
                for message in received
                for marker in message.markers
            ),
        )
        deleted = [
            marker
            for message in received
            for marker in message.markers
            if marker.action == Marker.DELETE
        ]
        assert len(deleted) == 2

        # In-place rotation also drives the motors even when linear velocity
        # is effectively zero.
        received.clear()
        odom.twist.twist.linear.x = 0.0
        odom.twist.twist.angular.z = 1.0
        motor._cb_odom("robot1", odom)
        assert motor._last_speed["robot1"] == pytest.approx(0.25)
        motor._publish_robot_sounds()
        _spin_until(
            rclpy,
            [motor, consumer],
            lambda: any(
                marker.action == Marker.ADD
                for message in received
                for marker in message.markers
            ),
        )

        # Disabling the source at runtime clears its marker and leaves the
        # robot available as a listener.
        received.clear()
        motor.set_parameters([
            Parameter(
                "enable_robot_sound",
                Parameter.Type.BOOL,
                False,
            )
        ])
        motor._publish_robot_sounds()
        _spin_until(
            rclpy,
            [motor, consumer],
            lambda: any(
                marker.action == Marker.DELETE
                for message in received
                for marker in message.markers
            ),
        )
        assert motor._robots["robot1"].moving is False
    finally:
        consumer.destroy_node()
        motor.destroy_node()


def test_human_sound_node_publishes_footstep_and_cone(rclpy_context):
    import rclpy
    from arena_people_msgs.msg import Pedestrians
    from rclpy.parameter import Parameter
    from task_generator.auditory.human_sound_node import HumanSoundNode
    from task_generator.auditory.qos_profiles import transient_event_qos
    from task_generator_msgs.msg import SoundEvent
    from visualization_msgs.msg import Marker, MarkerArray

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    sound_topic = f"/test/{suffix}/human_sound_events"
    marker_topic = f"/test/{suffix}/pedestrian_markers/extra"
    producer = HumanSoundNode(
        parameter_overrides=[
            Parameter(
                "sound_events_topic",
                Parameter.Type.STRING,
                sound_topic,
            ),
            Parameter(
                "sound_markers_topic",
                Parameter.Type.STRING,
                marker_topic,
            ),
        ]
    )
    consumer = rclpy.create_node(f"human_sound_consumer_{suffix}")
    sounds: list[SoundEvent] = []
    markers: list[MarkerArray] = []
    consumer.create_subscription(
        SoundEvent,
        sound_topic,
        sounds.append,
        transient_event_qos(),
    )
    consumer.create_subscription(
        MarkerArray,
        marker_topic,
        markers.append,
        transient_event_qos(depth=10),
    )
    pedestrians = Pedestrians()
    pedestrian = _make_pedestrian(7, 1.0, 2.0)
    pedestrian.twist.linear.x = 0.4
    pedestrians.pedestrians.append(pedestrian)

    try:
        _spin_until(
            rclpy,
            [producer, consumer],
            lambda: (
                producer._sound_publisher.get_subscription_count() > 0
                and producer._marker_publisher.get_subscription_count() > 0
            ),
        )
        producer._on_pedestrians(pedestrians)
        _spin_until(
            rclpy,
            [producer, consumer],
            lambda: bool(sounds and markers),
        )
        assert sounds[-1].sound_type == "footstep"
        assert sounds[-1].semantic_tags == ["walk", "default"]
        assert {marker.type for marker in markers[-1].markers} == {
            Marker.TRIANGLE_LIST,
            Marker.LINE_STRIP,
        }
    finally:
        consumer.destroy_node()
        producer.destroy_node()


def test_human_sound_detector_uses_pose_when_twist_is_zero():
    from arena_people_msgs.msg import Pedestrians
    from task_generator.simulators.human.auditory_events import (
        AuditoryEventDetector,
    )

    emitted: list[tuple[str, int]] = []
    detector = AuditoryEventDetector(
        lambda sound_type, ped: emitted.append((sound_type, int(ped.id))),
        walking_speed_threshold=0.05,
        footstep_interval_sec=0.45,
    )
    first = Pedestrians()
    first.pedestrians.append(_make_pedestrian(9, 1.0, 2.0))
    detector.update(first, 1.0)
    assert emitted == []

    moved = Pedestrians()
    moved.pedestrians.append(_make_pedestrian(9, 1.2, 2.0))
    detector.update(moved, 1.5)
    assert emitted == [("footstep", 9)]


def test_sound_propagation_uses_base_frame_when_listener_frame_is_empty(
    rclpy_context,
):
    from rclpy.parameter import Parameter
    from task_generator.auditory.sound_propagation_node import SoundPropagationNode

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    ns = f"/test/{suffix}"
    robot_fleet_topic = f"{ns}/state/robots"
    world_topic = f"{ns}/state/world"

    propagation = SoundPropagationNode(
        parameter_overrides=[
            Parameter("robot_fleet_topic", Parameter.Type.STRING, robot_fleet_topic),
            Parameter("world_topic", Parameter.Type.STRING, world_topic),
            Parameter("robot_listener_frame", Parameter.Type.STRING, ""),
        ],
    )

    try:
        propagation._cb_robot_fleet(_make_robot_fleet("robot1", f"{ns}/robot1"))
        assert (
            propagation._robot_base_frames["robot:robot1"]
            == "robot1/base_link"
        )
    finally:
        propagation.destroy_node()


def test_sound_propagation_resolves_relative_listener_frame_override(
    rclpy_context,
):
    from rclpy.parameter import Parameter
    from task_generator.auditory.sound_propagation_node import SoundPropagationNode

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    ns = f"/test/{suffix}"
    robot_fleet_topic = f"{ns}/state/robots"
    world_topic = f"{ns}/state/world"

    propagation = SoundPropagationNode(
        parameter_overrides=[
            Parameter("robot_fleet_topic", Parameter.Type.STRING, robot_fleet_topic),
            Parameter("world_topic", Parameter.Type.STRING, world_topic),
            Parameter(
                "robot_listener_frame",
                Parameter.Type.STRING,
                "oakd_rgb_camera_optical_frame",
            ),
        ],
    )

    try:
        propagation._cb_robot_fleet(_make_robot_fleet("robot1", f"{ns}/robot1"))
        assert (
            propagation._robot_base_frames["robot:robot1"]
            == "robot1/oakd_rgb_camera_optical_frame"
        )
    finally:
        propagation.destroy_node()


def test_sound_propagation_uses_tf_height_for_microphones(rclpy_context):
    from geometry_msgs.msg import Point
    from task_generator.auditory.sound_propagation_node import SoundPropagationNode

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    propagation = SoundPropagationNode(namespace=f"/test/{suffix}")
    try:
        listener_id = "microphone:zone:reception:ceiling:1"
        propagation._spawned_microphones[listener_id] = (Point(z=2.4), "map")
        assert propagation._listener_height(listener_id, Point(z=2.4)) == pytest.approx(2.4)
        assert propagation._listener_height("robot:robot1", Point(z=2.4)) == pytest.approx(0.35)
        assert propagation._listener_height("agent:1", Point(z=2.4)) == pytest.approx(1.60)
    finally:
        propagation.destroy_node()


def test_static_audio_uses_authored_source_height(rclpy_context):
    from task_generator.auditory.audio_playback_node import SoundPlaybackNode
    from task_generator.auditory.sound_propagation_node import (
        SoundPropagationNode,
    )
    from task_generator_msgs.msg import ContinuousHeardSoundState, SoundEvent

    source_height = 2.6
    sound_event = SoundEvent()
    sound_event.semantic_tags = ["environment", "static", "alarm"]
    sound_event.source_position.z = source_height

    heard_state = ContinuousHeardSoundState()
    heard_state.source_model = "static_audio_source"
    heard_state.source_position.z = source_height

    assert SoundPropagationNode._source_height(sound_event) == pytest.approx(
        source_height
    )
    assert SoundPlaybackNode._source_height(heard_state) == pytest.approx(
        source_height
    )


def test_propagation_visualizer_splits_pedestrian_and_robot_markers(
    rclpy_context,
):
    import rclpy
    from rclpy.parameter import Parameter
    from task_generator.auditory.sound_propagation_visualizer import (
        SoundPropagationVisualizer,
    )
    from visualization_msgs.msg import MarkerArray

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    pedestrian_topic = f"/test/{suffix}/pedestrian_propagation"
    robot_topic = f"/test/{suffix}/robot_propagation"
    visualizer = SoundPropagationVisualizer(
        parameter_overrides=[
            Parameter(
                "pedestrian_marker_topic",
                Parameter.Type.STRING,
                pedestrian_topic,
            ),
            Parameter(
                "robot_marker_topic",
                Parameter.Type.STRING,
                robot_topic,
            ),
        ]
    )
    consumer = rclpy.create_node(f"propagation_marker_consumer_{suffix}")
    pedestrian_markers: list[MarkerArray] = []
    robot_markers: list[MarkerArray] = []
    consumer.create_subscription(
        MarkerArray,
        pedestrian_topic,
        pedestrian_markers.append,
        10,
    )
    consumer.create_subscription(
        MarkerArray,
        robot_topic,
        robot_markers.append,
        10,
    )

    try:
        _spin_until(
            rclpy,
            [visualizer, consumer],
            lambda: (
                visualizer._pedestrian_publisher.get_subscription_count() > 0
                and visualizer._robot_publisher.get_subscription_count() > 0
            ),
        )
        not_ready_event = _make_heard_sound_event()
        not_ready_event.listener_id = "agent:3"
        not_ready_event.used_backend_fallback = True
        not_ready_event.backend_fallback_reason = "acoustic_scene_not_loaded"
        visualizer._callback(not_ready_event)
        for _ in range(3):
            rclpy.spin_once(consumer, timeout_sec=0.02)
        assert pedestrian_markers == []
        assert robot_markers == []

        pedestrian_event = _make_heard_sound_event()
        pedestrian_event.listener_id = "agent:3"
        visualizer._callback(pedestrian_event)
        _spin_until(
            rclpy,
            [visualizer, consumer],
            lambda: bool(pedestrian_markers),
        )
        assert robot_markers == []
        pedestrian_color = pedestrian_markers[-1].markers[0].color
        assert pedestrian_color.b == pytest.approx(1.0)
        assert pedestrian_color.r == pytest.approx(0.10)

        robot_event = _make_heard_sound_event()
        robot_event.listener_id = "robot:jackal"
        robot_event.propagation_backend = "pyroomacoustics_multi_portal"
        robot_event.portal_ids = ["door:a", "opening:b"]
        robot_event.traversed_zones = ["room_a", "hall", "room_b"]
        robot_event.portal_hop_count = 2
        robot_event.portal_route_loss_db = 3.5
        from geometry_msgs.msg import Point
        robot_event.portal_positions = [
            Point(x=0.25, y=0.0, z=1.0),
            Point(x=0.75, y=0.0, z=1.0),
        ]
        visualizer._callback(robot_event)
        _spin_until(
            rclpy,
            [visualizer, consumer],
            lambda: bool(robot_markers),
        )
        robot_color = robot_markers[-1].markers[0].color
        assert robot_color.r == pytest.approx(0.65)
        assert robot_color.b == pytest.approx(1.0)
        path = next(
            marker
            for marker in robot_markers[-1].markers
            if marker.ns == "robot_sound_propagation_path"
        )
        assert len(path.points) == 4
        portals = [
            marker
            for marker in robot_markers[-1].markers
            if marker.ns == "robot_acoustic_portal"
        ]
        assert len(portals) == 2
        text = next(
            marker
            for marker in robot_markers[-1].markers
            if marker.ns == "robot_sound_propagation_backend"
        )
        assert "2 portal(s)" in text.text
        assert "3.5 dB" in text.text

        previous_path_id = path.id
        latest_event = _make_heard_sound_event()
        latest_event.listener_id = "robot:jackal"
        latest_event.source_position.x = 2.0
        visualizer._callback(latest_event)
        _spin_until(
            rclpy,
            [visualizer, consumer],
            lambda: len(robot_markers) >= 2,
        )
        latest_path = next(
            marker
            for marker in robot_markers[-1].markers
            if marker.ns == "robot_sound_propagation_path"
        )
        assert latest_path.id == previous_path_id
        assert latest_path.points[0].x == pytest.approx(2.0)
        assert len(latest_path.points) == 2
        stale_portals = [
            marker
            for marker in robot_markers[-1].markers
            if (
                marker.ns == "robot_acoustic_portal"
                and marker.action == marker.DELETE
            )
        ]
        assert len(stale_portals) == 2
    finally:
        consumer.destroy_node()
        visualizer.destroy_node()
