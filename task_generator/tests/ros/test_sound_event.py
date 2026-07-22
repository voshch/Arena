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
    from task_generator_msgs.msg import RobotDescriptor, RobotFleet

    robot = RobotDescriptor()
    robot.name = robot_name
    robot.model = "jackal"
    robot.ns = namespace
    robot.frame = robot_name

    fleet = RobotFleet()
    fleet.robots.append(robot)
    return fleet

def test_sound_event_round_trips_to_heard_sound_event(rclpy_context):
    import rclpy
    from rclpy.parameter import Parameter
    from arena_people_msgs.msg import Pedestrians
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
        ],
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
        assert heard.received_volume_db == pytest.approx(
            60.0 - 20.0 * math.log10(5.0),
            abs=1e-3,
        )
    finally:
        emitter.destroy_node()
        consumer.destroy_node()
        propagation.destroy_node()


def test_auditory_round_trip_greeting_reaches_robot_marker(rclpy_context):
    import rclpy
    from rclpy.parameter import Parameter
    from nav_msgs.msg import Odometry
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


def test_motor_sound_publishes_colored_arcs_and_clears_them(rclpy_context):
    import rclpy
    from geometry_msgs.msg import Point
    from rclpy.parameter import Parameter
    from task_generator.auditory.qos_profiles import transient_event_qos
    from task_generator.auditory.robot_sound_node import (
        RobotSoundNode,
        RobotSoundSource,
    )
    from visualization_msgs.msg import Marker, MarkerArray

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    marker_topic = f"/test/{suffix}/human_sound_markers"
    motor = RobotSoundNode(
        parameter_overrides=[
            Parameter("motor_marker_topic", Parameter.Type.STRING, marker_topic),
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
    motor._robots["robot1"] = RobotSoundSource(
        name="robot1",
        namespace="/robot1",
        position=Point(x=2.0, y=3.0, z=0.0),
        marker_frame_id="robot1/base_link",
    )
    motor._last_speed["robot1"] = 0.2

    try:
        assert motor._robot_base_frame("jackal", "robot1") == "robot1/base_link"
        _spin_until(
            rclpy,
            [motor, consumer],
            lambda: motor._marker_pub.get_subscription_count() > 0,
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
        assert len(added) == 3
        assert all(marker.type == Marker.LINE_STRIP for marker in added)
        assert all(marker.header.frame_id == "robot1/base_link" for marker in added)
        assert all(len(marker.points) > 12 for marker in added)
        assert max(
            abs(point.x)
            for marker in added
            for point in marker.points
        ) < 1.1
        assert len(
            {(marker.color.r, marker.color.g, marker.color.b) for marker in added}
        ) == 3

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
        assert len(deleted) == 3
    finally:
        consumer.destroy_node()
        motor.destroy_node()
