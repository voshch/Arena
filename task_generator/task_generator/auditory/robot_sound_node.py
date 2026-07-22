from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import rclpy
from arena_robots.Robot import RobotIdentifier
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from task_generator.auditory.qos_profiles import acoustic_metadata_qos, transient_event_qos
from task_generator_msgs.msg import RobotFleet, SoundEvent
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class RobotSoundSource:
    name: str
    namespace: str
    position: Point | None = None
    yaw: float = 0.0
    moving: bool = False
    marker_frame_id: str = "map"


class RobotSoundNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("robot_sound_node", **kwargs)

        self.declare_parameter("robot_fleet_topic", "state/robots")
        self.declare_parameter("sound_events_topic", "human_sound_events")
        self.declare_parameter("sound_type", "motor")
        self.declare_parameter("odom_topic_template", "{namespace}/{name}_velocity_controller/odom")
        self.declare_parameter("motor_start_asset_id", "motor_start")
        self.declare_parameter("motor_stop_asset_id", "motor_stop")
        self.declare_parameter("source_volume_db", 55.0)
        self.declare_parameter("publish_period_sec", 0.5)
        self.declare_parameter("only_when_moving", False)
        self.declare_parameter("min_speed_mps", 0.02)
        self.declare_parameter("stop_speed_mps", 0.01)
        self.declare_parameter("publish_motor_markers", True)
        self.declare_parameter("motor_marker_topic", "human_sound_markers")
        self.declare_parameter("motor_marker_lifetime_sec", 0.8)
        self.declare_parameter("motor_marker_z_m", 0.16)
        self.declare_parameter("motor_marker_line_width_m", 0.055)
        self.declare_parameter("motor_marker_arc_degrees", 300.0)
        self.declare_parameter("motor_marker_radii_m", [0.45, 0.75, 1.05])

        self._robots: dict[str, RobotSoundSource] = {}
        self._odom_qos = QoSProfile( history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT )
        self._odom_subs = []
        self._last_speed: dict[str, float] = {}
        self._event_counter = 0

        self._sound_pub = self.create_publisher(
            SoundEvent,
            str(self.get_parameter("sound_events_topic").value),
            transient_event_qos(),
        )
        self._marker_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("motor_marker_topic").value),
            transient_event_qos(depth=10),
        )

        self.create_subscription(
            RobotFleet,
            str(self.get_parameter("robot_fleet_topic").value),
            self._cb_robot_fleet,
            acoustic_metadata_qos(),
        )

        self.create_timer(
            float(self.get_parameter("publish_period_sec").value),
            self._publish_robot_sounds,
        )

    def _cb_robot_fleet(self, msg: RobotFleet) -> None:
        for robot in msg.robots:
            name = str(robot.name)
            if name in self._robots:
                continue

            namespace = str(robot.ns).rstrip("/")
            marker_frame_id = self._robot_base_frame(
                str(robot.model),
                str(robot.frame),
            )
            self._robots[name] = RobotSoundSource(
                name=name,
                namespace=namespace,
                marker_frame_id=marker_frame_id,
            )

            odom_topic = str(self.get_parameter("odom_topic_template").value).format(namespace=namespace, name=name)
            self.get_logger().warning(f"subscribing to robot odom: {odom_topic}")
            sub = self.create_subscription(
                Odometry,
                odom_topic,
                lambda odom, robot_name=name: self._cb_odom(robot_name, odom),
                self._odom_qos,
            )
            self._odom_subs.append(sub)

    def _cb_odom(self, robot_name: str, msg: Odometry) -> None:
        source = self._robots.get(robot_name)
        if source is None:
            return

        source.position = msg.pose.pose.position
        source.yaw = self._yaw_from_quaternion(msg.pose.pose.orientation)
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self._last_speed[robot_name] = (vx * vx + vy * vy) ** 0.5

    def _publish_robot_sounds(self) -> None:
        only_when_moving = bool(self.get_parameter("only_when_moving").value)
        min_speed = float(self.get_parameter("min_speed_mps").value)
        stop_speed = float(self.get_parameter("stop_speed_mps").value)

        for robot_name, source in self._robots.items():
            if source.position is None:
                continue

            speed = self._last_speed.get(robot_name, 0.0)
            if not only_when_moving:
                moving = True
            elif source.moving:
                # Use a lower stop threshold to avoid start/stop chatter when
                # noisy odometry fluctuates around the start threshold.
                moving = speed > stop_speed
            else:
                moving = speed >= min_speed
            state_changed = moving != source.moving
            if moving:
                self._publish_motor_arc_markers(robot_name, source)
            elif state_changed:
                self._clear_motor_arc_markers(robot_name, source.marker_frame_id)

            if not state_changed:
                continue

            source.moving = moving
            parameter = (
                "motor_start_asset_id" if moving else "motor_stop_asset_id"
            )
            asset_id = str(self.get_parameter(parameter).value)
            self._sound_pub.publish(
                self._make_sound_event(robot_name, source.position, asset_id)
            )
            self.get_logger().info(
                f"robot {robot_name!r} motor "
                f"{'started' if moving else 'stopped'} "
                f"at {speed:.3f} m/s"
            )

    def _publish_motor_arc_markers(
        self,
        robot_name: str,
        source: RobotSoundSource,
    ) -> None:
        if (
            not bool(self.get_parameter("publish_motor_markers").value)
            or source.position is None
        ):
            return

        radii = [
            float(value)
            for value in self.get_parameter("motor_marker_radii_m").value
            if float(value) > 0.0
        ]
        if not radii:
            return

        arc_degrees = min(
            max(float(self.get_parameter("motor_marker_arc_degrees").value), 10.0),
            360.0,
        )
        arc_span = math.radians(arc_degrees)
        line_width = max(
            float(self.get_parameter("motor_marker_line_width_m").value),
            0.001,
        )
        z = float(self.get_parameter("motor_marker_z_m").value)
        period = float(self.get_parameter("publish_period_sec").value)
        lifetime = max(
            float(self.get_parameter("motor_marker_lifetime_sec").value),
            period * 1.5,
        )
        lifetime_sec = int(lifetime)
        lifetime_nanosec = int((lifetime % 1.0) * 1_000_000_000)
        base_id = self._motor_marker_base_id(robot_name)
        colors = (
            ColorRGBA(r=1.0, g=0.20, b=0.05, a=0.95),
            ColorRGBA(r=1.0, g=0.55, b=0.05, a=0.88),
            ColorRGBA(r=1.0, g=0.88, b=0.18, a=0.78),
        )

        markers = MarkerArray()
        robot_local_frame = source.marker_frame_id != "map"
        center_x = 0.0 if robot_local_frame else float(source.position.x)
        center_y = 0.0 if robot_local_frame else float(source.position.y)
        for index, radius in enumerate(radii):
            marker = Marker()
            marker.header.frame_id = source.marker_frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f"motor_sound_{robot_name}"
            marker.id = base_id + index
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = line_width
            marker.color = colors[index % len(colors)]
            marker.lifetime.sec = lifetime_sec
            marker.lifetime.nanosec = lifetime_nanosec

            # Leave a small rear-facing gap so these read as expanding sound
            # arcs rather than obstacle/range circles. Alternate the gap angle
            # slightly to keep all three lines visually distinct.
            center_angle = (
                0.0 if robot_local_frame else source.yaw
            ) + index * math.radians(7.0)
            start_angle = center_angle - arc_span / 2.0
            segments = max(int(arc_degrees / 6.0), 12)
            for step in range(segments + 1):
                angle = start_angle + arc_span * step / segments
                marker.points.append(
                    Point(
                        x=center_x + math.cos(angle) * radius,
                        y=center_y + math.sin(angle) * radius,
                        z=z,
                    )
                )
            markers.markers.append(marker)

        self._marker_pub.publish(markers)

    def _clear_motor_arc_markers(
        self,
        robot_name: str,
        marker_frame_id: str,
    ) -> None:
        markers = MarkerArray()
        base_id = self._motor_marker_base_id(robot_name)
        count = len(self.get_parameter("motor_marker_radii_m").value)
        for index in range(count):
            marker = Marker()
            marker.header.frame_id = marker_frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f"motor_sound_{robot_name}"
            marker.id = base_id + index
            marker.action = Marker.DELETE
            markers.markers.append(marker)
        self._marker_pub.publish(markers)

    @staticmethod
    def _motor_marker_base_id(robot_name: str) -> int:
        digest = hashlib.blake2b(robot_name.encode(), digest_size=4).digest()
        return (int.from_bytes(digest, "big") & 0x0FFFFFFF) * 8

    def _robot_base_frame(self, model_name: str, frame_prefix: str) -> str:
        prefix = frame_prefix.strip("/")
        try:
            base_frame = (
                RobotIdentifier(model_name)
                .resolve_sync()
                .model_params.base_frame
                .strip("/")
            )
        except Exception as exc:
            base_frame = "base_link"
            self.get_logger().warning(
                f"could not resolve base frame for robot model {model_name!r}: "
                f"{exc}; using {base_frame!r}"
            )
        return "/".join(part for part in (prefix, base_frame) if part)

    @staticmethod
    def _yaw_from_quaternion(quaternion) -> float:
        siny_cosp = 2.0 * (
            quaternion.w * quaternion.z + quaternion.x * quaternion.y
        )
        cosy_cosp = 1.0 - 2.0 * (
            quaternion.y * quaternion.y + quaternion.z * quaternion.z
        )
        return math.atan2(siny_cosp, cosy_cosp)

    def _make_sound_event(
        self,
        robot_name: str,
        position: Point,
        asset_id: str,
    ) -> SoundEvent:
        stamp = self.get_clock().now().to_msg()
        sound_type = str(self.get_parameter("sound_type").value)
        period = float(self.get_parameter("publish_period_sec").value)

        msg = SoundEvent()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.event_id = f"robot:{robot_name}:{stamp.sec}:{stamp.nanosec}:{self._event_counter}"
        self._event_counter += 1

        msg.source_agent_id = self._robot_numeric_id(robot_name)
        msg.source_agent_name = robot_name
        msg.sound_type = sound_type
        msg.label = sound_type
        msg.asset_id = asset_id
        msg.source_position = position
        msg.source_yaw = 0.0
        msg.source_volume_db = float(self.get_parameter("source_volume_db").value)
        msg.semantic_tags = ["robot", "motor", "mechanical"]
        msg.duration.sec = int(period)
        msg.duration.nanosec = int((period % 1.0) * 1_000_000_000)
        msg.loop = False
        return msg

    @staticmethod
    def _robot_numeric_id(robot_name: str) -> int:
        digest = hashlib.blake2b(robot_name.encode(), digest_size=4).digest()
        return -int.from_bytes(digest, "big")


def main() -> None:
    rclpy.init()
    node = RobotSoundNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
