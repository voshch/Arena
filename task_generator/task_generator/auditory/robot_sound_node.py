from __future__ import annotations

import hashlib
import math

import attrs
import rclpy
from arena_robots.Robot import RobotIdentifier
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, Header
from task_generator_msgs.msg import (
    ContinuousAudioSourceState,
    RobotFleet,
    SoundEvent,
)
from visualization_msgs.msg import Marker, MarkerArray

from task_generator.auditory.audio_playback_node import SoundPlaybackNode
from task_generator.auditory.qos_profiles import (
    acoustic_metadata_qos,
    continuous_audio_qos,
    transient_event_qos,
)


@attrs.define
class RobotSoundSource:
    name: str
    namespace: str
    model: str
    effective_wheel_separation_m: float
    base_frame_id: str
    position: Point | None = None
    position_header: Header | None = None
    linear_velocity_mps: float = 0.0
    angular_velocity_radps: float = 0.0
    left_velocity_mps: float = 0.0
    right_velocity_mps: float = 0.0
    moving: bool = False


class RobotSoundNode(SoundPlaybackNode):
    def __init__(self, **kwargs: object) -> None:
        super().__init__("robot_sound_node", "robot", **kwargs)

        self.declare_parameter("robot_fleet_topic", "state/robots")
        self.declare_parameter(
            "continuous_audio_sources_topic",
            "continuous_audio_sources",
        )
        self.declare_parameter("sound_type", "motor")
        self.declare_parameter("odom_topic_template", "{namespace}/{name}_velocity_controller/odom")
        self.declare_parameter("motor_start_asset_id", "motor_start")
        self.declare_parameter("motor_stop_asset_id", "motor_stop")
        self.declare_parameter("enable_robot_sound", True)
        self.declare_parameter("source_volume_db", 45.0)
        self.declare_parameter("publish_period_sec", 0.5)
        self.declare_parameter("only_when_moving", False)
        self.declare_parameter("min_speed_mps", 0.02)
        self.declare_parameter("stop_speed_mps", 0.01)
        self.declare_parameter("angular_speed_scale_m", 0.25)
        self.declare_parameter("publish_motor_markers", True)
        # A non-empty shared topic preserves the old/testing interface.
        # Normal launch leaves it empty and uses one topic per robot.
        self.declare_parameter("motor_marker_topic", "")
        self.declare_parameter(
            "motor_marker_topic_suffix",
            "motor_sound_markers",
        )
        self.declare_parameter("motor_marker_lifetime_sec", 0.8)
        self.declare_parameter("motor_marker_z_m", 0.16)
        self.declare_parameter("motor_marker_line_width_m", 0.055)
        self.declare_parameter("motor_marker_cone_degrees", 70.0)
        self.declare_parameter("motor_marker_range_m", 1.25)

        self._robots: dict[str, RobotSoundSource] = {}
        self._odom_qos = QoSProfile( history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT )
        self._odom_subs = []
        self._last_speed: dict[str, float] = {}
        self._event_counter = 0
        self._episode_seed = 0
        self._marker_pubs = {}

        self._sound_pub = self.create_publisher(
            SoundEvent,
            str(self.get_parameter("sound_events_topic").value),
            transient_event_qos(),
        )
        self._continuous_pub = self.create_publisher(
            ContinuousAudioSourceState,
            str(
                self.get_parameter(
                    "continuous_audio_sources_topic"
                ).value
            ),
            continuous_audio_qos(),
        )
        shared_marker_topic = str(
            self.get_parameter("motor_marker_topic").value
        ).strip()
        self._marker_pub = (
            self.create_publisher(
                MarkerArray,
                shared_marker_topic,
                transient_event_qos(depth=10),
            )
            if shared_marker_topic
            else None
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
        for state in msg.robots:
            robot = state.descriptor
            name = str(robot.name)
            if name in self._robots:
                continue

            namespace = str(robot.ns).rstrip("/")
            model = str(robot.model)
            base_frame_id = self._robot_base_frame(
                model,
                str(robot.frame),
            )
            self._robots[name] = RobotSoundSource(
                name=name,
                namespace=namespace,
                model=model,
                effective_wheel_separation_m=(
                    self._effective_wheel_separation(model)
                ),
                base_frame_id=base_frame_id,
            )
            if self._marker_pub is None:
                suffix = str(
                    self.get_parameter("motor_marker_topic_suffix").value
                ).strip("/")
                self._marker_pubs[name] = self.create_publisher(
                    MarkerArray,
                    f"{name}/{suffix}",
                    transient_event_qos(depth=10),
                )

            odom_topics = self._robot_odom_topics(
                model_name=str(robot.model),
                namespace=namespace,
                robot_name=name,
            )
            self.get_logger().info(
                f"robot {name!r} motor motion sources: "
                f"{', '.join(odom_topics)}"
            )
            for odom_topic in odom_topics:
                self._odom_subs.append(
                    self.create_subscription(
                        Odometry,
                        odom_topic,
                        lambda odom, robot_name=name: self._cb_odom(
                            robot_name,
                            odom,
                        ),
                        self._odom_qos,
                    )
                )

    def _robot_odom_topics(
        self,
        *,
        model_name: str,
        namespace: str,
        robot_name: str,
    ) -> tuple[str, ...]:
        """Return configured, model-specific, and standard odometry topics."""
        topics = [
            str(self.get_parameter("odom_topic_template").value).format(
                namespace=namespace,
                name=robot_name,
            ),
            f"{namespace}/odom",
        ]
        try:
            control = (
                RobotIdentifier(model_name)
                .resolve_sync()
                .model_params.control
            )
            model_odom = (
                control.odom_topic.strip("/") if control is not None else ""
            )
            if model_odom:
                topics.append(f"{namespace}/{model_odom}")
        except Exception as exc:
            self.get_logger().warning(
                f"could not resolve odometry topic for {model_name!r}: {exc}"
            )
        return tuple(dict.fromkeys(topic for topic in topics if topic))

    def _cb_odom(self, robot_name: str, msg: Odometry) -> None:
        source = self._robots.get(robot_name)
        if source is None:
            return

        source.position = Point()
        position_header = Header()
        position_header.stamp = msg.header.stamp
        position_header.frame_id = source.base_frame_id
        source.position_header = position_header
        linear_velocity = float(msg.twist.twist.linear.x)
        lateral_velocity = float(msg.twist.twist.linear.y)
        angular_velocity = float(msg.twist.twist.angular.z)
        half_separation = source.effective_wheel_separation_m * 0.5
        source.linear_velocity_mps = linear_velocity
        source.angular_velocity_radps = angular_velocity
        source.left_velocity_mps = (
            linear_velocity - angular_velocity * half_separation
        )
        source.right_velocity_mps = (
            linear_velocity + angular_velocity * half_separation
        )
        angular_scale = max(
            float(self.get_parameter("angular_speed_scale_m").value),
            0.0,
        )
        # Treat in-place rotation as motor motion. Multiplying angular speed
        # by an effective chassis radius expresses it in the same m/s scale
        # used by the existing start/stop thresholds.
        self._last_speed[robot_name] = max(
            math.hypot(linear_velocity, lateral_velocity),
            abs(angular_velocity) * angular_scale,
        )

    def _publish_robot_sounds(self) -> None:
        only_when_moving = bool(self.get_parameter("only_when_moving").value)
        min_speed = float(self.get_parameter("min_speed_mps").value)
        stop_speed = float(self.get_parameter("stop_speed_mps").value)

        for robot_name, source in self._robots.items():
            if source.position is None:
                continue

            if not bool(
                self.get_parameter("enable_robot_sound").value
            ):
                if self._uses_procedural_audio(source):
                    self._publish_continuous_state(source, active=False)
                if source.moving:
                    self._clear_motor_cone_markers(
                        robot_name,
                        source.base_frame_id,
                    )
                    source.moving = False
                    if self._uses_procedural_audio(source):
                        self.get_logger().info(
                            f"robot {robot_name!r} procedural motor sound disabled"
                        )
                        continue
                    asset_id = str(
                        self.get_parameter(
                            "motor_stop_asset_id"
                        ).value
                    )
                    self._sound_pub.publish(
                        self._make_sound_event(
                            source,
                            asset_id,
                        )
                    )
                    self.get_logger().info(
                        f"robot {robot_name!r} motor sound disabled"
                    )
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

            if self._uses_procedural_audio(source):
                self._publish_continuous_state(
                    source,
                    active=(
                        bool(
                            self.get_parameter(
                                "enable_robot_sound"
                            ).value
                        )
                        and moving
                    ),
                )
            state_changed = moving != source.moving
            if moving:
                self._publish_motor_cone_markers(robot_name, source)
            elif state_changed:
                self._clear_motor_cone_markers(robot_name, source.base_frame_id)

            if not state_changed:
                continue

            source.moving = moving
            if self._uses_procedural_audio(source):
                self.get_logger().info(
                    f"robot {robot_name!r} procedural motor "
                    f"{'started' if moving else 'stopped'} "
                    f"at {speed:.3f} m/s"
                )
                continue
            parameter = (
                "motor_start_asset_id" if moving else "motor_stop_asset_id"
            )
            asset_id = str(self.get_parameter(parameter).value)
            self._sound_pub.publish(
                self._make_sound_event(source, asset_id)
            )
            self.get_logger().info(
                f"robot {robot_name!r} motor "
                f"{'started' if moving else 'stopped'} "
                f"at {speed:.3f} m/s"
            )

    def _uses_procedural_audio(self, source: RobotSoundSource) -> bool:
        mode = str(
            self.get_parameter("motor_audio_mode").value
        ).strip().lower()
        return mode == "procedural" and source.model.lower() == "jackal"

    def _publish_continuous_state(
        self,
        source: RobotSoundSource,
        *,
        active: bool,
    ) -> None:
        if source.position is None or source.position_header is None:
            return
        msg = ContinuousAudioSourceState()
        msg.header.stamp = source.position_header.stamp
        msg.header.frame_id = source.position_header.frame_id
        msg.source_id = f"robot:{source.name}:motor"
        msg.source_agent_id = self._robot_numeric_id(source.name)
        msg.source_agent_name = source.name
        msg.source_model = source.model
        msg.sound_type = "motor"
        msg.source_backend = "drivetrain"
        msg.source_position = source.position
        msg.source_yaw = 0.0
        msg.linear_velocity_mps = source.linear_velocity_mps
        msg.angular_velocity_radps = source.angular_velocity_radps
        msg.left_velocity_mps = source.left_velocity_mps
        msg.right_velocity_mps = source.right_velocity_mps
        msg.source_volume_db = float(
            self.get_parameter("source_volume_db").value
        )
        msg.reference_distance_m = 1.0
        msg.active = active
        msg.deterministic_seed = self._source_seed(source.name)
        msg.has_normalized_load = False
        msg.normalized_load = 0.0
        msg.semantic_tags = ["robot", "motor", "mechanical", "procedural"]
        self._continuous_pub.publish(msg)

    def _source_seed(self, robot_name: str) -> int:
        payload = f"{self._episode_seed}:{robot_name}:motor".encode()
        return int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(),
            "little",
        )

    def _effective_wheel_separation(self, model_name: str) -> float:
        fallback = max(
            2.0 * float(
                self.get_parameter("angular_speed_scale_m").value
            ),
            1e-3,
        )
        try:
            view = RobotIdentifier(model_name).resolve_sync()
            controllers = view.control
            controller = next(
                value
                for name, value in controllers.items()
                if name != "controller_manager"
                and isinstance(value, dict)
                and isinstance(value.get("ros__parameters"), dict)
                and "wheel_separation" in value["ros__parameters"]
            )
            params = controller["ros__parameters"]
            separation = float(params["wheel_separation"])
            multiplier = float(
                params.get("wheel_separation_multiplier", 1.0)
            )
            effective = separation * multiplier
            if effective <= 0.0 or not math.isfinite(effective):
                raise ValueError(f"invalid effective separation {effective}")
            return effective
        except Exception as exc:
            self.get_logger().warning(
                f"could not resolve wheel separation for {model_name!r}: "
                f"{exc}; using {fallback:.5f} m"
            )
            return fallback

    def _publish_motor_cone_markers(
        self,
        robot_name: str,
        source: RobotSoundSource,
    ) -> None:
        if (
            not bool(self.get_parameter("publish_motor_markers").value)
            or source.position is None
        ):
            return

        cone_degrees = min(
            max(
                float(
                    self.get_parameter(
                        "motor_marker_cone_degrees"
                    ).value
                ),
                10.0,
            ),
            180.0,
        )
        cone_angle = math.radians(cone_degrees)
        marker_range = max(
            float(self.get_parameter("motor_marker_range_m").value),
            0.05,
        )
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
        source_x = 0.0
        source_y = 0.0
        yaw = 0.0
        apex = Point(
            x=source_x + math.cos(yaw) * 0.15,
            y=source_y + math.sin(yaw) * 0.15,
            z=z,
        )
        arc_points: list[Point] = []
        start_angle = yaw - cone_angle / 2.0
        for index in range(17):
            angle = start_angle + cone_angle * index / 16
            arc_points.append(
                Point(
                    x=source_x + math.cos(angle) * marker_range,
                    y=source_y + math.sin(angle) * marker_range,
                    z=z,
                )
            )

        stamp = self.get_clock().now().to_msg()
        fill = Marker()
        fill.header.frame_id = source.base_frame_id
        fill.header.stamp = stamp
        fill.ns = f"motor_sound_{robot_name}"
        fill.id = base_id
        fill.type = Marker.TRIANGLE_LIST
        fill.action = Marker.ADD
        fill.pose.orientation.w = 1.0
        fill.scale.x = fill.scale.y = fill.scale.z = 1.0
        fill.color = ColorRGBA(r=1.0, g=0.55, b=0.05, a=0.28)
        fill.lifetime.sec = lifetime_sec
        fill.lifetime.nanosec = lifetime_nanosec
        for left, right in zip(
            arc_points,
            arc_points[1:],
            strict=False,
        ):
            fill.points.extend([apex, left, right])

        outline = Marker()
        outline.header = fill.header
        outline.ns = f"motor_sound_{robot_name}_outline"
        outline.id = base_id + 1
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.pose.orientation.w = 1.0
        outline.scale.x = line_width
        outline.color = ColorRGBA(r=1.0, g=0.55, b=0.05, a=0.95)
        outline.lifetime = fill.lifetime
        outline.points = [apex, *arc_points, apex]

        marker_pub = self._marker_pub or self._marker_pubs.get(robot_name)
        if marker_pub is not None:
            marker_pub.publish(MarkerArray(markers=[fill, outline]))

    def _clear_motor_cone_markers(
        self,
        robot_name: str,
        marker_frame_id: str,
    ) -> None:
        markers = MarkerArray()
        base_id = self._motor_marker_base_id(robot_name)
        for index, namespace in enumerate(
            (
                f"motor_sound_{robot_name}",
                f"motor_sound_{robot_name}_outline",
            )
        ):
            marker = Marker()
            marker.header.frame_id = marker_frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = namespace
            marker.id = base_id + index
            marker.action = Marker.DELETE
            markers.markers.append(marker)
        marker_pub = self._marker_pub or self._marker_pubs.get(robot_name)
        if marker_pub is not None:
            marker_pub.publish(markers)

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

    def _make_sound_event(
        self,
        source: RobotSoundSource,
        asset_id: str,
    ) -> SoundEvent:
        stamp = self.get_clock().now().to_msg()
        sound_type = str(self.get_parameter("sound_type").value)
        period = float(self.get_parameter("publish_period_sec").value)
        if source.position is None or source.position_header is None:
            raise RuntimeError("robot sound source has no odometry update")

        msg = SoundEvent()
        msg.header.stamp = source.position_header.stamp
        msg.header.frame_id = source.position_header.frame_id
        msg.event_id = (
            f"robot:{source.name}:{stamp.sec}:{stamp.nanosec}:"
            f"{self._event_counter}"
        )
        self._event_counter += 1

        msg.source_agent_id = self._robot_numeric_id(source.name)
        msg.source_agent_name = source.name
        msg.sound_type = sound_type
        msg.label = sound_type
        msg.asset_id = asset_id
        msg.source_position = source.position
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
