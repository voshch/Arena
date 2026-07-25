from __future__ import annotations

import rclpy
import tf2_ros
from arena_robots.Robot import RobotIdentifier
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from task_generator_msgs.msg import HeardSoundEvent, RobotFleet
from visualization_msgs.msg import Marker, MarkerArray

from .qos_profiles import acoustic_metadata_qos, transient_event_qos


class SoundPropagationVisualizer(Node):
    """Publish source/portal/listener propagation paths for RViz."""

    def __init__(self, **kwargs) -> None:
        super().__init__("sound_propagation_visualizer", **kwargs)
        self.declare_parameter("heard_sound_events_topic", "heard_sound_events")
        self.declare_parameter(
            "pedestrian_marker_topic",
            "pedestrian_sound_propagation_markers",
        )
        self.declare_parameter(
            "robot_marker_topic",
            "robot_sound_propagation_markers",
        )
        self.declare_parameter("robot_fleet_topic", "state/robots")
        self.declare_parameter("sync_robot_listener_to_tf", True)
        self.declare_parameter("marker_lifetime_sec", 2.0)
        self.declare_parameter("path_z_m", 1.0)
        self._robot_base_frames: dict[str, str] = {}
        self._previous_portal_count = {
            "pedestrian": 0,
            "robot": 0,
        }
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(
            self._tf_buffer,
            self,
        )
        self._pedestrian_publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("pedestrian_marker_topic").value),
            10,
        )
        self._robot_publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("robot_marker_topic").value),
            10,
        )
        self.create_subscription(
            HeardSoundEvent,
            str(self.get_parameter("heard_sound_events_topic").value),
            self._callback,
            transient_event_qos(),
        )
        self.create_subscription(
            RobotFleet,
            str(self.get_parameter("robot_fleet_topic").value),
            self._on_robot_fleet,
            acoustic_metadata_qos(),
        )

    def _on_robot_fleet(self, msg: RobotFleet) -> None:
        frames: dict[str, str] = {}
        for state in msg.robots:
            descriptor = state.descriptor
            prefix = str(descriptor.frame).strip("/")
            try:
                base_frame = (
                    RobotIdentifier(str(descriptor.model))
                    .resolve_sync()
                    .model_params.base_frame
                    .strip("/")
                )
            except Exception as exc:
                base_frame = "base_link"
                self.get_logger().warning(
                    "could not resolve base frame for propagation "
                    f"visualization of {descriptor.name!r}: {exc}; "
                    f"using {base_frame!r}"
                )
            frames[f"robot:{descriptor.name}"] = "/".join(
                part for part in (prefix, base_frame) if part
            )
        self._robot_base_frames = frames

    def _callback(self, msg: HeardSoundEvent) -> None:
        if (
            bool(msg.used_backend_fallback)
            and str(msg.backend_fallback_reason).strip()
            == "acoustic_scene_not_loaded"
        ):
            return
        backend = str(msg.propagation_backend).strip() or "unknown"
        output = self._listener_output(str(msg.listener_id))
        if output is None:
            self.get_logger().warning(
                f"cannot visualize sound for unknown listener "
                f"{msg.listener_id!r}"
            )
            return
        publisher, listener_kind, color = output
        frame = str(msg.header.frame_id).strip() or "map"
        marker_id = 0
        lifetime = float(self.get_parameter("marker_lifetime_sec").value)
        path_z = float(self.get_parameter("path_z_m").value)

        source = Point(
            x=float(msg.source_position.x),
            y=float(msg.source_position.y),
            z=path_z,
        )
        listener = Point(
            x=float(msg.listener_position.x),
            y=float(msg.listener_position.y),
            z=path_z,
        )
        if listener_kind == "robot":
            listener = self._current_robot_position(
                str(msg.listener_id),
                frame,
                listener,
                path_z,
            )
        points = [source]
        portal_points = [
            Point(x=float(point.x), y=float(point.y), z=path_z)
            for point in getattr(msg, "portal_positions", ())
        ]
        if not portal_points and str(msg.portal_id).strip():
            portal_points = [
                Point(
                    x=float(msg.portal_position.x),
                    y=float(msg.portal_position.y),
                    z=path_z,
                )
            ]
        points.extend(portal_points)
        points.append(listener)

        path = self._marker(frame, marker_id, Marker.LINE_STRIP, lifetime)
        path.ns = f"{listener_kind}_sound_propagation_path"
        path.scale.x = 0.07
        path.color = color
        path.points = points

        source_marker = self._marker(frame, marker_id + 1, Marker.SPHERE, lifetime)
        source_marker.ns = f"{listener_kind}_sound_source"
        source_marker.pose.position = source
        source_marker.scale.x = source_marker.scale.y = source_marker.scale.z = 0.22
        source_marker.color = color

        listener_marker = self._marker(frame, marker_id + 2, Marker.SPHERE, lifetime)
        listener_marker.ns = f"{listener_kind}_sound_listener"
        listener_marker.pose.position = listener
        listener_marker.scale.x = listener_marker.scale.y = listener_marker.scale.z = 0.22
        listener_marker.color = color

        text = self._marker(frame, marker_id + 3, Marker.TEXT_VIEW_FACING, lifetime)
        text.ns = f"{listener_kind}_sound_propagation_backend"
        text.pose.position = listener
        text.pose.position.z += 0.35
        text.scale.z = 0.24
        text.color = color
        text.text = backend
        portal_ids = list(getattr(msg, "portal_ids", ()))
        if not portal_ids and msg.portal_id:
            portal_ids = [msg.portal_id]
        if portal_ids:
            text.text += (
                f"\n{len(portal_ids)} portal(s), "
                f"loss={float(getattr(msg, 'portal_route_loss_db', 0.0)):.1f} dB"
            )
            text.text += "\n" + " → ".join(portal_ids)
        if msg.used_backend_fallback:
            text.text += f"\nfallback: {msg.backend_fallback_reason}"

        markers = [path, source_marker, listener_marker, text]
        for index, portal_point in enumerate(portal_points):
            portal = self._marker(
                frame,
                marker_id + 4 + index,
                Marker.CUBE,
                lifetime,
            )
            portal.ns = f"{listener_kind}_acoustic_portal"
            portal.pose.position = portal_point
            portal.scale.x = portal.scale.y = portal.scale.z = 0.28
            portal.color = color
            markers.append(portal)
        previous_portals = self._previous_portal_count[listener_kind]
        for index in range(len(portal_points), previous_portals):
            stale = self._marker(
                frame,
                marker_id + 4 + index,
                Marker.CUBE,
                0.0,
            )
            stale.ns = f"{listener_kind}_acoustic_portal"
            stale.action = Marker.DELETE
            markers.append(stale)
        self._previous_portal_count[listener_kind] = len(portal_points)
        publisher.publish(MarkerArray(markers=markers))

    def _current_robot_position(
        self,
        listener_id: str,
        frame: str,
        fallback: Point,
        path_z: float,
    ) -> Point:
        if not bool(
            self.get_parameter("sync_robot_listener_to_tf").value
        ):
            return fallback
        base_frame = self._robot_base_frames.get(listener_id)
        if not base_frame:
            return fallback
        try:
            transform = self._tf_buffer.lookup_transform(
                frame,
                base_frame,
                rclpy.time.Time(),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return fallback
        translation = transform.transform.translation
        return Point(
            x=float(translation.x),
            y=float(translation.y),
            z=path_z,
        )

    def _marker(
        self, frame: str, marker_id: int, marker_type: int, lifetime_sec: float
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        seconds = max(lifetime_sec, 0.0)
        marker.lifetime.sec = int(seconds)
        marker.lifetime.nanosec = int((seconds % 1.0) * 1_000_000_000)
        return marker

    def _listener_output(
        self,
        listener_id: str,
    ) -> tuple[object, str, ColorRGBA] | None:
        if listener_id.startswith("agent:"):
            return (
                self._pedestrian_publisher,
                "pedestrian",
                ColorRGBA(r=0.10, g=0.45, b=1.0, a=0.92),
            )
        if listener_id.startswith("robot:"):
            return (
                self._robot_publisher,
                "robot",
                ColorRGBA(r=0.65, g=0.20, b=1.0, a=0.92),
            )
        return None


def main() -> None:
    rclpy.init()
    node = SoundPropagationVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
