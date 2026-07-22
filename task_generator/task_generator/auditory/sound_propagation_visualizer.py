from __future__ import annotations

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from task_generator_msgs.msg import HeardSoundEvent
from visualization_msgs.msg import Marker, MarkerArray

from .qos_profiles import transient_event_qos


class SoundPropagationVisualizer(Node):
    """Publish source/portal/listener propagation paths for RViz."""

    def __init__(self, **kwargs) -> None:
        super().__init__("sound_propagation_visualizer", **kwargs)
        self.declare_parameter("heard_sound_events_topic", "heard_sound_events")
        self.declare_parameter("marker_topic", "sound_propagation_markers")
        self.declare_parameter("marker_lifetime_sec", 2.0)
        self.declare_parameter("path_z_m", 1.0)
        self._next_marker_id = 0
        self._publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("marker_topic").value),
            10,
        )
        self.create_subscription(
            HeardSoundEvent,
            str(self.get_parameter("heard_sound_events_topic").value),
            self._callback,
            transient_event_qos(),
        )

    def _callback(self, msg: HeardSoundEvent) -> None:
        backend = str(msg.propagation_backend).strip() or "unknown"
        color = self._backend_color(backend, msg.used_backend_fallback)
        frame = str(msg.header.frame_id).strip() or "map"
        marker_id = self._next_marker_id
        self._next_marker_id = (self._next_marker_id + 10) % 2_000_000_000
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
        points = [source]
        if str(msg.portal_id).strip():
            points.append(
                Point(
                    x=float(msg.portal_position.x),
                    y=float(msg.portal_position.y),
                    z=path_z,
                )
            )
        points.append(listener)

        path = self._marker(frame, marker_id, Marker.LINE_STRIP, lifetime)
        path.ns = "sound_propagation_path"
        path.scale.x = 0.07
        path.color = color
        path.points = points

        source_marker = self._marker(frame, marker_id + 1, Marker.SPHERE, lifetime)
        source_marker.ns = "sound_source"
        source_marker.pose.position = source
        source_marker.scale.x = source_marker.scale.y = source_marker.scale.z = 0.22
        source_marker.color = ColorRGBA(r=0.15, g=0.95, b=0.25, a=0.95)

        listener_marker = self._marker(frame, marker_id + 2, Marker.SPHERE, lifetime)
        listener_marker.ns = "sound_listener"
        listener_marker.pose.position = listener
        listener_marker.scale.x = listener_marker.scale.y = listener_marker.scale.z = 0.22
        listener_marker.color = ColorRGBA(r=0.15, g=0.35, b=1.0, a=0.95)

        text = self._marker(frame, marker_id + 3, Marker.TEXT_VIEW_FACING, lifetime)
        text.ns = "sound_propagation_backend"
        text.pose.position = listener
        text.pose.position.z += 0.35
        text.scale.z = 0.24
        text.color = color
        text.text = backend
        if msg.portal_id:
            text.text += f"\n{msg.portal_id}"
        if msg.used_backend_fallback:
            text.text += f"\nfallback: {msg.backend_fallback_reason}"

        markers = [path, source_marker, listener_marker, text]
        if msg.portal_id:
            portal = self._marker(frame, marker_id + 4, Marker.CUBE, lifetime)
            portal.ns = "acoustic_portal"
            portal.pose.position = points[1]
            portal.scale.x = portal.scale.y = portal.scale.z = 0.28
            portal.color = ColorRGBA(r=1.0, g=0.85, b=0.05, a=0.95)
            markers.append(portal)
        self._publisher.publish(MarkerArray(markers=markers))

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

    @staticmethod
    def _backend_color(backend: str, fallback: bool) -> ColorRGBA:
        if fallback:
            return ColorRGBA(r=1.0, g=0.12, b=0.08, a=0.9)
        if backend == "pyroomacoustics_one_door":
            return ColorRGBA(r=0.75, g=0.15, b=1.0, a=0.9)
        if backend == "pyroomacoustics_same_room":
            return ColorRGBA(r=0.05, g=0.9, b=0.95, a=0.9)
        if backend == "level3":
            return ColorRGBA(r=1.0, g=0.55, b=0.05, a=0.9)
        return ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.9)


def main() -> None:
    rclpy.init()
    node = SoundPropagationVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
