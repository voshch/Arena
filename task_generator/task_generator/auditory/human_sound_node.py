from __future__ import annotations

import itertools
import math

import rclpy
from arena_people_msgs.msg import Pedestrian, Pedestrians
from arena_simulation_setup.tree.World import WorldIdentifier
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import Point, Quaternion
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import ColorRGBA, String
from task_generator_msgs.msg import SoundEvent
from visualization_msgs.msg import Marker, MarkerArray

from task_generator.auditory.acoustic_scene import AcousticScene
from task_generator.auditory.qos_profiles import (
    acoustic_metadata_qos,
    transient_event_qos,
)
from task_generator.simulators.human.auditory_events import AuditoryEventDetector


class HumanSoundNode(Node):
    """Derive human sounds and RViz cones from the shared pedestrian stream."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__("human_sound_node", **kwargs)
        self.declare_parameter("arena_peds_topic", "arena_peds")
        self.declare_parameter("world_topic", "state/world")
        self.declare_parameter("sound_events_topic", "human_sound_events")
        self.declare_parameter(
            "sound_markers_topic",
            "pedestrian_markers/extra",
        )
        self.declare_parameter("walking_speed_threshold", 0.05)
        self.declare_parameter("footstep_interval_sec", 0.45)
        self.declare_parameter("greeting_distance_m", 1.5)
        self.declare_parameter("greeting_fov_deg", 90.0)
        self.declare_parameter("greeting_cooldown_sec", 5.0)

        self._event_counter = itertools.count()
        self._marker_counter = itertools.count()
        self._acoustic_scene: AcousticScene | None = None
        self._sound_publisher = self.create_publisher(
            SoundEvent,
            str(self.get_parameter("sound_events_topic").value),
            transient_event_qos(),
        )
        self._marker_publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("sound_markers_topic").value),
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self._detector = AuditoryEventDetector(
            self._publish_sound_event,
            walking_speed_threshold=float(
                self.get_parameter("walking_speed_threshold").value
            ),
            footstep_interval_sec=float(
                self.get_parameter("footstep_interval_sec").value
            ),
            greeting_distance_m=float(
                self.get_parameter("greeting_distance_m").value
            ),
            greeting_fov_deg=float(
                self.get_parameter("greeting_fov_deg").value
            ),
            greeting_cooldown_sec=float(
                self.get_parameter("greeting_cooldown_sec").value
            ),
        )
        self._pedestrian_subscription = self.create_subscription(
            Pedestrians,
            str(self.get_parameter("arena_peds_topic").value),
            self._on_pedestrians,
            10,
        )
        self._world_subscription = self.create_subscription(
            String,
            str(self.get_parameter("world_topic").value),
            self._on_world,
            acoustic_metadata_qos(),
        )

    def _on_pedestrians(self, msg: Pedestrians) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self._detector.update(msg, now_sec)

    def _on_world(self, msg: String) -> None:
        world_name = msg.data.strip()
        if not world_name:
            return
        try:
            world = WorldIdentifier(world_name).resolve_sync().load()
            self._acoustic_scene = AcousticScene.from_world(world)
        except Exception as exc:
            self.get_logger().warning(
                "failed to load acoustic scene for footstep material mapping: "
                f"{exc!r}"
            )
            self._acoustic_scene = None

    def _floor_tag(self, ped: Pedestrian) -> str:
        if self._acoustic_scene is None:
            return "default"
        zone = self._acoustic_scene.zone_at(ped.pose.position)
        if zone is None:
            return "default"
        tag = (
            zone.floor_material_id.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        supported = {
            "walnut_planks",
            "oak_planks",
            "marble_tile",
            "smooth_concrete",
            "ceramic_tile",
        }
        return tag if tag in supported else "default"

    def _publish_sound_event(
        self,
        sound_type: str,
        ped: Pedestrian,
    ) -> None:
        sound_type = sound_type.strip()
        if not sound_type:
            return
        stamp = self.get_clock().now().to_msg()
        msg = SoundEvent()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.event_id = (
            f"human:{ped.id}:{stamp.sec}:{stamp.nanosec}:"
            f"{next(self._event_counter)}"
        )
        msg.source_agent_id = int(ped.id)
        msg.source_agent_name = ped.name
        msg.sound_type = sound_type
        msg.label = sound_type
        msg.asset_id = sound_type
        msg.source_position = ped.pose.position
        msg.source_yaw = self._yaw(ped.pose.orientation)
        msg.semantic_tags = (
            ["walk", self._floor_tag(ped)]
            if sound_type == "footstep"
            else ["human", sound_type]
        )
        msg.loop = False
        source_volume_db, duration_sec = {
            "footstep": (45.0, 0.2),
            "greeting": (60.0, 1.0),
        }.get(sound_type, (60.0, 1.0))
        msg.source_volume_db = source_volume_db
        msg.duration.sec = int(duration_sec)
        msg.duration.nanosec = int(
            (duration_sec % 1.0) * 1_000_000_000
        )
        self._sound_publisher.publish(msg)
        self._publish_cone(sound_type, ped, stamp)

    def _publish_cone(
        self,
        sound_type: str,
        ped: Pedestrian,
        stamp: Time,
    ) -> None:
        yaw = self._yaw(ped.pose.orientation)
        source_x = float(ped.pose.position.x)
        source_y = float(ped.pose.position.y)
        apex = Point(
            x=source_x + math.cos(yaw) * 0.15,
            y=source_y + math.sin(yaw) * 0.15,
            z=0.08,
        )
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns = f"human_sound_{sound_type}"
        marker.id = next(self._marker_counter)
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        marker.lifetime = Duration(sec=1, nanosec=200_000_000)
        marker.color = self._color(sound_type)

        arc_points: list[Point] = []
        cone_angle = math.radians(70.0)
        start_angle = yaw - cone_angle / 2.0
        for index in range(17):
            angle = start_angle + cone_angle * index / 16
            arc_points.append(
                Point(
                    x=source_x + math.cos(angle) * 1.25,
                    y=source_y + math.sin(angle) * 1.25,
                    z=0.08,
                )
            )
        for left, right in zip(arc_points, arc_points[1:], strict=False):
            marker.points.extend([apex, left, right])

        outline = Marker()
        outline.header = marker.header
        outline.ns = f"human_sound_{sound_type}_outline"
        outline.id = next(self._marker_counter)
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.pose.orientation.w = 1.0
        outline.scale.x = 0.035
        outline.lifetime = marker.lifetime
        outline.color = ColorRGBA(
            r=marker.color.r,
            g=marker.color.g,
            b=marker.color.b,
            a=0.95,
        )
        outline.points = [apex, *arc_points, apex]
        self._marker_publisher.publish(
            MarkerArray(markers=[marker, outline])
        )

    @staticmethod
    def _color(sound_type: str) -> ColorRGBA:
        if sound_type == "greeting":
            return ColorRGBA(r=0.2, g=0.75, b=1.0, a=0.35)
        if sound_type == "footstep":
            return ColorRGBA(r=1.0, g=0.8, b=0.25, a=0.28)
        return ColorRGBA(r=0.8, g=0.8, b=0.8, a=0.3)

    @staticmethod
    def _yaw(quaternion: Quaternion) -> float:
        siny_cosp = 2.0 * (
            quaternion.w * quaternion.z + quaternion.x * quaternion.y
        )
        cosy_cosp = 1.0 - 2.0 * (
            quaternion.y * quaternion.y + quaternion.z * quaternion.z
        )
        return math.atan2(siny_cosp, cosy_cosp)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = HumanSoundNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
