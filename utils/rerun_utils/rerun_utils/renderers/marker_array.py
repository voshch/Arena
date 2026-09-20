"""DisplayKind.MARKER_ARRAY renderer: generic visualization_msgs/MarkerArray → rerun.

Makes no pedestrian-namespace assumptions, so it suits debug overlays and static-geometry
layers. Also exposes `subscribe_marker_array`, reused by the PEDESTRIANS renderer.
"""

from __future__ import annotations

import rerun as rr
from arena_viz import DisplayKind, StyleSpec
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from task_generator_msgs.msg import AdapterDisplay, RobotDescriptor
from visualization_msgs.msg import Marker, MarkerArray

from rerun_utils.entity_paths import display_path
from rerun_utils.renderers._registry import RendererCtx, register


def _marker_color(m: Marker) -> tuple[int, int, int, int]:
    c = m.color
    return (int(c.r * 255), int(c.g * 255), int(c.b * 255), int(c.a * 255))


def subscribe_marker_array(ctx: RendererCtx, d: AdapterDisplay, base: str) -> None:
    """Subscribe to a MarkerArray topic and log box-like markers under `base`."""

    def cb(msg: MarkerArray) -> None:
        centers: list[tuple[float, float, float]] = []
        sizes: list[tuple[float, float, float]] = []
        colors: list[tuple[int, int, int, int]] = []
        for m in msg.markers:
            if m.type in (Marker.CYLINDER, Marker.CUBE):
                centers.append((m.pose.position.x, m.pose.position.y, m.pose.position.z))
                sizes.append((m.scale.x, m.scale.y, m.scale.z))
                colors.append(_marker_color(m))
        if centers:
            rr.log(f"{base}/bodies", rr.Boxes3D(centers=centers, sizes=sizes, colors=colors))

    if StyleSpec.from_json(d.style_json).latched:
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    else:
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
    ctx.node.create_subscription(MarkerArray, d.topic, cb, qos)


@register(DisplayKind.MARKER_ARRAY)
def render_marker_array(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    if not style.enabled:
        return
    subscribe_marker_array(ctx, d, display_path(ctx.env_id, robot, d.name))
