"""DisplayKind.LASER_SCAN renderer: sensor_msgs/LaserScan → rr.Points3D in the scan's frame."""

from __future__ import annotations

import math

import rerun as rr
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

from rerun_utils.entity_paths import display_path, in_frame
from rerun_utils.renderers._registry import RendererCtx, register


@register(DisplayKind.LASER_SCAN)
def render_laser_scan(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    if not style.enabled:
        return
    color = style.color or (255, 80, 80)
    entity = display_path(ctx.env_id, robot, d.name)

    def cb(msg: LaserScan) -> None:
        pts: list[tuple[float, float, float]] = []
        ang = msg.angle_min
        for r in msg.ranges:
            if math.isfinite(r) and msg.range_min <= r <= msg.range_max:
                pts.append((r * math.cos(ang), r * math.sin(ang), 0.0))
            ang += msg.angle_increment
        if not pts:
            return
        rr.log(entity, rr.Points3D(pts, colors=[color], radii=0.02), *in_frame(msg.header.frame_id))

    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
    ctx.node.create_subscription(LaserScan, d.topic, cb, qos)
