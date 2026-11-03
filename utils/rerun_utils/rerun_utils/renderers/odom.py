"""DisplayKind.ODOM renderer: nav_msgs/Odometry → rr.Arrows3D at current robot pose."""

from __future__ import annotations

import rerun as rr
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy

from rerun_utils.entity_paths import display_path
from rerun_utils.renderers._registry import RendererCtx, register
from rerun_utils.renderers.pose import _yaw_to_dir


@register(DisplayKind.ODOM)
def render_odom(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    if not style.enabled:
        return
    entity = display_path(ctx.env_id, robot, d.name)
    color = style.color or (0, 200, 255)

    def cb(msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        rr.log(
            entity,
            rr.Arrows3D(
                origins=[(p.x, p.y, p.z)],
                vectors=[_yaw_to_dir(q.x, q.y, q.z, q.w)],
                colors=[color],
            ),
        )

    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    ctx.node.create_subscription(Odometry, d.topic, cb, qos)
