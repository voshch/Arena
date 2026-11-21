"""DisplayKind.IMU renderer: sensor_msgs/Imu -> two scalar time series, linear acceleration and angular velocity per axis."""

from __future__ import annotations

import rerun as rr
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu

from rerun_utils.entity_paths import display_path
from rerun_utils.renderers._registry import RendererCtx, register


@register(DisplayKind.IMU)
def render_imu(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    if not style.enabled:
        return
    base = display_path(ctx.env_id, robot, d.name)

    for part in ("accel", "gyro"):
        ctx.stream.fixed(f"{base}/{part}", rr.SeriesLines(names=["x", "y", "z"]))

    def cb(msg: Imu) -> None:
        a = msg.linear_acceleration
        w = msg.angular_velocity
        ctx.stream.log(f"{base}/accel", rr.Scalars([a.x, a.y, a.z]), stamp=msg.header.stamp)
        ctx.stream.log(f"{base}/gyro", rr.Scalars([w.x, w.y, w.z]), stamp=msg.header.stamp)

    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
    ctx.node.create_subscription(Imu, d.topic, ctx.stream.subscriber(d.topic, cb), qos)
