"""DisplayKind.IMU renderer: sensor_msgs/Imu → scalar time series (lin accel + ang vel)."""

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

    def cb(msg: Imu) -> None:
        a = msg.linear_acceleration
        w = msg.angular_velocity
        rr.log(f"{base}/accel/x", rr.Scalars(a.x))
        rr.log(f"{base}/accel/y", rr.Scalars(a.y))
        rr.log(f"{base}/accel/z", rr.Scalars(a.z))
        rr.log(f"{base}/gyro/x", rr.Scalars(w.x))
        rr.log(f"{base}/gyro/y", rr.Scalars(w.y))
        rr.log(f"{base}/gyro/z", rr.Scalars(w.z))

    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
    ctx.node.create_subscription(Imu, d.topic, cb, qos)
