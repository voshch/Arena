"""DisplayKind.POINTS_3D renderer: sensor_msgs/PointCloud2 → rr.Points3D in sensor frame."""

from __future__ import annotations

import numpy as np
import rerun as rr
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from rerun_utils.entity_paths import tf_path
from rerun_utils.renderers._registry import RendererCtx, register


def _read_xyz(msg: PointCloud2) -> np.ndarray:
    offsets = {f.name: f.offset for f in msg.fields if f.name in ("x", "y", "z")}
    if len(offsets) < 3:
        return np.empty((0, 3), dtype=np.float32)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    pts = np.empty((buf.shape[0], 3), dtype=np.float32)
    for i, axis in enumerate(("x", "y", "z")):
        off = offsets[axis]
        pts[:, i] = buf[:, off : off + 4].copy().view(np.float32).reshape(-1)
    return pts


@register(DisplayKind.POINTS_3D)
def render_points3d(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    if not style.enabled:
        return
    color = style.color or (180, 180, 255)

    def cb(msg: PointCloud2) -> None:
        pts = _read_xyz(msg)
        if pts.size == 0:
            return
        entity = f"{tf_path(ctx.env_id, msg.header.frame_id)}/{d.name}"
        rr.log(entity, rr.Points3D(pts, colors=[color], radii=0.02))

    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
    ctx.node.create_subscription(PointCloud2, d.topic, cb, qos)
    del robot
