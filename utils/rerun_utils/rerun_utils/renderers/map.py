"""DisplayKind.MAP renderer: nav_msgs/OccupancyGrid → rr.SegmentationImage in map frame."""

from __future__ import annotations

import numpy as np
import rerun as rr
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from rerun_utils.entity_paths import display_path
from rerun_utils.renderers._registry import RendererCtx, register


@register(DisplayKind.MAP)
def render_map(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    if not style.enabled:
        return
    entity = display_path(ctx.env_id, robot, d.name)

    def cb(msg: OccupancyGrid) -> None:
        w, h = msg.info.width, msg.info.height
        data = np.asarray(msg.data, dtype=np.int8).reshape(h, w)
        labels = np.where(data < 0, 0, np.where(data > 50, 2, 1)).astype(np.uint8)
        rr.log(entity, rr.SegmentationImage(labels))

    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    ctx.node.create_subscription(OccupancyGrid, d.topic, cb, qos)
