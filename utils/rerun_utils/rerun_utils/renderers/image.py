"""DisplayKind.IMAGE renderer: sensor_msgs/Image → rr.Image."""

from __future__ import annotations

import numpy as np
import rerun as rr
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from rerun_utils.entity_paths import display_path
from rerun_utils.renderers._registry import RendererCtx, register

_ENCODING_CHANNELS = {
    "rgb8": (3, np.uint8),
    "rgba8": (4, np.uint8),
    "bgr8": (3, np.uint8),
    "bgra8": (4, np.uint8),
    "mono8": (1, np.uint8),
    "mono16": (1, np.uint16),
    "16UC1": (1, np.uint16),
    "32FC1": (1, np.float32),
}


@register(DisplayKind.IMAGE)
def render_image(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    if not style.enabled:
        return
    entity = display_path(ctx.env_id, robot, d.name)

    def cb(msg: Image) -> None:
        spec = _ENCODING_CHANNELS.get(msg.encoding)
        if spec is None:
            return
        ch, dt = spec
        arr = np.frombuffer(bytes(msg.data), dtype=dt).reshape(msg.height, msg.width, ch)
        if msg.encoding.startswith("bgr"):
            arr = arr[..., ::-1]
        if ch == 1:
            arr = arr.squeeze(-1)
        rr.log(entity, rr.Image(arr))

    qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
    ctx.node.create_subscription(Image, d.topic, cb, qos)
