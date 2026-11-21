"""DisplayKind.IMAGE renderer: sensor_msgs/Image -> JPEG for 8-bit color and mono, 16-bit PNG depth in millimeters for depth, rr.Image otherwise."""

from __future__ import annotations

import io

import numpy as np
import rerun as rr
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec
from PIL import Image as PILImage
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
JPEG_QUALITY = 80
DEPTH_MM = {"16UC1": 1.0, "32FC1": 1000.0}


def depth_png(arr: np.ndarray, scale: float) -> rr.EncodedDepthImage:
    """`arr` scaled to millimeters as a 16-bit PNG, invalid and out-of-range depths as 0."""
    mm = np.nan_to_num(arr.astype(np.float32) * scale, nan=0.0, posinf=0.0, neginf=0.0)
    out = io.BytesIO()
    PILImage.fromarray(np.clip(mm, 0, 65535).astype(np.uint16)).save(out, format="PNG")
    return rr.EncodedDepthImage(out.getvalue(), media_type="image/png", meter=1000.0)


def image_archetype(msg: Image) -> rr.Image | rr.EncodedImage | rr.EncodedDepthImage | None:
    spec = _ENCODING_CHANNELS.get(msg.encoding)
    if spec is None:
        return None
    ch, dt = spec
    arr = np.frombuffer(bytes(msg.data), dtype=dt).reshape(msg.height, msg.width, ch)
    if msg.encoding.startswith("bgr"):
        arr = arr[..., 2::-1]
    elif msg.encoding == "rgba8":
        arr = arr[..., :3]
    if ch == 1:
        arr = arr.squeeze(-1)
    if msg.encoding in DEPTH_MM:
        return depth_png(arr, DEPTH_MM[msg.encoding])
    image = rr.Image(arr)
    return image.compress(jpeg_quality=JPEG_QUALITY) if dt == np.uint8 else image


@register(DisplayKind.IMAGE)
def render_image(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    if not style.enabled:
        return
    entity = display_path(ctx.env_id, robot, d.name)

    def cb(msg: Image) -> None:
        archetype = image_archetype(msg)
        if archetype is not None:
            ctx.stream.log(entity, archetype, stamp=msg.header.stamp)

    qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
    ctx.node.create_subscription(Image, d.topic, ctx.stream.subscriber(d.topic, cb), qos)
