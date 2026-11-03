"""DisplayKind.ROBOT_MODEL renderer: log robot_description URDF as rr.Asset3D.

Requires `rerun-loader-urdf-python` (optional install: `pip install rerun-loader-urdf-python`).
Without it, the renderer logs a one-time warning and skips.
"""

from __future__ import annotations

import tempfile

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from rerun_utils.entity_paths import display_path
from rerun_utils.renderers._registry import RendererCtx, register

try:
    from rerun_loader_urdf_python.urdf_logger import URDFLogger  # type: ignore

    _HAVE_URDF = True
except ImportError:
    URDFLogger = None
    _HAVE_URDF = False


@register(DisplayKind.ROBOT_MODEL)
def render_robot_model(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    if not style.enabled:
        return
    if not _HAVE_URDF:
        ctx.node.get_logger().warning(f"rerun-loader-urdf-python not installed; skipping ROBOT_MODEL for {d.name!r}")
        return
    entity = display_path(ctx.env_id, robot, d.name)
    logged: dict[str, bool] = {"done": False}

    def cb(msg: String) -> None:
        if logged["done"]:
            return
        with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
            f.write(msg.data)
            urdf_path = f.name
        assert URDFLogger is not None
        try:
            logger = URDFLogger(urdf_path, entity_path_prefix=entity)
            logger.log()
            logged["done"] = True
        except Exception as e:
            ctx.node.get_logger().error(f"URDF log failed for {d.name!r}: {e}")

    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    ctx.node.create_subscription(String, d.topic, cb, qos)
