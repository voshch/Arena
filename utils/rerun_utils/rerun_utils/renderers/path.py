"""DisplayKind.PATH renderer: nav_msgs/Path → rr.LineStrips3D."""

from __future__ import annotations

import rerun as rr
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec
from nav_msgs.msg import Path

from rerun_utils.entity_paths import display_path
from rerun_utils.renderers._registry import RendererCtx, register


@register(DisplayKind.PATH)
def render_path(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    if not style.enabled:
        return
    entity = display_path(ctx.env_id, robot, d.name)
    color = style.color or (0, 200, 0)
    radius = max(style.line_width / 2.0, 0.005)

    def cb(msg: Path) -> None:
        pts = [(p.pose.position.x, p.pose.position.y, p.pose.position.z) for p in msg.poses]
        if not pts:
            return
        rr.log(entity, rr.LineStrips3D([pts], colors=[color], radii=[radius]))

    ctx.node.create_subscription(Path, d.topic, cb, 10)
