"""DisplayKind.POLYGON renderer: geometry_msgs/PolygonStamped → rr.LineStrips3D (closed loop)."""

from __future__ import annotations

import rerun as rr
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec
from geometry_msgs.msg import PolygonStamped

from rerun_utils.entity_paths import display_path, in_frame
from rerun_utils.renderers._registry import RendererCtx, register


@register(DisplayKind.POLYGON)
def render_polygon(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    if not style.enabled:
        return
    entity = display_path(ctx.env_id, robot, d.name)
    color = style.color or (255, 255, 0)
    radius = max(style.line_width / 2.0, 0.005)

    def cb(msg: PolygonStamped) -> None:
        pts = [(p.x, p.y, p.z) for p in msg.polygon.points]
        if len(pts) < 2:
            return
        pts.append(pts[0])
        rr.log(entity, rr.LineStrips3D([pts], colors=[color], radii=[radius]), *in_frame(msg.header.frame_id))

    ctx.node.create_subscription(PolygonStamped, d.topic, cb, 10)
