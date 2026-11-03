"""DisplayKind.FOOT_CONTACT renderer: log contact as scalar series. Skipped if msg type unavailable."""

from __future__ import annotations

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind

from rerun_utils.renderers._registry import RendererCtx, register


@register(DisplayKind.FOOT_CONTACT)
def render_foot_contact(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    ctx.node.get_logger().info(f"FOOT_CONTACT renderer not implemented; skipping {d.name!r} on {d.topic!r}")
    del robot
