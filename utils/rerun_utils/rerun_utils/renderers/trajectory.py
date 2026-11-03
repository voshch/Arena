"""DisplayKind.TRAJECTORY renderer: MoveIt DisplayTrajectory has no native rerun equivalent yet."""

from __future__ import annotations

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind

from rerun_utils.renderers._registry import RendererCtx, register


@register(DisplayKind.TRAJECTORY)
def render_trajectory(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    ctx.node.get_logger().info(f"TRAJECTORY renderer not implemented (MoveIt DisplayTrajectory); skipping {d.name!r}")
    del robot
