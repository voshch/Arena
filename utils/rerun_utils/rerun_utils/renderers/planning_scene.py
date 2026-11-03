"""DisplayKind.PLANNING_SCENE renderer: MoveIt PlanningScene has no native rerun equivalent yet."""

from __future__ import annotations

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind

from rerun_utils.renderers._registry import RendererCtx, register


@register(DisplayKind.PLANNING_SCENE)
def render_planning_scene(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    ctx.node.get_logger().info(f"PLANNING_SCENE renderer not implemented (MoveIt PlanningScene); skipping {d.name!r}")
    del robot
