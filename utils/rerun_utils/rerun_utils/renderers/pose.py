"""DisplayKind.POSE renderer: geometry_msgs/PoseStamped → rr.Arrows3D (goal/subgoal)."""

from __future__ import annotations

import rerun as rr
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec
from geometry_msgs.msg import PoseStamped

from rerun_utils.entity_paths import display_path
from rerun_utils.renderers._registry import RendererCtx, register


def _yaw_to_dir(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """Forward direction = unit X rotated by the quaternion (z-up world)."""
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fy = 2.0 * (qx * qy + qz * qw)
    fz = 2.0 * (qx * qz - qy * qw)
    return (fx, fy, fz)


@register(DisplayKind.POSE)
def render_pose(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    if not style.enabled:
        return
    entity = display_path(ctx.env_id, robot, d.name)
    color = style.color or (255, 200, 0)

    def cb(msg: PoseStamped) -> None:
        p = msg.pose.position
        q = msg.pose.orientation
        rr.log(
            entity,
            rr.Arrows3D(
                origins=[(p.x, p.y, p.z)],
                vectors=[_yaw_to_dir(q.x, q.y, q.z, q.w)],
                colors=[color],
            ),
        )

    ctx.node.create_subscription(PoseStamped, d.topic, cb, 10)
