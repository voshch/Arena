"""DisplayKind.PEDESTRIANS renderer: the env's arena_people_msgs/Pedestrians (`<env>/arena_peds`) → standing capsules.

The display's own topic is the ROS4HRI `humans` namespace, which rviz draws as skeletons; rerun has no skeleton
plugin, so this reads the env's pedestrian roster instead: one capsule per person in the roster's frame, labelled with
the name and agent type, its walking direction as an arrow.
"""

from __future__ import annotations

import rerun as rr
from arena_people_msgs.msg import Pedestrian, Pedestrians
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec
from rclpy.qos import QoSProfile, ReliabilityPolicy

from rerun_utils.entity_paths import display_path, in_frame
from rerun_utils.renderers._registry import RendererCtx, register

RADIUS, HEIGHT = 0.22, 1.7
WALKING, IDLE = (60, 130, 230), (150, 150, 160)


def _label(p: Pedestrian) -> str:
    name = p.name.rsplit("/", 1)[-1]
    return f"{name} ({p.agent_type})" if p.agent_type else name


def body_centers(peds: list[Pedestrian]) -> list[list[float]]:
    return [[p.pose.position.x, p.pose.position.y, p.pose.position.z + RADIUS] for p in peds]


def heading_origins(peds: list[Pedestrian]) -> list[list[float]]:
    return [[p.pose.position.x, p.pose.position.y, p.pose.position.z + HEIGHT * 0.6] for p in peds]


@register(DisplayKind.PEDESTRIANS)
def render_pedestrians(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    if not style.enabled:
        return
    entity = display_path(ctx.env_id, robot, d.name)
    color = style.color or WALKING

    def cb(msg: Pedestrians) -> None:
        peds = list(msg.pedestrians)
        frame = in_frame(msg.header.frame_id)
        if not peds:
            rr.log(entity, rr.Clear(recursive=True))
            return
        rr.log(
            f"{entity}/bodies",
            rr.Capsules3D(
                lengths=[HEIGHT - 2 * RADIUS] * len(peds),
                radii=[RADIUS] * len(peds),
                translations=body_centers(peds),
                colors=[color if p.animation_state != Pedestrian.IDLE else IDLE for p in peds],
                labels=[_label(p) for p in peds],
                fill_mode="solid",
            ),
            *frame,
        )
        rr.log(
            f"{entity}/headings",
            rr.Arrows3D(
                origins=heading_origins(peds),
                vectors=[[p.twist.linear.x * 0.6, p.twist.linear.y * 0.6, 0.0] for p in peds],
                colors=[color],
            ),
            *frame,
        )

    ctx.node.create_subscription(Pedestrians, peds_topic(ctx.env_namespace), cb, QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))


def peds_topic(env_namespace: str) -> str:
    """The env's pedestrian roster: `<env namespace>/arena_peds`, published there whatever the tf mode."""
    return f"{env_namespace.rstrip('/')}/arena_peds"
