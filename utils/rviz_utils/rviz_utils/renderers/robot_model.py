"""Renderer for DisplayKind.ROBOT_MODEL (std_msgs/String robot_description)."""

from __future__ import annotations

from arena_rclpy_mixins.shared import FrameNamespace
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec

from rviz_utils.renderers._registry import register
from rviz_utils.renderers._registry import rviz_extra as _rviz_extra


@register(DisplayKind.ROBOT_MODEL)
def render_robot_model(d: AdapterDisplay, robot: RobotDescriptor | None) -> dict[str, object] | None:
    style = StyleSpec.from_json(d.style_json)
    tf_prefix = ""
    if robot is not None:
        tf_prefix = FrameNamespace(robot.frame).raw()
    name = d.name
    if robot is not None:
        name = f"{robot.name}: {d.name}"
    result: dict[str, object] = {
        "Class": "rviz_default_plugins/RobotModel",
        "Name": name,
        "Enabled": style.enabled,
        "TF Prefix": tf_prefix,
        "Description Topic": {
            "Value": d.topic,
            "Depth": 20,
            "History Policy": "Keep Last",
            "Reliability Policy": "Reliable",
            "Durability Policy": "Transient Local",
        },
        "Visual Enabled": True,
        "Collision Enabled": False,
    }
    result.update(_rviz_extra(style))
    return result
