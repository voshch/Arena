"""Renderer for DisplayKind.POSE (geometry_msgs/PoseStamped)."""

from __future__ import annotations

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec

from rviz_utils.renderers._registry import register
from rviz_utils.renderers._registry import rviz_extra as _rviz_extra


@register(DisplayKind.POSE)
def render_pose(d: AdapterDisplay, robot: RobotDescriptor | None) -> dict[str, object] | None:
    style = StyleSpec.from_json(d.style_json)
    color = "255; 85; 255"
    if style.color is not None:
        r, g, b = style.color
        color = f"{r}; {g}; {b}"
    name = d.name
    if robot is not None:
        name = f"{robot.name}: {d.name}"
    result: dict[str, object] = {
        "Class": "rviz_default_plugins/Pose",
        "Name": name,
        "Enabled": style.enabled,
        "Topic": {
            "Value": d.topic,
            "Depth": 20,
            "History Policy": "Keep Last",
            "Reliability Policy": "Reliable",
            "Durability Policy": "Volatile",
        },
        "Color": color,
        "Alpha": style.alpha,
        "Axes Length": 1.0,
        "Axes Radius": 0.1,
        "Head Length": 0.1,
        "Head Radius": 0.15,
        "Shaft Length": 0.5,
        "Shaft Radius": 0.03,
        "Shape": "Arrow",
    }
    result.update(_rviz_extra(style))
    return result
