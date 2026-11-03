"""Renderer for DisplayKind.ODOM (nav_msgs/Odometry)."""

from __future__ import annotations

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec

from rviz_utils.renderers._registry import register
from rviz_utils.renderers._registry import rviz_extra as _rviz_extra


def _color_str(style: StyleSpec, fallback: str = "255; 255; 255") -> str:
    if style.color is None:
        return fallback
    r, g, b = style.color
    return f"{r}; {g}; {b}"


@register(DisplayKind.ODOM)
def render_odom(d: AdapterDisplay, robot: RobotDescriptor | None) -> dict[str, object] | None:
    style = StyleSpec.from_json(d.style_json)
    name = d.name
    if robot is not None:
        name = f"{robot.name}: {d.name}"
    result: dict[str, object] = {
        "Class": "rviz_default_plugins/Odometry",
        "Name": name,
        "Enabled": style.enabled,
        "Topic": {
            "Value": d.topic,
            "Depth": 20,
            "History Policy": "Keep Last",
            "Reliability Policy": "Reliable",
            "Durability Policy": "Volatile",
        },
        "Shape": "Arrow",
        "Color": _color_str(style),
        "Position Tolerance": 0.1,
        "Angle Tolerance": 0.1,
        "Keep": 1,
        "Shaft Length": 0.5,
        "Shaft Radius": 0.05,
        "Head Length": 0.2,
        "Head Radius": 0.1,
        "Covariance": {"Value": False},
    }
    result.update(_rviz_extra(style))
    return result
