"""Renderer for DisplayKind.PATH (nav_msgs/Path)."""

from __future__ import annotations

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec

from rviz_utils.renderers._registry import register
from rviz_utils.renderers._registry import rviz_extra as _rviz_extra


@register(DisplayKind.PATH)
def render_path(d: AdapterDisplay, robot: RobotDescriptor | None) -> dict[str, object] | None:
    style = StyleSpec.from_json(d.style_json)
    color = "255; 0; 0"
    if style.color is not None:
        r, g, b = style.color
        color = f"{r}; {g}; {b}"
    name = d.name
    if robot is not None:
        name = f"{robot.name}: {d.name}"
    result: dict[str, object] = {
        "Class": "rviz_default_plugins/Path",
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
        "Line Width": style.line_width,
    }
    result.update(_rviz_extra(style))
    return result
