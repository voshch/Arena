"""Renderer for DisplayKind.LASER_SCAN (sensor_msgs/LaserScan)."""

from __future__ import annotations

import os

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec

from rviz_utils.renderers._registry import register
from rviz_utils.renderers._registry import rviz_extra as _rviz_extra


@register(DisplayKind.LASER_SCAN)
def render_laser_scan(d: AdapterDisplay, robot: RobotDescriptor | None) -> dict[str, object] | None:
    style = StyleSpec.from_json(d.style_json)
    color = "255; 255; 255"
    if style.color is not None:
        r, g, b = style.color
        color = f"{r}; {g}; {b}"
    name = d.name if d.name else f"LaserScan: {os.path.basename(d.topic)}"
    result: dict[str, object] = {
        "Class": "rviz_default_plugins/LaserScan",
        "Name": name,
        "Enabled": style.enabled,
        "Topic": {
            "Value": d.topic,
            "Depth": 20,
            "History Policy": "Keep Last",
            "Reliability Policy": "Best Effort",
            "Durability Policy": "Volatile",
        },
        "Color": color,
        "Size (m)": 0.05,
        "Style": "Points",
        "Alpha": style.alpha,
        "Decay Time": style.decay,
    }
    result.update(_rviz_extra(style))
    return result
