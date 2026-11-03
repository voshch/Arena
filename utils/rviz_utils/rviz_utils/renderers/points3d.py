"""Renderer for DisplayKind.POINTS_3D (sensor_msgs/PointCloud2, legacy PointCloud)."""

from __future__ import annotations

import os

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec

from rviz_utils.renderers._registry import register
from rviz_utils.renderers._registry import rviz_extra as _rviz_extra

_LEGACY_TYPE = "sensor_msgs/msg/PointCloud"


@register(DisplayKind.POINTS_3D)
def render_points3d(d: AdapterDisplay, robot: RobotDescriptor | None) -> dict[str, object] | None:
    style = StyleSpec.from_json(d.style_json)
    color = "255; 255; 255"
    if style.color is not None:
        r, g, b = style.color
        color = f"{r}; {g}; {b}"
    name = d.name if d.name else f"PointCloud: {os.path.basename(d.topic)}"
    use_legacy = d.topic_type == _LEGACY_TYPE
    if use_legacy:
        result: dict[str, object] = {
            "Class": "rviz_default_plugins/PointCloud",
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
            "Size (m)": 0.03,
            "Alpha": style.alpha,
            "Decay Time": style.decay,
        }
    else:
        result = {
            "Class": "rviz_default_plugins/PointCloud2",
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
            "Size (m)": 0.03,
            "Style": "Flat Squares",
            "Alpha": style.alpha,
            "Decay Time": style.decay,
        }
    result.update(_rviz_extra(style))
    return result
