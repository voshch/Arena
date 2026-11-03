"""Renderer for DisplayKind.IMAGE (sensor_msgs/Image)."""

from __future__ import annotations

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec

from rviz_utils.renderers._registry import register
from rviz_utils.renderers._registry import rviz_extra as _rviz_extra


@register(DisplayKind.IMAGE)
def render_image(d: AdapterDisplay, robot: RobotDescriptor | None) -> dict[str, object] | None:
    style = StyleSpec.from_json(d.style_json)
    result: dict[str, object] = {
        "Class": "rviz_default_plugins/Image",
        "Name": d.name,
        "Enabled": style.enabled,
        "Topic": {
            "Value": d.topic,
            "Depth": 20,
            "History Policy": "Keep Last",
            "Reliability Policy": "Reliable",
            "Durability Policy": "Volatile",
        },
        "Max Value": 1,
        "Median window": 5,
        "Min Value": 0,
        "Normalize Range": True,
        "Value": True,
    }
    result.update(_rviz_extra(style))
    return result
