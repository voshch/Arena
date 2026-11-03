"""Renderer for DisplayKind.FOOT_CONTACT (foot_contact_msgs/FootContact)."""

from __future__ import annotations

import os

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec

from rviz_utils.renderers._registry import register
from rviz_utils.renderers._registry import rviz_extra as _rviz_extra


@register(DisplayKind.FOOT_CONTACT)
def render_foot_contact(d: AdapterDisplay, robot: RobotDescriptor | None) -> dict[str, object] | None:
    style = StyleSpec.from_json(d.style_json)
    color = "255; 140; 0"
    if style.color is not None:
        r, g, b = style.color
        color = f"{r}; {g}; {b}"
    name = d.name if d.name else f"FootContact: {os.path.basename(d.topic)}"
    result: dict[str, object] = {
        "Class": "rviz_default_plugins/Marker",
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
    }
    result.update(_rviz_extra(style))
    return result
