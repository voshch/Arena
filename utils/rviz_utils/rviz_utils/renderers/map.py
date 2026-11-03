"""Renderer for DisplayKind.MAP (nav_msgs/OccupancyGrid)."""

from __future__ import annotations

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec

from rviz_utils.renderers._registry import register
from rviz_utils.renderers._registry import rviz_extra as _rviz_extra


@register(DisplayKind.MAP)
def render_map(d: AdapterDisplay, robot: RobotDescriptor | None) -> dict[str, object] | None:
    style = StyleSpec.from_json(d.style_json)
    rviz_extra = _rviz_extra(style)
    color_scheme = rviz_extra.get("Color Scheme", "map")
    durability = rviz_extra.get("Durability Policy", "Transient Local")
    reliability = rviz_extra.get("Reliability Policy", "Reliable")
    topic: dict[str, object] = {
        "Value": d.topic,
        "Depth": 20,
        "History Policy": "Keep Last",
        "Reliability Policy": reliability,
        "Durability Policy": durability,
    }
    topic_override = rviz_extra.get("Topic")
    if isinstance(topic_override, dict):
        topic.update(topic_override)
    result: dict[str, object] = {
        "Class": "rviz_default_plugins/Map",
        "Name": d.name,
        "Enabled": style.enabled,
        "Topic": topic,
        "Alpha": style.alpha,
        "Color Scheme": color_scheme,
        "Draw Behind": color_scheme == "map",
        "Use Timestamp": False,
    }
    for k, v in rviz_extra.items():
        if k not in ("Color Scheme", "Durability Policy", "Reliability Policy", "Topic"):
            result[k] = v
    return result
