"""Renderer for DisplayKind.MARKER_ARRAY: generic visualization_msgs/MarkerArray passthrough.

Unlike DisplayKind.PEDESTRIANS this makes no namespace assumptions, so it suits debug
overlays and static-geometry layers that are not pedestrian data.
"""

from __future__ import annotations

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec

from rviz_utils.renderers._registry import register
from rviz_utils.renderers._registry import rviz_extra as _rviz_extra


@register(DisplayKind.MARKER_ARRAY)
def render_marker_array(d: AdapterDisplay, robot: RobotDescriptor | None) -> dict[str, object] | None:
    style = StyleSpec.from_json(d.style_json)
    result: dict[str, object] = {
        "Class": "rviz_default_plugins/MarkerArray",
        "Name": d.name,
        "Enabled": style.enabled,
        "Topic": {
            "Value": d.topic,
            "Depth": 20,
            "History Policy": "Keep Last",
            "Reliability Policy": "Reliable" if style.latched else "Best Effort",
            "Durability Policy": "Transient Local" if style.latched else "Volatile",
        },
        "Value": True,
    }
    result.update(_rviz_extra(style))
    return result
