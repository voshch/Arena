"""Renderer for DisplayKind.INTERACTIVE_MARKERS: the topic is the marker server namespace."""

from __future__ import annotations

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec

from rviz_utils.renderers._registry import register
from rviz_utils.renderers._registry import rviz_extra as _rviz_extra


@register(DisplayKind.INTERACTIVE_MARKERS)
def render_interactive_markers(d: AdapterDisplay, robot: RobotDescriptor | None) -> dict[str, object] | None:
    style = StyleSpec.from_json(d.style_json)
    result: dict[str, object] = {
        "Class": "rviz_default_plugins/InteractiveMarkers",
        "Name": d.name,
        "Enabled": style.enabled,
        "Interactive Markers Namespace": d.topic,
        "Show Descriptions": True,
        "Show Axes": False,
        "Show Visual Aids": False,
        "Enable Transparency": True,
        "Value": True,
    }
    result.update(_rviz_extra(style))
    return result
