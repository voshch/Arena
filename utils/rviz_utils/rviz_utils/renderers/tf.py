"""Renderer for DisplayKind.TF (transform tree)."""

from __future__ import annotations

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec

from rviz_utils.renderers._registry import register
from rviz_utils.renderers._registry import rviz_extra as _rviz_extra


@register(DisplayKind.TF)
def render_tf(d: AdapterDisplay, robot: RobotDescriptor | None) -> dict[str, object] | None:
    style = StyleSpec.from_json(d.style_json)
    result: dict[str, object] = {
        "Class": "rviz_default_plugins/TF",
        "Name": d.name,
        "Enabled": style.enabled,
        "Frame Timeout": 15,
        "Marker Scale": 1.0,
        "Show Arrows": True,
        "Show Axes": True,
        "Show Names": False,
    }
    result.update(_rviz_extra(style))
    return result
