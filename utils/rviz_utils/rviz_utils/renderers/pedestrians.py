"""Renderer for DisplayKind.PEDESTRIANS (hri_rviz/Skeletons3D)."""

from __future__ import annotations

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec

from rviz_utils.renderers._registry import register
from rviz_utils.renderers._registry import rviz_extra as _rviz_extra


@register(DisplayKind.PEDESTRIANS)
def render_pedestrians(d: AdapterDisplay, robot: RobotDescriptor | None) -> dict[str, object] | None:
    style = StyleSpec.from_json(d.style_json)
    rviz_extra = _rviz_extra(style)
    result: dict[str, object] = {
        "Class": "hri_rviz/Skeletons3D",
        "Name": d.name,
        "Enabled": style.enabled,
        "Alpha": rviz_extra.get("Alpha", 1.0),
        "Update Interval": rviz_extra.get("Update Interval", 0),
        "Visual Enabled": rviz_extra.get("Visual Enabled", True),
        "Collision Enabled": rviz_extra.get("Collision Enabled", False),
        "Tf Prefix": rviz_extra.get("Tf Prefix", ""),
        "Namespace": d.topic.removesuffix("/humans"),
    }
    for k, v in rviz_extra.items():
        if k not in ("Alpha", "Update Interval", "Visual Enabled", "Collision Enabled", "Tf Prefix", "Namespace"):
            result[k] = v
    return result
