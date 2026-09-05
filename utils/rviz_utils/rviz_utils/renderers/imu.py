"""Renderer for DisplayKind.IMU (sensor_msgs/Imu)."""

from __future__ import annotations

import functools
import os

from arena_viz import DisplayKind, StyleSpec
from task_generator_msgs.msg import AdapterDisplay, RobotDescriptor

from rviz_utils.renderers._registry import register


@functools.cache
def _imu_display_class() -> str | None:
    # ROS 2 has no Imu display in rviz_default_plugins (checked through jazzy);
    # the only provider is rviz_imu_plugin from imu_tools. Emitting a class that
    # is not installed makes every rviz session log a PluginlibFactory ERROR.
    from ament_index_python.packages import get_package_share_directory

    try:
        get_package_share_directory("rviz_imu_plugin")
    except Exception:
        return None
    return "rviz_imu_plugin/Imu"


@register(DisplayKind.IMU)
def render_imu(d: AdapterDisplay, robot: RobotDescriptor | None) -> dict[str, object] | None:
    display_class = _imu_display_class()
    if display_class is None:
        return None

    style = StyleSpec.from_json(d.style_json)
    color = "204; 51; 204"
    if style.color is not None:
        r, g, b = style.color
        color = f"{r}; {g}; {b}"
    name = d.name if d.name else f"IMU: {os.path.basename(d.topic)}"
    result: dict[str, object] = {
        "Class": display_class,
        "Name": name,
        "Enabled": style.enabled,
        "Topic": {
            "Value": d.topic,
            "Depth": 20,
            "History Policy": "Keep Last",
            "Reliability Policy": "Best Effort",
            "Durability Policy": "Volatile",
        },
        "Axes Length": 0.3,
        "Axes Radius": 0.03,
        "Color": color,
    }
    result.update(style.extra.get("rviz", {}))
    return result
