"""Renderer for DisplayKind.TRAJECTORY (moveit_msgs/DisplayTrajectory)."""

from __future__ import annotations

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec

from rviz_utils.renderers._registry import register
from rviz_utils.renderers._registry import rviz_extra as _rviz_extra


@register(DisplayKind.TRAJECTORY)
def render_trajectory(d: AdapterDisplay, robot: RobotDescriptor | None) -> dict[str, object] | None:
    style = StyleSpec.from_json(d.style_json)
    robot_desc = ""
    if robot is not None:
        robot_desc = f"{robot.name}.robot_description"
    name = d.name
    if robot is not None:
        name = f"{robot.name}: {d.name}"
    result: dict[str, object] = {
        "Class": "moveit_rviz_plugin/Trajectory",
        "Name": name,
        "Enabled": style.enabled,
        "Robot Description": robot_desc,
        "Trajectory Topic": {
            "Value": d.topic,
            "Depth": 20,
            "History Policy": "Keep Last",
            "Reliability Policy": "Reliable",
            "Durability Policy": "Volatile",
        },
    }
    result.update(_rviz_extra(style))
    return result
