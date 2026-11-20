"""Shared robot URDF serving for the simulators."""

from collections.abc import Collection, Mapping

import arena_robots.Robot
import arena_robots.Sensor
import attrs
from arena_robots.served_urdf import ServedURDF, serve

from arena_runtime.shared import Robot


def robot_xacro_args(robot: Robot) -> dict[str, object]:
    """The xacro arguments every simulator passes: the robot's extra entries and attributes."""
    args: dict[str, object] = {
        **robot.extra,
        **attrs.asdict(robot, filter=lambda attribute, _: attribute.name != 'extra'),
    }
    args.pop('resolved_assembly', None)
    return args


def render_robot(
    robot: Robot,
    view: arena_robots.Robot.RobotView,
    *,
    optim: Collection[str],
    sensor_prefix: str,
    xacro_args: Mapping[str, object] | None = None,
) -> ServedURDF:
    """The URDF a simulator loads for ``robot``, its sensor topics under ``sensor_prefix``."""
    return serve(
        view,
        robot.resolved_request,
        robot.frames,
        xacro_args={**robot_xacro_args(robot), **(xacro_args or {})},
        optim=optim,
        sensor_topic_patches=arena_robots.Sensor.topic_elements(view.effective_sensors(robot.resolved_request, frames=robot.frames), sensor_prefix),
    )
