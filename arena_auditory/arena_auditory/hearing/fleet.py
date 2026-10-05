"""Bind a hearing node to one robot announced on the task generator's fleet topic."""

from __future__ import annotations

import attrs
from arena_robots.Robot import RobotIdentifier
from task_generator_msgs.msg import RobotFleet

FLEET_SUFFIX = "/state/robots"


@attrs.frozen
class RobotBinding:
    name: str
    tg_node: str
    base_frame: str
    max_linear_mps: float


def bind_robot(fleet: RobotFleet, wanted: str, fleet_topic: str) -> RobotBinding | None:
    """The fleet entry named ``wanted``, or the first one when empty. None until the fleet carries it."""
    for state in fleet.robots:
        robot = state.descriptor
        name = str(robot.name).strip()
        if not name or (wanted and name != wanted):
            continue
        view = RobotIdentifier(str(robot.model)).resolve_sync()
        base = view.model_params.base_frame.strip("/")
        prefix = str(robot.frame).strip("/")
        mobile = view.mobile
        limits = mobile.velocity_limits if mobile is not None else None
        return RobotBinding(
            name=name,
            tg_node=fleet_topic.removesuffix(FLEET_SUFFIX),
            base_frame="/".join(part for part in (prefix, base) if part),
            max_linear_mps=float(limits.linear.max) if limits is not None else 0.0,
        )
    return None
