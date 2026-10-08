"""Pure edit operations on a guided waypoint chain."""

import math

from task_generator.shared import Orientation, Pose, Position

INSERT_AHEAD = 1.0


def scope_prefix(robot_names: list[str]) -> str:
    return f"guided/{'+'.join(sorted(robot_names))}/"


def waypoint_name(scope: str, index: int) -> str:
    return f"{scope}wp/{index}"


def line_name(scope: str) -> str:
    return f"{scope}line"


def delete(waypoints: list[Pose], index: int) -> list[Pose]:
    return [wp for i, wp in enumerate(waypoints) if i != index]


def replace(waypoints: list[Pose], index: int, pose: Pose) -> list[Pose]:
    return [pose if i == index else wp for i, wp in enumerate(waypoints)]


def insert_after(waypoints: list[Pose], index: int) -> list[Pose]:
    """Insert at the midpoint to the next waypoint, or INSERT_AHEAD along the heading after the last."""
    current = waypoints[index]
    if index + 1 < len(waypoints):
        following = waypoints[index + 1]
        x = (current.position.x + following.position.x) / 2
        y = (current.position.y + following.position.y) / 2
    else:
        yaw = current.orientation.to_yaw()
        x = current.position.x + INSERT_AHEAD * math.cos(yaw)
        y = current.position.y + INSERT_AHEAD * math.sin(yaw)
    inserted = Pose(position=Position(x, y), orientation=Orientation.from_yaw(current.orientation.to_yaw()))
    return [*waypoints[: index + 1], inserted, *waypoints[index + 1 :]]
