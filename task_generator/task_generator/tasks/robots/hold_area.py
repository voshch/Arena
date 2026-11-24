"""The move the hold and random task modes end on: drive to a goal, then stay in its target disc for a set time."""

from __future__ import annotations

import math

from arena_runtime.env.task_api import GoToPhase
from arena_runtime.shared import Pose

# A heading tolerance of pi accepts every heading, so the goal's heading never counts.
ANY_HEADING = math.pi


def hold_within(goal: Pose, radius: float, seconds: float) -> GoToPhase:
    """Drive to goal, then stay within radius of it for seconds.

    radius is the goal tolerance, the disc a goal is reached in: the count starts when the robot enters it and restarts whenever it leaves.
    """
    return GoToPhase(
        pose=goal,
        tolerance_radius=radius,
        tolerance_angle=ANY_HEADING,
        hold_time=0.0,
        until=f"robot within {radius:g} of {goal.position.x:.6f},{goal.position.y:.6f} for {seconds:g}",
    )
