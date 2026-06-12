"""Typed task requests submitted to robots."""

from __future__ import annotations

import collections
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Literal

import attrs
import geometry_msgs.msg
from arena_robots.task_kinds import TaskKind

from task_generator.shared import Pose

if TYPE_CHECKING:
    from task_generator.manager.robot_manager.robot_manager import RobotManager


@attrs.define
class TaskPhase(ABC):
    """One typed step within a :class:`TaskRequest`."""

    kind: ClassVar[TaskKind]

    on_failure: Literal["continue", "stop_task", "abort_episode"] = attrs.field(default="continue", kw_only=True)
    """Phase failure disposition: advance to next phase, stop the task, or abort the episode."""

    @abstractmethod
    def is_satisfied(self, robot_manager: RobotManager) -> bool:
        """Tier-3 default completion check; must not raise when pose is unavailable."""


@attrs.define
class GoToPhase(TaskPhase):
    """Navigate-to-pose phase with optional per-phase tolerance overrides."""

    kind: ClassVar[TaskKind] = TaskKind.GOTO_POSE

    pose: Pose
    tolerance_radius: float | None = None
    tolerance_angle: float | None = None

    def is_satisfied(self, robot_manager: RobotManager) -> bool:
        pose = robot_manager.pose
        if pose is None:
            return False

        conf = robot_manager.node.conf.Robot
        tol_dist = self.tolerance_radius if self.tolerance_radius is not None else conf.GOAL_TOLERANCE_RADIUS.value
        tol_ang = self.tolerance_angle if self.tolerance_angle is not None else conf.GOAL_TOLERANCE_ANGLE.value

        dx = pose.position.x - self.pose.position.x
        dy = pose.position.y - self.pose.position.y
        if math.hypot(dx, dy) > tol_dist:
            return False

        if tol_ang > 0 and robot_manager.controls_orientation:
            dyaw = pose.orientation.to_yaw() - self.pose.orientation.to_yaw()
            dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi
            if abs(dyaw) > tol_ang:
                return False

        return True


@attrs.define
class ReachPhase(TaskPhase):
    kind: ClassVar[TaskKind] = TaskKind.REACH_POSE
    target: geometry_msgs.msg.PoseStamped | None = None
    named_target: str | None = None
    random: bool = False
    position_tolerance: float | None = None
    orientation_tolerance: float | None = None
    planning_time: float | None = None

    def __attrs_post_init__(self):
        if sum([self.target is not None, self.named_target is not None, self.random]) != 1:
            raise ValueError("ReachPhase requires exactly one of target / named_target / random")

    def is_satisfied(self, robot_manager: RobotManager) -> bool:
        return False


@attrs.define
class PlayGesturePhase(TaskPhase):
    kind: ClassVar[TaskKind] = TaskKind.PLAY_GESTURE
    gesture: str | None = None  # None means random; adapter expands before dispatch

    def is_satisfied(self, robot_manager: RobotManager) -> bool:
        return False  # client-driven completion


DonePredicate = Callable[
    ["RobotManager", TaskPhase],
    bool | None,
]


@attrs.define
class TaskRequest:
    """Typed sequence of phases submitted to a robot."""

    phases: collections.abc.Sequence[TaskPhase]
    done_predicate: DonePredicate | None = None

    @property
    def kind(self) -> TaskKind | None:
        """Single homogeneous kind of all phases, or None if mixed/empty."""
        if not self.phases:
            return None
        first = self.phases[0].kind
        for phase in self.phases[1:]:
            if phase.kind is not first:
                return None
        return first


__all__ = [
    "TaskKind",
    "TaskPhase",
    "GoToPhase",
    "ReachPhase",
    "PlayGesturePhase",
    "TaskRequest",
    "DonePredicate",
]
