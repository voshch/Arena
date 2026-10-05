"""Typed task requests submitted to robots."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Literal

import attrs
import geometry_msgs.msg
from arena_robots.task_kinds import TaskKind

from task_generator.shared import Pose
from task_generator.utils.park import ParkTimer

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

    def with_defaults(self, robot_manager: RobotManager) -> TaskPhase:
        """This phase with every field it left unset taken from the node's launch parameters."""
        return self


@attrs.define
class GoToPhase(TaskPhase):
    """Navigate-to-pose phase, unset tolerances, hold time and signal take the node's goal_* launch parameters on submit."""

    kind: ClassVar[TaskKind] = TaskKind.GOTO_POSE

    pose: Pose
    tolerance_radius: float | None = None
    tolerance_angle: float | None = None
    instruction: str = ""  # natural-language goal for language-conditioned planners, empty = none
    hold_time: float | None = None
    signal: str | None = None
    _park: ParkTimer = attrs.field(factory=ParkTimer, init=False, eq=False, repr=False)

    def with_defaults(self, robot_manager: RobotManager) -> GoToPhase:
        conf = robot_manager.node.conf.Robot
        return attrs.evolve(
            self,
            tolerance_radius=float(conf.GOAL_TOLERANCE_RADIUS.value if self.tolerance_radius is None else self.tolerance_radius),
            tolerance_angle=float(conf.GOAL_TOLERANCE_ANGLE.value if self.tolerance_angle is None else self.tolerance_angle),
            hold_time=float(conf.GOAL_HOLD_TIME.value if self.hold_time is None else self.hold_time),
            signal=str(conf.GOAL_SIGNAL.value if self.signal is None else self.signal),
        )

    def is_satisfied(self, robot_manager: RobotManager) -> bool:
        assert self.hold_time is not None and self.signal is not None, "GoToPhase checked before with_defaults filled it"
        pose = robot_manager.pose
        if pose is None or self.signal:
            return False

        if not self.arrived(robot_manager, pose):
            self._park.reset()
            return False

        if self.hold_time <= 0:
            return True
        held = self._park.held_for(pose.position.x, pose.position.y, pose.orientation.to_yaw(), robot_manager.node.sim_time.to_seconds())
        return held >= self.hold_time

    def arrived(self, robot_manager: RobotManager, pose: Pose) -> bool:
        """`pose` is within tolerance_radius of the goal, and within tolerance_angle for robots that control orientation."""
        assert self.tolerance_radius is not None and self.tolerance_angle is not None, "GoToPhase checked before with_defaults filled it"
        tol_ang = self.tolerance_angle
        dx = pose.position.x - self.pose.position.x
        dy = pose.position.y - self.pose.position.y
        if math.hypot(dx, dy) > self.tolerance_radius:
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
    instance: str = ""  # arm cap instance (mount name), empty = sole arm

    def __attrs_post_init__(self):
        if sum([self.target is not None, self.named_target is not None, self.random]) != 1:
            raise ValueError("ReachPhase requires exactly one of target / named_target / random")

    def is_satisfied(self, robot_manager: RobotManager) -> bool:
        return False


@attrs.define
class PlayGesturePhase(TaskPhase):
    kind: ClassVar[TaskKind] = TaskKind.PLAY_GESTURE
    gesture: str | None = None  # None means random; adapter expands before dispatch
    instance: str = ""  # arm cap instance (mount name), empty = sole arm

    def is_satisfied(self, robot_manager: RobotManager) -> bool:
        return False  # client-driven completion


DonePredicate = Callable[
    ["RobotManager", TaskPhase],
    bool | None,
]


@attrs.define
class TaskRequest:
    """Typed sequence of phases submitted to a robot."""

    phases: list[TaskPhase]
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
