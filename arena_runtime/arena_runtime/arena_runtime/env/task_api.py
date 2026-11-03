"""The runtime types and helpers a task layer calls."""

from __future__ import annotations

import importlib
import typing

from arena_runtime._node import NodeInterface
from arena_runtime.acoustics import SoundEmission
from arena_runtime.env.constants import Constants
from arena_runtime.env.interactive import Apply, Menu, planar_marker, static_marker, visual
from arena_runtime.env.placement import random_robot_pose
from arena_runtime.env.realizer import Realizer
from arena_runtime.env.request import GoToPhase, LoopPhase, PlayGesturePhase, ReachPhase, TaskKind, TaskPhase, TaskRequest
from arena_runtime.env.robot_manager.robot_manager import RobotManager
from arena_runtime.env.robot_manager.task_runner import PhaseState
from arena_runtime.env.world_manager.utils import GridFrame, WorldMap, WorldOccupancy
from arena_runtime.humans.itinerary import Stop, plan_itinerary

if typing.TYPE_CHECKING:
    from arena_runtime.humans.hunav.agent import HunavDynamicObstacle

__all__ = [
    "Apply",
    "Constants",
    "GoToPhase",
    "GridFrame",
    "HunavDynamicObstacle",
    "LoopPhase",
    "Menu",
    "NodeInterface",
    "PhaseState",
    "PlayGesturePhase",
    "ReachPhase",
    "Realizer",
    "RobotManager",
    "SoundEmission",
    "Stop",
    "TaskKind",
    "TaskPhase",
    "TaskRequest",
    "WorldMap",
    "WorldOccupancy",
    "plan_itinerary",
    "planar_marker",
    "random_robot_pose",
    "static_marker",
    "visual",
]


def __getattr__(name: str) -> type:
    if name == "HunavDynamicObstacle":
        return importlib.import_module("arena_runtime.humans.hunav.agent").HunavDynamicObstacle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
