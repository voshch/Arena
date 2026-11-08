"""No-op gripper adapter (gripper: none, gripper cap)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from arena_robots.bringup.gripper.none import NoneGripperBringup

from arena_runtime.env.adapters import Adapter, AdapterMeta

if TYPE_CHECKING:
    from arena_runtime.env.request import TaskPhase
    from arena_runtime.env.robot_manager.robot_manager import RobotManager


@AdapterMeta.attach(accepts=(), bringup=NoneGripperBringup, clients={}, cap="gripper")
class NoneGripperAdapter(Adapter):
    kind: ClassVar[str] = "none"

    async def dispatch_phase(self, phase: TaskPhase, robot: RobotManager) -> None:
        raise RuntimeError(f"{type(self).__name__} accepts no task phases; got {type(phase).__name__}")
