"""No-op lift adapter (lift: none, lift cap)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from arena_robots.bringup.lift.none import NoneLiftBringup

from arena_runtime.env.adapters import Adapter, AdapterMeta

if TYPE_CHECKING:
    from arena_runtime.env.request import TaskPhase
    from arena_runtime.env.robot_manager.robot_manager import RobotManager


@AdapterMeta.attach(accepts=(), bringup=NoneLiftBringup, clients={}, cap="lift")
class NoneLiftAdapter(Adapter):
    kind: ClassVar[str] = "none"

    async def dispatch_phase(self, phase: TaskPhase, robot: RobotManager) -> None:
        raise RuntimeError(f"{type(self).__name__} accepts no task phases; got {type(phase).__name__}")
