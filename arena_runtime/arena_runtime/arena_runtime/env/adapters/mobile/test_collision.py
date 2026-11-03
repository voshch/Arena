"""Debug adapter that drives a constant cmd_vel.

No planner, no goal dispatch: the robot moves forward at a fixed linear
velocity until something stops it. Intended for validating the collision
tracker shim end-to-end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from arena_robots.bringup.mobile.test_collision import TestCollisionBringup
from arena_robots.clients.goto_pose import GotoPoseClient
from arena_robots.task_kinds import TaskKind

from arena_runtime.env.adapters import AdapterMeta
from arena_runtime.env.adapters.mobile import MobileAdapter
from arena_runtime.env.world_manager.shims import requires_map_server

if TYPE_CHECKING:
    from arena_runtime.env.request import TaskPhase
    from arena_runtime.env.robot_manager.robot_manager import RobotManager


@AdapterMeta.attach(
    accepts={TaskKind.GOTO_POSE},
    bringup=TestCollisionBringup,
    client=GotoPoseClient,
    cap="mobile",
)
@requires_map_server
class TestCollisionAdapter(MobileAdapter):
    kind: ClassVar[str] = "test-collision"

    async def dispatch_phase(self, phase: TaskPhase, robot: RobotManager) -> None:
        return None
