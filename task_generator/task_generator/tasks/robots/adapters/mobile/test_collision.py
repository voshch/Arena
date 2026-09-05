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

from task_generator.manager.world_manager.shims import requires_map_server
from task_generator.tasks.robots.adapters import AdapterMeta
from task_generator.tasks.robots.adapters.mobile import MobileAdapter

if TYPE_CHECKING:
    from task_generator.manager.robot_manager.robot_manager import RobotManager
    from task_generator.tasks.robots.request import TaskPhase


@AdapterMeta.attach(
    accepts={TaskKind.GOTO_POSE},
    bringup=TestCollisionBringup,
    client=GotoPoseClient,
    cap="mobile",
)
@requires_map_server
class TestCollisionAdapter(MobileAdapter):
    kind: ClassVar[str] = "test-collision"

    def is_phase_done(self, phase: TaskPhase, robot: RobotManager) -> bool | None:
        return False

    async def dispatch_phase(self, phase: TaskPhase, robot: RobotManager) -> None:
        return None
