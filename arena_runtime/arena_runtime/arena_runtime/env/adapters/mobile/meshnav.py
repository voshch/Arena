"""MeshNav adapter: MeshNavBringup + GotoPoseClient planning on a nav mesh of the spawned world in the simulator frame."""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from arena_robots.bringup.mobile.meshnav import MeshNavBringup
from arena_robots.clients.goto_pose import GotoPoseClient
from arena_robots.task_kinds import TaskKind
from arena_robots_msgs.action import GotoPose
from arena_simulation_setup.shared.navmesh import NavMesh, build_navmesh
from arena_viz.kinds import DisplayKind
from arena_viz.style import StyleSpec

from arena_runtime.env.adapters import AdapterDisplayHint, AdapterMeta
from arena_runtime.env.adapters.mobile import MobileAdapter
from arena_runtime.env.request import ElevatorRidePhase, GoToPhase, TaskPhase
from arena_runtime.env.world_manager.mechanism_map import static_shut_doors

if TYPE_CHECKING:
    import geometry_msgs.msg
    import numpy as np
    from arena_simulation_setup.utils.geometry import Position
    from launch.actions import GroupAction

    from arena_runtime.env.adapters import AdapterCtx, ResetContext
    from arena_runtime.env.robot_manager.robot_manager import RobotManager
    from arena_runtime.shared import Pose

_REDISPATCH_MIN_S = 2.0
_LOCATE_REACH = 1.0
NAVMESH_DIR = Path(tempfile.gettempdir()) / "arena_meshnav"


def navmesh_file(mesh: NavMesh, directory: Path = NAVMESH_DIR) -> Path:
    """The PLY of a nav mesh, written once under `directory` and named by its content."""
    digest = hashlib.sha1(mesh.vertices.tobytes() + mesh.faces.tobytes()).hexdigest()[:16]
    path = directory / f"{digest}.ply"
    if not path.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial.ply")
        mesh.write_ply(partial)
        partial.replace(path)
    return path


def separated(mesh: NavMesh, components: np.ndarray, start: Position, goal: Position, reach: float = _LOCATE_REACH) -> bool:
    """Whether two simulator-frame points lie on separate components of the mesh, False when either is farther than `reach` from it."""
    here, here_off = mesh.nearest(start.x, start.y, start.z)
    there, there_off = mesh.nearest(goal.x, goal.y, goal.z)
    return max(here_off, there_off) <= reach and bool(components[here] != components[there])


@AdapterMeta.attach(
    accepts={TaskKind.GOTO_POSE},
    bringup=MeshNavBringup,
    client=GotoPoseClient,
    cap="mobile",
    displays=[
        AdapterDisplayHint(
            name="Mesh Plan",
            topic="{ns}/move_base_flex/path",
            topic_type="nav_msgs/Path",
            kind=DisplayKind.PATH,
            style_json=StyleSpec(color=(0, 160, 255), line_width=0.05).to_json(),
        ),
    ],
)
class MeshNavAdapter(MobileAdapter):
    kind: ClassVar[str] = "meshnav"
    routes_levels: ClassVar[bool] = True
    bringup: MeshNavBringup

    def __init__(self, robot_manager: RobotManager, **kwargs: object):
        super().__init__(robot_manager, **kwargs)
        self._odom_frame = ""
        self._last_redispatch: float = -_REDISPATCH_MIN_S
        self._mesh: NavMesh | None = None
        self._components: np.ndarray | None = None

    async def ensure_services(self) -> None:
        levels = self.rm.environment_manager.nav_levels
        if not levels:
            raise RuntimeError(f"meshnav for robot {self.rm.name!r}: no world geometry spawned yet, the nav mesh needs floors, walls and stairs")
        settings = self.bringup.settings
        agents = [name for name in (self.rm.name, self.rm.robot.sim_path) if name]
        shut = static_shut_doors((door for level in levels for door in level.doors), agents)
        mesh = await asyncio.to_thread(build_navmesh, levels, float(settings["resolution"]), climb=float(settings["max_step"]), shut=shut)
        self._components = await asyncio.to_thread(mesh.components)
        self._mesh = mesh
        path = await asyncio.to_thread(navmesh_file, mesh)
        self._bringup_kwargs = {**self._bringup_kwargs, "mesh_file": str(path)}

    def launch_description(self, ctx: AdapterCtx) -> GroupAction:
        self._odom_frame = self.rm.frame(ctx.odom_frame).raw()
        self._bringup_kwargs = {
            **self._bringup_kwargs,
            "odom_frame": self._odom_frame,
            "base_frame": self.rm.frame(ctx.base_frame).raw(),
        }
        return super().launch_description(ctx)

    async def publish_goal_loop(self) -> None:
        return

    async def dispatch_phase(self, phase: TaskPhase, robot: RobotManager) -> None:
        assert isinstance(phase, GoToPhase), f"MeshNavAdapter only accepts GOTO_POSE phases; got {type(phase).__name__} (kind={phase.kind!r})"
        assert phase.pose is not None
        robot._goal_pos = phase.pose  # pylint: disable=protected-access
        if self.client.is_done() is False:
            self.client.cancel()
        reason = self.unreachable(phase, robot)
        if reason is not None:
            robot.node.get_logger().warning(f"robot {robot.name!r}: {reason}")
            robot.fail_phase(reason)
            return
        goal = GotoPose.Goal()
        goal.target = self.placed_target(phase.pose, robot)
        goal.pose_tolerance = float(phase.tolerance_radius or 0.0)
        goal.yaw_tolerance = float(phase.tolerance_angle or 0.0)
        await self.client.send_goal(goal)

    def unreachable(self, phase: GoToPhase, robot: RobotManager) -> str | None:
        """Why the mesh cannot carry the robot to a map-frame goal, None when it can or either end lies off the mesh."""
        pose = robot.pose
        if self._mesh is None or self._components is None or pose is None or phase.pose is None or isinstance(phase, ElevatorRidePhase):
            return None
        placement = robot.environment_manager.placement()
        start, goal = placement.place(pose.position), placement.place(phase.pose.position)
        if not separated(self._mesh, self._components, start, goal):
            return None
        return f"meshnav: no mesh path from ({start.x:.2f}, {start.y:.2f}, {start.z:.2f}) to the goal at ({goal.x:.2f}, {goal.y:.2f}, {goal.z:.2f}), they lie on separate parts of the nav mesh and no elevator route joins them"

    def placed_target(self, pose: Pose, robot: RobotManager) -> geometry_msgs.msg.PoseStamped:
        """An abstract-frame goal on its level, in the robot's odom frame, which is the simulator frame."""
        import geometry_msgs.msg

        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self._odom_frame
        msg.header.stamp = robot.node.sim_time.to_msg()
        msg.pose = pose.to_msg()
        placed = robot.environment_manager.placement().place(pose.position)
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = placed.x, placed.y, placed.z
        return msg

    def on_phase_tick(self, phase: TaskPhase, robot: RobotManager) -> None:
        if robot.waiting or not self.client.is_done():
            return
        now = robot.node.sim_time.to_seconds()
        if now - self._last_redispatch >= _REDISPATCH_MIN_S:
            self._last_redispatch = now
            robot.node.event_loop.create_task(self.dispatch_phase(phase, robot))

    async def on_reset(self, robot: RobotManager, ctx: ResetContext) -> None:
        if self.client.is_done() is False:
            self.client.cancel()
        await super().on_reset(robot, ctx)

    async def before_move(self, pose: Pose, robot: RobotManager) -> None:
        if self.client.is_done() is False:
            self.client.cancel()

    async def on_move(self, pose: Pose, robot: RobotManager) -> None:
        phase = robot.active_dispatch_phase()
        if phase is not None:
            await self.dispatch_phase(phase, robot)
