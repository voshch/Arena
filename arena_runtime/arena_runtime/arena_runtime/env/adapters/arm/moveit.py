"""MoveIt arm adapter: thin composer of MoveItArmBringup + ReachPoseClient."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, ClassVar

from arena_robots.bringup.arm.moveit import MoveItArmBringup
from arena_robots.clients.play_gesture import PlayGestureClient
from arena_robots.clients.reach_pose import ReachPoseClient
from arena_robots.task_kinds import TaskKind
from arena_robots_msgs.action import PlayGesture, ReachPose
from arena_viz.kinds import DisplayKind
from arena_viz.style import StyleSpec

from arena_runtime.env.adapters import Adapter, AdapterDisplayHint, AdapterMeta
from arena_runtime.env.adapters._reach_sampling import reach_frame, sample_reach_target
from arena_runtime.env.adapters.arm import park_arms
from arena_runtime.env.request import PlayGesturePhase, ReachPhase

if TYPE_CHECKING:
    from arena_runtime.env.adapters import ResetContext
    from arena_runtime.env.request import TaskPhase
    from arena_runtime.env.robot_manager.robot_manager import RobotManager

_log = logging.getLogger(__name__)


@AdapterMeta.attach(
    accepts={TaskKind.REACH_POSE, TaskKind.PLAY_GESTURE},
    bringup=MoveItArmBringup,
    clients={TaskKind.REACH_POSE: ReachPoseClient, TaskKind.PLAY_GESTURE: PlayGestureClient},
    cap="arm",
    displays=(
        AdapterDisplayHint(
            name="Planned Trajectory",
            topic="{ns}/move_group/display_planned_path",
            topic_type="moveit_msgs/DisplayTrajectory",
            kind=DisplayKind.TRAJECTORY,
        ),
        AdapterDisplayHint(
            name="Planning Scene",
            topic="{ns}/monitored_planning_scene",
            topic_type="moveit_msgs/PlanningScene",
            kind=DisplayKind.PLANNING_SCENE,
            style_json=StyleSpec(enabled=False).to_json(),
        ),
    ),
)
class MoveItArmAdapter(Adapter):
    kind: ClassVar[str] = "moveit"
    bringup: MoveItArmBringup

    async def dispatch_phase(self, phase: TaskPhase, robot: RobotManager) -> None:
        if isinstance(phase, ReachPhase):
            goal = ReachPose.Goal()
            goal.instance = phase.instance
            if phase.random:
                arms = robot.robot_view.effective_caps(robot.robot.resolved_request, frames=robot.robot.frames).arm
                if not arms:
                    raise ValueError(f"random ReachPhase requested but robot {robot.name!r} has no arm cap")
                if phase.instance:
                    if phase.instance not in arms:
                        raise ValueError(f"random ReachPhase instance {phase.instance!r} not found on robot {robot.name!r}; available: {sorted(arms)}")
                    arm = arms[phase.instance]
                elif len(arms) != 1:
                    raise ValueError(f"random ReachPhase requires an explicit arm instance on robot {robot.name!r}; available: {sorted(arms)}")
                else:
                    (arm,) = arms.values()
                rng = robot.node.conf.General.RNG.stream("arm", robot.name)
                goal.target = sample_reach_target(arm, robot.frame, rng)
            elif phase.target is not None:
                goal.target.header.frame_id = reach_frame(phase.frame or robot.base_frame, robot.frame)
                goal.target.header.stamp = robot.node.sim_time.to_msg()
                goal.target.pose = phase.target.to_msg()
            elif phase.named_target is not None:
                goal.named_target = phase.named_target
            goal.position_tolerance = float(phase.position_tolerance or 0.0)
            goal.orientation_tolerance = float(phase.orientation_tolerance or 0.0)
            goal.planning_time = float(phase.planning_time or 0.0)
            await self.client_for(TaskKind.REACH_POSE).send_goal(goal)
        elif isinstance(phase, PlayGesturePhase):
            gesture_name = phase.gesture
            if gesture_name is None:
                gesture_name = _pick_random_gesture(robot)
            goal = PlayGesture.Goal(gesture=gesture_name or "", instance=phase.instance)
            await self.client_for(TaskKind.PLAY_GESTURE).send_goal(goal)
        else:
            raise TypeError(f"MoveItArmAdapter: unsupported phase type {type(phase).__name__} (kind={phase.kind!r})")

    async def on_reset(self, robot: RobotManager, ctx: ResetContext) -> None:
        # MoveIt cannot plan while sim is paused (no /joint_states, no current state),
        # the TM is responsible for emitting a stow phase if it wants the arm parked.
        del robot, ctx

    async def on_controllers_active(self, robot: RobotManager) -> None:
        self._park_pubs = await park_arms(robot)

    async def wait_until_ready(self, robot: RobotManager, node_paths: set[str]) -> None:
        for mount in self.bringup.arms():
            mg = str(self.bringup.arm_namespace(mount)("move_group"))
            while mg not in node_paths:
                await asyncio.sleep(0.01)
        await super().wait_until_ready(robot, node_paths)


def _pick_random_gesture(robot: RobotManager) -> str:
    """Return a gesture name supported by this robot's arm, or empty string if none."""
    import yaml
    from arena_simulation_setup import ASS_DIR
    from arena_simulation_setup.tree.Gesture import GestureSpec

    arms = robot.robot_view.effective_caps(robot.robot.resolved_request, frames=robot.robot.frames).arm
    if not arms:
        _log.warning("random PlayGesturePhase: robot %r has no arm cap", robot.name)
        return ""
    if len(arms) != 1:
        _log.warning("random PlayGesturePhase: robot %r has multiple arm instances %s; explicit instance not yet supported", robot.name, sorted(arms))
        return ""
    (arm,) = arms.values()
    available_poses = set(arm.named_poses.keys())

    # shared library: glob ASS_DIR/configs/gestures/*.yaml
    gestures: dict[str, GestureSpec] = {}
    gestures_dir = ASS_DIR / "configs" / "gestures"
    for path in sorted(gestures_dir.glob("*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        gestures[path.stem] = GestureSpec.parse(data)

    # per-arm overrides shadow shared library
    gestures.update(arm.gestures)

    supported = [name for name, spec in gestures.items() if available_poses >= spec.required_poses()]
    if not supported:
        _log.warning(
            "random PlayGesturePhase: robot %r has no supported gestures (named_poses: %s)",
            robot.name,
            sorted(available_poses),
        )
        return ""
    rng = robot.node.conf.General.RNG.stream("arm", robot.name)
    return str(rng.choice(supported))
