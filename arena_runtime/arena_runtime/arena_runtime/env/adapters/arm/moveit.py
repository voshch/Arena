"""MoveIt arm adapter: thin composer of MoveItArmBringup + ReachPoseClient."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, ClassVar

import rclpy.qos
import rclpy.task
import rclpy.time
import tf2_ros
from arena_robots.bringup.arm.moveit import MoveItArmBringup
from arena_robots.clients.play_gesture import PlayGestureClient
from arena_robots.clients.reach_pose import ReachPoseClient
from arena_robots.task_kinds import TaskKind
from arena_robots_msgs.action import PlayGesture, ReachPose
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position
from arena_viz.kinds import DisplayKind
from arena_viz.style import StyleSpec
from moveit_msgs.srv import ApplyPlanningScene
from std_msgs.msg import String

from arena_runtime.env.adapters import Adapter, AdapterDisplayHint, AdapterMeta
from arena_runtime.env.adapters._reach_sampling import MAP_FRAME, reach_frame, reach_target, sample_reach_target
from arena_runtime.env.adapters.arm import park_arms
from arena_runtime.env.adapters.arm.scene import LINK_SEPARATOR, Articulated, StaticBox, attached_objects, collision_objects, detached_objects, object_id, read_urdf, scene_diff, urdf_root_link
from arena_runtime.env.request import PlayGesturePhase, ReachPhase
from arena_runtime.humans.utils import ObstacleLayer
from arena_runtime.shared import WORLD_ENTITY_PREFIX, DynamicObstacle
from arena_runtime.tf_frames import frames_below

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from arena_rclpy_mixins.Async import ClientWrapper

    from arena_runtime.env.adapters import ResetContext
    from arena_runtime.env.request import TaskPhase
    from arena_runtime.env.robot_manager.robot_manager import RobotManager

_log = logging.getLogger(__name__)

_DESCRIPTION_TIMEOUT = 10.0
_SCENE_TIMEOUT = 5.0
_ENTITY_WAIT = 10.0
_ENTITY_POLL = 0.2

type Entity = Articulated | StaticBox


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

    def __init__(self, robot_manager: RobotManager, **bringup_kwargs: object) -> None:
        super().__init__(robot_manager, **bringup_kwargs)
        self._scene_ids: dict[str, set[str]] = {}
        self._attached_ids: dict[str, set[str]] = {}
        self._scene_clients: dict[str, ClientWrapper] = {}
        self._root_frame: str | None = None
        self._deferred: asyncio.Task[None] | None = None

    async def dispatch_phase(self, phase: TaskPhase, robot: RobotManager) -> None:
        if isinstance(phase, ReachPhase):
            client = self.client_for(TaskKind.REACH_POSE)
            assert isinstance(client, ReachPoseClient)
            client.reset()
            self._drop_deferred()
            entities = await self._episode_entities(robot)
            if _frame_missing(phase, entities, robot):
                self._deferred = asyncio.create_task(self._deferred_reach(phase, robot, client))
                return
            await self._send_reach(phase, robot, client, entities)
        elif isinstance(phase, PlayGesturePhase):
            gesture_name = phase.gesture
            if gesture_name is None:
                gesture_name = _pick_random_gesture(robot)
            goal = PlayGesture.Goal(gesture=gesture_name or "", instance=phase.instance)
            await self.client_for(TaskKind.PLAY_GESTURE).send_goal(goal)
        else:
            raise TypeError(f"MoveItArmAdapter: unsupported phase type {type(phase).__name__} (kind={phase.kind!r})")

    async def cancel_and_wait(self, kinds: Iterable[TaskKind] | None = None) -> None:
        self._drop_deferred()
        await super().cancel_and_wait(kinds)

    def _drop_deferred(self) -> None:
        if self._deferred is not None and not self._deferred.done():
            self._deferred.cancel()
        self._deferred = None

    async def _deferred_reach(self, phase: ReachPhase, robot: RobotManager, client: ReachPoseClient) -> None:
        """Send the reach once the object its frame names is registered, at most `_ENTITY_WAIT` s of wall time later."""
        deadline = time.monotonic() + _ENTITY_WAIT
        entities = await self._episode_entities(robot)
        while _frame_missing(phase, entities, robot) and time.monotonic() < deadline:
            await asyncio.sleep(_ENTITY_POLL)
            entities = await self._episode_entities(robot)
        try:
            await self._send_reach(phase, robot, client, entities)
        except Exception as exc:
            robot.fail_phase(f"reach dispatch failed: {exc!r}")

    async def _send_reach(self, phase: ReachPhase, robot: RobotManager, client: ReachPoseClient, entities: dict[str, Entity]) -> None:
        await self._sync_scene(robot, self._mount(phase.instance), entities)
        goal = ReachPose.Goal()
        goal.instance = phase.instance
        goal.motion = phase.motion
        goal.speed = float(phase.speed or 0.0)
        goal.relative = phase.relative
        if phase.random:
            arms = robot.caps.arm
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
            located = _locate(phase.frame, entities, robot)
            if located is None:
                frame = phase.frame or robot.base_frame
                if frame != MAP_FRAME and not robot.node.tf_buffer.can_transform(reach_frame(frame, robot.frame), robot.base_frame, rclpy.time.Time()):
                    robot.fail_phase(f"reach frame {phase.frame!r} is no robot frame and no episode object ({robot.node._resolve_semantic_entity(phase.frame)!r} not among {sorted(entities)})")
                    return
                goal.target = reach_target(phase.target, frame, robot.frame, robot.base_frame, robot.node.tf_buffer)
                goal.target.header.stamp = robot.node.sim_time.to_msg()
            else:
                entity, link = located
                self._place_in_entity(goal, phase, robot, entity, link)
        elif phase.named_target is not None:
            goal.named_target = phase.named_target
        goal.position_tolerance = float(phase.position_tolerance or 0.0)
        goal.orientation_tolerance = float(phase.orientation_tolerance or 0.0)
        goal.planning_time = float(phase.planning_time or 0.0)
        await client.send_goal(goal)

    def _mount(self, instance: str) -> str | None:
        arms = self.bringup.arms()
        if instance:
            return instance if instance in arms else None
        return next(iter(arms)) if len(arms) == 1 else None

    def _place_in_entity(self, goal: ReachPose.Goal, phase: ReachPhase, robot: RobotManager, entity: Entity, link: str | None) -> None:
        assert phase.target is not None
        frame = entity.link_frame(link) if isinstance(entity, Articulated) and link is not None else entity.pose
        inside = _map_pose(robot, robot.base_frame).inverse() * frame
        if phase.relative:
            turn = inside.orientation
            placed = Pose(position=Position(*(turn * phase.target.position)), orientation=turn * phase.target.orientation * turn.inverse())
        else:
            placed = inside * phase.target
        goal.target.header.frame_id = robot.base_frame
        goal.target.header.stamp = robot.node.sim_time.to_msg()
        goal.target.pose = placed.to_msg()
        goal.contact = _contact_ids(entity)

    async def _episode_entities(self, robot: RobotManager) -> dict[str, Entity]:
        """Static and articulated objects of the env by realized name, static ones where the simulator last put them, articulated ones posed at their current joint values."""
        node = robot.node
        simulator = node._simulator
        entities: dict[str, Entity] = {}
        for runtime in simulator._articulated.values():
            obstacle = runtime.obstacle
            view = await obstacle.model.resolve()
            urdf = _urdf_path(view.path)
            if urdf is not None:
                entities[runtime.entity] = Articulated(name=obstacle.name, pose=obstacle.pose, scale=obstacle.scale, urdf=read_urdf(urdf), values=dict(runtime.current))
            elif view.extent is not None:
                entities[runtime.entity] = StaticBox(name=obstacle.name, pose=obstacle.pose, scale=obstacle.scale, extent=view.extent)
        for known in node._human_simulator._known_obstacles.values():
            obstacle = known.obstacle
            if isinstance(obstacle, DynamicObstacle) or not known.spawned or known.layer == ObstacleLayer.UNUSED or obstacle.name in simulator._articulated:
                continue
            try:
                view = await obstacle.model.resolve()
            except FileNotFoundError:
                continue
            if view.extent is not None:
                pose = simulator._carry_objects.get(obstacle.name, obstacle).pose
                entities[str(node._realizer.prefix(obstacle.name))] = StaticBox(name=obstacle.name, pose=pose, scale=obstacle.scale, extent=view.extent)
        return entities

    async def _root(self, robot: RobotManager) -> str:
        if self._root_frame is not None:
            return self._root_frame
        description = rclpy.task.Future()

        def received(msg: String) -> None:
            if not description.done():
                description.set_result(msg.data)

        qos = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE)
        subscription = robot.node.create_subscription(String, str(robot.namespace("robot_description")), received, qos)
        try:
            async with asyncio.timeout(_DESCRIPTION_TIMEOUT):
                root = urdf_root_link(await robot.node.await_ros(description))
        finally:
            robot.node.destroy_subscription(subscription)
        self._root_frame = reach_frame(root, robot.frame)
        return self._root_frame

    async def _sync_scene(self, robot: RobotManager, mount: str | None, entities: dict[str, Entity]) -> None:
        """Replace the arena-owned collision objects of the arm's MoveIt scene with the env's objects as seen from the robot root, the ones its gripper holds attached to the arm's tip link."""
        if mount is None:
            return
        tip = robot.robot.frame.tf(self.bringup.arms()[mount].tip_link)
        try:
            frame = await self._root(robot)
            root = _map_pose(robot, frame)
            carried = _carried(robot, tip, entities)
        except (TimeoutError, tf2_ros.TransformException) as error:
            _log.warning("robot %r: planning scene not synced, no root frame (%s)", robot.name, error)
            return
        touch_links = frames_below(robot.node.tf_buffer, tip)
        attached = [obj for entity, offset in carried for obj in attached_objects(entity, offset, tip, touch_links)]
        held = {obj.object.id for obj in attached}
        held_entities = [entity for entity, _ in carried]
        free = [entity for entity in entities.values() if not any(entity is held_entity for held_entity in held_entities)]
        objects = collision_objects(free, root, frame, self.bringup.arms()[mount].scene_radius)
        detached = detached_objects(self._attached_ids.get(mount, set()) - held, tip)
        client = self._scene_clients.get(mount)
        if client is None:
            client = self._scene_clients[mount] = robot.node.create_client_wrapper(ApplyPlanningScene, str(self.bringup.arm_namespace(mount)("apply_planning_scene")), timeout=_SCENE_TIMEOUT)
        request = ApplyPlanningScene.Request(scene=scene_diff(set(self._scene_ids.get(mount, ())) - held, objects, frame, [*detached, *attached]))
        response = await client.call_timeout(request, _SCENE_TIMEOUT) if await client.ensure(_SCENE_TIMEOUT) else None
        if response is None or not response.success:
            _log.warning("robot %r: planning scene of arm %r not synced", robot.name, mount)
            return
        self._scene_ids[mount] = {obj.id for obj in objects}
        self._attached_ids[mount] = held

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


def _urdf_path(directory: Path) -> Path | None:
    named = directory / f"{directory.name}.urdf"
    if named.is_file():
        return named
    found = sorted(directory.glob("*.urdf"))
    return found[0] if found else None


def _map_pose(robot: RobotManager, frame: str) -> Pose:
    transform = robot.node.tf_buffer.lookup_transform("map", frame, rclpy.time.Time()).transform
    return Pose(position=Position(transform.translation.x, transform.translation.y, transform.translation.z), orientation=Orientation.from_msg(transform.rotation))


def _frame_missing(phase: ReachPhase, entities: dict[str, Entity], robot: RobotManager) -> bool:
    """Whether the reach targets a frame that is neither a registered episode object nor a TF frame yet."""
    if phase.target is None or not phase.frame or phase.frame == MAP_FRAME or _locate(phase.frame, entities, robot) is not None:
        return False
    return not robot.node.tf_buffer.can_transform(reach_frame(phase.frame, robot.frame), robot.base_frame, rclpy.time.Time())


def _contact_ids(entity: Entity) -> list[str]:
    """Planning scene ids of the entity, one per link of an articulated one."""
    return [object_id(entity.name, part) for part in sorted(entity.urdf.links)] if isinstance(entity, Articulated) else [object_id(entity.name)]


def _carried(robot: RobotManager, hand: str, entities: dict[str, Entity]) -> list[tuple[Entity, Pose]]:
    """The entities the gripper of the `hand` frame carries with their pose in that frame."""
    carried: list[tuple[Entity, Pose]] = []
    for name, (carry_hand, offset) in sorted(robot.environment_manager.carry_offsets(robot.robot.name).items()):
        if carry_hand == hand and (located := _locate(name, entities, robot)) is not None:
            carried.append((located[0], offset))
    return carried


def _locate(frame: str, entities: dict[str, Entity], robot: RobotManager) -> tuple[Entity, str | None] | None:
    """The entity a reach frame names, a world object also by its bare name, with its link for `<entity>/<link>`, None for any other frame."""
    if not frame:
        return None
    for name in (frame, f"{WORLD_ENTITY_PREFIX}{frame}"):
        if (entity := entities.get(robot.node._resolve_semantic_entity(name))) is not None:
            return entity, None
    head, separator, tail = frame.rpartition(LINK_SEPARATOR)
    entity = entities.get(robot.node._resolve_semantic_entity(head)) if separator else None
    if isinstance(entity, Articulated) and tail in entity.urdf.links:
        return entity, tail
    return None


def _pick_random_gesture(robot: RobotManager) -> str:
    """Return a gesture name supported by this robot's arm, or empty string if none."""
    import yaml
    from arena_simulation_setup import ASS_DIR
    from arena_simulation_setup.tree.Gesture import GestureSpec

    arms = robot.caps.arm
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
