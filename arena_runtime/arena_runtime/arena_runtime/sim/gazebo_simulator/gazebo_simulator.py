"""
GazeboHost is constructed once by arena_runtime_node and owns process-singleton resources for Gazebo: the gz_services_bridge launch, the lifecycle (pause/unpause), and the control-world client.
GazeboSimulator is per-env on arena_env_node and adapts env-namespace state (spawned-name tracking, per-env service clients) over those shared resources. Sim_path prefixing is owned by the Realizer.
"""

import asyncio
import itertools
import math
import time
import traceback
import typing
import xml.etree.ElementTree
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import arena_robots.catalog
import arena_robots.Robot
import arena_robots.Sensor
import attrs
import gz.transport13
import launch
import launch_ros
import rclpy.impl.rcutils_logger
import rclpy.time
from arena_bringup.constants import SimSimulator
from arena_people_msgs.msg import Pedestrians
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.Async import ClientWrapper
from arena_rclpy_mixins.Time import Time
from arena_simulation_setup.shared import Ceiling, Light, cct_to_rgb
from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
from arena_simulation_setup.tree.Wall import WallSegment
from arena_simulation_setup.utils.articulation import Joint, JointType, Values
from arena_simulation_setup.utils.geometry import Scale, Vector3
from arena_simulation_setup.utils.material import MdlUtil
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Quaternion, TransformStamped
from geometry_msgs.msg import Pose as RosPose
from gz.msgs10 import boolean_pb2, double_v_pb2, light_pb2, material_color_pb2, pose_v_pb2, twist_pb2, world_stats_pb2
from launch import LaunchDescription
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from ros_gz_interfaces.msg import Entity as EntityMsg
from ros_gz_interfaces.msg import EntityFactory, WorldControl
from ros_gz_interfaces.srv import ControlWorld, DeleteEntity, SetEntityPose, SpawnEntity
from std_msgs.msg import Bool
from viewport_control_msgs.msg import ViewportView
from viewport_control_msgs.srv import ViewportSetProjection, ViewportSetReferenceFrame, ViewportSetView

from arena_runtime.flags import ObstaclesOptim, obstacles_optim_level
from arena_runtime.shared import (
    DynamicObstacle,
    Entity,
    Floor,
    Model,
    ModelType,
    Obstacle,
    Orientation,
    Pose,
    Position,
    Robot,
    Wall,
)
from arena_runtime.sim import BaseSim, SimKwargs, SimLifecycle, SimUnavailable
from arena_runtime.sim._articulation import ArticulatedObject, command_fields, surface
from arena_runtime.sim._articulation import JointCommand as ArticulatedCommand
from arena_runtime.sim._control import (
    effective_control_yaml,
    robot_controllers,
    twist_stamper_node,
)
from arena_runtime.sim._interface import resolve_obstacle_box
from arena_runtime.sim._walls import realize_renderable, sloped_box
from arena_runtime.static_tf import StaticTransformBroadcaster

from .robot_bridge import BridgeConfiguration, MappingDirection, _TopicMapping
from .scene import parse_scene_models

_VIEWPORT_STREAM_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)

# gz resolves set_pose and remove by name, first match wins, so a stale entity under a
# robot's sim_path swallows every later teleport while still acking success
_SPAWN_SWEEP_ROUNDS = 5
_SPAWN_SWEEP_INTERVAL = 0.05
_SPAWN_SWEEP_DRAIN_TIMEOUT = 2.0
_ENTITY_ID_TIMEOUT = 5.0
_ENTITY_ID_POLL = 0.25
_RTF_BOOST = 1e6
_SET_PHYSICS_TIMEOUT_MS = 5000
_SET_PHYSICS_ATTEMPTS = 3
_CLEANUP_ATTEMPTS = 5
_WORLD_STATS_TOPIC = '/world/default/stats'
_BATCH_STALL_S = 5.0
_BATCH_POLL_S = 0.05

_LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


async def _list_models() -> list[str]:
    proc = await asyncio.create_subprocess_exec(
        'gz',
        'model',
        '--list',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    out = stdout.decode()
    if proc.returncode != 0 or 'Available models:' not in out:
        raise SimUnavailable(f"gz model --list failed: {(stderr.decode() or out).strip()}")
    names: list[str] = []
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith('- '):
            names.append(stripped[2:].strip())
    return names


class GazeboHost(SimLifecycle):
    def __init__(
        self,
        node: ArenaMixinNode,
        semaphore: asyncio.Semaphore,
        service_control_world: ClientWrapper,
        service_delete_entity: ClientWrapper,
        logger: rclpy.impl.rcutils_logger.RcutilsLogger,
        physics_dt: float,
    ) -> None:
        self._node = node
        self._semaphore = semaphore
        self._service_control_world = service_control_world
        self._service_delete_entity = service_delete_entity
        self._logger = logger
        self._dt = physics_dt
        self._rtf_boosted = False
        self._batch_target: Time | None = None
        self._stats: world_stats_pb2.WorldStatistics | None = None
        self._gz_node = gz.transport13.Node()
        self._gz_node.subscribe(world_stats_pb2.WorldStatistics, _WORLD_STATS_TOPIC, self._on_stats)

    def _on_stats(self, msg: world_stats_pb2.WorldStatistics) -> None:
        self._stats = msg

    async def ensure_ready(self) -> None:
        await self._node.do_launch(
            LaunchDescription(
                [
                    launch_ros.actions.Node(
                        package="ros_gz_bridge",
                        executable="parameter_bridge",
                        name="gz_services_bridge",
                        output="screen",
                        arguments=[
                            "/world/default/create@ros_gz_interfaces/srv/SpawnEntity",
                            "/world/default/remove@ros_gz_interfaces/srv/DeleteEntity",
                            "/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose",
                            "/world/default/control@ros_gz_interfaces/srv/ControlWorld",
                        ],
                        parameters=[{"use_sim_time": True}],
                    )
                ]
            )
        )
        await asyncio.gather(
            self._service_control_world.ensure(),
            self._service_delete_entity.ensure(),
        )

    async def pause(self) -> bool:
        async with self._semaphore:
            self._logger.debug("Attempting to pause simulation")
            request = ControlWorld.Request()
            request.world_control = WorldControl()
            request.world_control.pause = True
            try:
                result = await self._service_control_world.call_timeout(request)
                if result is None:
                    self._logger.error("Pause service call failed")
                    return False
                self._logger.debug(f"Pause result: {result.success}")
                return result.success
            except Exception as e:
                self._logger.error(f"Error pausing simulation: {str(e)}")
                traceback.print_exc()
                return False

    async def _set_physics_rtf(self, value: float) -> bool:
        # a sparse request zeroes gravity (models levitate), so send the world defaults
        req = f'real_time_factor: {value}, gravity: {{x: 0, y: 0, z: -9.8}}, magnetic_field: {{x: 5.5645e-06, y: 2.28758e-05, z: -4.23884e-05}}'
        for attempt in range(1, _SET_PHYSICS_ATTEMPTS + 1):
            started = time.monotonic()
            process = await asyncio.create_subprocess_exec(
                'gz',
                'service',
                '-s',
                '/world/default/set_physics',
                '--reqtype',
                'gz.msgs.Physics',
                '--reptype',
                'gz.msgs.Boolean',
                '--timeout',
                str(_SET_PHYSICS_TIMEOUT_MS),
                '--req',
                req,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await process.communicate()
            reply = out.decode(errors='replace').strip()
            # the CLI exits 0 on a timed-out call, only the echoed reply proves delivery
            if process.returncode == 0 and 'data: true' in reply:
                self._logger.info(f"gz real_time_factor set to {value:g} in {time.monotonic() - started:.2f}s")
                return True
            self._logger.warning(f"set_physics real_time_factor={value:g} attempt {attempt}/{_SET_PHYSICS_ATTEMPTS} failed: {reply or f'rc={process.returncode}'}")
        return False

    async def unpause(self) -> bool:
        # returning to free-run: restore gz's realtime pacing
        if self._rtf_boosted:
            self._rtf_boosted = not await self._set_physics_rtf(1.0)
        if self._batch_target is not None:
            await self._await_batches_done(self._batch_target)
            self._batch_target = None
        async with self._semaphore:
            self._logger.debug("Attempting to unpause simulation")
            request = ControlWorld.Request()
            request.world_control = WorldControl()
            request.world_control.pause = False
            try:
                result = await self._service_control_world.call_timeout(request)
                if result is None:
                    self._logger.error("Unpause service call failed")
                    return False
                self._logger.debug(f"Unpause result: {result.success}")
                return result.success
            except Exception as e:
                self._logger.error(f"Error unpausing simulation: {str(e)}")
                traceback.print_exc()
                return False

    async def step_seconds(self, seconds: float) -> float:
        """Advance sim time by paused ControlWorld multi_step requests. Raises on service failure."""
        n = round(seconds / self._dt)
        if seconds > 0:
            n = max(1, n)
        if n <= 0:
            return 0.0
        # gz throttles stepped iterations to real_time_factor x wall clock, so
        # lift the limiter while stepping drives the sim, unpause() restores it
        if n > 1 and not self._rtf_boosted:
            self._rtf_boosted = await self._set_physics_rtf(_RTF_BOOST)
        # gz acks multi_step on acceptance and completion is observed via /clock, so
        # record the target before sending (half-tick slack for float vs ns rounding)
        target = self._node.sim_time + Time.from_float((n - 0.5) * self._dt)
        # send in chunks of at most one sim second: pending iterations cannot be
        # cancelled, so a caller that dies mid-step orphans at most one chunk
        max_chunk = max(1, round(1.0 / self._dt))
        # gz silently drops multi_step when the request lands while a previous
        # batch is still executing (Paused() is false mid-batch), so watch for
        # motion and resend the remaining ticks whenever the clock freezes
        while self._node.sim_time < target:
            start_sim = self._node.sim_time
            remaining = max(1, round((target - start_sim).to_seconds() / self._dt + 0.5))
            chunk = min(max_chunk, remaining)
            chunk_target = min(target, start_sim + Time.from_float((chunk - 0.5) * self._dt))
            await self._send_multi_step(chunk, seconds)
            await self._node.await_sim_time(chunk_target, freeze_timeout=1.0)
        self._batch_target = target
        return n * self._dt

    async def _await_batches_done(self, target: Time) -> None:
        """Wait until gz paused at or past target: an unpause sent while a multi_step batch is pending is undone when the batch ends."""
        target_s = target.to_seconds()
        last_sim = -1.0
        stalled_since = time.monotonic()
        while True:
            stats = self._stats
            sim = stats.sim_time.sec + stats.sim_time.nsec * 1e-9 if stats is not None else -1.0
            if stats is not None and stats.paused and sim >= target_s:
                return
            if sim > last_sim:
                last_sim, stalled_since = sim, time.monotonic()
            elif time.monotonic() - stalled_since > _BATCH_STALL_S:
                self._logger.warning(f"gz did not report a finished step batch within {_BATCH_STALL_S:.0f}s of stalling (sim {sim:.3f}s, target {target_s:.3f}s)")
                return
            await asyncio.sleep(_BATCH_POLL_S)

    async def _send_multi_step(self, n: int, seconds: float) -> None:
        async with self._semaphore:
            request = ControlWorld.Request()
            request.world_control = WorldControl()
            # pause must be set in the same request as multi_step, else gz free-runs.
            request.world_control.pause = True
            request.world_control.multi_step = n
            try:
                result = await self._service_control_world.call_timeout(request)
            except Exception as e:
                self._logger.error(f"Error stepping simulation by {n}: {str(e)}")
                traceback.print_exc()
                raise RuntimeError(f"step_seconds({seconds}) failed: {e}") from e
            if result is None or not result.success:
                raise RuntimeError(f"step_seconds({seconds}) service call failed")

    async def cleanup_namespace(self, prefix: str) -> int:
        removed = 0
        targets: list[str] = []
        for attempt in range(_CLEANUP_ATTEMPTS + 1):
            try:
                names = await _list_models()
            except SimUnavailable:
                if attempt == _CLEANUP_ATTEMPTS:
                    raise
                await asyncio.sleep(1.0 + attempt)
                continue
            targets = [n for n in names if n.startswith(prefix)]
            if not targets:
                return removed
            if attempt == _CLEANUP_ATTEMPTS:
                break
            results = await asyncio.gather(*(self._delete_model(n) for n in targets))
            removed += sum(1 for r in results if r)
        raise SimUnavailable(f"cleanup_namespace {prefix!r}: {len(targets)} model(s) left after {_CLEANUP_ATTEMPTS} attempts")

    async def _delete_model(self, name: str) -> bool:
        req = DeleteEntity.Request()
        req.entity = EntityMsg(name=name, type=EntityMsg.MODEL)
        async with self._semaphore:
            try:
                res = await self._service_delete_entity.call_timeout(req)
            except Exception as e:
                self._logger.warning(f"cleanup_namespace: delete {name} raised: {e!r}")
                return False
        return bool(res) and res.success


class GazeboSimulator(BaseSim):
    SIM_NAME = 'gazebo'
    STACKS_LEVELS = True
    ARTICULATED_PHYSICS = True

    def __init__(self, **kwargs: typing.Unpack[SimKwargs]) -> None:
        super().__init__(**kwargs)

        self._semaphore = asyncio.Semaphore(5)

        self._logger.info(f"Initializing GazeboSimulator with namespace: {self._namespace}")

        self.entities: dict[str, Entity] = {}
        self._walls_entities: list[str] = []
        self._wall_counter = itertools.count()
        self._light_entities: dict[str, list[str]] = {}
        self._default_lights_scaled = False
        self._gz_node = gz.transport13.Node()
        self._light_publisher = self._gz_node.advertise(_LIGHT_CONFIG_TOPIC, light_pb2.Light)
        self._material_publisher = self._gz_node.advertise(_MATERIAL_COLOR_TOPIC, material_color_pb2.MaterialColor)
        self._glow_visuals: dict[str, tuple[str, dict[str, tuple[str, tuple[float, float, float]]]]] = {}
        self._glow_lights: dict[str, tuple[Light, float]] = {}
        self._material_texture_cache: dict[str, dict[str, str]] = {}
        self._spawned_names: set[str] = set()
        self._entity_ids: dict[str, int] = {}
        self._id_wanted: set[str] = set()
        self._joint_commands: dict[str, gz.transport13.Publisher] = {}
        self._joint_states: dict[str, list[float]] = {}

        # gz leaves a model unactuated after a SetEntityPose issued while the
        # world is paused; re-issuing the same pose once it runs wakes it.
        self._repin: dict[str, Robot] = {}
        self._paused: bool | None = None

        self._viewport_camera_pose: Pose | None = None
        self._pose_batch: asyncio.Future | None = None
        self._moving_boxes: dict[str, Orientation] = {}
        self._odom_frames: dict[str, str] = {}
        self._odom_broadcaster = StaticTransformBroadcaster(self.node)
        self._velocity_publishers: dict[str, gz.transport13.Publisher] = {}

    def _robot_loader_args(self, robot: Robot) -> dict[str, object]:
        args: dict[str, object] = {
            **robot.extra,
            **attrs.asdict(robot, filter=lambda attribute, _: attribute.name != 'extra'),
            'sim_path': robot.sim_path,
            'optim': self.node.rosparam[str].get('optim', ''),
        }
        args.pop('resolved_assembly', None)
        robot_config = arena_robots.Robot.RobotIdentifier(robot.model.name).resolve_sync()
        args['sensor_topic_patches'] = arena_robots.Sensor.topic_elements(
            robot_config.effective_sensors(robot.resolved_request, frames=robot.frames),
            f"/model/{robot.sim_path}",
        )
        if robot.resolved_assembly is not None:
            assert robot_config.assembly is not None
            catalog = arena_robots.catalog.Catalog()
            args['xacro_wrapper'] = arena_robots.catalog.render_wrapper_xacro(robot_config, robot.resolved_assembly, catalog=catalog)
            args['control_joint_patch'] = arena_robots.catalog.render_control_joints(robot.resolved_assembly, catalog, prefix=robot_config.assembly.prefix)
        control_spec = robot_config.model_params.control
        if control_spec is None or not control_spec.is_ros2_control:
            return args
        args['namespace'] = str(self.node.service_namespace(robot.name))
        if control_spec.config is not None:
            args['gazebo_controllers'] = self._render_ros2_control_yaml(
                robot,
                control_spec.config,
                frame_prefix=robot.frame.tf(),
                control_prefix=(robot_config.assembly.prefix if robot_config.assembly is not None else 'robot_'),
            )
        return args

    def _render_ros2_control_yaml(self, robot: Robot, config_uri: str, *, frame_prefix: str, control_prefix: str) -> str:
        return effective_control_yaml(robot.resolved_assembly, config_uri, robot.sim_path, frame_prefix, sim=SimSimulator.GAZEBO, prefix=control_prefix)

    def _placed_light(self, params: dict[str, typing.Any]) -> dict[str, typing.Any]:
        x, y, z = params['pose']
        dx, dy, dz = self._shift(x, y)
        return {**params, 'pose': (x + dx, y + dy, z + dz)}

    def _publish_odom(self, sim_path: str) -> None:
        """Publish the robot's map -> odom, the inverse shift of its level."""
        frame = self._odom_frames.get(sim_path)
        if frame is None:
            return
        dx, dy, dz = self._robot_shift(sim_path)
        transform = TransformStamped()
        transform.header.frame_id = 'map'
        transform.child_frame_id = frame
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = -dx, -dy, -dz
        transform.transform.rotation.w = 1.0
        self._odom_broadcaster.sendTransform(transform)

    async def obstacle_spawn(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        level = obstacles_optim_level(self.node)
        return await asyncio.gather(*(self._spawn_obstacle(o, level) for o in obstacles))

    async def _spawn_obstacle(self, obstacle: Obstacle, level: ObstaclesOptim) -> bool:
        if level is ObstaclesOptim.BBOX and (box := await resolve_obstacle_box(obstacle)) is not None:
            return await self._spawn_box_obstacle(obstacle, *box)
        return await self._spawn_entity(obstacle)

    async def _spawn_box_obstacle(self, obstacle: Obstacle, size: tuple[float, float, float], center: tuple[float, float, float]) -> bool:
        sdf = _generate_box_sdf(obstacle.sim_path, size, center)
        async with self._semaphore:
            ok = await self._spawn_sdf(obstacle.sim_path, sdf, self._placed(obstacle.pose))
        if ok:
            self.entities[obstacle.name] = obstacle
        return ok

    async def pedestrian_spawn(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        # Ped actors are created and removed by PedSkeletonPlugin from arena_peds.
        return tuple(True for _ in pedestrians)

    def robot_controllers(self, robot: Robot) -> list[str]:
        return robot_controllers(arena_robots.Robot.RobotIdentifier(robot.model.name).resolve_sync(), robot.resolved_assembly)

    async def robot_spawn(self, robots: Sequence[Robot]) -> Sequence[bool]:

        async def impl(robot: Robot) -> bool:
            await self._sweep_entity_name(robot.sim_path)
            if not await self._spawn_entity(robot):
                return False
            robot_params = arena_robots.Robot.RobotIdentifier(robot.model.name).resolve_sync().model_params
            self._odom_frames[robot.sim_path] = robot.frame.tf(robot_params.odom_frame)
            self._track_level(robot.sim_path, robot.pose)
            _loader_args = self._robot_loader_args(robot)
            model = await (await robot.model.resolve()).model.get(ModelType.URDF, loader_args=_loader_args)
            if model.type is ModelType.UNKNOWN:
                self._logger.error(f"robot {robot.name!r} resolved to an unknown model type after spawning; removing the entity again")
                await self._delete_entity(robot.sim_path)
                return False
            self._id_wanted.add(robot.sim_path)
            if await self._resolve_entity_id(robot.sim_path, timeout=_ENTITY_ID_TIMEOUT) is None:
                self._logger.warning(f"entity id of {robot.sim_path!r} not in scene/info within {_ENTITY_ID_TIMEOUT}s, will retry at the first teleport")
            model_description = model.description
            self._robot_initialpose(robot)
            await self._robot_bridge(robot, model_description)
            robot_config = arena_robots.Robot.RobotIdentifier(robot.model.name).resolve_sync()
            self._register_agent_robot(robot, robot_config.model_params)
            return True

        success = await asyncio.gather(*map(impl, robots))
        return success

    async def obstacle_move(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        async def move(obstacle: Obstacle) -> bool:
            runtime = self._articulated.get(obstacle.name)
            if runtime is None or not runtime.physical:
                return await self._move_entity(obstacle)
            await self._delete_entity(obstacle.sim_path)
            return (await self._articulated_spawn([runtime]))[0]

        return await asyncio.gather(*map(move, obstacles))

    async def pedestrian_move(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        # Ped pose is owned by PedSkeletonPlugin (driven from the arena_peds topic).
        return tuple(True for _ in pedestrians)

    async def robot_move(self, robots: Sequence[Robot]) -> Sequence[bool]:
        async def impl(robot: Robot) -> bool:
            if not await self._move_entity(robot):
                return False
            self._track_level(robot.sim_path, robot.pose)
            if self._paused:
                self._repin[robot.sim_path] = robot
            return await self._robot_move(robot)

        return await asyncio.gather(*map(impl, robots))

    async def _on_paused(self, msg: Bool) -> None:
        was_paused, self._paused = self._paused, msg.data
        if not was_paused or msg.data or not self._repin:
            return
        robots = list(self._repin.values())
        self._repin.clear()
        await asyncio.gather(*map(self._move_entity, robots))

    async def obstacle_delete(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        for obstacle in obstacles:
            if self._joint_commands.pop(obstacle.sim_path, None) is not None:
                self._gz_node.unsubscribe(_joint_topic(obstacle.sim_path, "state"))
                self._joint_states.pop(obstacle.sim_path, None)
        return await asyncio.gather(*(self._delete_entity(o.sim_path) for o in obstacles))

    async def _articulated_spawn(self, objects: Sequence[ArticulatedObject]) -> Sequence[bool]:
        async def spawn(runtime: ArticulatedObject) -> bool:
            obstacle = runtime.obstacle
            links: list[tuple[str, Joint | None, Path | None]] = []
            for link, joint in runtime.tree():
                view = runtime.links.get(link)
                model = await view.model.get(ModelType.SDF) if view is not None else None
                links.append((link, joint, _sdf_file(model.path) if model is not None and model.type is ModelType.SDF and model.path is not None else None))
            name = obstacle.sim_path
            self._joint_states.pop(name, None)
            if name not in self._joint_commands:
                self._joint_commands[name] = self._gz_node.advertise(_joint_topic(name, "command"), double_v_pb2.Double_V)
                self._gz_node.subscribe(double_v_pb2.Double_V, _joint_topic(name, "state"), lambda msg, name=name: self._joint_states.__setitem__(name, list(msg.data)))
            async with self._semaphore:
                return await self._spawn_sdf(name, _articulated_sdf(name, runtime, links), obstacle.pose)

        return await asyncio.gather(*map(spawn, objects))

    async def _articulated_drive(self, commands: Sequence[ArticulatedCommand]) -> Sequence[Values | None]:
        readings: list[Values | None] = []
        for command in commands:
            name = command.runtime.obstacle.sim_path
            fields = command_fields(command)
            message = double_v_pb2.Double_V(data=[value for triple in zip(fields["reference"], fields["drive"], fields["place"], strict=True) for value in map(float, triple)])
            if (publisher := self._joint_commands.get(name)) is not None:
                publisher.publish(message)
            state = self._joint_states.get(name)
            readings.append(None if state is None or len(state) != len(fields["joints"]) else dict(zip(fields["joints"], state, strict=True)))
        return readings

    async def pedestrian_delete(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        # PedSkeletonPlugin removes a ped's actor when its name leaves arena_peds.
        return tuple(True for _ in pedestrians)

    async def robot_delete(self, robots: Sequence[Robot]) -> Sequence[bool]:
        for robot in robots:
            self._forget_agent_robot(robot.sim_path)
        return await asyncio.gather(*(self._delete_entity(robot.sim_path) for robot in robots))

    async def pedestrian_update(self, pedestrians: Pedestrians) -> Sequence[bool]:
        # Ped pose is owned by PedSkeletonPlugin (driven from the arena_peds topic).
        return tuple(True for _ in pedestrians.pedestrians)

    async def spawn_floors(self, floors: Sequence[Floor]) -> bool:
        for floor in floors:
            name = self._realizer.realize(f"floor_{next(self._wall_counter)}")
            textures = await self._resolve_wall_textures(floor.material)
            floor_sdf = _generate_floor_sdf(name, floor, textures)
            async with self._semaphore:
                if not await self._spawn_sdf(name, floor_sdf, self._offset(floor.pos.x, floor.pos.y)):
                    raise RuntimeError(f"spawn_floors: {name} did not spawn")
            self._walls_entities.append(name)
        return True

    async def spawn_ceilings(self, ceilings: Sequence[Ceiling]) -> bool:
        for ceiling in ceilings:
            name = self._realizer.realize(f"ceiling_{next(self._wall_counter)}")
            textures = await self._resolve_wall_textures(ceiling.material)
            ceiling_sdf = _generate_ceiling_sdf(name, ceiling, textures)
            async with self._semaphore:
                if not await self._spawn_sdf(name, ceiling_sdf, self._offset(ceiling.pos.x, ceiling.pos.y)):
                    raise RuntimeError(f"spawn_ceilings: {name} did not spawn")
            self._walls_entities.append(name)
        return True

    def _light_config(self, request: light_pb2.Light) -> bool:
        return self._light_publisher.has_connections() and self._light_publisher.publish(request)

    async def _light_link(self) -> bool:
        deadline = time.monotonic() + _LIGHT_LINK_TIMEOUT
        while not self._light_publisher.has_connections():
            if time.monotonic() > deadline:
                self._logger.warning(f"no subscriber on {_LIGHT_CONFIG_TOPIC}")
                return False
            await asyncio.sleep(0.05)
        return True

    @property
    def _owns_ambient(self) -> bool:
        return self._realizer.realize("light") == f"{_GZ_AMBIENT_ENV}/light"

    def _scale_default_lights(self, scale: float) -> bool:
        if not self._owns_ambient:
            return True
        return all([self._light_config(_default_light_request(name, direction, scale)) for name, direction in _GZ_DEFAULT_LIGHTS])

    async def _spawn_light_entity(self, name: str, params: dict[str, typing.Any]) -> bool:
        params = self._placed_light(params)
        async with self._semaphore:
            x, y, z = params['pose']
            return await self._spawn_sdf(name, _generate_light_sdf(name, params), Pose(position=Position(x=x, y=y, z=z)))

    async def _spawn_lights(self, lights: Sequence[Light]) -> bool:
        ok = await self._light_link()
        if any(not light.intrinsic for light in lights):
            scale = _default_light_scale([*self._lights.values(), *lights])
            assert scale is not None
            ok = self._scale_default_lights(scale) and ok
            self._default_lights_scaled = self._owns_ambient
        spawns = []
        for light in lights:
            if light.fixture == 'dome' or (light.spec.ambient and not self._owns_ambient):
                continue
            level = 0.0 if light.lit is False else light.dim
            names = []
            for params in _light_params(light, level, light.alive(light.dead_fraction)):
                names.append(self._realizer.realize(f"light_{next(self._wall_counter)}"))
                spawns.append(self._spawn_light_entity(names[-1], params))
            self._light_entities[light.name] = names
            self._glow(light, level)
        return all(await asyncio.gather(*spawns)) and ok

    async def _remove_lights(self, lights: Sequence[Light]) -> bool:
        results = [await self._delete_entity(name, EntityMsg.LIGHT) for light in lights for name in self._light_entities.pop(light.name, ())]
        for light in lights:
            self._glow_lights.pop(light.name, None)
        if self._default_lights_scaled and _default_light_scale(self._lights.values()) is None:
            self._default_lights_scaled = False
            results.append(self._scale_default_lights(1.0))
        return all(results)

    def _glow(self, light: Light, level: float) -> None:
        if not light.glow:
            return
        self._glow_lights[light.name] = (light, level)
        sim_path, visuals = self._glow_visuals.get(light.owner, ('', {}))
        if light.glow in visuals:
            visual, base = visuals[light.glow]
            self._material_publisher.publish(_glow_request(f'{sim_path}::{visual}', base, light.cct_K, level))

    async def _move_light(self, light: Light, position: Position, yaw: float) -> None:
        output = self._light_outputs.get(light.name)
        names = self._light_entities.get(light.name)
        if output is None or not names:
            return
        level, dead_fraction = output
        params = _light_params(light, level, light.alive(dead_fraction), position, yaw)
        if not all([self._light_config(_light_request(name, self._placed_light(p))) for name, p in zip(names, params, strict=True)]):
            raise RuntimeError(f"light {light.name!r} did not take its pose")

    async def _apply_light(self, light: Light, level: float, alive: Sequence[bool]) -> None:
        if light.fixture == 'dome':
            if not self._scale_default_lights(_gz_dome_scale(light.lux, level)):
                raise RuntimeError(f"light {light.name!r} did not take its state")
            return
        self._glow(light, level)
        names = self._light_entities.get(light.name)
        if not names:
            return
        pose = self._light_poses.get(light.name)
        params = _light_params(light, level, alive) if pose is None else _light_params(light, level, alive, *pose)
        if not all([self._light_config(_light_request(name, self._placed_light(p))) for name, p in zip(names, params, strict=True)]):
            raise RuntimeError(f"light {light.name!r} did not take its state")

    async def set_robot_pose(self, sim_path: str, pose: Pose) -> bool:
        if sim_path not in self._agent_robots:
            return False
        req = SetEntityPose.Request()
        req.entity = EntityMsg(name=sim_path, type=EntityMsg.MODEL)
        req.pose = self._placed(pose).to_msg()
        try:
            res = await self._service_set_entity_pose.call_timeout(req)
            if res and res.success:
                self._track_level(sim_path, pose)
            return bool(res and res.success)
        except Exception as e:
            self._logger.warning(f"set_robot_pose({sim_path!r}) failed: {e}")
            return False

    async def spawn_box(self, name: str, size: tuple[float, float, float], pose: Pose, material: MaterialIdentifier | None = None, *, velocity: tuple[float, float, float] | None = None, comb: bool = False) -> bool:
        textures = await self._resolve_wall_textures(material)
        if velocity is not None:
            self._moving_boxes[name] = pose.orientation
        sdf = _generate_box_sdf(name, size, textures=textures, velocity=None if velocity is None else _body_frame(pose.orientation, velocity), comb=comb)
        async with self._semaphore:
            return await self._spawn_sdf(name, sdf, self._placed(pose))

    async def set_box_velocities(self, velocities: Mapping[str, tuple[float, float, float]]) -> bool:
        ok = True
        for name, velocity in velocities.items():
            orientation = self._moving_boxes.get(name)
            if orientation is None:
                ok = False
                continue
            publisher = self._velocity_publishers.get(name)
            if publisher is None:
                publisher = self._velocity_publishers[name] = self._gz_node.advertise(f'/model/{name}/cmd_vel', twist_pb2.Twist)
            twist = twist_pb2.Twist()
            twist.linear.x, twist.linear.y, twist.linear.z = _body_frame(orientation, velocity)
            ok = publisher.publish(twist) and ok
        return ok

    async def move_box(self, name: str, pose: Pose) -> bool:
        # Fire-and-forget: animation is cosmetic, last-write-wins. Awaiting the service round-trip
        # would gate the shim's tick rate on bridge latency; skipping the semaphore lets parallel
        # door updates issue concurrently.
        request = SetEntityPose.Request()
        request.entity = EntityMsg(name=name, type=EntityMsg.MODEL)
        request.pose = self._placed(pose).to_msg()
        try:
            self._service_set_entity_pose.client.call_async(request)
        except Exception as e:
            self._logger.warning(f"move_box dispatch failed for {name}: {e}")
            return False
        return True

    async def move_boxes(self, poses: Mapping[str, Pose]) -> bool:
        if self._pose_batch is not None and not self._pose_batch.done():
            return False
        self._pose_batch = asyncio.get_running_loop().run_in_executor(
            None,
            self._gz_node.request,
            _SET_POSE_VECTOR_SERVICE,
            _pose_vector({name: self._placed(pose) for name, pose in poses.items()}),
            pose_v_pb2.Pose_V,
            boolean_pb2.Boolean,
            _POSE_BATCH_TIMEOUT_MS,
        )
        return True

    async def delete_box(self, name: str) -> bool:
        self._moving_boxes.pop(name, None)
        self._velocity_publishers.pop(name, None)
        return await self._delete_entity(name)

    # ViewportITF: drives the gz GUI user-camera over the ViewportCamera plugin's
    # /arena/viewport/* services. The viewport is global (one GUI), so every env's
    # adapter shares the same backend, and the services only exist while the GUI runs,
    # hence the short-timeout clients that fail soft when headless.

    async def viewport_set_view(
        self,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        fov: float = 0.0,
    ) -> bool:
        request = ViewportSetView.Request()
        request.eye = Point(x=float(eye[0]), y=float(eye[1]), z=float(eye[2]))
        request.target = Point(x=float(target[0]), y=float(target[1]), z=float(target[2]))
        request.fov = float(fov)
        return await self._viewport_call(self._service_viewport_set_view, request)

    _REFERENCE_MODES = {
        "full": ViewportSetReferenceFrame.Request.FULL,
        "yaw": ViewportSetReferenceFrame.Request.YAW_ONLY,
        "position": ViewportSetReferenceFrame.Request.POSITION_ONLY,
    }

    @staticmethod
    def _ros_pose(pose: Pose) -> RosPose:
        return RosPose(
            position=Point(x=pose.position.x, y=pose.position.y, z=pose.position.z),
            orientation=Quaternion(
                w=pose.orientation.w,
                x=pose.orientation.x,
                y=pose.orientation.y,
                z=pose.orientation.z,
            ),
        )

    async def viewport_set_reference_frame(
        self,
        entity: str = "",
        pose: Pose | None = None,
        mode: str = "full",
    ) -> bool:
        request = ViewportSetReferenceFrame.Request()
        request.entity = entity
        request.has_pose = pose is not None
        if pose is not None:
            request.pose = self._ros_pose(pose)
        request.mode = self._REFERENCE_MODES[mode]
        return await self._viewport_call(self._service_viewport_set_reference_frame, request)

    def viewport_stream_view(
        self,
        pose: Pose,
        world_orientation: bool = False,
        fov: float = 0.0,
    ) -> bool:
        msg = ViewportView()
        msg.pose = self._ros_pose(pose)
        msg.world_orientation = bool(world_orientation)
        msg.fov = float(fov)
        self._publisher_viewport_view.publish(msg)
        return True

    async def viewport_set_projection(self, projection: str) -> bool:
        request = ViewportSetProjection.Request()
        request.projection = projection
        return await self._viewport_call(self._service_viewport_set_projection, request)

    def viewport_camera_pose(self) -> Pose | None:
        return self._viewport_camera_pose

    async def _viewport_call(self, service: ClientWrapper, request: object) -> bool:
        try:
            result = await service.call_timeout(request)
        except Exception as e:
            self._logger.warning(f"viewport service call failed: {e}")
            return False
        if result is None:
            self._logger.warning("viewport service unavailable, is the gazebo GUI running?")
            return False
        return result.success

    def _on_viewport_camera_pose(self, msg: PoseStamped) -> None:
        position, orientation = msg.pose.position, msg.pose.orientation
        self._viewport_camera_pose = Pose(
            position=Position(x=position.x, y=position.y, z=position.z),
            orientation=Orientation(w=orientation.w, x=orientation.x, y=orientation.y, z=orientation.z),
        )

    # IMPL

    async def _move_entity(self, entity: Entity, entity_type: int = EntityMsg.MODEL) -> bool:
        async with self._semaphore:
            if entity.sim_path in self._id_wanted and entity.sim_path not in self._entity_ids:
                await self._resolve_entity_id(entity.sim_path, timeout=0.0)
            self._logger.debug(f"Attempting to move entity: {entity.sim_path}")
            self._logger.debug(f"Moving entity {entity.sim_path} to position: {entity.pose}")

            request = SetEntityPose.Request()
            request.entity = EntityMsg(
                id=self._entity_ids.get(entity.sim_path, 0),
                name=entity.sim_path,
                type=entity_type,
            )
            request.pose = self._placed(entity.pose).to_msg()

            try:
                await self._service_set_entity_pose.ensure()
                result = await self._service_set_entity_pose.call_timeout(request)

                if result is None:
                    self._logger.error(f"Move service call failed for {entity.sim_path}")
                    return False

                self._logger.debug(f"Move result for {entity.sim_path}: {result.success}")

                return result.success

            except Exception as e:
                self._logger.error(f"Error moving entity {entity.sim_path}: {str(e)}")
                traceback.print_exc()
                return False

    async def _spawn_entity(self, entity: Entity) -> bool:
        async with self._semaphore:
            try:
                try:
                    if isinstance(entity, Robot):
                        _loader_args = self._robot_loader_args(entity)
                        model = await (await entity.model.resolve()).model.get(ModelType.URDF, loader_args=_loader_args)
                    else:
                        model = await (await entity.model.resolve()).model.get(ModelType.SDF)
                except Exception:
                    self._logger.warning(f"Failed to resolve model for entity {entity.name}")
                    self._logger.debug(traceback.format_exc())
                    return False

                if model.type is ModelType.UNKNOWN:
                    self._logger.warning(f"Failed to resolve model for entity {entity.name}: unknown model type {model}")
                    return False

                ok = await self._spawn_model(entity.sim_path, model, self._placed(entity.pose))
                if ok and (model.path is None or model.type is ModelType.URDF):
                    self.entities[entity.name] = entity
                if ok and model.path and model.type is not ModelType.URDF:
                    self._bind_glow(entity, _sdf_file(model.path))
                return ok

            except SimUnavailable:
                raise
            except Exception as e:
                self._logger.error(f"Error spawning entity {entity.name}: {str(e)}")
                traceback.print_exc()
                return False

    def _bind_glow(self, entity: Entity, sdf_path: Path) -> None:
        visuals = _glow_visuals(sdf_path) if sdf_path.is_file() else {}
        if not visuals:
            self._glow_visuals.pop(entity.name, None)
            return
        self._glow_visuals[entity.name] = (entity.sim_path, visuals)
        for light, level in list(self._glow_lights.values()):
            if light.owner == entity.name:
                self._glow(light, level)

    async def _spawn_model(self, name: str, model: Model, pose: Pose) -> bool:
        """Spawn an already-resolved model under `name` at `pose`. Caller holds self._semaphore."""
        if model.path and model.type not in (ModelType.URDF,):
            return await self._spawn_factory(name, EntityFactory(sdf_filename=str(_sdf_file(model.path))), pose)
        return await self._spawn_sdf(name, model.description, pose)

    async def _spawn_sdf(self, name: str, sdf: str, pose: Pose) -> bool:
        """Spawn from an in-memory SDF via ros_gz_bridge. Caller holds self._semaphore."""
        return await self._spawn_factory(name, EntityFactory(sdf=sdf), pose)

    async def _spawn_factory(self, name: str, factory: EntityFactory, pose: Pose) -> bool:
        """Create `factory` as `name` at `pose` through the bridged create service. Caller holds self._semaphore."""
        factory.name = name
        factory.pose = pose.to_msg()
        request = SpawnEntity.Request(entity_factory=factory)

        try:
            result = await self._service_spawn_entity.call_timeout(request)
        except SimUnavailable:
            raise
        except Exception as e:
            self._logger.error(f"Spawn service call raised for {name}: {e}")
            return False

        if result is not None and result.success:
            self._spawned_names.add(name)
            return True

        if name in await _list_models():
            self._spawned_names.add(name)
            return True

        self._logger.error(f"Failed to spawn {name}: {'timed out' if result is None else 'rejected'}")
        req = DeleteEntity.Request()
        req.entity = EntityMsg(name=name, type=EntityMsg.MODEL)
        await self._service_delete_entity.call_timeout(req)
        return False

    async def _delete_entity(self, sim_path: str, entity_type: int = EntityMsg.MODEL) -> bool:
        async with self._semaphore:
            self._logger.debug(f"Attempting to delete entity: {sim_path}")

            request = DeleteEntity.Request()
            request.entity = EntityMsg(
                id=self._entity_ids.get(sim_path, 0),
                name=sim_path,
                type=entity_type,
            )

            try:
                result = await self._service_delete_entity.call_timeout(request)
            except SimUnavailable:
                raise
            except Exception as e:
                self._logger.error(f"Error deleting entity {sim_path}: {str(e)}")
                traceback.print_exc()
                return False

            if result is None:
                raise SimUnavailable(f"delete_entity({sim_path}) timed out")

            self._logger.debug(f"Delete result for {sim_path}: {result.success}")

            if result.success:
                self.entities.pop(sim_path, None)
                self._spawned_names.discard(sim_path)
                self._entity_ids.pop(sim_path, None)
                self._id_wanted.discard(sim_path)

            return result.success

    async def _sweep_entity_name(self, sim_path: str) -> None:
        """Bounded delete-by-name of whatever holds ``sim_path``, drained by one sim step before the spawn that follows."""
        for _ in range(_SPAWN_SWEEP_ROUNDS):
            await self._delete_entity(sim_path)
            await asyncio.sleep(_SPAWN_SWEEP_INTERVAL)
        await self.node.await_sim_step(_SPAWN_SWEEP_DRAIN_TIMEOUT)

    async def _scene_model_ids(self) -> dict[str, int]:
        proc = await asyncio.create_subprocess_exec(
            'gz',
            'service',
            '-s',
            '/world/default/scene/info',
            '--reqtype',
            'gz.msgs.Empty',
            '--reptype',
            'gz.msgs.Scene',
            '--timeout',
            str(int(_ENTITY_ID_TIMEOUT * 1000)),
            '--req',
            '',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            self._logger.warning(f"gz scene/info failed: {stderr.decode().strip()}")
            return {}
        return parse_scene_models(stdout.decode())

    async def _resolve_entity_id(self, sim_path: str, *, timeout: float) -> int | None:
        """Cache the gz entity id behind ``sim_path``, polling scene/info for up to ``timeout``."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            models = await self._scene_model_ids()
            entity_id = models.get(sim_path)
            if entity_id is not None:
                self._entity_ids[sim_path] = entity_id
                self._logger.info(f"entity id of {sim_path!r} is {entity_id} ({len(models)} models listed)")
                return entity_id
            if loop.time() >= deadline:
                self._logger.debug(f"entity id of {sim_path!r} not in scene/info ({len(models)} models listed)")
                return None
            await asyncio.sleep(_ENTITY_ID_POLL)

    async def step(self, n: int = 1) -> bool:
        async with self._semaphore:
            request = ControlWorld.Request()
            request.world_control = WorldControl()
            # gz queues multi_step only if already paused, so pause is applied first from
            # this same message, then the runner steps n iterations and re-pauses
            request.world_control.pause = True
            request.world_control.multi_step = n
            try:
                result = await self._service_control_world.call_timeout(request)
                if result is None:
                    self._logger.error("Step service call failed")
                    return False
                return result.success
            except Exception as e:
                self._logger.error(f"Error stepping simulation: {str(e)}")
                traceback.print_exc()
                return False

    async def _resolve_wall_textures(self, material: MaterialIdentifier | None) -> dict[str, str]:
        """Resolve a wall material to absolute PBR texture paths, cached per identifier. Empty for unset/unresolvable materials."""
        if material is None:
            return {}
        key = f"{material!r}|{material.modifiers!r}"
        cached = self._material_texture_cache.get(key)
        if cached is not None:
            return cached
        result: dict[str, str] = {}
        try:
            resolved = await material.resolve()
            mdl = MdlUtil(Path(resolved.path))
            for name, slot in (("albedo", "diffuse_texture"), ("normal", "normalmap_texture")):
                tex = mdl.texture(slot)
                if tex is not None and tex.exists():
                    result[name] = str(tex)
        except Exception as e:
            self._logger.warning(f"Failed to resolve wall material textures for {material!r}: {e}")
            result = {}
        self._material_texture_cache[key] = result
        return result

    async def spawn_walls(self, walls: Sequence[Wall], clear_existing: bool = True) -> bool:
        if clear_existing:
            await self.remove_world()
        for wall in walls:
            segments, obstacles = await realize_renderable(wall, (ModelType.SDF,))
            boxes = []
            for segment in segments:
                boxes.append((segment, await self._resolve_wall_textures(segment.material)))
            if boxes:
                wall_name = self._realizer.realize(f"wall_{next(self._wall_counter)}")
                wall_sdf = _generate_wall_sdf(wall_name, boxes)
                if wall_sdf:
                    async with self._semaphore:
                        if not await self._spawn_sdf(wall_name, wall_sdf, self._offset((wall.start.x + wall.end.x) / 2.0, (wall.start.y + wall.end.y) / 2.0)):
                            raise RuntimeError(f"spawn_walls: {wall_name} (wall from {wall.start} to {wall.end}) did not spawn")
                    self._walls_entities.append(wall_name)
            for obstacle, model in obstacles:
                obstacle.sim_path = self._realizer.realize(f"wall_obj_{next(self._wall_counter)}")
                async with self._semaphore:
                    ok = await self._spawn_model(obstacle.sim_path, model, self._placed(obstacle.pose))
                if ok:
                    self._walls_entities.append(obstacle.sim_path)
        return True

    async def remove_world(self) -> bool:
        for entity in self._walls_entities:
            await self._delete_entity(entity)
        self._walls_entities = []
        return True

    async def _robot_bridge(self, robot: Robot, description: str):
        launch_description = launch.LaunchDescription()

        launch_description.add_action(launch_ros.actions.PushRosNamespace(namespace=self.node.service_namespace(robot.name)))

        robot_config = arena_robots.Robot.RobotIdentifier(robot.model.name).resolve_sync()

        subs = {
            "robot_name": robot.sim_path,
            "world": "/world/default",
        }
        file_mappings = BridgeConfiguration.from_file(robot_config.mappings)
        # Derived gz topics are already concrete, so dedupe against the file mappings' own
        # substituted form rather than their raw placeholder strings.
        existing = {(m.gz_topic, m.ros_topic) for m in file_mappings.substitute(subs)}
        for gz_topic, ros_topic, ros_type, gz_type in arena_robots.Sensor.bridge_rows(robot_config.effective_sensors(robot.resolved_request, frames=robot.frames), f"/model/{robot.sim_path}"):
            if (gz_topic, ros_topic) in existing:
                continue
            file_mappings.append(_TopicMapping(gz_topic=gz_topic, ros_topic=ros_topic, ros_type=ros_type, gz_type=gz_type, direction=MappingDirection.GZ_TO_ROS))

        mappings = file_mappings.substitute(subs)

        bridge_arguments = mappings.as_args()
        remappings = mappings.as_remappings()

        # Add parameter_bridge node
        launch_description.add_action(
            launch_ros.actions.Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                output="screen",
                arguments=bridge_arguments,
                remappings=remappings,
                parameters=[{"use_sim_time": True}],
            )
        )
        launch_description.add_action(
            launch_ros.actions.Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {"use_sim_time": True},
                    {"robot_description": description},
                    {"frame_prefix": robot.frame.tf()},
                ],
            )
        )

        control_spec = robot_config.model_params.control
        is_ros2_control = control_spec is not None and control_spec.is_ros2_control
        ros2_control_spec = control_spec if is_ros2_control else None
        launch_description.add_action(
            launch_ros.actions.Node(
                package="pose_to_tf",
                executable="pose_to_tf",
                name="pose_to_tf",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                        "parent_frame": robot.frame.tf(robot_config.model_params.odom_frame),
                        "child_frame": robot.frame.tf(robot_config.model_params.base_frame),
                        "pose_topic": "pose",
                        "odom_in_topic": (ros2_control_spec.odom_topic if ros2_control_spec is not None and ros2_control_spec.odom_topic != "odom" else ""),
                    }
                ],
            )
        )

        if is_ros2_control:
            assert ros2_control_spec is not None
            if not ros2_control_spec.controllers:
                raise ValueError(f"control.mode=ros2_control but no controllers declared for {robot.name}")
            launch_description.add_action(
                twist_stamper_node(
                    ros2_control_spec.cmd_vel_topic,
                    robot.frame.tf(robot_config.model_params.base_frame),
                )
            )

        # launch_description.add_action(
        #     launch_ros.actions.Node(
        #         package='joint_state_publisher',
        #         executable='joint_state_publisher',
        #         output='screen',
        #         parameters=[
        #             {'use_sim_time': True},
        #             {'robot_description': description},  # Ensure URDF is passed here too
        #         ],
        #         remappings=[('/joint_states', '/joint_states')]
        #     )
        # )
        await self.node.do_launch(launch_description)

    def _robot_initialpose(self, robot: Robot):
        pose = PoseWithCovarianceStamped()
        pose.pose.pose = robot.pose.to_msg()
        pose.header.frame_id = "map"

        self.node.create_publisher(
            PoseWithCovarianceStamped,
            self.node.service_namespace(robot.name, "initialpose"),
            qos_profile=1,
        ).publish(pose)

    async def _robot_move(self, robot: Robot) -> bool:
        name = robot.name
        try:
            self._robot_initialpose(robot)

            max_attempts = 3
            attempt = 1
            initial_pose_triggered = False

            while attempt <= max_attempts and not initial_pose_triggered:
                self._logger.debug(f"Attempt {attempt}/{max_attempts}: Triggering initial pose update for robot {name}")
                try:
                    self._robot_initialpose(robot)
                    initial_pose_triggered = True
                    self._logger.debug(f"Initial pose update for {name} succeeded on attempt {attempt}")
                except Exception as e:
                    self._logger.error(f"Attempt {attempt}/{max_attempts} failed for {name}: {str(e)}")
                    traceback.print_exc()
                    if attempt < max_attempts:
                        self._logger.debug("Waiting 1 second before retrying...")
                        time.sleep(1)
                    attempt += 1

            if not initial_pose_triggered:
                self._logger.error(f"Failed to set initial pose for {name} after {max_attempts} attempts")

            return True

        except Exception as e:
            self._logger.error(f"Error moving robot {name}: {str(e)}")
            return False

    async def _set_up_services(self):
        futures: list[typing.Awaitable] = []

        # Initialize service clients
        # https://gazebosim.org/api/sim/8/entity_creation.html
        self._service_spawn_entity = self.node.create_client_wrapper(
            SpawnEntity,
            "/world/default/create",
        )
        self._service_delete_entity = self.node.create_client_wrapper(
            DeleteEntity,
            "/world/default/remove",
        )
        self._service_set_entity_pose = self.node.create_client_wrapper(
            SetEntityPose,
            "/world/default/set_pose",
        )
        self._service_control_world = self.node.create_client_wrapper(
            ControlWorld,
            "/world/default/control",
        )
        self._logger.info("Waiting for gazebo services...")
        services = (
            (self._service_spawn_entity, "spawn entity"),
            (self._service_delete_entity, "delete entity"),
            (self._service_set_entity_pose, "set entity pose"),
            (self._service_control_world, "control world"),
        )

        for service, name in services:
            self._logger.debug(f"Waiting for {name} service...")
            futures.append(service.ensure())

        await asyncio.gather(*futures)
        self._logger.info("All Gazebo services are available now.")

        # Viewport camera control is optional (GUI-only) and must not gate readiness:
        # short timeout so calls fail soft when no GUI is up, and no ensure() here.
        self._service_viewport_set_view = self.node.create_client_wrapper(ViewportSetView, "/arena/viewport/set_view", timeout=3.0)
        self._service_viewport_set_reference_frame = self.node.create_client_wrapper(ViewportSetReferenceFrame, "/arena/viewport/set_reference_frame", timeout=3.0)
        self._service_viewport_set_projection = self.node.create_client_wrapper(ViewportSetProjection, "/arena/viewport/set_projection", timeout=3.0)
        self._publisher_viewport_view = self.node.create_publisher(ViewportView, "/arena/viewport/cmd_view", _VIEWPORT_STREAM_QOS)
        self.node.create_subscription(PoseStamped, "/arena/viewport/camera_pose", self._on_viewport_camera_pose, 10)
        self.node.create_subscription(Bool, "/arena/state/paused", self._on_paused, _LATCHED_QOS)

    async def shutdown(self) -> None:
        await self.stop_mechanisms()

        async def _delete_one(name: str) -> None:
            async with self._semaphore:
                req = DeleteEntity.Request()
                req.entity = EntityMsg(id=self._entity_ids.get(name, 0), name=name, type=EntityMsg.MODEL)
                await self._service_delete_entity.call_timeout(req)

        names = list(self._spawned_names)
        await asyncio.gather(*(_delete_one(n) for n in names), return_exceptions=True)
        self._spawned_names.clear()
        self._entity_ids.clear()
        self._id_wanted.clear()

    @classmethod
    async def create(cls, **kwargs: typing.Unpack[SimKwargs]) -> "GazeboSimulator":
        simulator = cls(**kwargs)
        await simulator._set_up_services()
        return simulator


def _wall_material_sdf(textures: dict[str, str] | None) -> str:
    """PBR metal block when an albedo map is available, else the flat-gray fallback.

    The diffuse/ambient base colors double as the albedo under ogre2 (its PBR path renders
    an ambient-only material black), so they keep walls visible under both render engines.
    """
    if textures and textures.get("albedo"):
        normal = f"<normal_map>{textures['normal']}</normal_map>" if textures.get("normal") else ""
        return f"""<material>
                    <diffuse>1 1 1 1</diffuse>
                    <pbr>
                        <metal>
                            <albedo_map>{textures["albedo"]}</albedo_map>
                            {normal}
                            <roughness>0.8</roughness>
                            <metalness>0.0</metalness>
                        </metal>
                    </pbr>
                </material>"""
    return """<material>
                    <ambient>0.4 0.4 0.4 1</ambient>
                    <diffuse>0.7 0.7 0.7 1</diffuse>
                    <specular>0.2 0.2 0.2 1</specular>
                </material>"""


def _generate_wall_sdf(name: str, boxes: list[tuple[WallSegment, dict[str, str]]]) -> str:
    """SDF model with one box link per (WallSegment, textures) pair, each at the segment's own z/height."""
    sdf_template = """
        <sdf version="1.6">
            <model name="{name}">
                <pose>0 0 0 0 0 0</pose>
                {links}
                <static>true</static>
            </model>
        </sdf>
        """
    link_template = """
        <link name="wall_segment_{index}">
            <visual name="visual">
                <geometry>
                    <box>
                        <size>{length} {thickness} {height}</size>
                    </box>
                </geometry>
                <cast_shadows>{cast_shadows}</cast_shadows>
                {material}
            </visual>
            {collision}
            <pose>{x} {y} {z} {roll} {pitch} {yaw}</pose>
        </link>
        """
    collision_template = """<collision name="collision">
                <geometry>
                    <box>
                        <size>{length} {thickness} {height}</size>
                    </box>
                </geometry>
            </collision>"""
    links = []
    for i, (segment, textures) in enumerate(boxes):
        sloped = sloped_box(segment)
        if sloped is not None:
            (length, thickness, height), pose = sloped
            roll, pitch, yaw = pose.orientation.to_euler()
            links.append(
                link_template.format(
                    index=i,
                    length=length,
                    thickness=thickness,
                    height=height,
                    x=pose.position.x,
                    y=pose.position.y,
                    z=pose.position.z,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    material=_wall_material_sdf(textures),
                    cast_shadows=str(segment.shadows).lower(),
                    collision=collision_template.format(length=length, thickness=thickness, height=height) if segment.solid else "",
                )
            )
            continue
        x1, y1, x2, y2 = segment.start.x, segment.start.y, segment.end.x, segment.end.y
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length < 1e-9:
            continue
        orientation = math.atan2(y2 - y1, x2 - x1)
        links.append(
            link_template.format(
                index=i,
                length=length,
                thickness=segment.width,
                height=segment.height,
                x=(x1 + x2) / 2,
                y=(y1 + y2) / 2,
                z=segment.start.z + segment.height / 2.0,
                roll=0,
                pitch=0,
                yaw=orientation,
                material=_wall_material_sdf(textures),
                cast_shadows=str(segment.shadows).lower(),
                collision=collision_template.format(length=length, thickness=segment.width, height=segment.height) if segment.solid else "",
            )
        )

    if not links:
        return ""
    return sdf_template.format(name=name, links="\n".join(links))


def _generate_floor_sdf(name: str, floor: Floor, textures: dict[str, str]) -> str:
    """SDF model with one thin box link: visual raised above the level base, collision top flush with it."""
    thickness = 0.02
    collision_depth = 0.5
    return f"""
        <sdf version="1.6">
            <model name="{name}">
                <pose>0 0 0 0 0 0</pose>
                <link name="floor_link">
                    <visual name="visual">
                        <pose>0 0 {thickness / 2.0 + 0.001} 0 0 0</pose>
                        <geometry>
                            <box>
                                <size>{floor.x_length} {floor.y_length} {thickness}</size>
                            </box>
                        </geometry>
                        {_wall_material_sdf(textures)}
                    </visual>
                    <collision name="collision">
                        <pose>0 0 {-collision_depth / 2.0} 0 0 0</pose>
                        <geometry>
                            <box>
                                <size>{floor.x_length} {floor.y_length} {collision_depth}</size>
                            </box>
                        </geometry>
                        <surface><contact><collide_bitmask>{_FLOOR_BITMASK:#x}</collide_bitmask></contact></surface>
                    </collision>
                    <pose>{floor.pos.x} {floor.pos.y} {floor.pos.z} 0 0 0</pose>
                </link>
                <static>true</static>
            </model>
        </sdf>
        """


def _generate_ceiling_sdf(name: str, ceiling: Ceiling, textures: dict[str, str]) -> str:
    """SDF model with a single downward-facing plane, visible only from below."""
    cast_shadows = 'true' if ceiling.cast_shadows else 'false'
    return f"""
        <sdf version="1.6">
            <model name="{name}">
                <pose>0 0 0 0 0 0</pose>
                <link name="ceiling_link">
                    <visual name="visual">
                        <cast_shadows>{cast_shadows}</cast_shadows>
                        <geometry>
                            <plane>
                                <normal>0 0 -1</normal>
                                <size>{ceiling.x_length} {ceiling.y_length}</size>
                            </plane>
                        </geometry>
                        {_wall_material_sdf(textures)}
                    </visual>
                    <pose>{ceiling.pos.x} {ceiling.pos.y} {ceiling.z} 0 0 0</pose>
                </link>
                <static>true</static>
            </model>
        </sdf>
        """


_LIGHT_CONFIG_TOPIC = '/world/default/light_config'
_MATERIAL_COLOR_TOPIC = '/world/default/material_color'
_SET_POSE_VECTOR_SERVICE = '/world/default/set_pose_vector'
_POSE_BATCH_TIMEOUT_MS = 1000
_GLOW_VISUAL_PREFIX = 'glow_'
_GZ_GLOW_RESPONSE = 0.65
_LIGHT_LINK_TIMEOUT = 5.0
_GZ_AMBIENT_ENV = 'env_0'
_WORLD_PIN = 'arena_world_pin'
_FLOOR_BITMASK = 0x01
_MOVING_LINK_BITMASK = 0xFFFE
_GZ_DEFAULT_LIGHTS = (
    ('sun', (-1.0, 0.0, -0.35)),
    ('fill_west', (1.0, 0.0, -0.35)),
    ('fill_north', (0.0, -1.0, -0.35)),
    ('fill_south', (0.0, 1.0, -0.35)),
)
_GZ_DEFAULT_DIFFUSE = 0.45
_GZ_DOME_REFERENCE_LUX = 400.0
_GZ_LUX_GAIN = 0.004
_GZ_FIXTURE_CONE = 2.6
_GZ_FIXTURE_RANGE = 8.0
_GZ_RIG_CELL_FIXTURES = 2
_GZ_BOUNCE_REFLECTANCE = 0.4
_GZ_LIGHT_TYPES = {'point': light_pb2.Light.POINT, 'spot': light_pb2.Light.SPOT, 'directional': light_pb2.Light.DIRECTIONAL}


def _gz_dome_scale(lux: float | None, level: float) -> float:
    return (lux or 0.0) * level / _GZ_DOME_REFERENCE_LUX


def _default_light_scale(lights: Collection[Light]) -> float | None:
    """Scale of the default lights under all spawned lights: None while none is declared, else the dome's share."""
    if all(light.intrinsic for light in lights):
        return None
    dome = next((light for light in lights if light.fixture == 'dome'), None)
    return _gz_dome_scale(dome.lux, 1.0) if dome is not None else 0.0


def _rig_cells(light: Light) -> list[list[int]]:
    """Fixture indices of a rig grouped into square cells of neighboring fixtures."""
    points = [(p.x, p.y) for p in light.fixtures]
    pitches = [math.dist(a, b) for i, a in enumerate(points) for b in points[i + 1 :]]
    cell = _GZ_RIG_CELL_FIXTURES * min((d for d in pitches if d > 0.0), default=1.0)
    x_min = min(x for x, _ in points)
    y_min = min(y for _, y in points)
    cells: dict[tuple[int, int], list[int]] = {}
    for index, (x, y) in enumerate(points):
        cells.setdefault((int((x - x_min + 1e-6) // cell), int((y - y_min + 1e-6) // cell)), []).append(index)
    return list(cells.values())


def _rig_bounce(light: Light, level: float, alive: Sequence[bool]) -> dict[str, typing.Any]:
    """Unshadowed fill at the room center standing in for the light the room surfaces reflect, which ogre2 does not trace."""
    xs = [p.x for p in light.fixtures]
    ys = [p.y for p in light.fixtures]
    pitches = [math.dist((a.x, a.y), (b.x, b.y)) for i, a in enumerate(light.fixtures) for b in light.fixtures[i + 1 :]]
    pitch = min((d for d in pitches if d > 0.0), default=2.0)
    length = max(xs) - min(xs) + pitch
    width = max(ys) - min(ys) + pitch
    height = light.fixtures[0].z
    surface = 2.0 * (length * width + length * height + width * height)
    live = sum(1 for index in range(len(light.fixtures)) if index >= len(alive) or alive[index])
    flux = (light.lumens or 0.0) * live * level
    reflected_lux = _GZ_BOUNCE_REFLECTANCE * flux / (surface * (1.0 - _GZ_BOUNCE_REFLECTANCE))
    reach = 0.6 * math.hypot(length, width, height)
    return {'type': 'point', 'pose': (sum(xs) / len(xs), sum(ys) / len(ys), height / 2.0), 'intensity': reflected_lux / _GZ_DOME_REFERENCE_LUX, 'range': reach, 'constant': 1.0, 'quadratic': 0.0, 'cone': None, 'cast_shadows': False}


def _light_params(light: Light, level: float, alive: Sequence[bool], position: Position | None = None, yaw: float = 0.0) -> list[dict[str, typing.Any]]:
    """One parameter set per gz light: a rig becomes a point light per cell of fixtures, a frame light points along its frame's yaw."""
    r, g, b = cct_to_rgb(light.cct_K)
    dx, dy, dz = light.direction
    direction = (math.cos(yaw) * dx - math.sin(yaw) * dy, math.sin(yaw) * dx + math.cos(yaw) * dy, dz)
    params: dict[str, typing.Any] = {'diffuse': (r, g, b), 'cast_shadows': light.cast_shadows, 'direction': direction}
    if light.fixture == 'sun':
        return [params | {'type': 'directional', 'pose': (0.0, 0.0, 10.0), 'intensity': _gz_dome_scale(light.lux, level), 'range': 1000.0, 'constant': 1.0, 'quadratic': 0.0, 'cone': None}]
    if light.rig:
        result = []
        for cell in _rig_cells(light):
            live = sum(1 for index in cell if index >= len(alive) or alive[index])
            x = sum(light.fixtures[index].x for index in cell) / len(cell)
            y = sum(light.fixtures[index].y for index in cell) / len(cell)
            intensity = _GZ_LUX_GAIN * (light.lumens or 0.0) * live * level / math.tau
            result.append(params | {'type': 'point', 'pose': (x, y, light.fixtures[cell[0]].z), 'intensity': intensity, 'range': _GZ_FIXTURE_RANGE, 'constant': 1.0, 'quadratic': 1.0, 'cone': None})
        return [*result, params | _rig_bounce(light, level, alive)]
    if position is None:
        position = light.position
    if light.spec.shape == 'sphere':
        return [params | {'type': 'point', 'pose': (position.x, position.y, position.z), 'intensity': _GZ_LUX_GAIN * (light.lumens or 0.0) * level / (4.0 * math.pi), 'range': _GZ_FIXTURE_RANGE, 'constant': 1.0, 'quadratic': 1.0, 'cone': None}]
    cone = math.radians(light.spec.cone_deg) if light.spec.cone_deg < 180.0 else _GZ_FIXTURE_CONE
    solid_angle = 2.0 * math.pi * (1.0 - math.cos(cone / 2.0))
    return [params | {'type': 'spot', 'pose': (position.x, position.y, position.z), 'intensity': _GZ_LUX_GAIN * (light.lumens or 0.0) * level / solid_angle, 'range': _GZ_FIXTURE_RANGE, 'constant': 1.0, 'quadratic': 1.0, 'cone': cone}]


def _generate_light_sdf(name: str, p: dict[str, typing.Any]) -> str:
    r, g, b = p['diffuse']
    dx, dy, dz = p['direction']
    x, y, z = p['pose']
    spot = f"<spot><inner_angle>{p['cone'] * 0.5}</inner_angle><outer_angle>{p['cone']}</outer_angle><falloff>1</falloff></spot>" if p['cone'] is not None else ''
    return f"""
        <sdf version="1.9">
            <light type="{p['type']}" name="{name}">
                <pose>{x} {y} {z} 0 0 0</pose>
                <cast_shadows>{'true' if p['cast_shadows'] else 'false'}</cast_shadows>
                <diffuse>{r} {g} {b} 1</diffuse>
                <specular>0 0 0 1</specular>
                <intensity>{p['intensity']}</intensity>
                <direction>{dx} {dy} {dz}</direction>
                <attenuation><range>{p['range']}</range><constant>{p['constant']}</constant><linear>0</linear><quadratic>{p['quadratic']}</quadratic></attenuation>
                {spot}
            </light>
        </sdf>
        """


def _light_request(name: str, p: dict[str, typing.Any]) -> light_pb2.Light:
    msg = light_pb2.Light(
        name=name,
        type=_GZ_LIGHT_TYPES[p['type']],
        range=p['range'],
        attenuation_constant=p['constant'],
        attenuation_quadratic=p['quadratic'],
        cast_shadows=p['cast_shadows'],
        intensity=p['intensity'],
        is_light_off=p['intensity'] <= 0.0,
    )
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = p['pose']
    msg.pose.orientation.w = 1.0
    msg.diffuse.r, msg.diffuse.g, msg.diffuse.b = p['diffuse']
    msg.diffuse.a = 1.0
    msg.specular.a = 1.0
    msg.direction.x, msg.direction.y, msg.direction.z = p['direction']
    if p['cone'] is not None:
        msg.spot_inner_angle = p['cone'] * 0.5
        msg.spot_outer_angle = p['cone']
        msg.spot_falloff = 1.0
    return msg


def _joint_topic(model: str, kind: str) -> str:
    return f"/model/{model}/articulated/{kind}"


def _link_geometry(link: str, sdf: Path | None, scale: Scale | None) -> str:
    """The visuals and collisions of a link bundle's SDF, mesh uris made absolute and scaled, renamed under the link."""
    if sdf is None:
        return ""
    factors = "1 1 1" if scale is None else f"{scale.x} {scale.y} {scale.z}"
    parts: list[str] = []
    source = xml.etree.ElementTree.parse(sdf).getroot().find(".//link")
    for index, element in enumerate(source if source is not None else ()):
        if element.tag not in ("visual", "collision"):
            continue
        element.set("name", f"{link}_{element.tag}_{index}")
        for mesh in element.iter("mesh"):
            uri = mesh.find("uri")
            if uri is not None and uri.text and "://" not in uri.text and not uri.text.startswith("/"):
                uri.text = str(sdf.parent / uri.text)
            for old in mesh.findall("scale"):
                mesh.remove(old)
            xml.etree.ElementTree.SubElement(mesh, "scale").text = factors
        parts.append(xml.etree.ElementTree.tostring(element, encoding="unicode"))
    return "".join(parts)


def _articulated_sdf(name: str, runtime: ArticulatedObject, links: Sequence[tuple[str, Joint | None, Path | None]]) -> str:
    """A model of one link per articulation link at the model frame, the root fixed to the world, link surfaces from their materials, damped rubbing sprung joints and an ArticulatedDrive over the movable ones."""
    root = runtime.articulation.root
    scale = runtime.obstacle.scale
    factors = (1.0, 1.0, 1.0) if scale is None else (scale.x, scale.y, scale.z)
    body: list[str] = [f'<joint name="{_WORLD_PIN}" type="fixed"><parent>world</parent><child>{root}</child></joint>']
    drives: list[str] = []
    for link, joint, sdf in links:
        inertia = runtime.inertias[link]
        cx, cy, cz = inertia.center
        ixx, iyy, izz = inertia.moments
        geometry = _link_geometry(link, sdf, scale)
        contact = ""
        if (material := runtime.articulation.materials.get(link)) is not None:
            _, dynamic, restitution = surface(material)
            contact += f"<friction><ode><mu>{dynamic}</mu><mu2>{dynamic}</mu2></ode></friction><bounce><restitution_coefficient>{restitution}</restitution_coefficient></bounce>"
        if joint is not None:
            contact += f"<contact><collide_bitmask>{_MOVING_LINK_BITMASK:#x}</collide_bitmask></contact>"
        if contact:
            geometry = geometry.replace("</collision>", f"<surface>{contact}</surface></collision>")
        body.append(f'<link name="{link}"><inertial><pose>{cx} {cy} {cz} 0 0 0</pose><mass>{inertia.mass}</mass><inertia><ixx>{ixx}</ixx><iyy>{iyy}</iyy><izz>{izz}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>{geometry}</link>')
        if joint is None:
            continue
        ox, oy, oz = (component * factor for component, factor in zip(joint.origin, factors, strict=True))
        if joint.type is JointType.FIXED:
            body.append(f'<joint name="{joint.name}" type="fixed"><parent>{joint.parent}</parent><child>{link}</child></joint>')
            continue
        dynamics = runtime.dynamics[joint.name]
        lower, upper = joint.limit if joint.limit is not None else (-1e16, 1e16)
        kind = "prismatic" if joint.type is JointType.PRISMATIC else "revolute"
        body.append(
            f'<joint name="{joint.name}" type="{kind}"><parent>{joint.parent}</parent><child>{link}</child><pose>{ox} {oy} {oz} 0 0 0</pose>'
            f'<axis><xyz expressed_in="__model__">{joint.axis.x} {joint.axis.y} {joint.axis.z}</xyz><limit><lower>{lower}</lower><upper>{upper}</upper></limit>'
            f"<dynamics><damping>{dynamics.damping}</damping><friction>{dynamics.friction}</friction>"
            f"<spring_reference>{dynamics.spring_rest}</spring_reference><spring_stiffness>{dynamics.spring_stiffness}</spring_stiffness></dynamics></axis></joint>"
        )
        wraps = "true" if joint.type is JointType.CONTINUOUS else "false"
        drives.append(f'<joint name="{joint.name}" stiffness="{dynamics.stiffness}" damping="{dynamics.drive_damping}" effort="{dynamics.effort}" wraps="{wraps}" initial="{runtime.current[joint.name]}"/>')
    plugin = f'<plugin filename="ArticulatedDrive" name="arena_gz_plugins::ArticulatedDrive">{"".join(drives)}</plugin>'
    return f'<?xml version="1.0"?><sdf version="1.9"><model name="{name}">{"".join(body)}{plugin}</model></sdf>'


def _sdf_file(path: Path) -> Path:
    """The SDF file of a model path, Gazebo takes a file and not a directory."""
    if not path.is_dir():
        return path
    candidate = path / f"{path.name}.sdf"
    if candidate.exists():
        return candidate
    return next(iter(path.glob("*.sdf")), path)


def _glow_visuals(sdf_path: Path) -> dict[str, tuple[str, tuple[float, float, float]]]:
    """Per glow material of a model SDF, the scoped name of its visual below the model and its base color."""
    result: dict[str, tuple[str, tuple[float, float, float]]] = {}
    for link in xml.etree.ElementTree.parse(sdf_path).getroot().iter('link'):
        for visual in link.iter('visual'):
            name = visual.get('name', '')
            if not name.startswith(_GLOW_VISUAL_PREFIX):
                continue
            diffuse = visual.findtext('material/diffuse')
            red, green, blue = (float(channel) for channel in diffuse.split()[:3]) if diffuse else (1.0, 1.0, 1.0)
            result[name.removeprefix(_GLOW_VISUAL_PREFIX)] = (f"{link.get('name', '')}::{name}", (red, green, blue))
    return result


def _glow_request(entity: str, base: tuple[float, float, float], cct_K: float, level: float) -> material_color_pb2.MaterialColor:
    msg = material_color_pb2.MaterialColor(entity_match=material_color_pb2.MaterialColor.FIRST)
    msg.entity.name = entity
    emissive = tuple(level**_GZ_GLOW_RESPONSE * channel for channel in cct_to_rgb(cct_K))
    for color, (red, green, blue) in ((msg.ambient, base), (msg.diffuse, base), (msg.specular, (0.0, 0.0, 0.0)), (msg.emissive, emissive)):
        color.r, color.g, color.b, color.a = red, green, blue, 1.0
    return msg


def _default_light_request(name: str, direction: tuple[float, float, float], scale: float) -> light_pb2.Light:
    msg = light_pb2.Light(name=name, type=light_pb2.Light.DIRECTIONAL, intensity=scale, is_light_off=scale <= 0.0)
    msg.diffuse.r = msg.diffuse.g = msg.diffuse.b = _GZ_DEFAULT_DIFFUSE
    msg.diffuse.a = 1.0
    msg.specular.a = 1.0
    msg.direction.x, msg.direction.y, msg.direction.z = direction
    return msg


_BOX_SDF_TEMPLATE = """
<sdf version="1.6">
    <model name="{name}">
        <static>{static}</static>
        <link name="link">
            <pose>{cx} {cy} {cz} 0 0 0</pose>{body}
            <visual name="visual">
                <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
                {material}
            </visual>
            <collision name="collision">
                <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>{surface}
            </collision>
        </link>{plugin}
    </model>
</sdf>
"""

_MOVING_BOX_BODY_SDF = """
            <gravity>false</gravity>
            <inertial>
                <mass>{mass}</mass>
                <inertia><ixx>{ixx}</ixx><iyy>{iyy}</iyy><izz>{izz}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
            </inertial>"""

_MOVING_BOX_PLUGIN_SDF = """
        <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl">
            <initial_linear>{vx} {vy} {vz}</initial_linear>
        </plugin>"""

_BOX_SURFACE_SDF = """
                <surface><contact><collide_bitmask>{mask}</collide_bitmask></contact></surface>"""

_COMB_BOX_BITMASK = '0x01'
_MOVING_BOX_BITMASK = '0xfffe'

_MOVING_BOX_MASS = 1000.0


def _generate_box_sdf(
    name: str,
    size: tuple[float, float, float],
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    textures: dict[str, str] | None = None,
    velocity: tuple[float, float, float] | None = None,
    comb: bool = False,
) -> str:
    """A static box, or with `velocity` (model frame) a heavy gravity-free one that VelocityControl drives, which passes through `comb` boxes."""
    sx, sy, sz = size
    cx, cy, cz = center
    body = plugin = surface = ''
    if comb:
        surface = _BOX_SURFACE_SDF.format(mask=_COMB_BOX_BITMASK)
    if velocity is not None:
        surface = _BOX_SURFACE_SDF.format(mask=_MOVING_BOX_BITMASK)
        m = _MOVING_BOX_MASS
        body = _MOVING_BOX_BODY_SDF.format(mass=m, ixx=m * (sy * sy + sz * sz) / 12.0, iyy=m * (sx * sx + sz * sz) / 12.0, izz=m * (sx * sx + sy * sy) / 12.0)
        plugin = _MOVING_BOX_PLUGIN_SDF.format(vx=velocity[0], vy=velocity[1], vz=velocity[2])
    static = 'true' if velocity is None else 'false'
    return _BOX_SDF_TEMPLATE.format(name=name, static=static, sx=sx, sy=sy, sz=sz, cx=cx, cy=cy, cz=cz, material=_wall_material_sdf(textures), body=body, plugin=plugin, surface=surface)


def _body_frame(orientation: Orientation, velocity: tuple[float, float, float]) -> tuple[float, float, float]:
    """A world-frame vector in the frame of a body with `orientation`."""
    local = Orientation(orientation.w, -orientation.x, -orientation.y, -orientation.z) * Vector3(*velocity)
    return local.x, local.y, local.z


def _pose_vector(poses: Mapping[str, Pose]) -> pose_v_pb2.Pose_V:
    """Named model poses as one gz set_pose_vector request."""
    request = pose_v_pb2.Pose_V()
    for name, pose in poses.items():
        entry = request.pose.add()
        entry.name = name
        entry.position.x, entry.position.y, entry.position.z = pose.position.x, pose.position.y, pose.position.z
        entry.orientation.w, entry.orientation.x, entry.orientation.y, entry.orientation.z = pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z
    return request
