"""
GazeboHost is constructed once by arena_node and owns process-singleton resources for Gazebo: the gz_services_bridge launch, the lifecycle (pause/unpause), and the control-world client.
GazeboSimulator is per-env on task_generator_node and adapts env-namespace state (spawned-name tracking, per-env service clients) over those shared resources. Sim_path prefixing is owned by the Realizer.
"""

import asyncio
import itertools
import math
import time
import traceback
import typing
from collections.abc import Sequence
from pathlib import Path

import arena_robots.catalog
import arena_robots.Robot
import arena_robots.Sensor
import launch
import launch_ros
import rclpy.impl.rcutils_logger
import rclpy.time
from arena_people_msgs.msg import Pedestrians
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.Async import ClientWrapper
from arena_rclpy_mixins.shared import Namespace
from arena_simulation_setup.shared import Ceiling
from arena_simulation_setup.tree.Wall import WallSegment
from arena_simulation_setup.utils.material import MdlUtil
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Quaternion
from geometry_msgs.msg import Pose as RosPose
from launch import LaunchDescription
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from ros_gz_interfaces.msg import Entity as EntityMsg
from ros_gz_interfaces.msg import EntityFactory, WorldControl
from ros_gz_interfaces.srv import ControlWorld, DeleteEntity, SetEntityPose, SpawnEntity
from task_generator.shared import (
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
from task_generator.utils.flags import ObstaclesOptim, obstacles_optim_level
from viewport_control_msgs.msg import ViewportView
from viewport_control_msgs.srv import ViewportSetProjection, ViewportSetReferenceFrame, ViewportSetView

from arena_runtime.sim import BaseSim, SimLifecycle
from arena_runtime.sim._control import (
    controller_spawner_node,
    effective_control_yaml,
    effective_controllers,
    odom_relay_node,
    twist_stamper_node,
)
from arena_runtime.sim._interface import resolve_obstacle_box
from arena_runtime.sim._walls import realize_renderable

from .robot_bridge import BridgeConfiguration, MappingDirection, _TopicMapping

_VIEWPORT_STREAM_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)


class GazeboHost(SimLifecycle):
    def __init__(
        self,
        node: ArenaMixinNode,
        semaphore: asyncio.Semaphore,
        service_control_world: ClientWrapper,
        service_delete_entity: ClientWrapper,
        logger: rclpy.impl.rcutils_logger.RcutilsLogger,
    ) -> None:
        self._node = node
        self._semaphore = semaphore
        self._service_control_world = service_control_world
        self._service_delete_entity = service_delete_entity
        self._logger = logger

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

    async def unpause(self) -> bool:
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

    async def cleanup_namespace(self, prefix: str) -> int:
        names = await self._list_models()
        targets = [n for n in names if n.startswith(prefix)]
        if not targets:
            return 0

        async def _del(name: str) -> bool:
            req = DeleteEntity.Request()
            req.entity = EntityMsg(name=name, type=EntityMsg.MODEL)
            async with self._semaphore:
                try:
                    res = await self._service_delete_entity.call_timeout(req)
                except Exception as e:
                    self._logger.warning(f"cleanup_namespace: delete {name} raised: {e!r}")
                    return False
            return bool(res) and res.success

        results = await asyncio.gather(*(_del(n) for n in targets))
        return sum(1 for r in results if r)

    async def _list_models(self) -> list[str]:
        proc = await asyncio.create_subprocess_exec(
            'gz',
            'model',
            '--list',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            self._logger.warning(f"gz model --list failed: {stderr.decode().strip()}")
            return []
        names: list[str] = []
        for line in stdout.decode().splitlines():
            stripped = line.strip()
            if stripped.startswith('- '):
                names.append(stripped[2:].strip())
        return names


class GazeboSimulator(BaseSim):
    SIM_NAME = 'gazebo'

    # gz create CLI ack timeout (ms), kept generous so a slow server-side ack
    # under heavy load is not mistaken for a spawn failure, which would leak an
    # untracked orphan model.
    _SPAWN_TIMEOUT_MS = 60000

    def __init__(self, *args: object, namespace: Namespace, **kwargs: object) -> None:
        super().__init__(*args, namespace=namespace, **kwargs)

        self._semaphore = asyncio.Semaphore(5)

        self._logger.info(f"Initializing GazeboSimulator with namespace: {namespace}")

        self.entities: dict[str, Entity] = {}
        self._walls_entities: list[str] = []
        self._wall_counter = itertools.count()
        self._material_texture_cache: dict[str, dict[str, str]] = {}
        self._spawned_names: set[str] = set()

        self._viewport_camera_pose: Pose | None = None

    def _robot_loader_args(self, robot: Robot) -> dict[str, object]:
        args: dict[str, object] = {
            **robot.asdict(),
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
        return effective_control_yaml(robot.resolved_assembly, config_uri, robot.sim_path, frame_prefix, prefix=control_prefix)

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
            ok = await self._spawn_sdf(obstacle.sim_path, sdf, obstacle.pose)
        if ok:
            self.entities[obstacle.name] = obstacle
        return ok

    async def pedestrian_spawn(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        # Ped actors are created and removed by PedSkeletonPlugin from arena_peds.
        return tuple(True for _ in pedestrians)

    async def robot_spawn(self, robots: Sequence[Robot]) -> Sequence[bool]:

        async def impl(robot: Robot) -> bool:
            if not await self._spawn_entity(robot):
                return False
            _loader_args = self._robot_loader_args(robot)
            model = await (await robot.model.resolve()).model.get(ModelType.URDF, loader_args=_loader_args)
            if model.type is ModelType.UNKNOWN:
                return False
            model_description = model.description
            self._robot_initialpose(robot)
            await self._robot_bridge(robot, model_description)
            robot_config = arena_robots.Robot.RobotIdentifier(robot.model.name).resolve_sync()
            self._register_agent_robot(robot, robot_config.model_params)
            return True

        success = await asyncio.gather(*map(impl, robots))
        return success

    async def obstacle_move(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        return await asyncio.gather(*map(self._move_entity, obstacles))

    async def pedestrian_move(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        # Ped pose is owned by PedSkeletonPlugin (driven from the arena_peds topic).
        return tuple(True for _ in pedestrians)

    async def robot_move(self, robots: Sequence[Robot]) -> Sequence[bool]:
        async def impl(robot: Robot) -> bool:
            return (await self._move_entity(robot)) and (await self._robot_move(robot))

        async with self.node.unpause_window():
            result = await asyncio.gather(*map(impl, robots))
        return result

    async def obstacle_delete(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        return await asyncio.gather(*(self._delete_entity(o.sim_path) for o in obstacles))

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
                await self._spawn_sdf(name, floor_sdf, Pose())
            self._walls_entities.append(name)
        return True

    async def spawn_ceilings(self, ceilings: Sequence[Ceiling]) -> bool:
        for ceiling in ceilings:
            name = self._realizer.realize(f"ceiling_{next(self._wall_counter)}")
            textures = await self._resolve_wall_textures(ceiling.material)
            ceiling_sdf = _generate_ceiling_sdf(name, ceiling, textures)
            async with self._semaphore:
                await self._spawn_sdf(name, ceiling_sdf, Pose())
            self._walls_entities.append(name)
        return True

    async def set_robot_pose(self, sim_path: str, pose: Pose) -> bool:
        if sim_path not in self._agent_robots:
            return False
        req = SetEntityPose.Request()
        req.entity = EntityMsg(name=sim_path, type=EntityMsg.MODEL)
        req.pose = pose.to_msg()
        try:
            res = await self._service_set_entity_pose.call_timeout(req)
            return bool(res and res.success)
        except Exception as e:
            self._logger.warning(f"set_robot_pose({sim_path!r}) failed: {e}")
            return False

    async def spawn_box(self, name: str, size: tuple[float, float, float], pose: Pose) -> bool:
        sdf = _generate_box_sdf(name, size)
        async with self._semaphore:
            return await self._spawn_sdf(name, sdf, pose)

    async def move_box(self, name: str, pose: Pose) -> bool:
        # Fire-and-forget: animation is cosmetic, last-write-wins. Awaiting the service round-trip
        # would gate the shim's tick rate on bridge latency; skipping the semaphore lets parallel
        # door updates issue concurrently.
        request = SetEntityPose.Request()
        request.entity = EntityMsg(name=name, type=EntityMsg.MODEL)
        request.pose = pose.to_msg()
        try:
            self._service_set_entity_pose.client.call_async(request)
        except Exception as e:
            self._logger.warning(f"move_box dispatch failed for {name}: {e}")
            return False
        return True

    async def delete_box(self, name: str) -> bool:
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
            self._logger.debug(f"Attempting to move entity: {entity.sim_path}")
            self._logger.debug(f"Moving entity {entity.sim_path} to position: {entity.pose}")

            request = SetEntityPose.Request()
            request.entity = EntityMsg(
                name=entity.sim_path,
                type=entity_type,
            )
            request.pose = entity.pose.to_msg()

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

                ok = await self._spawn_model(entity.sim_path, model, entity.pose)
                if ok and (model.path is None or model.type is ModelType.URDF):
                    self.entities[entity.name] = entity
                return ok

            except Exception as e:
                self._logger.error(f"Error spawning entity {entity.name}: {str(e)}")
                traceback.print_exc()
                return False

    async def _spawn_model(self, name: str, model: Model, pose: Pose) -> bool:
        """Spawn an already-resolved model under `name` at `pose`. Caller holds self._semaphore."""
        if model.path and model.type not in (ModelType.URDF,):
            # direct path available, use gz cli call
            sdf_path = model.path
            # Resolve directory to actual SDF file (Gazebo requires a file, not a directory)
            if sdf_path.is_dir():
                candidate = sdf_path / f"{sdf_path.name}.sdf"
                if candidate.exists():
                    sdf_path = candidate
                else:
                    candidates = list(sdf_path.glob("*.sdf"))
                    if candidates:
                        sdf_path = candidates[0]

            req_payload = f'sdf_filename: "{sdf_path}", name: "{name}", pose: {{   position: {{ x: {pose.position.x}, y: {pose.position.y}, z: {pose.position.z} }}   orientation: {{ x: {pose.orientation.x}, y: {pose.orientation.y}, z: {pose.orientation.z}, w: {pose.orientation.w} }} }}'

            process = await asyncio.create_subprocess_exec('gz', 'service', '-s', '/world/default/create', '--reqtype', 'gz.msgs.EntityFactory', '--reptype', 'gz.msgs.Boolean', '--timeout', str(self._SPAWN_TIMEOUT_MS), '--req', req_payload, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

            _, stderr = await process.communicate()

            if process.returncode == 0:
                self._spawned_names.add(name)
                return True

            # A non-zero return covers both a timed-out ack and an outright
            # rejection, but the create may have landed server-side anyway.
            # Reconcile against the live model list: track it if it exists, else
            # best-effort delete in case the create lands just after this check,
            # so a slow ack never leaves an orphan that survives every reset.
            if name in await self._list_models():
                self._spawned_names.add(name)
                return True

            self._logger.error(f"Failed to spawn {name}. Error: {stderr.decode().strip()}")
            req = DeleteEntity.Request()
            req.entity = EntityMsg(name=name, type=EntityMsg.MODEL)
            await self._service_delete_entity.call_timeout(req)
            return False

        return await self._spawn_sdf(name, model.description, pose)

    async def _spawn_sdf(self, name: str, sdf: str, pose: Pose) -> bool:
        """Spawn from an in-memory SDF via ros_gz_bridge. Caller holds self._semaphore."""
        request = SpawnEntity.Request()
        request.entity_factory = EntityFactory()
        request.entity_factory.name = name
        request.entity_factory.sdf = sdf
        request.entity_factory.pose = pose.to_msg()

        try:
            result = await self._service_spawn_entity.call_timeout(request)
        except Exception as e:
            self._logger.error(f"Spawn service call raised for {name}: {e}")
            return False

        if result is None:
            self._logger.error(f"Spawn service call failed for {name}")
            return False

        if result.success:
            self._spawned_names.add(name)
        return result.success

    async def _delete_entity(self, sim_path: str, entity_type: int = EntityMsg.MODEL) -> bool:
        async with self._semaphore:
            self._logger.debug(f"Attempting to delete entity: {sim_path}")

            request = DeleteEntity.Request()
            request.entity = EntityMsg(
                name=sim_path,
                type=entity_type,
            )

            try:
                result = await self._service_delete_entity.call_timeout(request)

                if result is None:
                    self._logger.error(f"Delete service call failed for {sim_path}")
                    return False

                self._logger.debug(f"Delete result for {sim_path}: {result.success}")

                if result.success:
                    self.entities.pop(sim_path, None)
                    self._spawned_names.discard(sim_path)

                return result.success

            except Exception as e:
                self._logger.error(f"Error deleting entity {sim_path}: {str(e)}")
                traceback.print_exc()
                return False

    async def step(self, n: int = 1) -> bool:
        async with self._semaphore:
            request = ControlWorld.Request()
            request.world_control = WorldControl()
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

    async def _resolve_wall_textures(self, material: object) -> dict[str, str]:
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
                        await self._spawn_sdf(wall_name, wall_sdf, Pose())
                    self._walls_entities.append(wall_name)
            for obstacle, model in obstacles:
                obstacle.sim_path = self._realizer.realize(f"wall_obj_{next(self._wall_counter)}")
                async with self._semaphore:
                    ok = await self._spawn_model(obstacle.sim_path, model, obstacle.pose)
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
                    }
                ],
            )
        )

        control_spec = robot_config.model_params.control
        if control_spec is not None and control_spec.is_ros2_control:
            if not control_spec.controllers:
                raise ValueError(f"control.mode=ros2_control but no controllers declared for {robot.name}")
            for controller_name in effective_controllers(robot.resolved_assembly, control_spec.controllers, prefix=(robot_config.assembly.prefix if robot_config.assembly is not None else 'robot_')):
                launch_description.add_action(controller_spawner_node(controller_name))
            launch_description.add_action(
                twist_stamper_node(
                    control_spec.cmd_vel_topic,
                    robot.frame.tf(robot_config.model_params.base_frame),
                )
            )
            if control_spec.odom_topic != "odom":
                launch_description.add_action(odom_relay_node(control_spec.odom_topic))

        launch_description.add_action(
            launch_ros.actions.Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="map_to_odom_publisher",
                arguments=[
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "1",
                    "map",
                    robot.frame.tf(robot_config.model_params.odom_frame),
                ],
                parameters=[{"use_sim_time": True}],
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

    async def shutdown(self) -> None:
        await self.stop_mechanisms()

        async def _delete_one(name: str) -> None:
            async with self._semaphore:
                req = DeleteEntity.Request()
                req.entity = EntityMsg(name=name, type=EntityMsg.MODEL)
                await self._service_delete_entity.call_timeout(req)

        names = list(self._spawned_names)
        await asyncio.gather(*(_delete_one(n) for n in names), return_exceptions=True)
        self._spawned_names.clear()

    @classmethod
    async def create(cls, *args: object, namespace: Namespace, **kwargs: object) -> "GazeboSimulator":
        simulator = cls(*args, namespace=namespace, **kwargs)
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
                {material}
            </visual>
            <collision name="collision">
                <geometry>
                    <box>
                        <size>{length} {thickness} {height}</size>
                    </box>
                </geometry>
            </collision>
            <pose>{x} {y} {z} 0 0 {orientation}</pose>
        </link>
        """
    links = []
    for i, (segment, textures) in enumerate(boxes):
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
                orientation=orientation,
                material=_wall_material_sdf(textures),
            )
        )

    if not links:
        return ""
    return sdf_template.format(name=name, links="\n".join(links))


def _generate_floor_sdf(name: str, floor: Floor, textures: dict[str, str]) -> str:
    """SDF model with one thin box link: visual raised above the level base, collision top flush with it."""
    thickness = 0.02
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
                        <pose>0 0 {-thickness / 2.0} 0 0 0</pose>
                        <geometry>
                            <box>
                                <size>{floor.x_length} {floor.y_length} {thickness}</size>
                            </box>
                        </geometry>
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


_BOX_SDF_TEMPLATE = """
<sdf version="1.6">
    <model name="{name}">
        <static>true</static>
        <link name="link">
            <pose>{cx} {cy} {cz} 0 0 0</pose>
            <visual name="visual">
                <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
                {material}
            </visual>
            <collision name="collision">
                <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
            </collision>
        </link>
    </model>
</sdf>
"""


def _generate_box_sdf(name: str, size: tuple[float, float, float], center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> str:
    sx, sy, sz = size
    cx, cy, cz = center
    return _BOX_SDF_TEMPLATE.format(name=name, sx=sx, sy=sy, sz=sz, cx=cx, cy=cy, cz=cz, material=_wall_material_sdf(None))
