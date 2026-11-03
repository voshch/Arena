import asyncio
import copy
import itertools
import os
import traceback
import types
import typing
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence

import arena_mujoco_msgs.msg
import arena_robots.catalog
import arena_robots.Robot
import arena_robots.Sensor
import arena_simulation_setup.tree.assets.Material
import attrs
import geometry_msgs.msg
import launch
import launch.event_handlers
import launch_ros
import rclpy.task
import std_srvs.srv
from arena_bringup.constants import SimSimulator
from arena_mujoco_msgs.msg import (
    ArticulatedLink,
    Ceiling,
    Floor,
    JointCommand,
    JointDynamics,
    Light,
    LightState,
    Material,
    Prim,
    PrimVisual,
    Scale,
    Wall,
)
from arena_mujoco_msgs.srv import (
    DeleteLights,
    DeletePrims,
    DriveArticulated,
    EditPrims,
    SetLights,
    SetMapOffset,
    SpawnArticulated,
    SpawnCeilings,
    SpawnFloors,
    SpawnLights,
    SpawnPrims,
    SpawnUrdf,
    SpawnWalls,
    Step,
)
from arena_people_msgs.msg import Pedestrian, Pedestrians, SpawnPedestrian
from arena_people_msgs.srv import (
    DeletePedestrians,
    MovePedestrians,
    SpawnPedestrians,
)
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.Async import ClientWrapper
from arena_rclpy_mixins.shared import Namespace
from arena_rclpy_mixins.Time import Time
from arena_simulation_setup.shared import Light as LightDefinition
from arena_simulation_setup.shared import Obstacle as ObstacleDefinition
from arena_simulation_setup.shared import cct_to_rgb
from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
from arena_simulation_setup.tree.Wall import WallSegment
from arena_simulation_setup.utils.articulation import Values
from launch_ros.parameters_type import SomeParameterFile, SomeParametersDict

from arena_runtime._node import NodeInterface
from arena_runtime.flags import ObstaclesOptim, obstacles_optim_level
from arena_runtime.shared import Ceiling as CeilingDefinition
from arena_runtime.shared import (
    DynamicObstacle,
    Model,
    ModelType,
    Obstacle,
    Pose,
    Position,
    Robot,
)
from arena_runtime.shared import Floor as FloorDefinition
from arena_runtime.shared import Wall as WallDefinition
from arena_runtime.sim import BaseSim, SimKwargs, SimLifecycle
from arena_runtime.sim._articulation import ArticulatedObject, command_fields, link_message
from arena_runtime.sim._articulation import JointCommand as ArticulatedCommand
from arena_runtime.sim._control import (
    effective_control_yaml,
    odom_relay_node,
    robot_controllers,
    twist_stamper_node,
)
from arena_runtime.sim._interface import offset_pose, resolve_obstacle_box
from arena_runtime.sim._urdf import transform_urdf_for_bridge
from arena_runtime.sim._walls import realize_renderable, sloped_box

"""
MujocoHost is constructed once by arena_runtime_node and owns process-singleton resources for MuJoCo: the lifecycle (pause/unpause/cleanup/step) and its service clients.
MujocoSimulator is per-env on arena_env_node and adapts env-namespace state (per-robot publishers, env-prefixed entity names) over those shared resources.
"""

_MUJOCO_PHYSICS_DT = 0.002
_MESH_MODEL_TYPES = (ModelType.SDF, ModelType.OBJ)
_SPAWN_ASSETS_TIMEOUT_S = 300.0
_RESEND_AFTER_S = 10.0
_PRIM_BATCH_WINDOW_S = 0.02


class _PrimBatch:
    """Sends the prims of spawn_box calls that arrive within a short window as one SpawnPrims request."""

    def __init__(self, client: ClientWrapper, env_id: int) -> None:
        self._client = client
        self._env_id = env_id
        self._pending: list[tuple[Prim, asyncio.Future[bool]]] = []
        self._flush: asyncio.Task[None] | None = None

    async def submit(self, prim: Prim) -> bool:
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending.append((prim, future))
        if self._flush is None:
            self._flush = asyncio.create_task(self._send())
        return await future

    async def _send(self) -> None:
        await asyncio.sleep(_PRIM_BATCH_WINDOW_S)
        batch, self._pending, self._flush = self._pending, [], None
        try:
            res = await self._client.call_timeout(SpawnPrims.Request(env_id=self._env_id, prims=[prim for prim, _ in batch]))
        except Exception as e:
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)
            return
        ret = list(res.ret) if res is not None else []
        for i, (_, future) in enumerate(batch):
            if not future.done():
                future.set_result(i < len(ret) and bool(ret[i]))


class MujocoHost(SimLifecycle):
    def __init__(self, node: ArenaMixinNode) -> None:
        self._node = node
        self._logger = node.get_logger().get_child(type(self).__name__)
        self._pause_client: ClientWrapper = node.create_client_wrapper(
            std_srvs.srv.Trigger,
            "/mujoco/PauseSimulation",
        )
        self._unpause_client: ClientWrapper = node.create_client_wrapper(
            std_srvs.srv.Trigger,
            "/mujoco/UnpauseSimulation",
        )
        self._delete_prims_client: ClientWrapper = node.create_client_wrapper(
            DeletePrims,
            "/mujoco/DeletePrims",
        )
        self._step_client: ClientWrapper = node.create_client_wrapper(
            Step,
            "/mujoco/Step",
        )

    async def ensure_ready(self) -> None:
        await asyncio.gather(
            self._pause_client.ensure(),
            self._unpause_client.ensure(),
            self._delete_prims_client.ensure(),
            self._step_client.ensure(),
        )

    async def pause(self) -> bool:
        """Pause, resending the request until it is answered."""
        while (res := await self._pause_client.call_timeout(std_srvs.srv.Trigger.Request(), timeout_sec=_RESEND_AFTER_S)) is None:
            pass
        return res.success

    async def unpause(self) -> bool:
        while (res := await self._unpause_client.call_timeout(std_srvs.srv.Trigger.Request(), timeout_sec=_RESEND_AFTER_S)) is None:
            pass
        return res.success

    async def cleanup_namespace(self, prefix: str) -> int:
        env_id = int(prefix.rstrip("/").removeprefix("env_"))
        request = DeletePrims.Request(env_id=env_id, names=[prefix])
        while (res := await self._delete_prims_client.call_timeout(request, timeout_sec=_RESEND_AFTER_S)) is None:
            pass
        if not res.ret:
            return 0
        return 1 if res.ret[0] else 0

    async def step_seconds(self, seconds: float) -> float:
        n = max(1, round(seconds / _MUJOCO_PHYSICS_DT))
        start = self._node.sim_time
        res = await self._step_client.call_forever(Step.Request(n=n))
        if not res.success:
            raise RuntimeError(f"mujoco/Step({n}) failed")
        target = start + Time.from_float((n - 0.5) * _MUJOCO_PHYSICS_DT)
        while not await self._node.await_sim_time(target, freeze_timeout=10.0):
            self._logger.warning(f"waiting on mujoco sim clock >= {target.to_seconds():.3f}s (at {self._node.sim_time.to_seconds():.3f}s)")
        return (self._node.sim_time - start).to_seconds()


def _mesh_files(model: Model) -> list[tuple[str, str]]:
    """(visual name, mesh file) behind an obstacle model: the OBJ itself or every mesh visual of an SDF, empty without one."""
    if model.path is None:
        return []
    if model.type is ModelType.OBJ:
        return [("visual", str(model.path))]
    if model.type is ModelType.SDF:
        return [(visual.get("name", ""), str(model.path.parent / uri.removeprefix("file://"))) for visual in ET.fromstring(model.description).iter("visual") if (uri := visual.findtext(".//mesh/uri"))]
    return []


def material_to_msg(material: arena_simulation_setup.tree.assets.Material.Material) -> arena_mujoco_msgs.msg.Material:
    return Material(
        name=material.name,
        path=material.path,
    )


def _vector(velocity: tuple[float, float, float]) -> geometry_msgs.msg.Vector3:
    return geometry_msgs.msg.Vector3(x=float(velocity[0]), y=float(velocity[1]), z=float(velocity[2]))


def _shifted(point: geometry_msgs.msg.Point, shift: tuple[float, float, float]) -> geometry_msgs.msg.Point:
    return geometry_msgs.msg.Point(x=point.x + shift[0], y=point.y + shift[1], z=point.z + shift[2])


class MujocoSimulator(BaseSim, NodeInterface):
    ARTICULATED_PHYSICS = True
    SIM_NAME = 'mujoco'
    STACKS_LEVELS = True

    def __init__(self, **kwargs: typing.Unpack[SimKwargs]) -> None:
        """Initialize MujocoSimulator"""
        super().__init__(**kwargs)

        env_prefix = f"env_{self._env_id}"
        self._NS_PRIM = Namespace(env_prefix)('Obstacles')
        self._NS_ROBOT = Namespace(env_prefix)('Robots')
        self._NS_WALL = Namespace(env_prefix)('Walls')
        self._NS_FLOOR = Namespace(env_prefix)('Floors')
        self._NS_CEILING = Namespace(env_prefix)('Ceilings')

        self.wall_counter = itertools.count()
        self.floor_counter = itertools.count()
        self._clients = types.SimpleNamespace(
            DeleteLights=self.node.create_client_wrapper(DeleteLights, "/mujoco/DeleteLights"),
            DeletePedestrians=self.node.create_client_wrapper(DeletePedestrians, "/mujoco/DeletePedestrians"),
            DeletePrims=self.node.create_client_wrapper(DeletePrims, "/mujoco/DeletePrims"),
            DriveArticulated=self.node.create_client_wrapper(DriveArticulated, "/mujoco/DriveArticulated"),
            EditPrims=self.node.create_client_wrapper(EditPrims, "/mujoco/EditPrims"),
            MovePedestrians=self.node.create_client_wrapper(MovePedestrians, "/mujoco/MovePedestrians"),
            SetLights=self.node.create_client_wrapper(SetLights, "/mujoco/SetLights"),
            SetMapOffset=self.node.create_client_wrapper(SetMapOffset, "/mujoco/SetMapOffset"),
            SpawnArticulated=self.node.create_client_wrapper(SpawnArticulated, "/mujoco/SpawnArticulated", timeout=_SPAWN_ASSETS_TIMEOUT_S),
            SpawnCeilings=self.node.create_client_wrapper(SpawnCeilings, "/mujoco/SpawnCeilings"),
            SpawnFloors=self.node.create_client_wrapper(SpawnFloors, "/mujoco/SpawnFloors"),
            SpawnLights=self.node.create_client_wrapper(SpawnLights, "/mujoco/SpawnLights"),
            SpawnPedestrians=self.node.create_client_wrapper(SpawnPedestrians, "/mujoco/SpawnPedestrians", timeout=_SPAWN_ASSETS_TIMEOUT_S),
            SpawnPrims=self.node.create_client_wrapper(SpawnPrims, "/mujoco/SpawnPrims", timeout=_SPAWN_ASSETS_TIMEOUT_S),
            SpawnUrdf=self.node.create_client_wrapper(SpawnUrdf, "/mujoco/SpawnUrdf"),
            SpawnWalls=self.node.create_client_wrapper(SpawnWalls, "/mujoco/SpawnWalls"),
        )
        self._prim_batch = _PrimBatch(self._clients.SpawnPrims, self._env_id)
        self._peds_publisher = self.node.create_publisher(Pedestrians, "/mujoco/arena_peds", 10)

        self._robot_prims: dict[str, str] = {}
        self._map_offsets: dict[str, tuple[float, float, float]] = {}
        self._box_centers: dict[str, tuple[float, float, float]] = {}

    def _robot_loader_args(self, robot: Robot) -> dict[str, object]:
        robot_config = arena_robots.Robot.RobotIdentifier(robot.model.name).resolve_sync()
        args: dict[str, object] = {
            **robot.extra,
            **attrs.asdict(robot, filter=lambda attribute, _: attribute.name != 'extra'),
            'optim': self.node.rosparam[str].get('optim', ''),
            'sensor_topic_patches': arena_robots.Sensor.topic_elements(robot_config.effective_sensors(robot.resolved_request, frames=robot.frames), ''),
        }
        args.pop('resolved_assembly', None)
        if robot.resolved_assembly is not None:
            assert robot_config.assembly is not None
            catalog = arena_robots.catalog.Catalog()
            args['xacro_wrapper'] = arena_robots.catalog.render_wrapper_xacro(robot_config, robot.resolved_assembly, catalog=catalog)
            args['control_joint_patch'] = arena_robots.catalog.render_control_joints(robot.resolved_assembly, catalog, prefix=robot_config.assembly.prefix)
        return args

    def robot_controllers(self, robot: Robot) -> list[str]:
        return robot_controllers(arena_robots.Robot.RobotIdentifier(robot.model.name).resolve_sync(), robot.resolved_assembly)

    async def robot_spawn(self, robots: Sequence[Robot]) -> Sequence[bool]:
        async def impl(robot: Robot) -> bool:
            try:
                model = await (await robot.model.resolve()).model.get(
                    (ModelType.URDF,),
                    loader_args=self._robot_loader_args(robot),
                )

                if model.type == ModelType.URDF:
                    assert model.path is not None, f"URDF model {model.name} must have a valid file path"
                    robot_config = await arena_robots.Robot.RobotIdentifier(robot.model.name).resolve()
                    robot_params = robot_config.model_params

                    fq_name = self._NS_ROBOT(robot.name)
                    ns = str(self.node.service_namespace(robot.name))

                    control_spec = robot_params.control
                    is_ros2_control = control_spec is not None and control_spec.is_ros2_control
                    relays_odom = control_spec is not None and is_ros2_control and control_spec.odom_topic != "odom"

                    bridge_urdf_path = os.path.join('/tmp', f"arena_bridge_{robot.frame.sanitize()}.urdf")
                    transform_urdf_for_bridge(
                        str(model.path),
                        {
                            'velocity': f"{ns}/mujoco/joint_commands_velocity",
                            'position': f"{ns}/mujoco/joint_commands_position",
                        },
                        f"{ns}/mujoco/joint_states",
                        bridge_urdf_path,
                    )

                    spawn_res = await self._clients.SpawnUrdf.call_timeout(
                        SpawnUrdf.Request(
                            env_id=self._env_id,
                            name=fq_name,
                            urdf_path=str(model.path),
                            robot_model=robot.model.name,
                            localization=True,
                            tf_prefix=robot.frame.tf(),
                            base_frame=robot_params.base_frame,
                            odom_frame=robot_params.odom_frame,
                            pose=self._placed(robot.pose).to_msg(),
                            cmd_vel_topic=self.node.service_namespace(robot.name, 'cmd_vel'),
                            joint_states_topic=self.node.service_namespace(robot.name, 'joint_states'),
                            odom_topic='' if relays_odom else self.node.service_namespace(robot.name, 'odom'),
                            sensors=[
                                arena_mujoco_msgs.msg.Sensor(
                                    name=spec.name,
                                    type=str(spec.type),
                                    topic=arena_robots.Sensor.resolve_topic(spec, ns),
                                    frame=spec.frame,
                                    sensor=spec.sensor or '',
                                )
                                for spec in robot_config.effective_sensors(robot.resolved_request, frames=robot.frames)
                            ],
                        )
                    )
                    if spawn_res is None or not spawn_res.path:
                        self._logger.error(f"SpawnUrdf failed for {fq_name!r}: {'timeout' if spawn_res is None else 'spawn error, check mujoco log'}")
                        return False

                    self._track_level(robot.sim_path, robot.pose)
                    self._robot_prims[robot.sim_path] = spawn_res.path
                    self._map_offsets.pop(robot.sim_path, None)
                    if (offset := self._map_offset_request(robot.sim_path)) is not None:
                        offset_res = await self._clients.SetMapOffset.call_timeout(offset)
                        if offset_res is None or not offset_res.ok:
                            self._map_offsets.pop(robot.sim_path, None)
                            self._logger.error(f"SetMapOffset failed for {fq_name!r}: {'timeout' if offset_res is None else 'no odom entry, check mujoco log'}")
                            return False

                    rsp_urdf_path = bridge_urdf_path if is_ros2_control else str(model.path)

                    await self._launch_robot_stack(robot, robot_params, rsp_urdf_path)

                    self._register_agent_robot(robot, robot_params)
                    return True

                raise NotImplementedError(f"robot model of type {model.type} can't be spawned by {self.__class__.__name__}")

            except Exception as e:
                self._logger.error(f"{repr(e)}\n{traceback.format_exc()}")
                return False

        return await asyncio.gather(*map(impl, robots))

    async def _launch_robot_stack(
        self,
        robot: Robot,
        robot_params: arena_robots.Robot.ModelParams,
        urdf_path: str,
    ) -> None:
        """Launch RSP, and (if ros2_control) controller_manager + twist_stamper."""
        ns = str(self.node.service_namespace(robot.name))
        with open(urdf_path) as f:
            description = f.read()

        ld = launch.LaunchDescription()
        ld.add_action(launch_ros.actions.PushRosNamespace(namespace=ns))

        ld.add_action(
            launch_ros.actions.Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                output='screen',
                parameters=[
                    {'use_sim_time': True},
                    {'robot_description': description},
                    {'frame_prefix': robot.frame.tf()},
                ],
            )
        )

        control_spec = robot_params.control
        if control_spec is not None and control_spec.is_ros2_control:
            if not control_spec.controllers:
                raise ValueError(f"control.mode=ros2_control but no controllers declared for {robot.name}")

            robot_config = arena_robots.Robot.RobotIdentifier(robot.model.name).resolve_sync()
            control_prefix = robot_config.assembly.prefix if robot_config.assembly is not None else 'robot_'

            rendered_yaml = (
                effective_control_yaml(
                    robot.resolved_assembly,
                    control_spec.config,
                    robot.sim_path,
                    robot.frame.tf(),
                    sim=SimSimulator.MUJOCO,
                    prefix=control_prefix,
                )
                if control_spec.config is not None
                else None
            )

            cm_params: list[SomeParameterFile | SomeParametersDict] = [{'use_sim_time': True}]
            if rendered_yaml is not None:
                cm_params.append(rendered_yaml)

            cm_node = launch_ros.actions.Node(
                package='controller_manager',
                executable='ros2_control_node',
                name='controller_manager',
                output='screen',
                parameters=cm_params,
            )
            ld.add_action(cm_node)
            urdf_pub_node = launch_ros.actions.Node(
                package='arena_runtime',
                executable='urdf_publisher',
                name='urdf_publisher',
                output='screen',
                parameters=[{'robot_description': description}],
            )
            ld.add_action(
                launch.actions.RegisterEventHandler(
                    launch.event_handlers.OnProcessStart(
                        target_action=cm_node,
                        on_start=[urdf_pub_node],
                    )
                )
            )
            ld.add_action(
                twist_stamper_node(
                    control_spec.cmd_vel_topic,
                    robot.frame.tf(robot_params.base_frame),
                )
            )
            if control_spec.odom_topic != "odom":
                ld.add_action(odom_relay_node(control_spec.odom_topic))

        await self.node.do_launch(ld)

    async def obstacle_spawn(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        level = obstacles_optim_level(self.node)

        async def plan(obstacle: Obstacle) -> tuple[list[tuple[str, str]], tuple | None]:
            """(visual name, mesh file) list and bbox to spawn from, both empty without a usable asset."""
            box = await resolve_obstacle_box(obstacle)
            if level is ObstaclesOptim.BBOX and box is not None:
                return [], box
            try:
                meshes = _mesh_files(await (await obstacle.model.resolve()).model.get(_MESH_MODEL_TYPES))
            except Exception:
                self._logger.debug(traceback.format_exc())
                meshes = []
            return (meshes, None) if meshes else ([], box)

        plans = await asyncio.gather(*map(plan, obstacles))

        req = SpawnPrims.Request(env_id=self._env_id)
        indices: list[int] = []
        skipped: list[str] = []
        centers: dict[int, tuple[float, float, float]] = {}
        for i, (obstacle, (meshes, box)) in enumerate(zip(obstacles, plans, strict=True)):
            scale = (1.0, 1.0, 1.0) if obstacle.scale is None else (obstacle.scale.x, obstacle.scale.y, obstacle.scale.z)
            pose = self._placed(obstacle.pose)
            if not meshes:
                if box is None:
                    skipped.append(obstacle.name)
                    continue
                size, center = box
                centers[i] = (center[0] * scale[0], center[1] * scale[1], center[2] * scale[2])
                pose = offset_pose(pose, centers[i])
                scale = tuple(d * f for d, f in zip(size, scale, strict=True))
            req.prims.append(  # pyright: ignore[reportAttributeAccessIssue]
                Prim(
                    name=self._NS_PRIM(obstacle.name),
                    pose=pose.to_msg(),
                    scale=Scale(x=scale[0], y=scale[1], z=scale[2]),
                    mesh_path=meshes[0][1] if meshes else "",
                    visuals=[PrimVisual(name=name, mesh_path=path) for name, path in meshes[1:]],
                    is_static=True,
                )
            )
            indices.append(i)
        if skipped:
            self._logger.warning(f"no mesh or bounding box for {len(skipped)} obstacle(s), not spawned: {skipped[:5]}")

        results: list[bool] = [False] * len(obstacles)
        if req.prims and (response := await self._clients.SpawnPrims.call_timeout(req)) is not None:
            for i, ok in zip(indices, response.ret, strict=True):
                results[i] = bool(ok)
                name = self._NS_PRIM(obstacles[i].name)
                if ok and i in centers:
                    self._box_centers[name] = centers[i]
                else:
                    self._box_centers.pop(name, None)
        return tuple(results)

    async def _articulated_spawn(self, objects: Sequence[ArticulatedObject]) -> Sequence[bool]:
        async def spawn(runtime: ArticulatedObject) -> bool:
            obstacle = runtime.obstacle
            links: list[ArticulatedLink] = []
            for link, joint in runtime.tree():
                view = runtime.links.get(link)
                meshes = _mesh_files(await view.model.get(_MESH_MODEL_TYPES)) if view is not None else []
                message = link_message(ArticulatedLink(), runtime, link, joint, JointDynamics)
                message.mesh_path = meshes[0][1] if meshes else ""
                message.visuals = [PrimVisual(name=name, mesh_path=path) for name, path in meshes[1:]]
                links.append(message)
            scale = obstacle.scale
            request = SpawnArticulated.Request(
                env_id=self._env_id,
                name=self._NS_PRIM(obstacle.name),
                pose=obstacle.pose.to_msg(),
                scale=Scale(x=1.0, y=1.0, z=1.0) if scale is None else Scale(x=scale.x, y=scale.y, z=scale.z),
                links=links,
            )
            response = await self._clients.SpawnArticulated.call_timeout(request)
            return response is not None and bool(response.ret)

        return await asyncio.gather(*map(spawn, objects))

    async def _articulated_drive(self, commands: Sequence[ArticulatedCommand]) -> Sequence[Values | None]:
        request = DriveArticulated.Request(env_id=self._env_id, commands=[JointCommand(name=self._NS_PRIM(command.runtime.name), **command_fields(command)) for command in commands])
        response = await self._clients.DriveArticulated.call_timeout(request)
        if response is None:
            return [None] * len(commands)
        return [dict(zip(reading.joints, reading.position, strict=True)) if reading.found else None for reading in response.readings]

    def _placement(self, obstacle: Obstacle) -> tuple[str, Pose]:
        name = self._NS_PRIM(obstacle.name)
        center = self._box_centers.get(name)
        return name, obstacle.pose if center is None else offset_pose(obstacle.pose, center)

    async def obstacle_move(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        return await self._move_entities([self._placement(o) for o in obstacles])

    async def pedestrian_move(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        req = MovePedestrians.Request(
            pedestrians=[
                Pedestrian(
                    name=p.sim_path,
                    pose=self._placed(p.pose).to_msg(),
                )
                for p in pedestrians
            ]
        )
        res = await self._clients.MovePedestrians.call_timeout(req)
        if res is None:
            return tuple(False for _ in pedestrians)
        return tuple(r == MovePedestrians.Response.SUCCESS for r in res.results)

    async def robot_move(self, robots: Sequence[Robot]) -> Sequence[bool]:
        async def move_robot(robot: Robot) -> bool:
            try:
                if not await self._move_entity(self._robot_prims.get(robot.sim_path, self._NS_ROBOT(robot.name)), robot.pose):
                    return False
                self._track_level(robot.sim_path, robot.pose)
                return True
            except Exception as e:
                self._logger.error(f"Failed to move robot {robot.name}: {e}\n{traceback.format_exc()}")
                return False

        return await asyncio.gather(*map(move_robot, robots))

    async def obstacle_delete(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        names = [self._NS_PRIM(o.name) for o in obstacles]
        for name in names:
            self._box_centers.pop(name, None)
        return await self._delete_entities(names)

    async def pedestrian_delete(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        res = await self._clients.DeletePedestrians.call_timeout(DeletePedestrians.Request(names=[p.sim_path for p in pedestrians]))
        if res is None:
            return tuple(False for _ in pedestrians)
        return tuple(r == DeletePedestrians.Response.SUCCESS for r in res.results)

    async def robot_delete(self, robots: Sequence[Robot]) -> Sequence[bool]:
        names: list[str] = []
        for robot in robots:
            self._forget_agent_robot(robot.sim_path)
            self._map_offsets.pop(robot.sim_path, None)
            names.append(self._robot_prims.pop(robot.sim_path, self._NS_ROBOT(robot.name)))
        return await asyncio.gather(*(self._delete_entity(name) for name in names))

    async def remove_world(self) -> bool:
        res = await self._clients.DeletePrims.call_timeout(DeletePrims.Request(env_id=self._env_id, names=[f"{ns}/" for ns in (self._NS_WALL, self._NS_FLOOR, self._NS_CEILING)]))
        return res is not None

    async def spawn_walls(self, walls: Sequence[WallDefinition], clear_existing: bool = True) -> bool:
        if clear_existing:
            await self.remove_world()
        self._logger.debug("Attempting to spawn walls")

        async def create_segment(segment: WallSegment, shift: tuple[float, float, float]) -> Wall | Prim | None:
            end = segment.end.to_msg()
            end.z += segment.height
            try:
                wall_name = self._NS_WALL(self._realizer.realize(f"wall_{next(self.wall_counter)}"))
                material = material_to_msg(await segment.material.resolve())
                if (sloped := sloped_box(segment)) is not None:
                    (sx, sy, sz), pose = sloped
                    pose_msg = pose.to_msg()
                    pose_msg.position = _shifted(pose_msg.position, shift)
                    return Prim(name=wall_name, pose=pose_msg, scale=Scale(x=sx, y=sy, z=sz), mesh_path="", material=material, is_static=True)
                return Wall(
                    name=wall_name,
                    start=_shifted(segment.start.to_msg(), shift),
                    end=_shifted(end, shift),
                    material=material,
                    thickness=segment.width,
                )

            except Exception as e:
                self._logger.error(f"Failed to spawn wall: {e}\n{traceback.format_exc()}")
                return None

        async def create_obstacle(item: tuple[ObstacleDefinition, Model]) -> Prim | None:
            obstacle, model = item
            try:
                prim_name = self._realizer.realize(f"obstacle_{next(self.wall_counter)}")
                meshes = _mesh_files(model)
                prim = Prim(is_static=True)
                prim.mesh_path = meshes[0][1] if meshes else ""
                prim.visuals = [PrimVisual(name=name, mesh_path=path) for name, path in meshes[1:]]
                prim.name = self._NS_WALL(prim_name)
                prim.pose = self._placed(obstacle.pose).to_msg()
                prim.scale = Scale(x=1.0, y=1.0, z=1.0)
                return prim

            except Exception as e:
                self._logger.error(f"Failed to spawn wall obstacle: {e}\n{traceback.format_exc()}")
                return None

        async def create_wall(wall: WallDefinition) -> tuple[typing.Iterator[object], typing.Iterator[object]]:
            segments, obstacles = await realize_renderable(wall, _MESH_MODEL_TYPES)
            shift = self._shift((wall.start.x + wall.end.x) / 2.0, (wall.start.y + wall.end.y) / 2.0)
            return (create_segment(segment, shift) for segment in segments), map(create_obstacle, obstacles)

        wall_futures = await asyncio.gather(*map(create_wall, walls))
        segment_futures, obstacle_futures = zip(*wall_futures, strict=False) if wall_futures else ((), ())

        walls_req = SpawnWalls.Request(env_id=self._env_id)
        prims_req = SpawnPrims.Request(env_id=self._env_id)
        segment_msgs = list(filter(None, await asyncio.gather(*itertools.chain.from_iterable(segment_futures))))
        walls_req.walls = [msg for msg in segment_msgs if isinstance(msg, Wall)]
        prims_req.prims = [msg for msg in segment_msgs if isinstance(msg, Prim)] + list(filter(None, await asyncio.gather(*itertools.chain.from_iterable(obstacle_futures))))

        walls_res = await self._clients.SpawnWalls.call_timeout(walls_req)
        prims_res = await self._clients.SpawnPrims.call_timeout(prims_req)
        res = bool(walls_res) and all(walls_res.ret) and bool(prims_res) and all(prims_res.ret)

        self._logger.debug("All walls spawned.")
        return res

    async def spawn_floors(self, floors: Sequence[FloorDefinition]) -> bool:
        self._logger.debug("Attempting to spawn floors")

        async def impl(floor: FloorDefinition) -> Floor | None:
            try:
                return Floor(
                    name=self._NS_FLOOR(floor.name),
                    x_length=floor.x_length,
                    y_length=floor.y_length,
                    pos=self._placed_point(floor.pos.to_msg()),
                    material=material_to_msg(await floor.material.resolve()),
                )

            except Exception:
                self._logger.error(f"Failed to spawn floor: {floor.name}\n{traceback.format_exc()}")
                return None

        floors_req = SpawnFloors.Request(env_id=self._env_id)
        floors_req.floors = list(filter(None, await asyncio.gather(*map(impl, floors))))
        floors_res = await self._clients.SpawnFloors.call_timeout(floors_req)

        res = bool(floors_res) and all(floors_res.ret)
        self._logger.debug("All floors spawned successfully.")
        return res

    async def spawn_ceilings(self, ceilings: Sequence[CeilingDefinition]) -> bool:
        self._logger.debug("Attempting to spawn ceilings")

        async def impl(ceiling: CeilingDefinition) -> Ceiling | None:
            try:
                pos = ceiling.pos.to_msg()
                pos.z = ceiling.z
                return Ceiling(
                    name=self._NS_CEILING(ceiling.name),
                    x_length=ceiling.x_length,
                    y_length=ceiling.y_length,
                    pos=self._placed_point(pos),
                    cast_shadows=ceiling.cast_shadows,
                    material=material_to_msg(await ceiling.material.resolve()),
                )

            except Exception:
                self._logger.error(f"Failed to spawn ceiling: {ceiling.name}\n{traceback.format_exc()}")
                return None

        ceilings_req = SpawnCeilings.Request(env_id=self._env_id)
        ceilings_req.ceilings = list(filter(None, await asyncio.gather(*map(impl, ceilings))))
        ceilings_res = await self._clients.SpawnCeilings.call_timeout(ceilings_req)

        res = bool(ceilings_res) and all(ceilings_res.ret)
        self._logger.debug("All ceilings spawned successfully.")
        return res

    def _light_msg(self, light: LightDefinition) -> Light:
        spec = light.spec
        fixtures = [] if spec.ambient else (light.fixtures if light.rig else [light.position])
        return Light(
            name=light.name,
            shape=spec.shape,
            cone_deg=spec.cone_deg,
            positions=[self._placed_point(position.to_msg()) for position in fixtures],
            direction=geometry_msgs.msg.Vector3(x=light.direction[0], y=light.direction[1], z=light.direction[2]),
            flux=light.lux if spec.ambient else light.lumens,
            color=cct_to_rgb(light.cct_K),
            cast_shadows=light.cast_shadows,
            level=0.0 if light.lit is False else light.dim,
            alive=light.alive(light.dead_fraction),
            intrinsic=light.intrinsic,
            rig=light.rig,
            glow=light.glow,
            owner=self._NS_PRIM(light.owner) if light.glow else "",
        )

    async def _spawn_lights(self, lights: Sequence[LightDefinition]) -> bool:
        res = await self._clients.SpawnLights.call_timeout(SpawnLights.Request(env_id=self._env_id, lights=[self._light_msg(light) for light in lights]))
        return res is not None and all(res.ret)

    async def _remove_lights(self, lights: Sequence[LightDefinition]) -> bool:
        res = await self._clients.DeleteLights.call_timeout(DeleteLights.Request(env_id=self._env_id, names=[light.name for light in lights]))
        return res is not None and all(res.ret)

    async def _move_light(self, light: LightDefinition, position: Position, yaw: float) -> None:
        state = LightState(name=light.name, move=True, position=self._placed_point(position.to_msg()), yaw=yaw)
        res = await self._clients.SetLights.call_timeout(SetLights.Request(env_id=self._env_id, states=[state]))
        if res is None or not all(res.ret):
            raise RuntimeError(f"light {light.name!r} did not move")

    async def _apply_light(self, light: LightDefinition, level: float, alive: Sequence[bool]) -> None:
        state = LightState(name=light.name, level=level, alive=list(alive))
        res = await self._clients.SetLights.call_timeout(SetLights.Request(env_id=self._env_id, states=[state]))
        if res is None or not all(res.ret):
            raise RuntimeError(f"light {light.name!r} did not take its state")

    async def spawn_box(self, name: str, size: tuple[float, float, float], pose: Pose, material: MaterialIdentifier | None = None, *, velocity: tuple[float, float, float] | None = None, comb: bool = False) -> bool:
        """Spawn a box as a single primitive Prim (empty mesh, scale = size), in its material when one is given, moving when `velocity` is given."""
        sx, sy, sz = size
        prim = Prim(
            name=name,
            pose=self._placed(pose).to_msg(),
            scale=Scale(x=sx, y=sy, z=sz),
            mesh_path="",
            comb=comb,
            is_static=velocity is None,
        )
        if velocity is not None:
            prim.velocity = _vector(velocity)
        if material is not None:
            prim.material = material_to_msg(await material.resolve())
        return await self._prim_batch.submit(prim)

    async def move_box(self, name: str, pose: Pose) -> bool:
        req = EditPrims.Request(env_id=self._env_id, prims=[Prim(name=name, pose=self._placed(pose).to_msg())], pose=True)
        self._clients.EditPrims.client.call_async(req)
        return True

    async def move_boxes(self, poses: Mapping[str, Pose]) -> bool:
        req = EditPrims.Request(env_id=self._env_id, prims=[Prim(name=name, pose=self._placed(pose).to_msg()) for name, pose in poses.items()], pose=True)
        self._clients.EditPrims.client.call_async(req)
        return True

    async def set_box_velocities(self, velocities: Mapping[str, tuple[float, float, float]]) -> bool:
        req = EditPrims.Request(env_id=self._env_id, prims=[Prim(name=name, velocity=_vector(velocity)) for name, velocity in velocities.items()], velocity=True)
        res = await self._clients.EditPrims.call_timeout(req)
        return bool(res) and all(res.ret)

    async def delete_box(self, name: str) -> bool:
        return await self._delete_entity(name)

    async def set_robot_pose(self, sim_path: str, pose: Pose) -> bool:
        prim_name = self._robot_prims.get(sim_path)
        if prim_name is None or not await self._move_entity(prim_name, pose):
            return False
        self._track_level(sim_path, pose)
        return True

    async def step(self, n: int = 1) -> bool:
        """Unused wall-clock fake, superseded by SimLifecycle.step_seconds."""
        del n
        raise NotImplementedError('use sim_lifecycle/step')

    async def pedestrian_spawn(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        req = SpawnPedestrians.Request(
            pedestrians=[
                SpawnPedestrian(
                    pedestrian=Pedestrian(
                        name=pedestrian.sim_path,
                        pose=self._placed(pedestrian.pose).to_msg(),
                    ),
                    model_ref=pedestrian.model.name,
                )
                for pedestrian in pedestrians
            ]
        )
        res = await self._clients.SpawnPedestrians.call_timeout(req)
        if res is None:
            return tuple(False for _ in pedestrians)

        success = tuple(r == SpawnPedestrians.Response.SUCCESS for r in res.results)

        await self.pedestrian_update(
            Pedestrians(
                pedestrians=[
                    Pedestrian(
                        name=ped.sim_path,
                        pose=ped.pose.to_msg(),
                    )
                    for status, ped in zip(success, pedestrians, strict=False)
                    if status
                ]
            )
        )

        return success

    async def pedestrian_update(self, pedestrians: Pedestrians) -> Sequence[bool]:
        self._peds_publisher.publish(self._placed_peds(pedestrians))
        return tuple(True for _ in pedestrians.pedestrians)

    def _placed_peds(self, pedestrians: Pedestrians) -> Pedestrians:
        """The pedestrians in the simulator frame, the message itself while the levels are not stacked."""
        if not self._realizer.placement().stacked:
            return pedestrians
        placed = copy.deepcopy(pedestrians)
        for ped in placed.pedestrians:
            ped.pose = self._placed(Pose.from_msg(ped.pose)).to_msg()
        return placed

    def _placed_point(self, point: geometry_msgs.msg.Point) -> geometry_msgs.msg.Point:
        """An abstract point in the simulator frame."""
        return _shifted(point, self._shift(point.x, point.y))

    def _map_offset_request(self, sim_path: str) -> SetMapOffset.Request | None:
        """The request that moves a spawned robot's map -> odom to the inverse of its level's shift, None while that is what the server holds."""
        prim_name = self._robot_prims.get(sim_path)
        dx, dy, dz = self._robot_shift(sim_path)
        offset = (-dx, -dy, -dz)
        if prim_name is None or self._map_offsets.get(sim_path, (0.0, 0.0, 0.0)) == offset:
            return None
        self._map_offsets[sim_path] = offset
        return SetMapOffset.Request(env_id=self._env_id, name=prim_name, translation=_vector(offset))

    def _publish_odom(self, sim_path: str) -> None:
        if (request := self._map_offset_request(sim_path)) is not None:
            self._clients.SetMapOffset.client.call_async(request).add_done_callback(lambda done: self._map_offset_done(sim_path, done))

    def _map_offset_done(self, sim_path: str, call: rclpy.task.Future) -> None:
        if call.cancelled() or call.exception() is not None or (result := call.result()) is None or not result.success:
            self._map_offsets.pop(sim_path, None)
            self._logger.warning(f"SetMapOffset failed for {sim_path!r}, its map -> odom is stale")

    async def _delete_entity(self, name: str) -> bool:
        return (await self._delete_entities([name]))[0]

    async def _delete_entities(self, names: Sequence[str]) -> Sequence[bool]:
        if not names:
            return ()
        res = await self._clients.DeletePrims.call_timeout(DeletePrims.Request(env_id=self._env_id, names=list(names)))
        if res is None:
            return (False,) * len(names)
        return tuple(res.ret)

    async def _move_entity(self, name: str, pose: Pose) -> bool:
        return (await self._move_entities([(name, pose)]))[0]

    async def _move_entities(self, actions: Sequence[tuple[str, Pose]]) -> Sequence[bool]:
        req = EditPrims.Request(
            env_id=self._env_id,
            prims=[
                Prim(
                    name=name,
                    pose=self._placed(pose).to_msg(),
                )
                for name, pose in actions
            ],
            pose=True,
        )

        response = await self._clients.EditPrims.call_timeout(req)
        if response is None:
            return [False] * len(actions)

        return response.ret

    async def setup(self):
        """
        Initialize all ROS 2 service clients and wait for their availability.
        """
        self._logger.info("Setting up MujocoSimulator service clients...")
        futures: list[typing.Awaitable] = []
        for client in self._clients.__dict__.values():
            client = typing.cast(ClientWrapper, client)
            self._logger.debug(f"Initializing service client: {client.client.srv_name}")
            futures.append(client.ensure())
        await asyncio.gather(*futures)

        self._logger.info("All service clients initialized and available.")

    @classmethod
    async def create(cls, **kwargs: typing.Unpack[SimKwargs]) -> "MujocoSimulator":
        self = cls(**kwargs)
        self._logger.info("Creating MujocoSimulator instance...")
        await self.setup()
        return self
