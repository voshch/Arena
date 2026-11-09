import asyncio
import copy
import itertools
import math
import os
import traceback
import types
import typing
from collections.abc import Mapping, Sequence

import arena_people_msgs.msg
import arena_robots.catalog
import arena_robots.Robot
import arena_robots.Sensor
import arena_simulation_setup.tree.assets.Material
import attrs
import geometry_msgs.msg
import isaacsim_msgs.msg
import launch
import launch.event_handlers
import launch_ros
import rclpy.task
import std_msgs.msg
import std_srvs.srv
from arena_bringup.constants import SimSimulator
from arena_people_msgs.msg import Pedestrian, SpawnPedestrian
from arena_people_msgs.srv import (
    DeletePedestrians,
    MovePedestrians,
    SpawnPedestrians,
    UpdatePedestrians,
)
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.Async import AsyncUtil, ClientWrapper
from arena_rclpy_mixins.params import declare
from arena_rclpy_mixins.shared import Namespace
from arena_rclpy_mixins.Time import Time
from arena_simulation_setup.shared import Light as LightDefinition
from arena_simulation_setup.shared import Obstacle as ObstacleDefinition
from arena_simulation_setup.shared import cct_to_rgb
from arena_simulation_setup.tree.Wall import WallSegment
from arena_simulation_setup.utils.articulation import Values
from isaacsim_msgs.msg import (
    ArticulatedLink,
    Ceiling,
    Floor,
    JointCommand,
    JointDynamics,
    Light,
    LightState,
    Material,
    Prim,
    Wall,
)
from isaacsim_msgs.srv import (
    DeleteLights,
    DeletePrims,
    DriveArticulated,
    EditPrims,
    ResetWorld,
    SetLights,
    SetMapOffset,
    SpawnArticulated,
    SpawnCeilings,
    SpawnFloors,
    SpawnLights,
    SpawnPrims,
    SpawnUrdf,
    SpawnWalls,
    StepSimulation,
)
from launch_ros.parameters_type import SomeParameterFile, SomeParametersDict

from arena_runtime._node import NodeInterface
from arena_runtime.flags import ObstaclesOptim, obstacles_optim_level
from arena_runtime.node_params import NodeParams
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
IsaacHost is constructed once by arena_runtime_node and owns process-singleton resources for Isaac: the lifecycle (pause/unpause/cleanup) and the pause/unpause/delete service clients.
IsaacSimulator is per-env on arena_env_node and adapts env-namespace state (per-robot publishers, env-prefixed entity names) over those shared resources.
"""


# matches World(physics_dt=rendering_dt=1/60) pinned in run_isaacsim.py
_ISAAC_PHYSICS_HZ = 60.0
_ISAAC_EXPOSURE_GAIN = 5.0
_TILT_EPS = 1e-9
_MAP_OFFSET_TIMEOUT_S = 10.0


class IsaacHost(SimLifecycle):
    def __init__(self, node: ArenaMixinNode) -> None:
        self._node = node
        self._logger = node.get_logger().get_child(type(self).__name__)
        self._pause_client: ClientWrapper = node.create_client_wrapper(
            std_srvs.srv.Trigger,
            "/isaac/PauseSimulation",
        )
        self._unpause_client: ClientWrapper = node.create_client_wrapper(
            std_srvs.srv.Trigger,
            "/isaac/UnpauseSimulation",
        )
        self._delete_prims_client: ClientWrapper = node.create_client_wrapper(
            DeletePrims,
            "/isaac/DeletePrims",
        )
        self._step_client: ClientWrapper = node.create_client_wrapper(
            StepSimulation,
            "/isaac/StepSimulationN",
        )
        self._clock_skew_logged = False

    async def ensure_ready(self) -> None:
        await asyncio.gather(
            self._pause_client.ensure(),
            self._unpause_client.ensure(),
            self._delete_prims_client.ensure(),
            self._step_client.ensure(),
        )

    async def pause(self) -> bool:
        res = await self._pause_client.call_forever(std_srvs.srv.Trigger.Request())
        return res.success

    async def unpause(self) -> bool:
        res = await self._unpause_client.call_forever(std_srvs.srv.Trigger.Request())
        return res.success

    async def cleanup_namespace(self, prefix: str) -> int:
        res = await self._delete_prims_client.call_forever(DeletePrims.Request(names=[prefix]))
        if not res.ret:
            return 0
        return 1 if res.ret[0] else 0

    async def step_seconds(self, seconds: float) -> float:
        n = max(1, round(seconds * _ISAAC_PHYSICS_HZ))
        start = self._node.sim_time
        res = await self._step_client.call_forever(StepSimulation.Request(steps=n))
        if not res.success:
            raise RuntimeError(res.error_msg)
        reported = res.target_sim_time - n / _ISAAC_PHYSICS_HZ
        if not self._clock_skew_logged and abs(reported - start.to_seconds()) > 1.0 / _ISAAC_PHYSICS_HZ:
            self._clock_skew_logged = True
            self._logger.warning(f"isaac sim time {reported:.3f}s vs /clock {start.to_seconds():.3f}s, stepping on /clock")
        target = start + Time.from_float((n - 0.5) / _ISAAC_PHYSICS_HZ)
        while not await self._node.await_sim_time(target, freeze_timeout=10.0):
            self._node.get_logger().warning(f"waiting on isaac sim clock >= {target.to_seconds():.3f}s (at {self._node.sim_time.to_seconds():.3f}s)")
        return (self._node.sim_time - start).to_seconds()


def material_to_msg(material: arena_simulation_setup.tree.assets.Material.Material) -> isaacsim_msgs.msg.Material:
    return Material(
        name=material.name,
        path=material.path,
    )


def _vector3(values: tuple[float, float, float]) -> geometry_msgs.msg.Vector3:
    x, y, z = values
    return geometry_msgs.msg.Vector3(x=float(x), y=float(y), z=float(z))


def _box_wall(name: str, size: tuple[float, float, float], pose: Pose) -> Wall:
    """Wall message of a box of `size` at a simulator-frame pose, yawed only, a tilt goes through EditPrims."""
    sx, sy, sz = size
    yaw = pose.orientation.to_yaw()
    cx, cy, cz = pose.position.x, pose.position.y, pose.position.z
    half = sx / 2.0
    return Wall(
        name=name,
        start=geometry_msgs.msg.Point(x=cx - half * math.cos(yaw), y=cy - half * math.sin(yaw), z=cz - sz / 2.0),
        end=geometry_msgs.msg.Point(x=cx + half * math.cos(yaw), y=cy + half * math.sin(yaw), z=cz + sz / 2.0),
        thickness=sy,
        is_static=True,
    )


class IsaacSimulator(BaseSim, NodeInterface):
    SIM_NAME = 'isaac'
    STACKS_LEVELS = True
    ARTICULATED_PHYSICS = True

    def __init__(self, **kwargs: typing.Unpack[SimKwargs]) -> None:
        """Initialize IsaacSimulator"""
        super().__init__(**kwargs)

        env_prefix = f"env_{self._env_id}"
        self._NS_PRIM = Namespace(env_prefix)('Obstacles')
        self._NS_ROBOT = Namespace(env_prefix)('Robots')
        self._NS_WALL = Namespace(env_prefix)('Walls')
        self._NS_FLOOR = Namespace(env_prefix)('Floors')
        self._NS_CEILING = Namespace(env_prefix)('Ceilings')
        self._NS_LIGHT = Namespace(env_prefix)('Lights')

        self.wall_counter = itertools.count()
        self.floor_counter = itertools.count()
        self._clients = types.SimpleNamespace(
            DeleteLights=self.node.create_client_wrapper(DeleteLights, "/isaac/DeleteLights"),
            DeletePedestrians=self.node.create_client_wrapper(DeletePedestrians, "/isaac/DeletePedestrians"),
            DeletePrims=self.node.create_client_wrapper(DeletePrims, "/isaac/DeletePrims"),
            DriveArticulated=self.node.create_client_wrapper(DriveArticulated, "/isaac/DriveArticulated"),
            EditPrims=self.node.create_client_wrapper(EditPrims, "/isaac/EditPrims"),
            MovePedestrians=self.node.create_client_wrapper(MovePedestrians, "/isaac/MovePedestrians"),
            ResetWorld=self.node.create_client_wrapper(ResetWorld, "/isaac/ResetWorld"),
            SetLights=self.node.create_client_wrapper(SetLights, "/isaac/SetLights"),
            SetMapOffset=self.node.create_client_wrapper(SetMapOffset, "/isaac/SetMapOffset"),
            SpawnArticulated=self.node.create_client_wrapper(SpawnArticulated, "/isaac/SpawnArticulated"),
            UpdatePedestrians=self.node.create_client_wrapper(UpdatePedestrians, "/isaac/UpdatePedestrians"),
            SpawnCeilings=self.node.create_client_wrapper(SpawnCeilings, "/isaac/SpawnCeilings"),
            SpawnFloors=self.node.create_client_wrapper(SpawnFloors, "/isaac/SpawnFloors"),
            SpawnLights=self.node.create_client_wrapper(SpawnLights, "/isaac/SpawnLights"),
            SpawnPedestrians=self.node.create_client_wrapper(SpawnPedestrians, "/isaac/SpawnPedestrians"),
            SpawnPrims=self.node.create_client_wrapper(SpawnPrims, "/isaac/SpawnPrims"),
            SpawnUrdf=self.node.create_client_wrapper(SpawnUrdf, "/isaac/SpawnUrdf"),
            SpawnWalls=self.node.create_client_wrapper(SpawnWalls, "/isaac/SpawnWalls"),
        )
        self._peds_publisher = self.node.create_publisher(arena_people_msgs.msg.Pedestrians, "/isaac/arena_peds", 10)

        self._robot_prims: dict[str, str] = {}
        self._map_offsets: dict[str, tuple[float, float, float]] = {}
        self._map_offset_calls: dict[str, rclpy.task.Future] = {}
        self._shared_lights: set[str] = set()
        self._box_centers: dict[str, tuple[float, float, float]] = {}

    def _robot_loader_args(self, robot: Robot) -> dict[str, object]:
        robot_config = arena_robots.Robot.RobotIdentifier(robot.model.name).resolve_sync()
        args: dict[str, object] = {
            **robot.extra,
            **attrs.asdict(robot, filter=lambda attribute, _: attribute.name != 'extra'),
            'optim': declare(self.node, NodeParams.OPTIM).value,
            # no world-scoped topic namespace in Isaac, empty prefix keeps the
            # injected sensor topics namespace-relative
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
                    (
                        ModelType.URDF,
                        # ModelType.USD
                    ),
                    loader_args=self._robot_loader_args(robot),
                )

                if model.type == ModelType.URDF:
                    assert model.path is not None, f"URDF model {model.name} must have a valid file path"
                    robot_params = (await arena_robots.Robot.RobotIdentifier(robot.model.name).resolve()).model_params

                    fq_name = self._NS_ROBOT(robot.name)

                    control_spec = robot_params.control
                    is_ros2_control = control_spec is not None and control_spec.is_ros2_control
                    relays_odom = control_spec is not None and is_ros2_control and control_spec.odom_topic != "odom"

                    spawn_res = await self._clients.SpawnUrdf.call_timeout(
                        SpawnUrdf.Request(
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
                        )
                    )
                    if spawn_res is None or not spawn_res.path:
                        self._logger.error(f"SpawnUrdf failed for {fq_name!r}: {'timeout' if spawn_res is None else 'spawn error, check isaac log'}")
                        return False

                    # Jazzy controller_manager reads URDF from the robot_description topic, not
                    # its parameter; feed RSP the bridge URDF so the same topic serves both
                    # RViz/Nav2 (which only care about link/joint geometry) and the bridge
                    # controller_manager (which needs TopicBasedSystem as the hw plugin).
                    rsp_urdf_path = str(model.path)
                    if is_ros2_control:
                        rsp_urdf_path = os.path.join('/tmp', f"arena_bridge_{robot.frame.sanitize()}.urdf")
                        ns = str(self.node.service_namespace(robot.name))
                        transform_urdf_for_bridge(
                            str(model.path),
                            {
                                'velocity': f"{ns}/isaac/joint_commands_velocity",
                                'position': f"{ns}/isaac/joint_commands_position",
                            },
                            f"{ns}/isaac/joint_states",
                            rsp_urdf_path,
                        )

                    await self._launch_robot_stack(robot, robot_params, rsp_urdf_path)

                    self._register_agent_robot(robot, robot_params)
                    self._robot_prims[robot.sim_path] = fq_name
                    self._map_offsets[robot.sim_path] = (0.0, 0.0, 0.0)
                    self._map_offset_calls.pop(robot.sim_path, None)
                    self._track_level(robot.sim_path, robot.pose)
                    if (call := self._map_offset_calls.get(robot.sim_path)) is not None and await AsyncUtil.timeout(self.node.await_ros(call), timeout_sec=_MAP_OFFSET_TIMEOUT_S) is None:
                        self._logger.warning(f"SetMapOffset for {fq_name!r} timed out, its map -> odom is stale")
                    return True

                # TODO
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
                    sim=SimSimulator.ISAAC,
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

        async def build_prim(obstacle: Obstacle) -> Prim | None:
            try:
                model = await (await obstacle.model.resolve()).model.get([ModelType.USD])
                if model.type is ModelType.UNKNOWN:
                    raise ValueError(f"obstacle model {obstacle.model.name} has no USD representation")
            except Exception:
                self._logger.warning(f"Failed to resolve model for obstacle {obstacle.name}")
                self._logger.debug(traceback.format_exc())
                return None
            assert model.path is not None, f"USD model {model.name} must have a valid file path"
            prim = Prim(is_static=True)
            prim.usd_path = str(model.path)
            prim.name = self._NS_PRIM(obstacle.name)
            prim.pose = self._placed(obstacle.pose).to_msg()
            if obstacle.scale is not None:
                prim.scale.x = obstacle.scale.x
                prim.scale.y = obstacle.scale.y
                prim.scale.z = obstacle.scale.z
            return prim

        level = obstacles_optim_level(self.node)
        boxes = await asyncio.gather(*(resolve_obstacle_box(o) for o in obstacles)) if level is ObstaclesOptim.BBOX else [None] * len(obstacles)
        box_indices = [i for i, box in enumerate(boxes) if box is not None]
        mesh_indices = [i for i, box in enumerate(boxes) if box is None]

        results: list[bool] = [False] * len(obstacles)

        async def spawn_one_box(i: int) -> bool:
            obstacle = obstacles[i]
            size, center = typing.cast(tuple, boxes[i])
            if obstacle.scale is not None:
                s = (obstacle.scale.x, obstacle.scale.y, obstacle.scale.z)
                size = tuple(d * f for d, f in zip(size, s, strict=True))
                center = tuple(c * f for c, f in zip(center, s, strict=True))
            name = self._NS_PRIM(obstacle.name)
            ok = bool(await self.spawn_box(name, size, offset_pose(obstacle.pose, center)))
            if ok:
                self._box_centers[name] = center
            return ok

        for i, ok in zip(box_indices, await asyncio.gather(*(spawn_one_box(i) for i in box_indices)), strict=True):
            results[i] = bool(ok)

        prims = await asyncio.gather(*(build_prim(obstacles[i]) for i in mesh_indices))
        req = SpawnPrims.Request()
        req.prims = list(filter(None, prims))
        if req.prims and (response := await self._clients.SpawnPrims.call_timeout(req)) is not None:
            response_iter = iter(response.ret)
            for i, prim in zip(mesh_indices, prims, strict=True):
                results[i] = (prim is not None) and bool(next(response_iter))

        return tuple(results)

    async def _articulated_spawn(self, objects: Sequence[ArticulatedObject]) -> Sequence[bool]:
        async def spawn(runtime: ArticulatedObject) -> bool:
            obstacle = runtime.obstacle
            links: list[ArticulatedLink] = []
            for link, joint in runtime.tree():
                view = runtime.links.get(link)
                model = await view.model.get([ModelType.USD]) if view is not None else None
                message = link_message(ArticulatedLink(), runtime, link, joint, JointDynamics)
                message.usd_path = str(model.path) if model is not None and model.type is ModelType.USD and model.path is not None else ""
                links.append(message)
            request = SpawnArticulated.Request(name=self._NS_PRIM(obstacle.name), pose=obstacle.pose.to_msg(), links=links)
            if obstacle.scale is not None:
                request.scale.x, request.scale.y, request.scale.z = obstacle.scale.x, obstacle.scale.y, obstacle.scale.z
            response = await self._clients.SpawnArticulated.call_timeout(request)
            return response is not None and bool(response.ret)

        return await asyncio.gather(*map(spawn, objects))

    async def _articulated_drive(self, commands: Sequence[ArticulatedCommand]) -> Sequence[Values | None]:
        request = DriveArticulated.Request(commands=[JointCommand(name=self._NS_PRIM(command.runtime.name), **command_fields(command)) for command in commands])
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
                if not await self._move_entity(self._NS_ROBOT(robot.name), robot.pose):
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
        return await asyncio.gather(*(self._delete_entity(name) for name in names))

    async def pedestrian_delete(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        res = await self._clients.DeletePedestrians.call_timeout(DeletePedestrians.Request(names=[p.sim_path for p in pedestrians]))
        if res is None:
            return tuple(False for _ in pedestrians)
        return tuple(r == DeletePedestrians.Response.SUCCESS for r in res.results)

    async def robot_delete(self, robots: Sequence[Robot]) -> Sequence[bool]:
        for robot in robots:
            self._forget_agent_robot(robot.sim_path)
            self._robot_prims.pop(robot.sim_path, None)
            self._map_offsets.pop(robot.sim_path, None)
            self._map_offset_calls.pop(robot.sim_path, None)
        return await asyncio.gather(*(self._delete_entity(self._NS_ROBOT(r.name)) for r in robots))

    async def remove_world(self) -> bool:
        res = await self._clients.ResetWorld.call_timeout(ResetWorld.Request())
        return bool(res) and res.ret

    async def spawn_walls(self, walls: Sequence[WallDefinition], clear_existing: bool = True) -> bool:
        if clear_existing:
            await self.remove_world()
        self._logger.debug("Attempting to spawn walls")
        tilted: list[Prim] = []

        async def create_segment(segment: WallSegment, shift: tuple[float, float, float]) -> Wall | None:
            dx, dy, dz = shift
            start = Position(segment.start.x + dx, segment.start.y + dy, segment.start.z + dz)
            end = Position(segment.end.x + dx, segment.end.y + dy, segment.end.z + dz + segment.height)
            try:
                wall_name = self._NS_WALL(self._realizer.realize(f"wall_{next(self.wall_counter)}"))
                if (sloped := sloped_box(segment)) is not None:
                    size, pose = sloped
                    placed = Pose(position=Position(pose.position.x + dx, pose.position.y + dy, pose.position.z + dz), orientation=pose.orientation)
                    wall = _box_wall(wall_name, size, placed)
                    tilted.append(Prim(name=wall_name, pose=placed.to_msg()))
                else:
                    wall = Wall(name=wall_name, start=start.to_msg(), end=end.to_msg(), thickness=segment.width, is_static=True)
                wall.material = material_to_msg(await segment.material.resolve())
                wall.visible = segment.visible
                wall.solid = segment.solid
                wall.shadows = segment.shadows
                return wall

            except Exception as e:
                self._logger.error(f"Failed to spawn wall: {e}\n{traceback.format_exc()}")
                return None

        async def create_obstacle(item: tuple[ObstacleDefinition, Model]) -> Prim | None:
            obstacle, model = item
            try:
                prim_name = self._realizer.realize(f"obstacle_{next(self.wall_counter)}")
                assert model.path is not None, f"USD model {model.name} must have a valid file path"
                prim = Prim(is_static=True)
                prim.usd_path = str(model.path)
                prim.name = self._NS_WALL(prim_name)
                prim.pose = self._placed(obstacle.pose).to_msg()
                return prim

            except Exception as e:
                self._logger.error(f"Failed to spawn wall obstacle: {e}\n{traceback.format_exc()}")
                return None

        async def create_wall(wall: WallDefinition) -> tuple[typing.Iterator[object], typing.Iterator[object]]:
            segments, obstacles = await realize_renderable(wall, (ModelType.USD,))
            shift = self._shift((wall.start.x + wall.end.x) / 2.0, (wall.start.y + wall.end.y) / 2.0)
            return (create_segment(segment, shift) for segment in segments), map(create_obstacle, obstacles)

        wall_futures = await asyncio.gather(*map(create_wall, walls))
        segment_futures, obstacle_futures = zip(*wall_futures, strict=False) if wall_futures else ((), ())

        walls_req = SpawnWalls.Request()
        prims_req = SpawnPrims.Request()
        walls_req.walls = list(filter(None, await asyncio.gather(*itertools.chain.from_iterable(segment_futures))))
        prims_req.prims = list(filter(None, await asyncio.gather(*itertools.chain.from_iterable(obstacle_futures))))

        walls_res = await self._clients.SpawnWalls.call_timeout(walls_req)
        prims_res = await self._clients.SpawnPrims.call_timeout(prims_req)
        res = bool(walls_res) and all(walls_res.ret) and bool(prims_res) and all(prims_res.ret)
        if tilted:
            edit = await self._clients.EditPrims.call_timeout(EditPrims.Request(prims=tilted, pose=True))
            res = res and bool(edit) and all(edit.ret)

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
                    pos=self._realizer.placement().place(floor.pos).to_msg(),
                    material=material_to_msg(await floor.material.resolve()),
                )

            except Exception:
                self._logger.error(f"Failed to spawn floor: {floor.name}\n{traceback.format_exc()}")
                return None

        floors_req = SpawnFloors.Request()
        floors_req.floors = list(filter(None, await asyncio.gather(*map(impl, floors))))
        floors_res = await self._clients.SpawnFloors.call_timeout(floors_req)

        res = bool(floors_res) and all(floors_res.ret)
        self._logger.debug("All floors spawned successfully.")
        return res

    async def spawn_ceilings(self, ceilings: Sequence[CeilingDefinition]) -> bool:
        self._logger.debug("Attempting to spawn ceilings")

        async def impl(ceiling: CeilingDefinition) -> Ceiling | None:
            try:
                pos = self._realizer.placement().place(Position(ceiling.pos.x, ceiling.pos.y, ceiling.z))
                return Ceiling(
                    name=self._NS_CEILING(ceiling.name),
                    x_length=ceiling.x_length,
                    y_length=ceiling.y_length,
                    pos=pos.to_msg(),
                    cast_shadows=ceiling.cast_shadows,
                    material=material_to_msg(await ceiling.material.resolve()),
                )

            except Exception:
                self._logger.error(f"Failed to spawn ceiling: {ceiling.name}\n{traceback.format_exc()}")
                return None

        ceilings_req = SpawnCeilings.Request()
        ceilings_req.ceilings = list(filter(None, await asyncio.gather(*map(impl, ceilings))))
        ceilings_res = await self._clients.SpawnCeilings.call_timeout(ceilings_req)

        res = bool(ceilings_res) and all(ceilings_res.ret)
        self._logger.debug("All ceilings spawned successfully.")
        return res

    def _light_name(self, light: LightDefinition) -> str:
        if light.spec.ambient:
            return Namespace('Ambient')(os.path.basename(light.name))
        return self._NS_LIGHT(light.name)

    def _light_msg(self, light: LightDefinition) -> Light:
        spec = light.spec
        placement = self._realizer.placement()
        if spec.shape == 'dome':
            assert light.lux is not None
            intensity = light.lux / math.pi
        elif spec.shape == 'distant':
            assert light.lux is not None
            intensity = light.lux
        else:
            assert light.lumens is not None
            intensity = light.lumens / (math.pi * spec.area)
            if spec.cone_deg < 180.0:
                intensity /= math.sin(math.radians(spec.cone_deg) / 2.0) ** 2
        return Light(
            name=self._light_name(light),
            shape=spec.shape,
            x_length=spec.x_length,
            y_length=spec.y_length,
            radius=spec.radius,
            cone_deg=spec.cone_deg,
            positions=[placement.place(position).to_msg() for position in (light.fixtures if light.rig else [light.position])],
            direction=geometry_msgs.msg.Vector3(x=light.direction[0], y=light.direction[1], z=light.direction[2]),
            intensity=_ISAAC_EXPOSURE_GAIN * intensity,
            cct_k=light.cct_K,
            cast_shadows=light.cast_shadows,
            level=0.0 if light.lit is False else light.dim,
            alive=light.alive(light.dead_fraction),
            intrinsic=light.intrinsic,
            glow_prim=self._NS_PRIM(light.owner) if light.glow else '',
            glow=light.glow,
            glow_color=cct_to_rgb(light.cct_K),
        )

    async def _spawn_lights(self, lights: Sequence[LightDefinition]) -> bool:
        res = await self._clients.SpawnLights.call_timeout(SpawnLights.Request(lights=[self._light_msg(light) for light in lights]))
        if res is None:
            return False
        self._shared_lights.update(light.name for light, shared in zip(lights, res.shared, strict=True) if shared)
        return all(res.ret)

    async def _remove_lights(self, lights: Sequence[LightDefinition]) -> bool:
        names = [self._light_name(light) for light in lights if light.name not in self._shared_lights]
        self._shared_lights.difference_update(light.name for light in lights)
        if not names:
            return True
        res = await self._clients.DeleteLights.call_timeout(DeleteLights.Request(names=names))
        return bool(res) and all(res.ret)

    async def _move_light(self, light: LightDefinition, position: Position, yaw: float) -> None:
        state = LightState(name=self._light_name(light), move=True, position=self._realizer.placement().place(position).to_msg(), yaw=yaw)
        res = await self._clients.SetLights.call_timeout(SetLights.Request(states=[state]))
        if res is None or not all(res.ret):
            raise RuntimeError(f"light {light.name!r} did not move")

    async def _apply_light(self, light: LightDefinition, level: float, alive: Sequence[bool]) -> None:
        state = LightState(name=self._light_name(light), level=level, alive=list(alive))
        res = await self._clients.SetLights.call_timeout(SetLights.Request(states=[state]))
        if res is None or not all(res.ret):
            raise RuntimeError(f"light {light.name!r} did not take its state")

    async def spawn_box(
        self,
        name: str,
        size: tuple[float, float, float],
        pose: Pose,
        material: arena_simulation_setup.tree.assets.Material.MaterialIdentifier | None = None,
        *,
        velocity: tuple[float, float, float] | None = None,
        comb: bool = False,
    ) -> bool:
        """Spawn a box using the SpawnWalls service.

        SpawnWalls creates a cube via create_cube(scale=(length, thickness, height)).
        Encoding: the AB vector carries sx (length) and yaw; thickness carries sy; the z
        extent of the start/end points carries sz. This lets a single Wall message
        fully describe a yawed box, a tilted one takes its full orientation from EditPrims.
        """
        del comb
        pose = self._placed(pose)
        wall = _box_wall(name, size, pose)
        wall.velocity = _vector3(velocity or (0.0, 0.0, 0.0))
        wall.is_static = velocity is None
        if material is not None:
            wall.material = material_to_msg(await material.resolve())
        req = SpawnWalls.Request()
        req.walls = [wall]
        res = await self._clients.SpawnWalls.call_timeout(req)
        if not (bool(res) and bool(res.ret) and res.ret[0]):
            return False
        roll, pitch, _ = pose.orientation.to_euler()
        if max(abs(roll), abs(pitch)) <= _TILT_EPS:
            return True
        edit = await self._clients.EditPrims.call_timeout(EditPrims.Request(prims=[Prim(name=name, pose=pose.to_msg())], pose=True))
        return bool(edit) and all(edit.ret)

    async def move_box(self, name: str, pose: Pose) -> bool:
        # Fire-and-forget: animation is cosmetic, last-write-wins, don't gate tick rate on bridge latency.
        req = EditPrims.Request(prims=[Prim(name=name, pose=self._placed(pose).to_msg())], pose=True)
        self._clients.EditPrims.client.call_async(req)
        return True

    async def move_boxes(self, poses: Mapping[str, Pose]) -> bool:
        req = EditPrims.Request(prims=[Prim(name=name, pose=self._placed(pose).to_msg()) for name, pose in poses.items()], pose=True)
        self._clients.EditPrims.client.call_async(req)
        return True

    async def set_box_velocities(self, velocities: Mapping[str, tuple[float, float, float]]) -> bool:
        req = EditPrims.Request(prims=[Prim(name=name, velocity=_vector3(velocity)) for name, velocity in velocities.items()], velocity=True)
        res = await self._clients.EditPrims.call_timeout(req)
        return bool(res) and all(res.ret)

    async def delete_box(self, name: str) -> bool:
        return await self._delete_entity(name)

    async def set_robot_pose(self, sim_path: str, pose: Pose) -> bool:
        prim_name = self._robot_prims.get(sim_path)
        if prim_name is None:
            return False
        if not await self._move_entity(prim_name, pose):
            return False
        self._track_level(sim_path, pose)
        return True

    async def step(self, n: int = 1) -> bool:
        """Unused wall-clock fake, superseded by SimLifecycle.step_seconds."""
        del n
        raise NotImplementedError('use sim_lifecycle/step')

    async def pedestrian_spawn(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:

        async def resolve_uri(pedestrian: DynamicObstacle) -> str:
            try:
                model = await (await pedestrian.model.resolve()).model.get(ModelType.SDF)
            except Exception as e:
                self._logger.warning(f"pedestrian {pedestrian.sim_path} model {pedestrian.model.name!r} unresolved: {e}")
                return ""
            if model.type is ModelType.UNKNOWN or model.path is None:
                return ""
            return str(model.path)

        uris = await asyncio.gather(*(resolve_uri(pedestrian) for pedestrian in pedestrians))
        items = [
            SpawnPedestrian(
                pedestrian=Pedestrian(
                    name=pedestrian.sim_path,
                    pose=self._placed(pedestrian.pose).to_msg(),
                    model_uri=uri,
                ),
                model_ref=pedestrian.model.name,
            )
            for pedestrian, uri in zip(pedestrians, uris, strict=True)
        ]

        req = SpawnPedestrians.Request(pedestrians=items)
        res = await self._clients.SpawnPedestrians.call_timeout(req)
        if res is None:
            return tuple(False for _ in pedestrians)

        success = tuple(r == SpawnPedestrians.Response.SUCCESS for r in res.results)

        await self.pedestrian_update(
            arena_people_msgs.msg.Pedestrians(
                pedestrians=[
                    arena_people_msgs.msg.Pedestrian(
                        name=ped.sim_path,
                        pose=ped.pose.to_msg(),
                    )
                    for status, ped in zip(success, pedestrians, strict=False)
                    if status
                ]
            )
        )

        return success

    async def pedestrian_update(self, pedestrians: arena_people_msgs.msg.Pedestrians) -> Sequence[bool]:
        # Published ped names are already sim_paths, matching the spawn key.
        # Latest-wins state stream rides the topic, mirroring gazebo's contract.
        if self._realizer.placement().stacked:
            pedestrians = copy.deepcopy(pedestrians)
            for ped in pedestrians.pedestrians:
                ped.pose = self._placed(Pose.from_msg(ped.pose)).to_msg()
        self._peds_publisher.publish(pedestrians)
        return tuple(True for _ in pedestrians.pedestrians)

    def _publish_odom(self, sim_path: str) -> None:
        """Send the robot's map -> odom, the inverse shift of its level, to its odom graph when it changed."""
        name = self._robot_prims.get(sim_path)
        if name is None:
            return
        dx, dy, dz = self._robot_shift(sim_path)
        translation = (-dx, -dy, -dz)
        if self._map_offsets.get(sim_path) == translation:
            return
        self._map_offsets[sim_path] = translation
        call = self._clients.SetMapOffset.client.call_async(SetMapOffset.Request(name=name, translation=_vector3(translation)))
        call.add_done_callback(lambda done: self._map_offset_done(sim_path, name, done))
        self._map_offset_calls[sim_path] = call

    def _map_offset_done(self, sim_path: str, name: str, call: rclpy.task.Future) -> None:
        if call.cancelled() or call.exception() is not None or (result := call.result()) is None or not result.success:
            self._map_offsets.pop(sim_path, None)
            self._logger.warning(f"SetMapOffset failed for {name!r}, its map -> odom is stale")

    async def _delete_entity(self, name: str) -> bool:
        self._logger.debug(f"Attempting to delete prim {name}")

        res = await self._clients.DeletePrims.call_timeout(DeletePrims.Request(names=[name]))
        if res is None:
            return False

        return res.ret[0]

    async def _move_entity(self, name: str, pose: Pose) -> bool:
        return (await self._move_entities([(name, pose)]))[0]

    async def _move_entities(self, actions: Sequence[tuple[str, Pose]]) -> Sequence[bool]:
        req = EditPrims.Request(
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
        self._logger.info("Setting up IsaacSimulator service clients...")
        futures: list[typing.Awaitable] = []
        # Define services with their corresponding client attributes
        for client in self._clients.__dict__.values():
            client = typing.cast(ClientWrapper, client)
            self._logger.debug(f"Initializing service client: {client.client.srv_name}")
            futures.append(client.ensure())
        await asyncio.gather(*futures)

        self._logger.info("All service clients are available.")

        self.node.create_publisher(std_msgs.msg.String, '/isaac/add_pedestrians_topic', 10).publish(std_msgs.msg.String(data=self._namespace('arena_peds')))

        self._logger.info("All service clients initialized and available.")

    @classmethod
    async def create(cls, **kwargs: typing.Unpack[SimKwargs]) -> "IsaacSimulator":
        self = cls(**kwargs)
        self._logger.info("Creating IsaacSimulator instance...")
        await self.setup()
        return self
