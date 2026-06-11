import asyncio
import itertools
import os
import traceback
import types
import typing
import xml.etree.ElementTree as ET
from collections.abc import Sequence

import arena_mujoco_msgs.msg
import arena_robots.catalog
import arena_robots.Robot
import arena_robots.Sensor
import arena_simulation_setup.tree.assets.Material
import launch
import launch_ros
import std_srvs.srv
from arena_mujoco_msgs.msg import (
    Ceiling,
    Floor,
    Material,
    Prim,
    Scale,
    Wall,
)
from arena_mujoco_msgs.srv import (
    DeletePrims,
    EditPrims,
    SpawnCeilings,
    SpawnFloors,
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
from arena_simulation_setup.shared import Obstacle as ObstacleDefinition
from arena_simulation_setup.tree.Wall import WallSegment
from task_generator.shared import Ceiling as CeilingDefinition
from task_generator.shared import (
    DynamicObstacle,
    Model,
    ModelType,
    Obstacle,
    Pose,
    Robot,
)
from task_generator.shared import Floor as FloorDefinition
from task_generator.shared import Wall as WallDefinition
from task_generator.utils.flags import ObstaclesOptim, obstacles_optim_level

from arena_runtime._node import NodeInterface
from arena_runtime.sim import BaseSim, SimLifecycle
from arena_runtime.sim._control import (
    effective_control_yaml,
    odom_relay_node,
    robot_controllers,
    twist_stamper_node,
)
from arena_runtime.sim._interface import offset_pose, resolve_obstacle_box
from arena_runtime.sim._urdf import transform_urdf_for_bridge
from arena_runtime.sim._walls import realize_renderable

"""
MujocoHost is constructed once by arena_node and owns process-singleton resources for MuJoCo: the lifecycle (pause/unpause/cleanup/step) and its service clients.
MujocoSimulator is per-env on task_generator_node and adapts env-namespace state (per-robot publishers, env-prefixed entity names) over those shared resources.
"""

_MUJOCO_PHYSICS_DT = 0.002


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


_MESH_MODEL_TYPES = (ModelType.OBJ, ModelType.SDF)

_SPAWN_ASSETS_TIMEOUT_S = 300.0
_RESEND_AFTER_S = 10.0


def _mesh_file(model: Model) -> str:
    """Mesh file behind an obstacle model: the OBJ itself or an SDF's visual mesh, empty without one."""
    if model.path is None:
        return ""
    if model.type is ModelType.OBJ:
        return str(model.path)
    if model.type is ModelType.SDF:
        uri = ET.fromstring(model.description).findtext(".//visual//mesh/uri")
        if uri:
            return str(model.path.parent / uri.removeprefix("file://"))
    return ""


def material_to_msg(material: arena_simulation_setup.tree.assets.Material.Material) -> arena_mujoco_msgs.msg.Material:
    return Material(
        name=material.name,
        path=material.path,
    )


class MujocoSimulator(BaseSim, NodeInterface):
    SIM_NAME = 'mujoco'

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize MujocoSimulator"""
        super().__init__(*args, **kwargs)

        env_prefix = f"env_{self._env_id}"
        self._NS_PRIM = Namespace(env_prefix)('Obstacles')
        self._NS_ROBOT = Namespace(env_prefix)('Robots')
        self._NS_WALL = Namespace(env_prefix)('Walls')
        self._NS_FLOOR = Namespace(env_prefix)('Floors')
        self._NS_CEILING = Namespace(env_prefix)('Ceilings')

        self.wall_counter = itertools.count()
        self.floor_counter = itertools.count()
        self._clients = types.SimpleNamespace(
            DeletePedestrians=self.node.create_client_wrapper(DeletePedestrians, "/mujoco/DeletePedestrians"),
            DeletePrims=self.node.create_client_wrapper(DeletePrims, "/mujoco/DeletePrims"),
            EditPrims=self.node.create_client_wrapper(EditPrims, "/mujoco/EditPrims"),
            MovePedestrians=self.node.create_client_wrapper(MovePedestrians, "/mujoco/MovePedestrians"),
            SpawnCeilings=self.node.create_client_wrapper(SpawnCeilings, "/mujoco/SpawnCeilings"),
            SpawnFloors=self.node.create_client_wrapper(SpawnFloors, "/mujoco/SpawnFloors"),
            SpawnPedestrians=self.node.create_client_wrapper(SpawnPedestrians, "/mujoco/SpawnPedestrians", timeout=_SPAWN_ASSETS_TIMEOUT_S),
            SpawnPrims=self.node.create_client_wrapper(SpawnPrims, "/mujoco/SpawnPrims", timeout=_SPAWN_ASSETS_TIMEOUT_S),
            SpawnUrdf=self.node.create_client_wrapper(SpawnUrdf, "/mujoco/SpawnUrdf"),
            SpawnWalls=self.node.create_client_wrapper(SpawnWalls, "/mujoco/SpawnWalls"),
        )
        self._peds_publisher = self.node.create_publisher(Pedestrians, "/mujoco/arena_peds", 10)

        self._robot_prims: dict[str, str] = {}

    def _robot_loader_args(self, robot: Robot) -> dict[str, object]:
        robot_config = arena_robots.Robot.RobotIdentifier(robot.model.name).resolve_sync()
        args: dict[str, object] = {
            **robot.asdict(),
            'optim': self.node.rosparam[str].get('optim', ''),
            'sensor_topic_patches': arena_robots.Sensor.topic_elements(robot_config.effective_sensors(robot.resolved_request, frames=robot.frames), ''),
        }
        args.pop('resolved_assembly', None)
        if robot.resolved_assembly is not None:
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
                            pose=robot.pose.to_msg(),
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

                    rsp_urdf_path = bridge_urdf_path if is_ros2_control else str(model.path)

                    await self._launch_robot_stack(robot, robot_params, rsp_urdf_path)

                    self._register_agent_robot(robot, robot_params)
                    self._robot_prims[robot.sim_path] = spawn_res.path
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
                    prefix=control_prefix,
                )
                if control_spec.config is not None
                else None
            )

            cm_params: list[object] = [{'use_sim_time': True}]
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

        async def plan(obstacle: Obstacle) -> tuple[str, tuple | None]:
            """(mesh file, bbox) to spawn from, both empty without a usable asset."""
            box = await resolve_obstacle_box(obstacle)
            if level is ObstaclesOptim.BBOX and box is not None:
                return "", box
            try:
                mesh_path = _mesh_file(await (await obstacle.model.resolve()).model.get(_MESH_MODEL_TYPES))
            except Exception:
                self._logger.debug(traceback.format_exc())
                mesh_path = ""
            return (mesh_path, None) if mesh_path else ("", box)

        plans = await asyncio.gather(*map(plan, obstacles))

        req = SpawnPrims.Request(env_id=self._env_id)
        indices: list[int] = []
        skipped: list[str] = []
        for i, (obstacle, (mesh_path, box)) in enumerate(zip(obstacles, plans, strict=True)):
            scale = (1.0, 1.0, 1.0) if obstacle.scale is None else (obstacle.scale.x, obstacle.scale.y, obstacle.scale.z)
            pose = obstacle.pose
            if not mesh_path:
                if box is None:
                    skipped.append(obstacle.name)
                    continue
                size, center = box
                pose = offset_pose(pose, tuple(c * f for c, f in zip(center, scale, strict=True)))
                scale = tuple(d * f for d, f in zip(size, scale, strict=True))
            req.prims.append(
                Prim(
                    name=self._NS_PRIM(obstacle.name),
                    pose=pose.to_msg(),
                    scale=Scale(x=scale[0], y=scale[1], z=scale[2]),
                    mesh_path=mesh_path,
                )
            )
            indices.append(i)
        if skipped:
            self._logger.warning(f"no mesh or bounding box for {len(skipped)} obstacle(s), not spawned: {skipped[:5]}")

        results: list[bool] = [False] * len(obstacles)
        if req.prims and (response := await self._clients.SpawnPrims.call_timeout(req)) is not None:
            for i, ok in zip(indices, response.ret, strict=True):
                results[i] = bool(ok)
        return tuple(results)

    async def obstacle_move(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        return await self._move_entities([(self._NS_PRIM(o.name), o.pose) for o in obstacles])

    async def pedestrian_move(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        req = MovePedestrians.Request(
            pedestrians=[
                Pedestrian(
                    name=p.sim_path,
                    pose=p.pose.to_msg(),
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
                return await self._move_entity(self._robot_prims.get(robot.sim_path, self._NS_ROBOT(robot.name)), robot.pose)
            except Exception as e:
                self._logger.error(f"Failed to move robot {robot.name}: {e}\n{traceback.format_exc()}")
                return False

        return await asyncio.gather(*map(move_robot, robots))

    async def obstacle_delete(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        return await self._delete_entities([self._NS_PRIM(o.name) for o in obstacles])

    async def pedestrian_delete(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        res = await self._clients.DeletePedestrians.call_timeout(DeletePedestrians.Request(names=[p.sim_path for p in pedestrians]))
        if res is None:
            return tuple(False for _ in pedestrians)
        return tuple(r == DeletePedestrians.Response.SUCCESS for r in res.results)

    async def robot_delete(self, robots: Sequence[Robot]) -> Sequence[bool]:
        names: list[str] = []
        for robot in robots:
            self._forget_agent_robot(robot.sim_path)
            names.append(self._robot_prims.pop(robot.sim_path, self._NS_ROBOT(robot.name)))
        return await asyncio.gather(*(self._delete_entity(name) for name in names))

    async def remove_world(self) -> bool:
        res = await self._clients.DeletePrims.call_timeout(DeletePrims.Request(env_id=self._env_id, names=[f"{ns}/" for ns in (self._NS_WALL, self._NS_FLOOR, self._NS_CEILING)]))
        return res is not None

    async def spawn_walls(self, walls: Sequence[WallDefinition], clear_existing: bool = True) -> bool:
        if clear_existing:
            await self.remove_world()
        self._logger.debug("Attempting to spawn walls")

        async def create_segment(segment: WallSegment) -> Wall | None:
            end = segment.end.to_msg()
            end.z += segment.height
            try:
                wall_name = self._realizer.realize(f"wall_{next(self.wall_counter)}")
                return Wall(
                    name=self._NS_WALL(wall_name),
                    start=segment.start.to_msg(),
                    end=end,
                    material=material_to_msg(await segment.material.resolve()),
                    thickness=segment.width,
                )

            except Exception as e:
                self._logger.error(f"Failed to spawn wall: {e}\n{traceback.format_exc()}")
                return None

        async def create_obstacle(item: tuple[ObstacleDefinition, Model]) -> Prim | None:
            obstacle, model = item
            try:
                prim_name = self._realizer.realize(f"obstacle_{next(self.wall_counter)}")
                prim = Prim()
                prim.mesh_path = _mesh_file(model)
                prim.name = self._NS_WALL(prim_name)
                prim.pose = obstacle.pose.to_msg()
                prim.scale = Scale(x=1.0, y=1.0, z=1.0)
                return prim

            except Exception as e:
                self._logger.error(f"Failed to spawn wall obstacle: {e}\n{traceback.format_exc()}")
                return None

        async def create_wall(wall: WallDefinition) -> tuple[typing.Iterator[object], typing.Iterator[object]]:
            segments, obstacles = await realize_renderable(wall, _MESH_MODEL_TYPES)
            return map(create_segment, segments), map(create_obstacle, obstacles)

        wall_futures = await asyncio.gather(*map(create_wall, walls))
        segment_futures, obstacle_futures = zip(*wall_futures, strict=False) if wall_futures else ((), ())

        walls_req = SpawnWalls.Request(env_id=self._env_id)
        prims_req = SpawnPrims.Request(env_id=self._env_id)
        walls_req.walls = list(filter(None, await asyncio.gather(*itertools.chain.from_iterable(segment_futures))))
        prims_req.prims = list(filter(None, await asyncio.gather(*itertools.chain.from_iterable(obstacle_futures))))

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
                    pos=floor.pos.to_msg(),
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
                    pos=pos,
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

    async def spawn_box(self, name: str, size: tuple[float, float, float], pose: Pose) -> bool:
        """Spawn an axis-aligned box as a single primitive Prim (empty mesh, scale = size)."""
        sx, sy, sz = size
        req = SpawnPrims.Request(
            env_id=self._env_id,
            prims=[
                Prim(
                    name=name,
                    pose=pose.to_msg(),
                    scale=Scale(x=sx, y=sy, z=sz),
                    mesh_path="",
                )
            ],
        )
        res = await self._clients.SpawnPrims.call_timeout(req)
        return bool(res) and bool(res.ret) and res.ret[0]

    async def move_box(self, name: str, pose: Pose) -> bool:
        req = EditPrims.Request(env_id=self._env_id, prims=[Prim(name=name, pose=pose.to_msg())], pose=True)
        self._clients.EditPrims.client.call_async(req)
        return True

    async def delete_box(self, name: str) -> bool:
        return await self._delete_entity(name)

    async def set_robot_pose(self, sim_path: str, pose: Pose) -> bool:
        prim_name = self._robot_prims.get(sim_path)
        if prim_name is None:
            return False
        return await self._move_entity(prim_name, pose)

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
                        pose=pedestrian.pose.to_msg(),
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
        self._peds_publisher.publish(pedestrians)
        return tuple(True for _ in pedestrians.pedestrians)

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
                    pose=pose.to_msg(),
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
    async def create(cls, *args: object, namespace: Namespace, **kwargs: object) -> "MujocoSimulator":
        self = cls(*args, namespace=namespace, **kwargs)
        self._logger.info("Creating MujocoSimulator instance...")
        await self.setup()
        return self
