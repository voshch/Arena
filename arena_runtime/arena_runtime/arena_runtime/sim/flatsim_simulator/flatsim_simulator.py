"""
FlatsimHost is constructed once by arena_runtime_node and owns process-singleton resources for flatsim: the lifecycle (pause/unpause/step/cleanup) over the control clients of the server, which owns /clock.
FlatsimSimulator is per-env on arena_env_node and mirrors env-namespace state (robots, outlines, discs and boxes under the env prefix) into the one world of that server.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import traceback
import typing
from collections.abc import Iterable, Mapping, Sequence

import arena_robots.catalog
import arena_robots.Robot
import arena_robots.Sensor
import attrs
import launch
import launch_ros.actions
import rclpy.qos
from arena_people_msgs.msg import Pedestrians
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.Async import ClientWrapper, LaunchHandle
from arena_rclpy_mixins.qos import reliable
from arena_rclpy_mixins.Time import Time
from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
from arena_simulation_setup.tree.Wall import WallSegment
from flatsim_msgs.msg import Box, Disc, Discs, EntityPose, EntityPoses, Fan, Outline, Ring, SensorSpec
from flatsim_msgs.msg import Floor as FloorMsg
from flatsim_msgs.srv import AddFloors, AddOutlines, ControlWorld, DeleteEntities, DeletePrefix, DeleteRobot, Republish, SetRobotPose, SpawnBoxes, SpawnRobot

from arena_runtime.humans import PED_RADIUS
from arena_runtime.shared import DynamicObstacle, Floor, ModelType, Obstacle, Pose, Robot, Wall
from arena_runtime.sim import BaseSim, SimKwargs, SimLifecycle, SimUnavailable
from arena_runtime.sim._interface import robot_footprint_radius

from .planar import drive_model

if typing.TYPE_CHECKING:
    import shapely

SERVER = '/flatsim'
SERVICE_QOS = rclpy.qos.QoSProfile(depth=100)

_EPS = 1e-6
_PED_HEIGHT = 1.8
_OBSTACLE_HEIGHT = 1.0
_TOPIC_QUEUE = 100
_SERVER_TIMEOUT = 30.0
_SHUTDOWN_TIMEOUT = 2.0
_SENSOR_TYPES = (SensorSpec.LASERSCAN, SensorSpec.POINTCLOUD, SensorSpec.IMU, SensorSpec.CONTACT)

Extent = tuple[float, float]
Point = tuple[float, float]


def env_prefix(env_id: int) -> str:
    """Name prefix of everything env_id puts into the world."""
    return f'env_{env_id}/'


class FlatsimHost(SimLifecycle):
    """Pauses and steps the flatsim server, which owns /clock, and removes what a departed env left in its world."""

    CLOCK_TIMEOUT = 5.0

    def __init__(self, physics_dt: float, node: ArenaMixinNode) -> None:
        self._dt = physics_dt
        self._node = node
        self._control = node.create_client_wrapper(ControlWorld, f'{SERVER}/control_world', qos_profile=SERVICE_QOS)
        self._delete_prefix = node.create_client_wrapper(DeletePrefix, f'{SERVER}/delete_prefix', qos_profile=SERVICE_QOS)

    async def ensure_ready(self) -> None:
        await asyncio.gather(self._control.ensure(), self._delete_prefix.ensure())

    async def _send(self, pause: bool, steps: int = 0) -> Time:
        response = await self._control.call_timeout(ControlWorld.Request(pause=pause, multi_step=steps))
        if response is None:
            raise SimUnavailable(f'flatsim: no answer from {SERVER}/control_world')
        if not response.success:
            raise RuntimeError(f'flatsim: control_world(pause={pause}, multi_step={steps}) refused')
        return Time.from_msg(response.time)

    async def pause(self) -> bool:
        await self._send(True)
        return True

    async def unpause(self) -> bool:
        await self._send(False)
        return True

    async def step_seconds(self, seconds: float) -> float:
        """Take whole physics steps on the paused server, returns once /clock carries the last one."""
        n = round(seconds / self._dt)
        if seconds > 0:
            n = max(1, n)
        if n <= 0:
            return 0.0
        chunk = max(1, round(1.0 / self._dt))
        reached = self._node.sim_time
        for start in range(0, n, chunk):
            reached = await self._send(True, min(chunk, n - start))
        if not await self._node.await_sim_time(reached - Time.from_float(_EPS), freeze_timeout=self.CLOCK_TIMEOUT):
            raise SimUnavailable(f'flatsim: /clock silent for {self.CLOCK_TIMEOUT}s after a step')
        return n * self._dt

    def env_prefix(self, env_id: int) -> str:
        return env_prefix(env_id)

    async def cleanup_namespace(self, prefix: str) -> int:
        response = await self._delete_prefix.call_timeout(DeletePrefix.Request(prefix=prefix))
        if response is None:
            raise SimUnavailable(f'flatsim: no answer from {SERVER}/delete_prefix')
        return response.removed


def _ped_extent(z: float) -> Extent:
    return z, z + _PED_HEIGHT


def _box_extent(pose: Pose, height: float) -> Extent:
    return pose.position.z - height / 2.0, pose.position.z + height / 2.0


def _ring(ring: shapely.LinearRing) -> list[Point]:
    return [(float(c[0]), float(c[1])) for c in ring.coords[:-1]]


def _segment_ring(segment: WallSegment) -> list[Point]:
    """Rectangle of the segment's width around start to end, empty for a zero-length segment."""
    sx, sy, ex, ey = segment.start.x, segment.start.y, segment.end.x, segment.end.y
    length = math.hypot(ex - sx, ey - sy)
    if length == 0.0:
        return []
    nx = -(ey - sy) / length * segment.width / 2.0
    ny = (ex - sx) / length * segment.width / 2.0
    return [(sx + nx, sy + ny), (ex + nx, ey + ny), (ex - nx, ey - ny), (sx - nx, sy - ny)]


def _outline(name: str, rings: Iterable[Sequence[Point]], extent: Extent) -> Outline:
    return Outline(name=name, rings=[Ring(x=[point[0] for point in ring], y=[point[1] for point in ring]) for ring in rings], z_min=extent[0], z_max=extent[1])


def _disc(name: str, pose: Pose, velocity: Point = (0.0, 0.0)) -> Disc:
    z_min, z_max = _ped_extent(pose.position.z)
    return Disc(name=name, radius=PED_RADIUS, x=pose.position.x, y=pose.position.y, vx=velocity[0], vy=velocity[1], z_min=z_min, z_max=z_max)


def _answer[T](response: T | None) -> T:
    if response is None:
        raise SimUnavailable('flatsim: the server did not answer')
    return response


class FlatsimSimulator(BaseSim):
    """Mirrors one env into the world of the flatsim server under the env's name prefix, the server steps it and publishes the robots' topics."""

    SIM_NAME = 'flatsim'

    def __init__(self, **kwargs: typing.Unpack[SimKwargs]) -> None:
        super().__init__(**kwargs)
        self._prefix = env_prefix(self._env_id)
        self._clients: list[ClientWrapper] = []
        self._add_floors = self._client(AddFloors, 'add_floors')
        self._add_outlines = self._client(AddOutlines, 'add_outlines')
        self._spawn_boxes = self._client(SpawnBoxes, 'spawn_boxes')
        self._delete_entities = self._client(DeleteEntities, 'delete_entities')
        self._delete_prefix = self._client(DeletePrefix, 'delete_prefix')
        self._spawn_robot_client = self._client(SpawnRobot, 'spawn_robot')
        self._set_robot_pose = self._client(SetRobotPose, 'set_robot_pose')
        self._delete_robot = self._client(DeleteRobot, 'delete_robot')
        self._republish = self._client(Republish, 'republish')
        self._discs = self.node.create_publisher(Discs, f'{SERVER}/discs', reliable(_TOPIC_QUEUE))
        self._poses = self.node.create_publisher(EntityPoses, f'{SERVER}/poses', reliable(_TOPIC_QUEUE))

        self._robots: dict[str, LaunchHandle] = {}
        self._pedestrians: set[str] = set()
        self._wall_counter = itertools.count()
        self._wall_names: list[str] = []
        self._floor_names: list[str] = []
        self._box_sizes: dict[str, tuple[float, float, float]] = {}

    def _name(self, name: str) -> str:
        """The world-wide name of an entity of this env."""
        return name if name.startswith(self._prefix) else self._prefix + name

    def _client(self, srv_type: type, name: str) -> ClientWrapper:
        client = self.node.create_client_wrapper(srv_type, f'{SERVER}/{name}', timeout=_SERVER_TIMEOUT, qos_profile=SERVICE_QOS)
        self._clients.append(client)
        return client

    @classmethod
    async def create(cls, **kwargs: typing.Unpack[SimKwargs]) -> FlatsimSimulator:
        self = cls(**kwargs)
        self._logger.info(f'Creating FlatsimSimulator instance (prefix {self._prefix})...')
        if not all(await asyncio.gather(*(client.ensure(_SERVER_TIMEOUT) for client in self._clients))):
            raise SimUnavailable(f'flatsim: no server at {SERVER} after {_SERVER_TIMEOUT}s')
        await self.node.poll(lambda: self._discs.get_subscription_count() > 0 and self._poses.get_subscription_count() > 0, f'the subscribers of {SERVER}/discs and {SERVER}/poses')
        return self

    async def step(self, n: int = 1) -> bool:
        """Republish every robot's state and every sensor at the current time, the server owns time."""
        del n
        return _answer(await self._republish.call_timeout(Republish.Request(prefix=self._prefix))).success

    async def after_reset_episode(self) -> bool:
        """Publish the state and every sensor of this env's robots after a reset."""
        await self.step()
        return True

    def _robot_loader_args(self, robot: Robot, robot_config: arena_robots.Robot.RobotView) -> dict[str, object]:
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

    def _sensors(self, robot: Robot, robot_config: arena_robots.Robot.RobotView) -> list[SensorSpec]:
        """Sensors the server simulates, a laserscan carrying the caps laser block for a URDF without its sensor."""
        namespace = str(self.node.service_namespace(robot.name))
        laser = robot_config.model_params.caps.mobile.laser
        fan = [Fan(angle_min=laser.angle.min, angle_increment=laser.angle.increment, beams=laser.num_beams, range_max=laser.range, update_rate=float(laser.update_rate or 0.0))] if laser is not None else []
        return [
            SensorSpec(
                name=sensor.name,
                type=str(sensor.type),
                topic=arena_robots.Sensor.resolve_topic(sensor, namespace),
                frame=sensor.frame,
                sensor=sensor.sensor or '',
                fan=fan if str(sensor.type) == SensorSpec.LASERSCAN else [],
            )
            for sensor in robot_config.effective_sensors(robot.resolved_request, frames=robot.frames)
            if str(sensor.type) in _SENSOR_TYPES
        ]

    async def _launch_robot_stack(self, robot: Robot, description: str) -> LaunchHandle:
        """Launch robot_state_publisher in the robot namespace."""
        ld = launch.LaunchDescription()
        ld.add_action(launch_ros.actions.PushRosNamespace(namespace=str(self.node.service_namespace(robot.name))))
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
        return await self.node.do_launch_tracked(ld)

    async def _release(self, sim_path: str) -> bool:
        """Remove one robot from the world and stop its launched nodes."""
        handle = self._robots.pop(sim_path, None)
        if handle is None:
            return False
        self._forget_agent_robot(sim_path)
        removed, _ = await asyncio.gather(self._delete_robot.call_timeout(DeleteRobot.Request(name=self._name(sim_path))), handle.shutdown(), return_exceptions=True)
        return removed is not None and not isinstance(removed, BaseException)

    async def _spawn_robot(self, robot: Robot) -> bool:
        handle: LaunchHandle | None = None
        try:
            await self._release(robot.sim_path)

            robot_config = arena_robots.Robot.RobotIdentifier(robot.model.name).resolve_sync()
            model = await (await robot.model.resolve()).model.get(
                (ModelType.URDF,),
                loader_args=self._robot_loader_args(robot, robot_config),
            )
            if model.type != ModelType.URDF:
                raise NotImplementedError(f"robot model of type {model.type} can't be spawned by {self.__class__.__name__}")
            assert model.path is not None, f'URDF model {model.name} must have a valid file path'
            with open(model.path) as f:
                description = f.read()

            robot_params = robot_config.model_params
            mobile = robot_params.caps.mobile
            footprint = mobile.footprint or []
            position = robot.pose.position
            request = SpawnRobot.Request(
                name=self._name(robot.sim_path),
                robot_namespace=str(self.node.service_namespace(robot.name)),
                tf_topic=self.node.resolve_topic_name('/tf'),
                world_frame='map',
                frame_prefix=robot.frame.tf(),
                odom_frame=robot_params.odom_frame,
                base_frame=robot_params.base_frame,
                x=position.x,
                y=position.y,
                z=position.z,
                yaw=robot.pose.orientation.to_yaw(),
                footprint=Ring(x=[float(point[0]) for point in footprint], y=[float(point[1]) for point in footprint]),
                radius=robot_footprint_radius(robot_params),
                drive=drive_model(mobile.planar, holonomic=mobile.is_holonomic),
                urdf=description,
                sensors=self._sensors(robot, robot_config),
            )

            handle = await self._launch_robot_stack(robot, description)
            response = _answer(await self._spawn_robot_client.call_timeout(request))
            if not response.success:
                raise RuntimeError(f'flatsim refused robot {robot.sim_path!r}: {response.message}')
            self._robots[robot.sim_path] = handle
            self._register_agent_robot(robot, robot_params)
            return True

        except Exception as e:
            self._logger.error(f'{e!r}\n{traceback.format_exc()}')
            if handle is not None:
                await handle.shutdown()
            return False

    async def robot_spawn(self, robots: Sequence[Robot]) -> Sequence[bool]:
        return await asyncio.gather(*map(self._spawn_robot, robots))

    async def set_robot_pose(self, sim_path: str, pose: Pose) -> bool:
        if sim_path not in self._robots:
            return False
        request = SetRobotPose.Request(name=self._name(sim_path), x=pose.position.x, y=pose.position.y, z=pose.position.z, yaw=pose.orientation.to_yaw())
        return _answer(await self._set_robot_pose.call_timeout(request)).success

    async def robot_move(self, robots: Sequence[Robot]) -> Sequence[bool]:
        return await asyncio.gather(*(self.set_robot_pose(robot.sim_path, robot.pose) for robot in robots))

    async def robot_delete(self, robots: Sequence[Robot]) -> Sequence[bool]:
        return await asyncio.gather(*(self._release(robot.sim_path) for robot in robots))

    async def _obstacle_outline(self, key: str, obstacle: Obstacle) -> Outline | None:
        poly = await obstacle.footprint()
        if poly is None:
            self._logger.warning(f'obstacle {obstacle.name!r} has no annotated footprint, not spawned')
            return None
        z = obstacle.pose.position.z
        span = await obstacle.span() or (z, z + _OBSTACLE_HEIGHT)
        return _outline(key, [_ring(poly.exterior), *map(_ring, poly.interiors)], span)

    async def _send_outlines(self, outlines: Sequence[Outline]) -> bool:
        if not outlines:
            return True
        response = _answer(await self._add_outlines.call_timeout(AddOutlines.Request(outlines=list(outlines))))
        if not response.success:
            self._logger.error(f'flatsim refused {len(outlines)} outlines: {response.message}')
        return response.success

    async def _add_obstacles(self, obstacles: Sequence[tuple[str, Obstacle]]) -> tuple[bool, ...]:
        outlines = await asyncio.gather(*(self._obstacle_outline(key, obstacle) for key, obstacle in obstacles))
        sent = await self._send_outlines([outline for outline in outlines if outline is not None])
        return tuple(sent and outline is not None for outline in outlines)

    async def _remove(self, names: Sequence[str]) -> list[bool]:
        if not names:
            return []
        return list(_answer(await self._delete_entities.call_timeout(DeleteEntities.Request(names=list(names)))).removed)

    async def obstacle_spawn(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        return await self._add_obstacles([(self._name(obstacle.sim_path), obstacle) for obstacle in obstacles])

    async def obstacle_move(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        return await self._add_obstacles([(self._name(obstacle.sim_path), obstacle) for obstacle in obstacles])

    async def obstacle_delete(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        return await self._remove([self._name(obstacle.sim_path) for obstacle in obstacles])

    async def pedestrian_spawn(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        discs = [_disc(self._name(pedestrian.sim_path), pedestrian.pose) for pedestrian in pedestrians]
        self._pedestrians.update(disc.name for disc in discs)
        self._discs.publish(Discs(discs=discs))
        return tuple(True for _ in pedestrians)

    async def pedestrian_move(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        known = [self._name(pedestrian.sim_path) in self._pedestrians for pedestrian in pedestrians]
        self._discs.publish(Discs(discs=[_disc(self._name(pedestrian.sim_path), pedestrian.pose) for pedestrian, ok in zip(pedestrians, known, strict=True) if ok]))
        return tuple(known)

    async def pedestrian_delete(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        names = [self._name(pedestrian.sim_path) for pedestrian in pedestrians]
        known = tuple(name in self._pedestrians for name in names)
        self._pedestrians.difference_update(names)
        self._discs.publish(Discs(removed=names))
        return known

    async def pedestrian_update(self, pedestrians: Pedestrians) -> Sequence[bool]:
        discs = [_disc(self._name(pedestrian.name), Pose.from_msg(pedestrian.pose), (pedestrian.twist.linear.x, pedestrian.twist.linear.y)) for pedestrian in pedestrians.pedestrians]
        self._pedestrians.update(disc.name for disc in discs)
        self._discs.publish(Discs(discs=discs))
        return tuple(True for _ in pedestrians.pedestrians)

    async def spawn_walls(self, walls: Sequence[Wall], clear_existing: bool = True) -> bool:
        if clear_existing:
            await self.remove_world()

        outlines: list[Outline] = []
        embedded: list[tuple[str, Obstacle]] = []
        for wall_segments, wall_obstacles in await asyncio.gather(*(wall.assets() for wall in walls)):
            for segment in wall_segments:
                name = self._name(self._realizer.realize(f'wall_{next(self._wall_counter)}'))
                outlines.append(_outline(name, [_segment_ring(segment)], (min(segment.start.z, segment.end.z), max(segment.start.z, segment.end.z) + segment.height)))
            for obstacle in wall_obstacles:
                embedded.append((self._name(self._realizer.realize(f'obstacle_{next(self._wall_counter)}')), obstacle))

        outlines.extend(outline for outline in await asyncio.gather(*(self._obstacle_outline(key, obstacle) for key, obstacle in embedded)) if outline is not None)
        if await self._send_outlines(outlines):
            self._wall_names.extend(outline.name for outline in outlines)
        return True

    async def spawn_floors(self, floors: Sequence[Floor]) -> bool:
        named = [FloorMsg(name=self._name(self._realizer.realize(f'floor_{next(self._wall_counter)}')), z=floor.pos.z) for floor in floors]
        response = _answer(await self._add_floors.call_timeout(AddFloors.Request(floors=named)))
        if not response.success:
            self._logger.error(f'flatsim refused the floors: {response.message}')
            return False
        self._floor_names.extend(floor.name for floor in named)
        return True

    async def remove_world(self) -> bool:
        names, self._wall_names, self._floor_names = [*self._wall_names, *self._floor_names], [], []
        await self._remove(names)
        return True

    async def spawn_box(self, name: str, size: tuple[float, float, float], pose: Pose, material: MaterialIdentifier | None = None, *, velocity: tuple[float, float, float] | None = None, comb: bool = False) -> bool:
        del material, velocity, comb
        z_min, z_max = _box_extent(pose, size[2])
        box = Box(name=self._name(name), size_x=size[0], size_y=size[1], x=pose.position.x, y=pose.position.y, yaw=pose.orientation.to_yaw(), z_min=z_min, z_max=z_max)
        response = _answer(await self._spawn_boxes.call_timeout(SpawnBoxes.Request(boxes=[box])))
        if response.success:
            self._box_sizes[name] = size
        return response.success

    def _box_pose(self, name: str, pose: Pose) -> EntityPose | None:
        size = self._box_sizes.get(name)
        if size is None:
            return None
        z_min, z_max = _box_extent(pose, size[2])
        return EntityPose(name=self._name(name), x=pose.position.x, y=pose.position.y, yaw=pose.orientation.to_yaw(), z_min=z_min, z_max=z_max)

    async def move_box(self, name: str, pose: Pose) -> bool:
        placed = self._box_pose(name, pose)
        if placed is None:
            return False
        self._poses.publish(EntityPoses(poses=[placed]))
        return True

    async def move_boxes(self, poses: Mapping[str, Pose]) -> bool:
        placed = [entity for name, pose in poses.items() if (entity := self._box_pose(name, pose)) is not None]
        if placed:
            self._poses.publish(EntityPoses(poses=placed))
        return len(placed) == len(poses)

    async def set_box_velocities(self, velocities: Mapping[str, tuple[float, float, float]]) -> bool:
        """Box2D boxes follow move_boxes alone."""
        return all(name in self._box_sizes for name in velocities)

    async def delete_box(self, name: str) -> bool:
        self._box_sizes.pop(name, None)
        return all(await self._remove([self._name(name)]))

    async def _leave_world(self) -> None:
        if self._delete_prefix.client.service_is_ready():
            await self._delete_prefix.call_timeout(DeletePrefix.Request(prefix=self._prefix), _SHUTDOWN_TIMEOUT)

    async def shutdown(self) -> None:
        await self.stop_mechanisms()
        handles = list(self._robots.values())
        for sim_path in self._robots:
            self._forget_agent_robot(sim_path)
        self._robots.clear()
        await asyncio.gather(self._leave_world(), *(handle.shutdown() for handle in handles), return_exceptions=True)
        self.node.destroy_publisher(self._discs)
        self.node.destroy_publisher(self._poses)
        self._pedestrians.clear()
        self._wall_names.clear()
        self._floor_names.clear()
        self._box_sizes.clear()
