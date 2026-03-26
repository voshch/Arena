import asyncio
import itertools
import os
import random
import traceback
import types
import typing
from collections.abc import Sequence

import arena_people_msgs.msg
import arena_robots.Robot
import arena_simulation_setup.tree.assets.Material
import isaacsim_msgs.msg
import numpy as np
import std_msgs.msg
import std_srvs.srv
from arena_rclpy_mixins.Async import ClientWrapper
from arena_simulation_setup.tree.Wall import WallSegment
from arena_simulation_setup.shared import Obstacle as ObstacleDefinition
from isaacsim_msgs.msg import (
    Door,
    Elevator,
    Floor,
    Material,
    Pedestrian,
    PedestrianGoal,
    Prim,
    Scale,
    Wall,
)
from isaacsim_msgs.srv import (
    DeletePrims,
    EditPrims,
    NavigatePedestrians,
    SpawnDoors,
    SpawnElevators,
    SpawnFloors,
    SpawnPedestrians,
    SpawnPrims,
    SpawnUrdf,
    SpawnUsd,
    SpawnWalls,
)
from std_msgs.msg import String as StdString

from task_generator.shared import Door as DoorDefinition
from task_generator.shared import (
    DynamicObstacle,
    ModelType,
    Namespace,
    Obstacle,
    Pose,
    Robot,
)
from task_generator.shared import Elevator as ElevatorDefinition
from task_generator.shared import Floor as FloorDefinition
from task_generator.shared import Wall as WallDefinition
from task_generator.simulators.sim import BaseSim, NodeInterface


def material_to_msg(material: arena_simulation_setup.tree.assets.Material.Material) -> isaacsim_msgs.msg.Material:
    return Material(
        name=material.name,
        path=material.path,
    )


class IsaacSimulator(BaseSim, NodeInterface):

    _NS_PRIM = Namespace('Obstacles')
    _NS_PEDESTRIAN = Namespace('Pedestrians')
    _NS_ROBOT = Namespace('Robots')
    _NS_WALL = Namespace('Walls')
    _NS_FLOOR = Namespace('Floors')
    _NS_DOOR = Namespace('Doors')

    def __init__(self, *args, **kwargs):
        """Initialize IsaacSimulator
        """
        super().__init__(*args, **kwargs)

        self.wall_counter = itertools.count()
        self.floor_counter = itertools.count()
        self._spawned_doors = []
        self._clients = types.SimpleNamespace(
            DeletePedestrians=self.node.create_client_wrapper(DeletePrims, "/isaac/DeletePedestrians"),
            DeletePrims=self.node.create_client_wrapper(DeletePrims, "/isaac/DeletePrims"),
            EditPrims=self.node.create_client_wrapper(EditPrims, "/isaac/EditPrims"),
            NavigatePedestrians=self.node.create_client_wrapper(NavigatePedestrians, "/isaac/NavigatePedestrians"),
            SpawnDoors=self.node.create_client_wrapper(SpawnDoors, "/isaac/SpawnDoors"),
            SpawnFloors=self.node.create_client_wrapper(SpawnFloors, "/isaac/SpawnFloors"),
            SpawnPedestrians=self.node.create_client_wrapper(SpawnPedestrians, "/isaac/SpawnPedestrians"),
            SpawnPrims=self.node.create_client_wrapper(SpawnPrims, "/isaac/SpawnPrims"),
            SpawnUrdf=self.node.create_client_wrapper(SpawnUrdf, "/isaac/SpawnUrdf"),
            SpawnUsd=self.node.create_client_wrapper(SpawnUsd, "/isaac/SpawnUsd"),
            SpawnWalls=self.node.create_client_wrapper(SpawnWalls, "/isaac/SpawnWalls"),
            SpawnElevators=self.node.create_client_wrapper(SpawnElevators, "/isaac/SpawnElevators"),
            PauseSimulation=self.node.create_client_wrapper(std_srvs.srv.Trigger, "/isaac/PauseSimulation"),
            UnpauseSimulation=self.node.create_client_wrapper(std_srvs.srv.Trigger, "/isaac/UnpauseSimulation"),
        )

        # Publisher for external registration messages so IsaacSim's DoorManager
        # can be informed about spawned entities in the IsaacSim process.
        self._reg_pub = self.node.create_publisher(StdString, '/isaac/register_entity', 10)

    async def robot_spawn(self, robots):
        async def impl(robot: Robot) -> bool:
            try:
                model = await (await robot.model.resolve()).model.get(
                    (
                        ModelType.URDF,
                        # ModelType.USD
                    ),
                    loader_args=robot.asdict()
                )

                if model.type == ModelType.URDF:
                    assert model.path is not None, f"URDF model {model.name} must have a valid file path"
                    robot_params = (await arena_robots.Robot.RobotIdentifier(robot.model.name).resolve()).model_params

                    fq_name = self._NS_ROBOT(robot.sim_path)
                    
                    await self._clients.SpawnUrdf.call_timeout(
                        SpawnUrdf.Request(
                            name=fq_name,
                            urdf_path=str(model.path),
                            robot_model=robot.model.name,
                            localization=True,
                            tf_prefix=robot.name,
                            base_frame=robot_params.base_frame,
                            odom_frame=robot_params.odom_frame,
                            pose=robot.pose.to_msg(),
                            cmd_vel_topic=self.node.service_namespace(robot.name, 'cmd_vel'),
                            joint_states_topic=self.node.service_namespace(robot.name, 'joint_states'),
                            odom_topic=self.node.service_namespace(robot.name, 'odom'),
                        )
                    )

                    base_frame = robot_params.base_frame
                    robot_prim_path = os.path.join("/World", fq_name, base_frame)

                    # Publish registration message so DoorManager in IsaacSim process
                    # registers the robot. This avoids cross-process direct calls.
                    try:
                        if self._reg_pub:
                            self._reg_pub.publish(StdString(data=f"robot|{robot_prim_path}"))
                            self._logger.debug(f"Published registration for robot: {robot_prim_path}")
                        else:
                            self._logger.warning('Registration publisher not available; robot not registered with IsaacSim DoorManager')
                    except Exception as e:
                        self._logger.warning(f'Failed to publish robot registration: {e}\n{traceback.format_exc()}')

                    return True

                # TODO
                raise NotImplementedError(
                    f"robot model of type {model.type} can't be spawned by {self.__class__.__name__}"
                )

            except Exception as e:
                self._logger.error(f"{repr(e)}\n{traceback.format_exc()}")
                return False

        return await asyncio.gather(*map(impl, robots))

    async def obstacle_spawn(self, obstacles):

        async def impl(obstacle: Obstacle) -> Prim | None:
            try:
                model = await (await obstacle.model.resolve()).get([ModelType.USD])
                if model.type is ModelType.UNKNOWN:
                    raise ValueError(f"obstacle model {obstacle.model.name} has no USD representation")
            except Exception as e:
                self._logger.error(f"Failed to load model for obstacle {obstacle.name}: {e}\n{traceback.format_exc()}")
                return None
            assert model.path is not None, f"USD model {model.name} must have a valid file path"
            prim = Prim()
            prim.usd_path = str(model.path)
            prim.name = self._NS_PRIM(obstacle.sim_path)
            prim.pose = obstacle.pose.to_msg()
            if obstacle.scale is not None:
                prim.scale.x = obstacle.scale.x
                prim.scale.y = obstacle.scale.y
                prim.scale.z = obstacle.scale.z
            return prim

        prims = await asyncio.gather(*map(impl, obstacles))

        req = SpawnPrims.Request()
        req.prims = list(filter(None, prims))
        response = await self._clients.SpawnPrims.call_timeout(req)
        if response is None:
            return tuple(False for _ in obstacles)

        response_iter = iter(response.ret)

        return tuple((a is not None) and next(response_iter) for a in prims)

    async def obstacle_move(self, obstacles):
        return await self._move_entities([(self._NS_PRIM(o.sim_path), o.pose) for o in obstacles])

    async def pedestrian_move(self, pedestrians):
        await self._clients.DeletePedestrians.call_timeout(
            DeletePrims.Request(names=[self._NS_PEDESTRIAN(p.sim_path) for p in pedestrians])
        )
        res = await self.pedestrian_spawn(pedestrians)
        if res is None:
            return tuple(False for _ in pedestrians)
        return tuple(res)

        # # tmp: restore when pedestrian move works within isaac sim
        # return await self._move_entities([(self._NS_PEDESTRIAN(p.name), p.pose) for p in pedestrians])

    async def robot_move(self, robots):
        async def move_robot(robot: Robot) -> bool:
            try:
                robot_params = (await arena_robots.Robot.RobotIdentifier(robot.model.name).resolve()).model_params
                res1 = await self._move_entity(self._NS_ROBOT(robot.sim_path), robot.pose)
                res2 = await self._move_entity(self._NS_ROBOT(robot.sim_path, robot_params.base_frame), robot.pose)
                return res1 and res2
            except Exception as e:
                self._logger.error(f"Failed to move robot {robot.name}: {e}\n{traceback.format_exc()}")
                return False
        return await asyncio.gather(*map(move_robot, robots))

    async def obstacle_delete(self, obstacles):
        return await asyncio.gather(*(self._delete_entity(self._NS_PRIM(o.sim_path)) for o in obstacles))

    async def pedestrian_delete(self, pedestrians):
        res = await self._clients.DeletePedestrians.call_timeout(
            DeletePrims.Request(names=[self._NS_PEDESTRIAN(p.sim_path) for p in pedestrians])
        )
        if res is None:
            ret = tuple(False for _ in pedestrians)
        else:
            ret = tuple(res.ret)
        return ret

    async def robot_delete(self, robots):
        return await asyncio.gather(*(self._delete_entity(self._NS_ROBOT(r.sim_path)) for r in robots))

    async def remove_world(self):
        await self._delete_entity(self._NS_WALL(self.node._environment_manager._prefix()))
        await self._delete_entity(self._NS_DOOR(self.node._environment_manager._prefix()))
        await self._delete_entity(self._NS_FLOOR(self.node._environment_manager._prefix()))
        return True

    async def spawn_walls(self, walls):
        # return True
        self._logger.debug("Attempting to spawn walls")

        async def create_segment(segment: WallSegment) -> Wall | None:
            end = segment.end.to_msg()
            end.z += segment.height
            try:
                wall_name = self.node._environment_manager.realize(f"wall_{next(self.wall_counter)}")
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

        async def create_obstacle(obstacle: ObstacleDefinition) -> Prim | None:
            try:
                prim_name = self.node._environment_manager.realize(f"obstacle_{next(self.wall_counter)}")
                model = await (await obstacle.model.resolve()).get(ModelType.USD)
                if model.type is ModelType.UNKNOWN:
                    return None
                assert model.path is not None, f"USD model {model.name} must have a valid file path"
                prim = Prim()
                prim.usd_path = str(model.path)
                prim.name = self._NS_WALL(prim_name)
                prim.pose = obstacle.pose.to_msg()
                return prim

            except Exception as e:
                self._logger.error(f"Failed to spawn wall obstacle: {e}\n{traceback.format_exc()}")
                return None

        async def create_wall(wall: WallDefinition):
            segments, obstacles = await wall.assets()
            return map(create_segment, segments), map(create_obstacle, obstacles)

        wall_futures = await asyncio.gather(*map(create_wall, walls))
        segment_futures, obstacle_futures = zip(*wall_futures)

        walls_req = SpawnWalls.Request()
        prims_req = SpawnPrims.Request()
        walls_req.walls = list(filter(None, await asyncio.gather(*itertools.chain.from_iterable(segment_futures))))
        prims_req.prims = list(filter(None, await asyncio.gather(*itertools.chain.from_iterable(obstacle_futures))))

        walls_res = await self._clients.SpawnWalls.call_timeout(walls_req)
        prims_res = await self._clients.SpawnPrims.call_timeout(prims_req)
        res = bool(walls_res) and all(walls_res.ret) and bool(prims_res) and all(prims_res.ret)

        self._logger.info("All walls spawned.")
        return res

    async def spawn_floors(self, floors) -> bool:
        self._logger.info("Attempting to spawn floors")

        async def impl(floor: FloorDefinition) -> Floor | None:
            try:
                return Floor(
                    name=self._NS_FLOOR(floor.sim_path),
                    x_length=floor.x_length,
                    y_length=floor.y_length,
                    pos=floor.pos.to_msg(),
                    material=material_to_msg(await floor.material.resolve()),
                )

            except Exception:
                self._logger.error(f"Failed to spawn floor: {floor.name}\n{traceback.format_exc()}")
                return None

        floors_req = SpawnFloors.Request()
        floors_req.floors = list(filter(None, await asyncio.gather(*map(impl, floors))))
        floors_res = await self._clients.SpawnFloors.call_timeout(floors_req)

        res = bool(floors_res) and all(floors_res.ret)
        self._logger.info("All floors spawned successfully.")
        return res

    async def spawn_doors(self, doors) -> bool:
        async def impl(door: DoorDefinition) -> Door | None:
            try:
                end = door.end.to_msg()
                end.z += door.height
                return Door(
                    name=self._NS_DOOR(door.name),
                    start=door.start.to_msg(),
                    end=end,
                    material=material_to_msg(await door.material.resolve()),
                    thickness=0.1,
                    kind=door.kind,
                )
            except Exception as e:
                self._logger.error(f"Failed to spawn door: {e}\n{traceback.format_exc()}")
                return None

        doors_req = SpawnDoors.Request()
        doors_req.doors = list(filter(None, await asyncio.gather(*map(impl, doors))))
        doors_res = await self._clients.SpawnDoors.call_timeout(doors_req)

        res = bool(doors_res) and all(doors_res.ret)
        self._logger.info("All doors spawned successfully.")
        return res
    async def spawn_elevators(self, elevators) -> bool:
        self._logger.debug(f"IsaacSimulator.spawn_elevators ENTRY, elevators: {elevators}")
        self._logger.debug(f"IsaacSimulator.spawn_elevators called with: {[e.name for e in elevators]}")
        for e in elevators:
            self._logger.debug(f"Elevator data: {e}")

        req = SpawnElevators.Request()

        async def impl(elevator: ElevatorDefinition) -> Elevator | None:
            try:
                pos = elevator.position
                size = elevator.size
                size = Scale(x=size[0], y=size[1], z=size[2])
                des = elevator.destination
                material_resolved = await elevator.material.resolve()
                result = Elevator(
                    name=elevator.sim_path,
                    position=pos.to_msg(),
                    size=size,
                    height_min=elevator.height_min,
                    height_max=elevator.height_max,
                    material=material_to_msg(material_resolved),
                    destination=des if hasattr(elevator, 'destination') else '',
                )
                return result
            except Exception as e:
                self._logger.error(f"Failed to append elevator: {elevator.name}: {e}\n{traceback.format_exc()}")
                return None
        
        req.elevators = list(filter(None, await asyncio.gather(*map(impl, elevators))))
        elevators_res = await self._clients.SpawnElevators.call_timeout(req)
        res = bool(elevators_res) and all(elevators_res.ret)
        self._logger.debug("All elevators spawned successfully.")
        return res

    async def before_reset_task(self):
        await self._pause()
        return True

    async def after_reset_task(self):
        await self._unpause()
        return True

    async def _pause(self):
        await self._clients.PauseSimulation.call_timeout(std_srvs.srv.Trigger.Request())

    async def _unpause(self):
        await self._clients.UnpauseSimulation.call_timeout(std_srvs.srv.Trigger.Request())

    async def pedestrian_spawn(self, pedestrians):

        on_success: list[tuple[str, str]] = []

        # TODO implement targeted pedestrian models
        async def impl(pedestrian: DynamicObstacle) -> Pedestrian | None:
            available_models: dict[str, str] = {
                # "F_Business_02",
                # "F_Medical_01",
                # "M_Medical_01",
                # "biped_demo",
                # "female_adult_police_01_new",
                # "female_adult_police_02",
                # "female_adult_police_03_new",
                # "male_adult_construction_01_new",
                # "male_adult_construction_03",
                # "male_adult_construction_05_new",
                # "male_adult_police_04",
                "female_adult_business_02": "original_female_adult_business_02",
                "female_adult_medical_01": "original_female_adult_medical_01",
                "female_adult_police_01": "original_female_adult_police_01",
                "female_adult_police_02": "original_female_adult_police_02",
                "female_adult_police_03": "original_female_adult_police_03",
                "male_adult_construction_01": "original_male_adult_construction_01",
                "male_adult_construction_02": "original_male_adult_construction_02",
                "male_adult_construction_03": "original_male_adult_construction_03",
                "male_adult_construction_05": "original_male_adult_construction_05",
                "male_adult_medical_01": "original_male_adult_medical_01",
                "male_adult_police_04": "original_male_adult_police_04",
            }
            if pedestrian.model.name in available_models:
                model_name = pedestrian.model.name
            else:
                model_name = random.choice(tuple(available_models.keys()))

            ped = Pedestrian()
            ped.name = self._NS_PEDESTRIAN(pedestrian.sim_path)
            ped.character_name = available_models[model_name]
            ped.pose = pedestrian.pose.to_msg()
            ped.controller_stats = False

            on_success.append((pedestrian.name, model_name))
            return ped

        req = SpawnPedestrians.Request()
        req.pedestrians = list(filter(None, await asyncio.gather(*map(impl, pedestrians))))
        res = await self._clients.SpawnPedestrians.call_timeout(req)
        if res is None:
            return tuple(False for _ in pedestrians)

        await self.pedestrian_update(
            arena_people_msgs.msg.Pedestrians(pedestrians=[
                arena_people_msgs.msg.Pedestrian(
                    name=ped.sim_path,
                    pose=ped.pose.to_msg(),
                )
                for status, ped
                in zip(res.ret, pedestrians)
                if status
            ])
        )

        return res.ret

    async def pedestrian_update(self, pedestrians):

        async def impl(ped: arena_people_msgs.msg.Pedestrian) -> PedestrianGoal | None:
            goal = PedestrianGoal()
            goal.name = self._NS_PEDESTRIAN(ped.name)
            goal.pose = ped.pose
            goal.twist = ped.twist
            return goal

        goals = list(filter(None, await asyncio.gather(*map(impl, pedestrians.pedestrians))))
        req = NavigatePedestrians.Request()
        req.goals = goals
        res = await self._clients.NavigatePedestrians.call_timeout(req)

        return tuple(a and b for a, b in zip(goals, res and res.ret or ()))

    async def _delete_entity(self, name: str) -> bool:
        self._logger.debug(f"Attempting to delete prim {name}")

        res = await self._clients.DeletePrims.call_timeout(
            DeletePrims.Request(
                names=[name]
            )
        )
        if res is None:
            return False

        return res.ret[0]

    async def _delete_pedestrians(self, prim_path):
        self._logger.info(f"Attempting to delete prim named {prim_path}")

        res = await self._clients.DeletePedestrians.call_timeout(
            DeletePrims.Request(names=[prim_path])
        )
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
        self._logger.info("Setting up IsaacSimulator service clients...")
        futures: list[typing.Awaitable] = []
        # Define services with their corresponding client attributes
        for client in self._clients.__dict__.values():
            client = typing.cast(ClientWrapper, client)
            self._logger.info(f"Initializing service client: {client.client.srv_name}")
            futures.append(client.ensure())
        await asyncio.gather(*futures)

        self._logger.info("All service clients are available.")

        self.node.create_publisher(std_msgs.msg.String, '/isaac/add_pedestrians_topic', 10).publish(
            std_msgs.msg.String(data=self.node.service_namespace('arena_peds'))
        )

        self._logger.info("All service clients initialized and available.")

    @classmethod
    async def create(cls, *args, namespace, **kwargs):
        self = cls(*args, namespace=namespace, **kwargs)
        self._logger.info("Creating IsaacSimulator instance...")
        await self.setup()
        return self
