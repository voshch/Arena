import asyncio
import copy
import math
import time
import traceback
import typing
from collections.abc import Sequence

import attrs
import geometry_msgs.msg
import numpy as np
import rclpy.client
import rclpy.node
import rclpy.qos
import rclpy.time
import tf2_ros
from arena_people_msgs.msg import Pedestrian, Pedestrians
from arena_rclpy_mixins.Async import ClientWrapper
from arena_rclpy_mixins.shared import Namespace
from arena_runtime.constants import SimSimulator
from arena_runtime.sim import BaseSim
from geometry_msgs.msg import Point
from hunav_msgs.msg import Agent, AgentBehavior, Agents, WallSegment
from hunav_msgs.srv import ComputeAgent, ComputeAgents, GetAgents, GetWalls, MoveAgent
from std_srvs.srv import Trigger
from task_generator_msgs.msg import RobotFleet
from visualization_msgs.msg import Marker, MarkerArray

from task_generator.shared import (
    DynamicObstacle,
    Obstacle,
    Orientation,
    Pose,
    Position,
    Robot,
    Wall,
)
from task_generator.simulators.human import BaseHumanSimulator
from task_generator.simulators.human.noop import NoopHumanSimulator

from . import HunavDynamicObstacle

# robotless-env park position, far enough that all robot forces vanish
_ROBOT_PARK_XY = 1.0e4

_FLEET_QOS = rclpy.qos.QoSProfile(
    depth=1,
    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
)


@attrs.define
class _TrackedRobot:
    robot: Robot
    tf_frame: str | None
    pose: Pose
    velocity: tuple[float, float] = (0.0, 0.0)
    stamp: float | None = None


class HunavHumanSimulator(BaseHumanSimulator if typing.TYPE_CHECKING else NoopHumanSimulator):
    """HunavManager with debug logging for tracking execution flow"""

    @classmethod
    def _register_task_modes(cls):
        from task_generator.constants import Constants
        from task_generator.tasks.obstacles.prompt import NS, declare_schema
        from task_generator.tasks.registry import OBSTACLES_MODES

        @OBSTACLES_MODES.register(Constants.TaskMode.TM_Obstacles.PROMPT, namespace=NS, schema=declare_schema)
        def _prompt() -> type:
            from task_generator.tasks.obstacles.prompt.hunav import TM_Prompt

            return TM_Prompt

    # Service Names
    SERVICE_COMPUTE_AGENT = "compute_agent"
    SERVICE_COMPUTE_AGENTS = "compute_agents"
    SERVICE_MOVE_AGENT = "move_agent"
    SERVICE_CLEAR_AGENTS = "clear_agents"
    SERVICE_GET_AGENTS = "get_agents"
    SERVICE_GET_WALLS = "get_walls"

    # Service Clients
    _compute_agent_client: ClientWrapper
    _compute_agents_client: ClientWrapper
    _move_agent_client: ClientWrapper
    _clear_agents_client: ClientWrapper

    # Service Servers
    _get_agents_service: rclpy.node.Service
    _get_walls_service: rclpy.node.Service

    def __init__(self, *args: object, namespace: Namespace, simulator: BaseSim, **kwargs: object) -> None:
        """Initialize HunavManager with debug logging"""
        super().__init__(*args, namespace=namespace, simulator=simulator, **kwargs)
        # Detect Simulator Type to decide between Plugin or move_entity callback
        self._logger.info(f"Detected simulator type: {self._simulator_type}")

        self._logger.debug("=== HUNAVMANAGER INIT START ===")
        self._logger.debug("Parent class initialized")

        # Initialize collections
        self._logger.debug("Initializing collections...")
        self._pedestrians: dict[int, dict] = {}
        self._explicit_wall_segments: dict[str, WallSegment] = {}
        self._explicit_wall_points: dict[str, list[Point]] = {}
        self._obstacle_wall_segments: dict[str, WallSegment] = {}
        self._obstacle_wall_points: dict[str, list[Point]] = {}
        self._wall_segments: dict[str, WallSegment] = {}
        self._wall_points: dict[str, list[Point]] = {}
        self._agents_lock: asyncio.Lock = asyncio.Lock()
        self._robots: dict[str, _TrackedRobot] = {}
        self._fleet_order: list[str] = []
        self._agents_container: Agents = Agents()  # Container to hold all registered agents
        self._get_agents_container: Agents = Agents()  # Container specifically just to send the Agent attributes to Hunavsystemplugin
        self._agents_container.header.frame_id = "map"
        self._arena_pedestrians_container: Pedestrians = Pedestrians()
        self._arena_pedestrians_container.header.frame_id = "map"

        self._update_loop_task: asyncio.Task
        self._publish_loop_task: asyncio.Task

        self._last_updated_agents = None
        self._last_smooth_yaws = {}

        self._agent_previous_orientations = {}
        self._orientation_smoothing_factor = 0.15  # 0.05-0.3 range

        self._logger.debug("Collections initialized")

        self.node.create_subscription(
            RobotFleet,
            self.node.service_namespace("state", "robots"),
            self._on_robot_fleet,
            _FLEET_QOS,
        )

        # forward hunav's force arrows to the debug overlay
        self.node.create_subscription(
            MarkerArray,
            self.node.service_namespace("sfm_forces"),
            self.publish_markers,
            5,
        )

        self._compute_agent_client = self.node.create_client_wrapper(ComputeAgent, self.node.service_namespace(self.SERVICE_COMPUTE_AGENT))
        self._compute_agents_client = self.node.create_client_wrapper(ComputeAgents, self.node.service_namespace(self.SERVICE_COMPUTE_AGENTS))
        self._move_agent_client = self.node.create_client_wrapper(MoveAgent, self.node.service_namespace(self.SERVICE_MOVE_AGENT))
        self._clear_agents_client = self.node.create_client_wrapper(Trigger, self.node.service_namespace(self.SERVICE_CLEAR_AGENTS))

    @property
    def _simulator_type(self) -> SimSimulator:
        """Detect which simulator is being used"""
        return self.node.conf.Arena.SIM.value

    async def setup(self):

        # Setup services
        self._logger.debug("Setting up services...")
        if await self._setup_services():
            self._logger.debug("Services setup complete")
        else:
            self._logger.error("Service setup failed!")

        self._logger.debug("Waiting for services to be ready...")
        await asyncio.sleep(2.0)
        self._logger.debug("Service wait complete")

        self._logger.debug("=== HUNAVMANAGER INIT COMPLETE ===")

        self._update_loop_task = asyncio.create_task(self._move_entity_loop())
        self._publish_loop_task = asyncio.create_task(self._publish_arena_peds_loop())

    @classmethod
    async def create(cls, *args: object, namespace: Namespace, simulator: BaseSim, **kwargs: object) -> "HunavHumanSimulator":
        self = cls(*args, namespace=namespace, simulator=simulator, **kwargs)
        await self.setup()
        return self

    async def _setup_services(self) -> bool:
        """Initialize all required services with debug logging"""
        self._logger.debug("=== SETUP_SERVICES START ===")

        # Debug namespace information
        self._logger.debug(f"Node namespace: {self.node.get_namespace()}")
        self._logger.debug(f"Task generator namespace: {self._namespace}")

        # Create service names with full namespace path

        # Create GetAgents service provider
        self._get_agents_service = self.node.create_service(
            GetAgents,
            self.node.service_namespace(self.SERVICE_GET_AGENTS),
            self._get_agents_callback,
        )

        # Create GetWalls service provider
        self._get_walls_service = self.node.create_service(
            GetWalls,
            self.node.service_namespace(self.SERVICE_GET_WALLS),
            self._get_walls_callback,
        )

        futures: list[typing.Awaitable] = []
        for client in (
            self._compute_agent_client,
            self._compute_agents_client,
            self._move_agent_client,
            self._clear_agents_client,
        ):
            futures.append(typing.cast(ClientWrapper, client).ensure())
        await asyncio.gather(*futures)

        self._logger.debug("All required services are available")
        self._logger.debug("=== SETUP_SERVICES COMPLETE ===")
        return True

    def _on_robot_fleet(self, msg: RobotFleet) -> None:
        self._fleet_order = [state.descriptor.name for state in msg.robots]

    async def _spawn_robot_impl(self, robots: Sequence[Robot]) -> Sequence[bool]:
        for robot in robots:
            tf_frame: str | None = None
            try:
                view = await robot.model.resolve()
                tf_frame = robot.frame(view.model_params.base_frame).raw()
            except Exception as e:
                self._logger.warning(f"robot {robot.name!r}: base frame unresolved ({e}), using spawn pose only")
            self._robots[robot.name] = _TrackedRobot(
                robot=robot,
                tf_frame=tf_frame,
                pose=copy.deepcopy(robot.pose),
            )
        return (True,) * len(robots)

    async def _move_robot_impl(self, robots: Sequence[Robot]) -> Sequence[bool]:
        for robot in robots:
            tracked = self._robots.get(robot.name)
            if tracked is None:
                continue
            tracked.pose = copy.deepcopy(robot.pose)
            tracked.velocity = (0.0, 0.0)
            tracked.stamp = None
        return (True,) * len(robots)

    async def _remove_robot_impl(self, robots: Sequence[Robot]) -> Sequence[bool]:
        for robot in robots:
            self._robots.pop(robot.name, None)
        return (True,) * len(robots)

    def _primary_robot(self) -> _TrackedRobot | None:
        """First fleet-order robot present in the roster."""
        for name in self._fleet_order:
            tracked = self._robots.get(name)
            if tracked is not None:
                return tracked
        return next(iter(self._robots.values()), None)

    def _refresh_robot(self, tracked: _TrackedRobot) -> None:
        """Update cached pose and finite-difference velocity from the map->base TF."""
        if tracked.tf_frame is None:
            return
        try:
            t = self.node.tf_buffer.lookup_transform('map', tracked.tf_frame, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return
        stamp = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
        if tracked.stamp is not None and stamp <= tracked.stamp:
            return
        position = Position(x=t.transform.translation.x, y=t.transform.translation.y)
        if tracked.stamp is not None:
            dt = stamp - tracked.stamp
            tracked.velocity = (
                (position.x - tracked.pose.position.x) / dt,
                (position.y - tracked.pose.position.y) / dt,
            )
        tracked.pose = Pose(position, Orientation.from_msg(t.transform.rotation))
        tracked.stamp = stamp

    def _robot_agent_msg(self) -> Agent:
        msg = Agent()
        msg.id = 0
        msg.type = Agent.ROBOT
        msg.name = "robot"
        msg.radius = 0.3
        tracked = self._primary_robot()
        if tracked is None:
            msg.position.position.x = _ROBOT_PARK_XY
            msg.position.position.y = _ROBOT_PARK_XY
            return msg
        self._refresh_robot(tracked)
        msg.name = tracked.robot.name
        msg.radius = self.node.rosparam[float].get('robot_radius', msg.radius)
        msg.position.position.x = tracked.pose.position.x
        msg.position.position.y = tracked.pose.position.y
        msg.position.orientation = tracked.pose.orientation.to_msg()
        msg.yaw = tracked.pose.orientation.to_yaw()
        vx, vy = tracked.velocity
        msg.velocity.linear.x = vx
        msg.velocity.linear.y = vy
        msg.linear_vel = math.hypot(vx, vy)
        return msg

    def _update_agent_obstacles(self, current_agents: Agents) -> Agents:
        """Update agent closest_obs with latest obstacle data before HuNav call"""
        from itertools import chain

        all_wall_points = list(chain.from_iterable(self._wall_points.values()))

        for agent in current_agents.agents:
            agent.closest_obs.extend(all_wall_points)
            self._logger.debug(f"Updated agent {agent.name} with {len(agent.closest_obs)} obstacles")

        return current_agents

    async def _get_agents_callback(self, request: GetAgents.Request, response: GetAgents.Response) -> GetAgents.Response:
        """Handle get_agents service request - return UNMODIFIED agents"""
        del request
        try:
            self._logger.debug("=== GET AGENTS CALLBACK ===")
            self._logger.debug(f"Returning {len(self._get_agents_container.agents)} agents")

            # Update timestamp
            self._get_agents_container.header.stamp = self.node.sim_time.to_msg()
            self._get_agents_container.header.frame_id = "map"

            # Return the UNMODIFIED container
            response.agents = self._get_agents_container

            # Debug output
            # for agent in self._get_agents_container.agents:
            #     self._logger.debug(f"Sending agent {agent.name}: desired_velocity={agent.desired_velocity}, goal_force_factor={agent.behavior.goal_force_factor}")

            return response

        except Exception as e:
            self._logger.error(f"Error in get_agents_callback: {e}")
            response.agents = Agents()  # Empty response on error
            return response

    async def _get_walls_callback(self, request: GetWalls.Request, response: GetWalls.Response) -> GetWalls.Response:
        """Service callback für Wall-Segments"""
        del request
        response.walls = list(self._wall_segments.values())
        self._logger.debug(f"Sent {len(self._wall_segments)} wall segments")
        return response

    async def _move_entity_loop(self):
        """Pedestrian Move Entity Callback for non gazebo simulators"""

        try:
            with self.node.sim_time_rate(10.0) as (done, rate):
                while not done.is_set():
                    await rate.get()
                    async with self._agents_lock:
                        await self.relay_pedestrian_update(self._arena_pedestrians_container)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.error(f"Failed to update agent positions: {e}\n{traceback.format_exc()}")

    async def _spawn_obstacles_impl(self, obstacles: Sequence[Obstacle]) -> Sequence[Obstacle | None]:
        self._logger.debug(f"Spawning walls for {len(obstacles)} obstacles")

        async def obstacle_to_walls(obstacle: Obstacle) -> list[tuple[str, Wall]]:
            try:
                view = await obstacle.model.resolve()
                bounds = view.bounds
                if bounds is None:
                    return []

                yaw = obstacle.pose.orientation.to_yaw()
                rotation_mat = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                corners = np.array(bounds) @ rotation_mat.T
                corners += np.array([[obstacle.pose.position.x, obstacle.pose.position.y]])

                obstacle_walls: list[tuple[str, Wall]] = []
                for i, (start, end) in enumerate(zip(corners, np.roll(corners, -1, axis=0), strict=False)):
                    wall = Wall(
                        start=Position(x=start[0], y=start[1]),
                        end=Position(x=end[0], y=end[1]),
                    )
                    obstacle_walls.append((f"{obstacle.name}_wall_{i}", wall))

                self._logger.debug(f"spawning wall segments for obstacle {obstacle.name}")
                for _name, wall in obstacle_walls:
                    self._logger.debug(f"  Wall from ({wall.start.x:.2f}, {wall.start.y:.2f}) to ({wall.end.x:.2f}, {wall.end.y:.2f})")

                return obstacle_walls
            except Exception as e:
                self._logger.warning(f"failed to spawn bounding walls for obstacle '{obstacle.name}': {e}")
                return []

        self._logger.debug("Spawning walls for obstacles...")
        all_walls = await asyncio.gather(*(obstacle_to_walls(obs) for obs in obstacles))
        self._logger.debug("Finished spawning walls for obstacles.")

        for name, wall in [entry for walls in all_walls for entry in walls]:
            self._add_wall(
                name,
                wall,
                self._obstacle_wall_segments,
                self._obstacle_wall_points,
            )
        self._rebuild_wall_cache()
        return obstacles

    async def _spawn_dynamic_obstacles_impl(self, obstacles: Sequence[DynamicObstacle]) -> Sequence[DynamicObstacle | None]:
        async with self._agents_lock:
            results: list[DynamicObstacle | None] = []
            new_agent_msgs: list = []

            for obstacle in obstacles:
                try:
                    unique_id = len(self._agents_container.agents) + 1
                    self._logger.debug(f"Preparing to spawn dynamic obstacle '{obstacle.name}' with ID {unique_id}")
                    hunav_obstacle = HunavDynamicObstacle.from_dynamic_obstacle(obstacle)
                    hunav_obstacle = attrs.evolve(hunav_obstacle, id=unique_id)

                    agent_msg = hunav_obstacle.to_msg()

                    self._get_agents_container.agents.append(agent_msg)  # type: ignore
                    self._agents_container.agents.append(agent_msg)  # type: ignore
                    new_agent_msgs.append(agent_msg)

                    arena_pedestrian = self._create_arena_pedestrian(hunav_obstacle, unique_id)
                    self._arena_pedestrians_container.pedestrians.append(arena_pedestrian)  # type: ignore

                    self._pedestrians[agent_msg.id] = {
                        "last_update": time.time(),
                        "current_state": agent_msg.behavior.state,
                        "agent": agent_msg,
                        "animation_time": 0.0,
                    }

                    results.append(obstacle)

                except Exception as e:
                    self._logger.error(f"Error preparing agent: {str(e)}")
                    self._logger.error(traceback.format_exc())
                    results.append(None)

            if not new_agent_msgs:
                return results

            # HuNav's BTnode initializes per-agent behavior trees only on the
            # first ComputeAgents call (guarded by `initialized_`). Subsequent
            # calls with new agent ids leave `trees_` unchanged, so the new
            # agent is never ticked. Workaround: call `clear_agents` (which
            # flips `initialized_` back to false) and then ComputeAgents with
            # the full agent set so HuNav re-runs `initializeBehaviorTrees`.
            #
            # `_last_updated_agents` (HuNav's response) drops `behavior_tree`,
            # `goals`, `cyclic_goals`, `goal_radius`, `desired_velocity`,
            # `radius`, `linear_vel`, `angular_vel`, so we cannot use it as
            # the resend payload directly: existing agents would come back
            # with empty goals and stop. Instead, take the full original
            # Agent msg cached in `_pedestrians[id]['agent']` and overlay the
            # live pose/velocity from `_last_updated_agents`. Trade-off: BT
            # internal node bookkeeping (timers, cyclic-goal index) resets on
            # every incremental spawn; visible motion continues from the live
            # pose toward the unchanged goals.
            live_by_id = {}
            if self._last_updated_agents is not None:
                live_by_id = {a.id: a for a in self._last_updated_agents.agents}

            merged = Agents()
            merged.header.frame_id = "map"
            merged.header.stamp = self.node.sim_time.to_msg()
            for ped_id, info in self._pedestrians.items():
                full = copy.deepcopy(info["agent"])
                live = live_by_id.get(ped_id)
                if live is not None:
                    full.position = live.position
                    full.yaw = live.yaw
                    full.velocity = live.velocity
                    full.linear_vel = live.linear_vel
                    full.angular_vel = live.angular_vel
                    full.behavior.state = live.behavior.state
                merged.agents.append(full)  # type: ignore

            if not await self._reset_hunav():
                self._logger.error("clear_agents failed; new pedestrian(s) will not move")
                return results

            request = ComputeAgents.Request()
            request.robot = self._robot_agent_msg()
            request.current_agents = merged
            response = await self._compute_agents_client.call_timeout(request)

            if response and response.updated_agents:
                self._last_updated_agents = response.updated_agents
            else:
                self._logger.error("ComputeAgents after clear_agents returned no updated agents")

            return results

    def _wall_to_points(self, start: Position, end: Position, spacing: float = 0.2) -> list[Point]:
        points: list[Point] = []
        v = (end - start).normalized()
        for i in np.arange(0, (end - start).norm(), spacing):
            pt_pos = start + v * i
            points.append(pt_pos.to_msg())  # convert Position -> geometry_msgs.msg.Point
        points.append(end.to_msg())
        return points

    def _rebuild_wall_cache(self):
        """Merge explicit and obstacle wall dicts into the cached view."""
        self._wall_segments = {
            **self._explicit_wall_segments,
            **self._obstacle_wall_segments,
        }
        self._wall_points = {**self._explicit_wall_points, **self._obstacle_wall_points}

    def _add_wall(
        self,
        name: str,
        wall: Wall,
        segments: dict[str, WallSegment],
        points: dict[str, list[Point]],
    ):
        """Add a single wall segment to the given dicts."""
        segment = WallSegment()
        segment.id = len(segments)
        segment.start = wall.start.to_msg()
        segment.end = wall.end.to_msg()
        segment.length = (wall.start - wall.end).norm()

        segments[name] = segment
        points[name] = self._wall_to_points(wall.start, wall.end)

        self._logger.debug(f"Added wall segment '{name}' from ({segment.start.x:.2f}, {segment.start.y:.2f}) to ({segment.end.x:.2f}, {segment.end.y:.2f})")

    async def _spawn_walls_impl(self, walls: typing.Mapping[str, Wall]) -> bool:
        self._explicit_wall_segments = {}
        self._explicit_wall_points = {}

        for name, wall in walls.items():
            self._add_wall(
                name,
                wall,
                self._explicit_wall_segments,
                self._explicit_wall_points,
            )

        self._rebuild_wall_cache()
        self._logger.debug(f"Cached {len(self._wall_segments)} wall segments")
        return True

    async def _remove_walls_impl(self, names: Sequence[str]) -> bool:
        for name in names:
            self._explicit_wall_segments.pop(name, None)
            self._explicit_wall_points.pop(name, None)
        self._rebuild_wall_cache()
        self._logger.debug(f"Removed {len(names)} wall segments, {len(self._wall_segments)} remaining")
        return True

    async def _remove_obstacles_impl(self, names: Sequence[str]) -> bool:
        """Remove obstacle-derived wall segments by obstacle name."""
        for name in names:
            keys = [k for k in self._obstacle_wall_segments if k.startswith(f"{name}_wall_")]
            for k in keys:
                self._obstacle_wall_segments.pop(k, None)
                self._obstacle_wall_points.pop(k, None)
        self._rebuild_wall_cache()
        self._logger.debug(f"Removed obstacle walls for {len(names)} obstacles, {len(self._wall_segments)} wall segments remaining")
        return True

    async def _remove_pedestrians_impl(self) -> bool:
        """Remove all spawned pedestrians from simulation safely."""
        self._logger.debug(f"=== REMOVING {len(self._pedestrians)} PEDESTRIANS ===")

        async with self._agents_lock:
            self._agents_container = Agents()
            self._get_agents_container = Agents()
            self._last_updated_agents: Agents | None = None
            self._logger.debug("Cleared local agents container")

            success = await self._reset_hunav()
            if not success:
                self._logger.error("Failed to reset HuNav agents - continuing anyway")

            self._clear_local_data()

        self._logger.debug(f"Complete reset completed: {success}")
        return success

    async def _reset_hunav(self) -> bool:
        """Reset HuNav by calling ClearAgents service"""
        try:
            if not await self._clear_agents_client.ensure(timeout_sec=2.0):
                self._logger.error("ClearAgents service not available")
                return False

            request = Trigger.Request()

            self._logger.debug("Calling HuNav ClearAgents service...")
            response = await self._clear_agents_client.call_timeout(request)

            if response and response.success:
                self._logger.debug("HuNav clear successful - ready for new agents")
                return True
            else:
                self._logger.error("HuNav clear failed")
                return False

        except Exception as e:
            self._logger.error(f"Error calling ClearAgents: {e}")
            return False

    def _clear_local_data(self):
        """Clear all local data structures safely"""
        # Clear pedestrians dictionary
        self._pedestrians.clear()

        # Clear agents container (if not already done)
        self._agents_container.agents.clear()  # type: ignore
        self._arena_pedestrians_container.pedestrians.clear()  # type: ignore
        self._ped_model_uris.clear()

        self._last_updated_agents = None
        self._last_smooth_yaws = {}
        self._agent_previous_orientations = {}

        self._logger.debug("All local data structures cleared")

    def _create_arena_pedestrian(self, hunav_obstacle: HunavDynamicObstacle, unique_id: int) -> Pedestrian:
        """Create arena_people_msgs.Pedestrian (separate from hunav)"""

        arena_ped = Pedestrian()

        arena_ped.name = hunav_obstacle.name
        arena_ped.id = unique_id

        arena_ped.pose.position.x = hunav_obstacle.init_pose.x
        arena_ped.pose.position.y = hunav_obstacle.init_pose.y
        arena_ped.pose.position.z = 1.25

        arena_ped.pose.orientation = Orientation.from_yaw(hunav_obstacle.yaw).to_msg()

        # Initial twist (zero at spawn)
        arena_ped.twist.linear.x = 0.0
        arena_ped.twist.linear.y = 0.0
        arena_ped.twist.angular.z = 0.0

        # Animation state from behavior
        arena_ped.animation_state = self._map_hunav_behavior_to_arena_state(hunav_obstacle.behavior.type)

        self._logger.debug(f"Created arena pedestrian: {arena_ped.name}")
        return arena_ped

    def _map_hunav_behavior_to_arena_state(self, behavior_type: int) -> int:
        """Map hunav behavior to arena animation state"""

        # Import AgentBehavior constants
        behavior_mapping = {
            AgentBehavior.BEH_REGULAR: Pedestrian.WALKING,
            AgentBehavior.BEH_IMPASSIVE: Pedestrian.IDLE,
            AgentBehavior.BEH_SURPRISED: Pedestrian.SURPRISED,
            AgentBehavior.BEH_SCARED: Pedestrian.PANIC,
            AgentBehavior.BEH_CURIOUS: Pedestrian.CURIOUS,
            AgentBehavior.BEH_THREATENING: Pedestrian.THREATENING,
        }

        return behavior_mapping.get(behavior_type, Pedestrian.WALKING)

    async def _publish_arena_peds_loop(self):
        """Use last updated agents as current agents (like Plugin does)"""
        try:
            with self.node.sim_time_rate(10.0) as (done, rate):  # 10 Hz
                while not done.is_set():
                    await rate.get()
                    async with self._agents_lock:
                        self._logger.debug(f"arena_peds_callback: publishing {len(self._arena_pedestrians_container.pedestrians)} pedestrians")
                        if not self._arena_pedestrians_container.pedestrians:
                            continue

                        # Use last updated agents as current agents
                        if self._last_updated_agents:
                            current_agents = self._last_updated_agents
                        else:
                            current_agents = self._agents_container

                        # Ensure frame_id is set
                        current_agents.header.frame_id = "map"
                        current_agents.header.stamp = self.node.sim_time.to_msg()

                        # Update obstacles BEFORE sending to HuNav
                        current_agents = self._update_agent_obstacles(current_agents)

                        # Smooth yaw values before sending to HuNav
                        current_agents = self._smooth_agents_before_hunav(current_agents)

                        # Create request
                        request = ComputeAgents.Request()
                        request.current_agents = current_agents
                        request.robot = self._robot_agent_msg()
                        response = await self._compute_agents_client.call_timeout(request)

                        if response and response.updated_agents:
                            # Fix frame_id
                            response.updated_agents.header.frame_id = "map"
                            response.updated_agents.header.stamp = self.node.sim_time.to_msg()

                            self._last_updated_agents = response.updated_agents

                            self._logger.debug(f"Updated agents: {self._last_updated_agents}")

                            # Update arena pedestrians
                            for arena_ped in self._arena_pedestrians_container.pedestrians:
                                for updated_agent in response.updated_agents.agents:
                                    if updated_agent.id == arena_ped.id:
                                        calculated_vel_x, calculated_vel_y = self._calculate_velocity_from_position_change(updated_agent, arena_ped)

                                        arena_ped.pose = self._round_coordinates(updated_agent.position, 2)

                                        arena_ped.twist.linear.x = updated_agent.velocity.linear.x
                                        arena_ped.twist.linear.y = updated_agent.velocity.linear.y
                                        arena_ped.twist.linear.z = 0.0

                                        arena_ped.twist.angular.x = 0.0
                                        arena_ped.twist.angular.y = 0.0
                                        arena_ped.twist.angular.z = 0.0

                                        import math

                                        # Orientation through CALCULATED VELOCITY!
                                        vel_x = calculated_vel_x
                                        vel_y = calculated_vel_y
                                        velocity_magnitude = math.sqrt(vel_x**2 + vel_y**2)

                                        if velocity_magnitude > 0.05:
                                            target_yaw = math.atan2(vel_y, vel_x)
                                        else:
                                            target_yaw = updated_agent.yaw

                                        smoothed_yaw = self._smooth_yaw_slerp(target_yaw, arena_ped.id)

                                        arena_ped.pose.orientation = Orientation.from_yaw(smoothed_yaw).to_msg()

                                        updated_agent.velocity.linear.x = calculated_vel_x
                                        updated_agent.velocity.linear.y = calculated_vel_y
                                        updated_agent.yaw = smoothed_yaw
                                        updated_agent.position.orientation = Orientation.from_yaw(smoothed_yaw).to_msg()

                                        break

                        # Publish
                        for ped in self._arena_pedestrians_container.pedestrians:
                            self._logger.debug(f"Publishing pedestrian {ped.name} at ({ped.pose.position.x}, {ped.pose.position.y}) with velocity ({ped.twist.linear.x}, {ped.twist.linear.y})")

                        self.publish_arena_peds(self._arena_pedestrians_container)
                        self._publish_wall_markers()

        except asyncio.CancelledError:
            pass

        except Exception as e:
            self._logger.error(f"Error in arena_peds loop: {e}\n{traceback.format_exc()}")

    def _publish_wall_markers(self):
        """Publish wall segments as visualization markers"""
        marker_array = MarkerArray()

        for wall in self._wall_segments.values():
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.node.sim_time.to_msg()
            marker.ns = f"hunav_walls_{wall.id}"
            marker.id = wall.id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05  # Line width
            marker.color.a = 1.0  # Alpha
            marker.color.r = 1.0  # Red
            marker.lifetime.sec = 1

            # Add start and end points of the wall segment
            marker.points.append(wall.start)
            marker.points.append(wall.end)

            marker_array.markers.append(marker)

        self.publish_static_markers(marker_array)

    def _smooth_yaw(self, new_yaw: float, current_yaw: float) -> float:
        import math

        def normalize_angle(angle: float) -> float:
            return math.atan2(math.sin(angle), math.cos(angle))

        new_yaw = normalize_angle(new_yaw)
        current_yaw = normalize_angle(current_yaw)
        diff = normalize_angle(new_yaw - current_yaw)

        if abs(diff) > math.radians(25):
            return normalize_angle(current_yaw + (diff * 0.1))
        else:
            return current_yaw

    def _smooth_agents_before_hunav(self, agents: Agents) -> Agents:
        """Yaw smoothing before sending back to hunav"""
        for agent in agents.agents:
            if agent.id in self._last_smooth_yaws:
                agent.yaw = self._smooth_yaw(agent.yaw, self._last_smooth_yaws[agent.id])
            self._last_smooth_yaws[agent.id] = agent.yaw
        return agents

    def _round_coordinates(self, pose: geometry_msgs.msg.Pose, decimals: int = 2) -> geometry_msgs.msg.Pose:
        """Round coordinates to avoid floating point errors"""
        pose.position.x = round(pose.position.x, decimals)
        pose.position.y = round(pose.position.y, decimals)
        pose.position.z = round(pose.position.z, decimals)
        return pose

    def _calculate_movement_yaw(self, agent: Agent) -> float | None:
        import math

        # Velocity-basierte OrientTION
        vel_x = agent.velocity.linear.x
        vel_y = agent.velocity.linear.y

        # Only if the Agent moves
        velocity_magnitude = math.sqrt(vel_x**2 + vel_y**2)
        if velocity_magnitude > 0.05:
            movement_yaw = math.atan2(vel_y, vel_x)
            return movement_yaw
        else:
            # If no Movement, use same Orientation dont change it
            return None

    def _calculate_velocity_from_position_change(self, updated_agent: Agent, arena_ped: Pedestrian, dt: float = 0.1) -> tuple[float, float]:
        """Calculate Position from the velocity change"""
        import math

        # Previous position
        prev_x = arena_ped.pose.position.x
        prev_y = arena_ped.pose.position.y

        # Current position
        curr_x = updated_agent.position.position.x
        curr_y = updated_agent.position.position.y

        vel_x = (curr_x - prev_x) / dt
        vel_y = (curr_y - prev_y) / dt

        # Speed limiting
        velocity_magnitude = math.sqrt(vel_x**2 + vel_y**2)
        if velocity_magnitude > updated_agent.desired_velocity:
            scale_factor = updated_agent.desired_velocity / velocity_magnitude
            vel_x *= scale_factor
            vel_y *= scale_factor

        return vel_x, vel_y

    def _slerp_quaternions(self, q1_list: list[float], q2_list: list[float], t: float) -> list[float]:
        """Spherical linear interpolation between two quaternions"""
        import math

        # Manual quaternion normalization
        def normalize_quat(q: list[float]) -> list[float]:
            norm = math.sqrt(sum(x * x for x in q))
            return [x / norm for x in q] if norm > 0 else [0, 0, 0, 1]

        # Convert lists to normalized quaternions
        q1 = normalize_quat(q1_list)
        q2 = normalize_quat(q2_list)

        # Calculate dot product
        dot = sum(a * b for a, b in zip(q1, q2, strict=False))

        # If dot product is negative, negate one quaternion for shorter path
        if dot < 0.0:
            q2 = [-x for x in q2]
            dot = -dot

        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = [q1[i] + t * (q2[i] - q1[i]) for i in range(4)]
            return normalize_quat(result)

        # Calculate spherical interpolation
        theta_0 = math.acos(abs(dot))
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * t
        sin_theta = math.sin(theta)

        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        result = [s0 * q1[i] + s1 * q2[i] for i in range(4)]
        return normalize_quat(result)

    def _smooth_yaw_slerp(self, target_yaw: float, agent_id: int) -> float:
        """Smooth yaw transitions to avoid orientation errors of Hunavs inner calculations"""
        import math

        def normalize_angle(angle: float) -> float:
            return math.atan2(math.sin(angle), math.cos(angle))

        target_yaw = normalize_angle(target_yaw)

        if agent_id in self._agent_previous_orientations:
            prev_yaw = self._agent_previous_orientations[agent_id]
            yaw_diff = normalize_angle(target_yaw - prev_yaw)

            # Smooth interpolation
            smoothed_yaw = prev_yaw + yaw_diff * self._orientation_smoothing_factor
            smoothed_yaw = normalize_angle(smoothed_yaw)
        else:
            smoothed_yaw = target_yaw

        # Store for next frame
        self._agent_previous_orientations[agent_id] = smoothed_yaw

        return smoothed_yaw
