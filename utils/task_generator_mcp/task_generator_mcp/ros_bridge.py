from __future__ import annotations

import os
import threading
from typing import Any

import rclpy
import rclpy.qos
from arena_rclpy_mixins import ActionClientWrapper, AsyncNode, ClientWrapper
from arena_runtime_msgs.srv import LifecycleHold
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.srv import DescribeParameters, GetParameters, ListParameters, SetParameters
from std_msgs.msg import Bool as BoolMsg
from std_msgs.msg import String
from std_srvs.srv import Empty
from task_generator_msgs.action import RunEpisode
from task_generator_msgs.msg import EpisodeRecord
from task_generator_msgs.srv import (
    DespawnRobot,
    GetTaskModes,
    QueryDynamicObstacles,
    QueryEnvironments,
    QueryParametrizeds,
    QueryRobots,
    QueryScenarios,
    QueryStaticObstacles,
    QueryWorlds,
    QueueEpisode,
    ResetEpisode,
    SpawnDynamic,
    SpawnRobot,
    SpawnStatic,
)

_TRANSIENT_LOCAL_1 = rclpy.qos.QoSProfile(
    depth=1,
    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
)


class RosBridge(AsyncNode):
    """Async access to the task_generator services, spinning on a background thread
    while the MCP server runs on the main asyncio loop. Service paths come from
    TASK_GENERATOR_NODE_NAME (default: /task_generator_node)."""

    def __init__(self) -> None:
        super().__init__("task_generator_mcp_bridge")

        node_name: str = os.environ.get("TASK_GENERATOR_NODE_NAME", "/task_generator_node")

        def _path(relative: str) -> str:
            return f"{node_name}/{relative}"

        self.client_reset_episode: ClientWrapper[ResetEpisode] = self.create_client_wrapper(ResetEpisode, _path("lifecycle/reset_episode"))
        self.client_pause: ClientWrapper[LifecycleHold] = self.create_client_wrapper(LifecycleHold, "/arena/sim_lifecycle/hold")
        self.client_wait_for_world: ClientWrapper[Empty] = self.create_client_wrapper(Empty, _path("lifecycle/wait_for_world"))

        self._arena_paused: bool = False

        self.client_query_worlds: ClientWrapper[QueryWorlds] = self.create_client_wrapper(QueryWorlds, _path("query/worlds"))
        self.client_query_scenarios: ClientWrapper[QueryScenarios] = self.create_client_wrapper(QueryScenarios, _path("query/scenarios"))
        self.client_query_robots: ClientWrapper[QueryRobots] = self.create_client_wrapper(QueryRobots, _path("query/robots"))
        self.client_query_static_obstacles: ClientWrapper[QueryStaticObstacles] = self.create_client_wrapper(QueryStaticObstacles, _path("query/static_obstacles"))
        self.client_query_dynamic_obstacles: ClientWrapper[QueryDynamicObstacles] = self.create_client_wrapper(QueryDynamicObstacles, _path("query/dynamic_obstacles"))
        self.client_query_environments: ClientWrapper[QueryEnvironments] = self.create_client_wrapper(QueryEnvironments, _path("query/environments"))
        self.client_query_parametrizeds: ClientWrapper[QueryParametrizeds] = self.create_client_wrapper(QueryParametrizeds, _path("query/parametrizeds"))

        self.client_queue_episode: ClientWrapper[QueueEpisode] = self.create_client_wrapper(QueueEpisode, _path("config/queue_episode"))
        self.client_get_task_modes: ClientWrapper[GetTaskModes] = self.create_client_wrapper(GetTaskModes, _path("config/get_task_modes"))

        self.client_spawn_static: ClientWrapper[SpawnStatic] = self.create_client_wrapper(SpawnStatic, _path("runtime/spawn_static"))
        self.client_spawn_dynamic: ClientWrapper[SpawnDynamic] = self.create_client_wrapper(SpawnDynamic, _path("runtime/spawn_dynamic"))
        self.client_spawn_robot: ClientWrapper[SpawnRobot] = self.create_client_wrapper(SpawnRobot, _path("runtime/spawn_robot"))
        self.client_despawn_robot: ClientWrapper[DespawnRobot] = self.create_client_wrapper(DespawnRobot, _path("runtime/despawn_robot"))

        self.action_run_episode: ActionClientWrapper[RunEpisode] = self.create_action_client_wrapper(RunEpisode, _path("lifecycle/run_episode"))

        # parameter clients against the task_generator node
        self.client_set_parameters: ClientWrapper[SetParameters] = self.create_client_wrapper(SetParameters, f"{node_name}/set_parameters")
        self.client_get_parameters: ClientWrapper[GetParameters] = self.create_client_wrapper(GetParameters, f"{node_name}/get_parameters")
        self.client_describe_parameters: ClientWrapper[DescribeParameters] = self.create_client_wrapper(DescribeParameters, f"{node_name}/describe_parameters")
        self.client_list_parameters: ClientWrapper[ListParameters] = self.create_client_wrapper(ListParameters, f"{node_name}/list_parameters")

        # latched state cache
        self._state_world: str = ""
        self._current_episode: EpisodeRecord | None = None
        self._queued_episode: EpisodeRecord | None = None

        self.create_subscription(
            BoolMsg,
            "/arena/state/paused",
            self._on_arena_paused,
            _TRANSIENT_LOCAL_1,
        )
        self.create_subscription(
            String,
            _path("state/world"),
            self._on_state_world,
            _TRANSIENT_LOCAL_1,
        )
        self.create_subscription(
            EpisodeRecord,
            _path("state/episode"),
            self._on_state_current,
            _TRANSIENT_LOCAL_1,
        )
        self.create_subscription(
            EpisodeRecord,
            _path("state/queue"),
            self._on_state_queue,
            _TRANSIENT_LOCAL_1,
        )

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def _on_arena_paused(self, msg: BoolMsg) -> None:
        self._arena_paused = msg.data

    def _on_state_world(self, msg: String) -> None:
        self._state_world = msg.data

    def _on_state_current(self, msg: EpisodeRecord) -> None:
        self._current_episode = msg

    def _on_state_queue(self, msg: EpisodeRecord) -> None:
        self._queued_episode = msg

    def _spin(self) -> None:
        rclpy.spin(self)

    @property
    def state_world(self) -> str:
        return self._state_world

    @property
    def current_episode(self) -> EpisodeRecord | None:
        return self._current_episode

    @property
    def queued_episode(self) -> EpisodeRecord | None:
        return self._queued_episode

    @property
    def arena_paused(self) -> bool:
        return self._arena_paused

    def pose_dict_to_msg(self, pose: dict[str, Any] | None) -> tuple[PoseStamped, bool]:
        """Convert an optional pose dict to a PoseStamped and use_pose flag."""
        msg = PoseStamped()
        if pose is None:
            return msg, False

        pos = pose.get("position", {})
        msg.pose.position.x = float(pos.get("x", 0.0))
        msg.pose.position.y = float(pos.get("y", 0.0))
        msg.pose.position.z = float(pos.get("z", 0.0))

        ori = pose.get("orientation", {})
        msg.pose.orientation.x = float(ori.get("x", 0.0))
        msg.pose.orientation.y = float(ori.get("y", 0.0))
        msg.pose.orientation.z = float(ori.get("z", 0.0))
        msg.pose.orientation.w = float(ori.get("w", 1.0))

        frame_id = pose.get("frame_id", "map")
        msg.header.frame_id = str(frame_id)

        return msg, True
