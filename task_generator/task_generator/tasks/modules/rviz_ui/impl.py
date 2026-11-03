import math
import typing
from collections.abc import Awaitable, Callable

import geometry_msgs.msg as geometry_msgs
import rclpy.qos
import rclpy.time
from arena_rclpy_mixins.Time import Time
from arena_runtime.env.task_api import GoToPhase, RobotManager, TaskRequest, planar_marker, visual
from arena_runtime.shared import Orientation, Pose, Position
from arena_runtime_msgs.msg import RobotFleet
from visualization_msgs.msg import Marker

from task_generator.tasks.modules import TM_Module, TM_ModuleKwargs

_FLEET_QOS = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)


def _handle_visuals(color: tuple[float, float, float, float]) -> list[Marker]:
    return [visual(Marker.CYLINDER, (0.5, 0.5, 0.02), color), visual(Marker.ARROW, (0.6, 0.1, 0.1), color)]


class Mod_OverrideRobot(TM_Module):
    TOPIC_SET_POSITION = "initialpose"
    TOPIC_SET_GOAL = "goal_pose"
    TOPIC_NEW_SCENARIO = "clicked_point"
    PARAM_WAYPOINTS = "guided_waypoints"

    _timeouts: dict[int, Time]
    _watched: dict[str, RobotManager]

    def __init__(self, **kwargs: typing.Unpack[TM_ModuleKwargs]) -> None:
        TM_Module.__init__(self, **kwargs)

        self._timeouts = {}
        self._watched = {}

        self.node.create_subscription(geometry_msgs.PoseWithCovarianceStamped, self.node.service_namespace(self.TOPIC_SET_POSITION), self._cb_set_position, 1)

        self.node.create_subscription(geometry_msgs.PoseStamped, self.node.service_namespace(self.TOPIC_SET_GOAL), self._cb_set_goal, 1)

        self.node.create_subscription(geometry_msgs.PointStamped, self.node.service_namespace(self.TOPIC_NEW_SCENARIO), self._cb_new_scenario, 1)

        self.node.create_subscription(RobotFleet, self.node.service_namespace("state", "robots"), self._cb_fleet, _FLEET_QOS)

    def after_reset(self) -> None:
        self._sync_handles()

    async def _cb_fleet(self, msg: RobotFleet) -> None:
        del msg
        if not self._ctx.reset_lock.locked():
            self._sync_handles()

    def _sync_handles(self) -> None:
        interactive = self._ctx.interactive
        assert interactive is not None, "rviz_ui needs the env's interactive context"
        hub = interactive.markers
        robots = self._ctx.robots

        for prefix in ("robot/", "goal/"):
            for marker in hub.names(prefix):
                if marker.removeprefix(prefix) not in robots:
                    hub.erase(marker)
        for name, manager in robots.items():
            if self._watched.get(name) is not manager or f"robot/{name}" not in hub.names(f"robot/{name}"):
                self._put_robot(name, manager)
            if self._watched.get(name) is not manager:
                manager.watch(self._on_robot)
                self._watched[name] = manager
            self._on_robot(manager)

    def _put_robot(self, name: str, manager: RobotManager) -> None:
        interactive = self._ctx.interactive
        assert interactive is not None, "rviz_ui needs the env's interactive context"
        reach = 2 * manager.radius
        interactive.markers.put(
            planar_marker(
                f"robot/{name}",
                geometry_msgs.Pose(),
                description=name,
                scale=reach + 0.6,
                visuals=[visual(Marker.CYLINDER, (reach + 0.3, reach + 0.3, 0.02), interactive.robot_colors.rgba(name, 0.35))],
                frame=manager.base_frame,
            ),
            on_pose=self._robot_mover(manager),
        )

    def _on_robot(self, manager: RobotManager) -> None:
        interactive = self._ctx.interactive
        assert interactive is not None, "rviz_ui needs the env's interactive context"
        name = manager.name
        if self._ctx.robots.get(name) is not manager:
            return
        if name in self._task.tm_robots.goal_editing_robots():
            interactive.markers.erase(f"goal/{name}")
            return
        goal = manager.goal
        if goal is not None:
            self._place(f"goal/{name}", goal.to_msg(), f"{name} goal", interactive.robot_colors.rgba(name, 0.95), self._goal_setter(name))

    def _place(self, marker: str, pose: geometry_msgs.Pose, description: str, color: tuple[float, float, float, float], on_pose: Callable[[geometry_msgs.Pose], Awaitable[None]]) -> None:
        interactive = self._ctx.interactive
        assert interactive is not None, "rviz_ui needs the env's interactive context"
        hub = interactive.markers
        if marker in hub.names(marker):
            hub.set_pose(marker, pose)
            return
        hub.put(planar_marker(marker, pose, description=description, scale=0.8, visuals=_handle_visuals(color)), on_pose=on_pose)

    def _robot_mover(self, manager: RobotManager) -> Callable[[geometry_msgs.Pose], Awaitable[geometry_msgs.Pose]]:
        interactive = self._ctx.interactive
        assert interactive is not None, "rviz_ui needs the env's interactive context"

        async def move(msg: geometry_msgs.Pose) -> geometry_msgs.Pose:
            base = interactive.tf_buffer.lookup_transform("map", manager.base_frame, rclpy.time.Time()).transform
            yaw = Orientation.from_msg(base.rotation).to_yaw()
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            target = Pose(
                Position(
                    base.translation.x + cos_yaw * msg.position.x - sin_yaw * msg.position.y,
                    base.translation.y + sin_yaw * msg.position.x + cos_yaw * msg.position.y,
                ),
                Orientation.from_yaw(yaw + Orientation.from_msg(msg.orientation).to_yaw()),
            )
            await manager.move(self._ctx.realizer.ezilear(target))
            return geometry_msgs.Pose()

        return move

    def _goal_setter(self, name: str) -> Callable[[geometry_msgs.Pose], Awaitable[None]]:
        async def go(msg: geometry_msgs.Pose) -> None:
            pose = self._ctx.realizer.ezilear(Pose.from_msg(msg))
            await self._task.submit_task(TaskRequest(phases=[GoToPhase(pose=pose)]), name)

        return go

    def _reset_timeout(self, index: int):
        self._timeouts[index] = self.node.sim_time

    def _to_abstract(self, frame_id: str, pose: Pose) -> Pose:
        return self._ctx.realizer.ezilear(pose) if frame_id == 'map' else pose

    async def _cb_set_position(self, pos: geometry_msgs.PoseWithCovarianceStamped):
        await self._task.set_robot_position(self._to_abstract(pos.header.frame_id, Pose.from_msg(pos.pose.pose)))

    async def _cb_set_goal(self, pos: geometry_msgs.PoseStamped):
        await self._task.set_robot_goal(self._to_abstract(pos.header.frame_id, Pose.from_msg(pos.pose)))

    def _cb_new_scenario(self, *args: object, **kwargs: object) -> None:
        self._ctx.episode.force_reset()
