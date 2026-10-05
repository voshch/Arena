from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import geometry_msgs.msg
import nav_msgs.msg
import rclpy.qos
from arena_rclpy_mixins.qos import best_effort
from arena_viz.kinds import DisplayKind
from arena_viz.style import StyleSpec

from task_generator.manager.robot_manager.collision_tracker import CollisionTrackerNode
from task_generator.tasks.robots.adapters import ADAPTERS, Adapter, AdapterDisplayHint

if TYPE_CHECKING:
    from task_generator.manager.robot_manager.robot_manager import RobotManager
    from task_generator.tasks.robots.adapters import ResetContext
    from task_generator.tasks.robots.request import GoToPhase

_TRAIL_STEP = 0.05
_TRAIL_POINTS = 2000
_TRAIL_QOS = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)


class MobileAdapter(Adapter):
    def __init__(self, *args: object, **kwargs: object):
        super().__init__(*args, **kwargs)
        self._trail = nav_msgs.msg.Path()
        self._trail.header.frame_id = "map"
        self._trail_start = self.rm.start_pos
        self._trail_pub = self.rm.node.create_publisher(nav_msgs.msg.Path, str(self.rm.namespace("trail")), _TRAIL_QOS)
        self._trail_sub = self.rm.node.create_subscription(nav_msgs.msg.Odometry, str(self.rm.namespace("odom")), self._extend_trail, best_effort(1))
        self.rm.watch(self._restart_trail)
        self._collision: CollisionTrackerNode | None = None
        mobile = self.rm.robot.model.resolve_sync().caps.mobile
        polys = mobile.polygons_dict
        footprint = mobile.footprint
        if footprint is None:
            self.rm.node.get_logger().warning(f'{self.rm.namespace}: caps/mobile.yaml has no footprint, collisions are not tracked')
        if not polys and footprint is None:
            return
        self._collision = CollisionTrackerNode(self.rm, polys or {}, footprint=footprint)
        self.rm.node.executor.add_node(self._collision)

    @property
    def controls_orientation(self) -> bool:
        return True

    cap_displays: ClassVar[tuple[AdapterDisplayHint, ...]] = (
        *Adapter.cap_displays,
        AdapterDisplayHint(
            name="Goal Pose",
            topic="{ns}/goal_pose",
            topic_type="geometry_msgs/PoseStamped",
            kind=DisplayKind.POSE,
        ),
        AdapterDisplayHint(
            name="Plan",
            topic="{ns}/plan",
            topic_type="nav_msgs/Path",
            kind=DisplayKind.PATH,
            style_json=StyleSpec(alpha=0.5, line_width=0.03).to_json(),
            topic_must_exist=True,
            robot_colored=True,
        ),
        AdapterDisplayHint(
            name="Trail",
            topic="{ns}/trail",
            topic_type="nav_msgs/Path",
            kind=DisplayKind.PATH,
            style_json=StyleSpec(line_width=0.04).to_json(),
            robot_colored=True,
        ),
    )

    async def _extend_trail(self, msg: nav_msgs.msg.Odometry) -> None:
        del msg
        pose = self.rm.pose
        if pose is None:
            return
        poses = self._trail.poses
        if poses and math.hypot(pose.position.x - poses[-1].pose.position.x, pose.position.y - poses[-1].pose.position.y) < _TRAIL_STEP:
            return
        point = geometry_msgs.msg.PoseStamped()
        point.header.frame_id = "map"
        point.pose = pose.to_msg()
        poses.append(point)
        del poses[:-_TRAIL_POINTS]
        self._publish_trail()

    def _restart_trail(self, robot: RobotManager) -> None:
        if robot.start_pos is self._trail_start:
            return
        self._trail_start = robot.start_pos
        self._trail.poses = []
        self._publish_trail()

    def _publish_trail(self) -> None:
        self._trail.header.stamp = self.rm.node.sim_time.to_msg()
        self._trail_pub.publish(self._trail)

    async def teardown(self) -> None:
        self.rm.node.destroy_subscription(self._trail_sub)
        self.rm.node.destroy_publisher(self._trail_pub)
        await super().teardown()

    async def on_reset(self, robot: RobotManager, ctx: ResetContext) -> None:
        if ctx.start_pose is not None:
            await robot.move(ctx.start_pose)

    def _resolve_tolerances(self, phase: GoToPhase, robot: RobotManager) -> tuple[float, float]:
        """Effective (distance, yaw) tolerances, resolved exactly as the tier-3
        completion check resolves them."""
        assert phase.tolerance_radius is not None and phase.tolerance_angle is not None
        return phase.tolerance_radius, phase.tolerance_angle if self.controls_orientation else 0.0

    async def publish_goal_loop(self) -> None:
        """Republish `<ns>/goal_pose` at 1Hz until the goal object changes. Override to noop if the adapter has its own goal transport."""
        rm = self.rm
        target = rm._goal_pos  # noqa: SLF001

        def publish() -> None:
            msg = geometry_msgs.msg.PoseStamped()
            msg.header.frame_id = "map"
            msg.header.stamp = rm.node.sim_time.to_msg()
            msg.pose = target.to_msg()
            rm._goal_pub.publish(msg)  # noqa: SLF001

        publish()
        with rm.node.sim_time_rate(1.0, 60) as (done, rate):
            while not done.is_set():
                await rate.get()
                if rm._goal_pos is not target:  # noqa: SLF001
                    break
                publish()


@ADAPTERS["mobile"].register("nav2")
def _load_nav2() -> type[Adapter]:
    from .nav2 import Nav2Adapter

    return Nav2Adapter


@ADAPTERS["mobile"].register("external")
def _load_external() -> type[Adapter]:
    from .external import ExternalAdapter

    return ExternalAdapter


@ADAPTERS["mobile"].register("manual")
def _load_manual() -> type[Adapter]:
    from .manual import ManualAdapter

    return ManualAdapter


@ADAPTERS["mobile"].register("rosnav_rl")
def _load_rosnav_rl() -> type[Adapter]:
    from .rosnav_rl import RosnavRlAdapter

    return RosnavRlAdapter


@ADAPTERS["mobile"].register("drl")
def _load_drl() -> type[Adapter]:
    from .drl import DrlAdapter

    return DrlAdapter


@ADAPTERS["mobile"].register("vla")
def _load_vla() -> type[Adapter]:
    from .vla import VlaAdapter

    return VlaAdapter


@ADAPTERS["mobile"].register("none")
def _load_none() -> type[Adapter]:
    from .none import NoneAdapter

    return NoneAdapter


@ADAPTERS["mobile"].register("test-collision")
def _load_test_collision() -> type[Adapter]:
    from .test_collision import TestCollisionAdapter

    return TestCollisionAdapter
