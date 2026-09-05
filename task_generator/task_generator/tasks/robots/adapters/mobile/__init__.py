from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import geometry_msgs.msg
from arena_viz.kinds import DisplayKind

from task_generator.manager.robot_manager.collision_tracker import CollisionTrackerNode
from task_generator.tasks.robots.adapters import ADAPTERS, Adapter, AdapterDisplayHint

if TYPE_CHECKING:
    from task_generator.manager.robot_manager.robot_manager import RobotManager
    from task_generator.tasks.robots.adapters import ResetContext
    from task_generator.tasks.robots.request import GoToPhase


class MobileAdapter(Adapter):
    def __init__(self, *args: object, **kwargs: object):
        super().__init__(*args, **kwargs)
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
            topic_must_exist=True,
        ),
    )

    async def on_reset(self, robot: RobotManager, ctx: ResetContext) -> None:
        if ctx.start_pose is not None:
            await robot.move(ctx.start_pose)

    def _resolve_tolerances(self, phase: GoToPhase, robot: RobotManager) -> tuple[float, float]:
        """Effective (distance, yaw) tolerances, resolved exactly as the tier-3
        completion check resolves them."""
        conf = robot.node.conf.Robot
        dist = phase.tolerance_radius if phase.tolerance_radius is not None else conf.GOAL_TOLERANCE_RADIUS.value
        if phase.tolerance_angle is not None:
            ang = phase.tolerance_angle
        elif self.controls_orientation:
            ang = conf.GOAL_TOLERANCE_ANGLE.value
        else:
            ang = 0.0
        return float(dist), float(ang)

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


@ADAPTERS["mobile"].register("none")
def _load_none() -> type[Adapter]:
    from .none import NoneAdapter

    return NoneAdapter


@ADAPTERS["mobile"].register("test-collision")
def _load_test_collision() -> type[Adapter]:
    from .test_collision import TestCollisionAdapter

    return TestCollisionAdapter
