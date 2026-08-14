"""TM_Characterization: open-loop maneuver sweep publishing exact ``cmd_vel`` profiles
through each robot's rated envelope, without navigation goals or planners. Every maneuver
is tagged on ``<robot_ns>/characterization_phase`` so offline calculators can map
energy/acoustic samples to working points. Odom silent beyond ``ODOM_STALL_TIMEOUT_S``
zeroes ``cmd_vel`` and aborts the episode.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import traceback
from collections.abc import Callable

import rclpy.publisher
import rclpy.subscription
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from task_generator.shared import Pose
from task_generator.tasks.robots import TM_Robots

from .schedule import (
    CONTROL_RATE_HZ,
    MAX_SCHEDULE_DURATION_S,
    ODOM_STALL_TIMEOUT_S,
    Phase,
    build_schedule,
    resolve_envelope,
    schedule_duration,
)

# The robot stack's cmd_vel consumers (twist stamper / velocity smoother)
# subscribe RELIABLE and reject a BEST_EFFORT publisher ("No messages will be
# sent to it", verified in env logs). The recorder's BEST_EFFORT subscription
# is compatible with a RELIABLE publisher.
_CMD_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
_ODOM_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)


@dataclasses.dataclass
class _Run:
    """Per-robot progress through that robot's own schedule."""

    schedule: list[Phase]
    phase_idx: int = 0
    phase_start: float = 0.0
    published: set[str] = dataclasses.field(default_factory=set)
    finished: bool = False


class TM_Characterization(TM_Robots):
    """Open-loop sweep: exact cmd_vel profiles through each robot's envelope."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._runs: dict[str, _Run] = {}
        self._finished = False
        self._driver: asyncio.Task | None = None
        self._twist_pubs: dict[str, rclpy.publisher.Publisher] = {}
        self._phase_pubs: dict[str, rclpy.publisher.Publisher] = {}
        self._odom_subs: dict[str, rclpy.subscription.Subscription] = {}
        self._last_odom: dict[str, float] = {}

    async def reset(self, **kwargs: object) -> None:
        await super().reset(**kwargs)
        await self._cancel_driver()
        self._finished = False
        self._runs = {}

        # Place every robot near the map centre (a free cell) so the sweep's
        # out-and-back legs stay inside the arena.
        self._start_poses = {}
        robot_positions = list(kwargs.get("ROBOT_POSITIONS", []) or [])
        for idx, manager in enumerate(self._ctx.robots.values()):
            if idx < len(robot_positions):
                self._start_poses[manager.name] = robot_positions[idx][0]
            else:
                self._start_poses[manager.name] = await self._centre_placement()
            self._logger.info(f"TM_Characterization: {manager.name} placed at {self._start_poses[manager.name].position}")

        self._sync_entities()

        now = self._sim_now()
        for manager in self._ctx.robots.values():
            envelope = resolve_envelope(manager.model_name)
            schedule = build_schedule(vx_max=envelope["vx_max"], wz_max=envelope["wz_max"])
            self._runs[manager.name] = _Run(schedule=schedule, phase_start=now)
            self._last_odom[manager.name] = now
            self._logger.info(f"TM_Characterization: {manager.name} (model={manager.model_name}) {len(schedule)} phases, {schedule_duration(schedule):.0f}s (vx up to {envelope['vx_max']:.2f} m/s, wz up to {envelope['wz_max']:.2f} rad/s)")

        if self._runs:
            self._driver = asyncio.create_task(self._drive())

    def _sync_entities(self) -> None:
        """Create this mode's ROS entities once per robot and drop departed ones."""
        current = self._ctx.robots
        for name in [n for n in self._twist_pubs if n not in current]:
            self.node.destroy_publisher(self._twist_pubs.pop(name))
            self.node.destroy_publisher(self._phase_pubs.pop(name))
            self.node.destroy_subscription(self._odom_subs.pop(name))
            self._last_odom.pop(name, None)

        for manager in current.values():
            if manager.name in self._twist_pubs:
                continue
            # manager.namespace is the full robot namespace (the fleet itself
            # publishes ns=str(mgr.namespace)); manager.frame is only the
            # robot's short name and would point at the wrong topics.
            ns = str(manager.namespace)
            self._twist_pubs[manager.name] = self.node.create_publisher(Twist, f"{ns}/cmd_vel", _CMD_QOS)
            self._phase_pubs[manager.name] = self.node.create_publisher(String, f"{ns}/characterization_phase", _CMD_QOS)
            self._odom_subs[manager.name] = self.node.create_subscription(Odometry, f"{ns}/odom", self._make_odom_cb(manager.name), _ODOM_QOS)

    async def _centre_placement(self) -> Pose:
        """A free cell closest to the map centre (fallback: random placement)."""
        from task_generator.manager.world_manager.world_manager import _occupancy_to_available
        from task_generator.shared import Orientation, Position
        from task_generator.tasks.robots._placement import random_placement

        try:
            import numpy as np

            wm = self._ctx.world_manager
            grid = wm.map.occupancy.grid
            rows, cols = grid.shape
            available = _occupancy_to_available(grid, 0.0)
            if len(available) > 0:
                dist = np.abs(available[:, 0] - rows // 2) + np.abs(available[:, 1] - cols // 2)
                row, col = available[int(np.argmin(dist))]
                pos = wm.map.tf_grid2pos((int(row), int(col)))
                return Pose(Position(pos.x, pos.y, 0.0), orientation=Orientation.from_yaw(0.0))
        except Exception as e:
            self._logger.warning(f"TM_Characterization centre placement failed ({e!r}), using random")
        return await random_placement(self._ctx)

    def _make_odom_cb(self, robot_name: str) -> Callable[[Odometry], None]:
        def _cb(msg: Odometry) -> None:
            del msg
            # A queued callback must not resurrect a robot dropped at reset.
            if robot_name in self._last_odom:
                self._last_odom[robot_name] = self._sim_now()

        return _cb

    def _sim_now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _publish_zero(self, robot_name: str | None = None) -> None:
        pubs = self._twist_pubs.values() if robot_name is None else [self._twist_pubs[robot_name]]
        for pub in pubs:
            pub.publish(Twist())

    def _abort(self, reason: str) -> None:
        """Zero cmd_vel and fail the episode so the record is not a completion."""
        self._finished = True
        self._publish_zero()
        self._logger.error(f"TM_Characterization aborting sweep: {reason}")
        self._ctx.abort_episode(f"characterization sweep aborted: {reason}")

    async def _cancel_driver(self) -> None:
        if self._driver is None:
            return
        self._driver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._driver
        self._driver = None

    async def _drive(self) -> None:
        """Advance every robot's schedule on sim time and publish exact cmd_vel profiles."""
        run_start = self._sim_now()

        try:
            while not self._finished:
                now = self._sim_now()

                stalled = [name for name, last in self._last_odom.items() if now - last > ODOM_STALL_TIMEOUT_S]
                if stalled:
                    self._abort(f"odometry of {', '.join(sorted(stalled))} silent for >{ODOM_STALL_TIMEOUT_S}s")
                    return

                if now - run_start > MAX_SCHEDULE_DURATION_S:
                    self._abort(f"schedule exceeded the {MAX_SCHEDULE_DURATION_S:.0f}s ceiling")
                    return

                for name, run in self._runs.items():
                    self._step(name, run, now)

                if all(run.finished for run in self._runs.values()):
                    self._publish_zero()
                    self._finished = True
                    self._logger.info("TM_Characterization: all schedules complete")
                    return

                await asyncio.sleep(1.0 / CONTROL_RATE_HZ)
        except asyncio.CancelledError:
            self._publish_zero()
            raise
        except Exception as e:
            self._logger.error(f"TM_Characterization driver crashed:\n{traceback.format_exc()}")
            self._abort(f"driver crashed: {e!r}")

    def _step(self, robot_name: str, run: _Run, now: float) -> None:
        """Publish one control tick for a single robot from its own schedule."""
        if run.finished:
            return

        while run.phase_idx < len(run.schedule) and now - run.phase_start >= run.schedule[run.phase_idx].duration_s:
            run.phase_idx += 1
            run.phase_start = now

        if run.phase_idx >= len(run.schedule):
            run.finished = True
            self._publish_zero(robot_name)
            self._logger.info(f"TM_Characterization: {robot_name} schedule complete")
            return

        phase = run.schedule[run.phase_idx]
        if phase.name not in run.published:
            run.published.add(phase.name)
            marker = String()
            marker.data = phase.name
            self._phase_pubs[robot_name].publish(marker)
            self._logger.info(f"TM_Characterization {robot_name} phase {phase.name} (vx={phase.vx_target} wz={phase.wz_target} dt={phase.duration_s}s)")

        self._twist_pubs[robot_name].publish(self._target_twist(phase, now - run.phase_start))

    def _target_twist(self, phase: Phase, elapsed_s: float) -> Twist:
        twist = Twist()
        if phase.kind.value == "angular":
            twist.angular.z = phase.wz_target
        elif phase.ramp_s > 0.0:
            frac = min(max(elapsed_s / phase.ramp_s, 0.0), 1.0)
            if phase.kind.value == "ramp_down":
                twist.linear.x = phase.vx_target * (1.0 - frac)
            else:
                twist.linear.x = phase.vx_target * frac
        else:
            twist.linear.x = phase.vx_target
        return twist

    async def set_goal(self, pose: Pose):
        """Open-loop mode ignores navigation goals."""
        del pose

    async def set_position(self, pose: Pose):
        """Open-loop mode ignores pose overrides."""
        del pose

    @property
    async def done(self) -> bool:
        return self._finished

    async def teardown(self) -> None:
        """Cancel the driver, zero cmd_vel and drop this mode's ROS entities."""
        self._finished = True
        await self._cancel_driver()
        self._publish_zero()
        self._runs = {}
        self._last_odom = {}
        for name in list(self._twist_pubs):
            self.node.destroy_publisher(self._twist_pubs.pop(name))
            self.node.destroy_publisher(self._phase_pubs.pop(name))
            self.node.destroy_subscription(self._odom_subs.pop(name))
