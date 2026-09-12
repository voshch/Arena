"""Rewriting agent routes *during* an episode.

The ROS half of :mod:`waypoints`. Built on `arena_humansim_msgs/srv/SetWaypoints` through
:meth:`BaseHumanSimulator.set_agent_waypoints`. Modelled on :mod:`object_driver`, and for the
same reasons:

* **Its own node, on sim time.** The tick is gated while the simulator is paused, so an
  `at: 8.0` rally does not fire the instant a long pause ends.
* **Its own failure boundary.** A reroute that cannot be sent disables the driver and says
  so; it never takes the episode down.

Several plans are armed at once, each with its own gate, clock and outcome: a crowd rally at
8 s and a hold on the injected walker when the robot reaches it are independent effects and
are reported independently.

**The case clock starts when the robot is under way** (`depart_after_m`): every plan's `at`
counts from the tick the robot has moved that far from where it stood when the plans were
armed, not from the reset. nav2 takes a few seconds to produce its first plan and the robot
mode may hold the robot on purpose; either way the case waits for the robot rather than the
robot for the case - an onset "eight seconds in" is eight seconds into the drive.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import Any

import attrs
import rclpy.node
from rclpy.parameter import Parameter

from .triggers import Trigger, distance_to
from .waypoints import Mode, Plan, Point, WaypointOutcome

Work = Coroutine[Any, Any, object]

#: How often plans are checked, Hz. Deliberately slower than the object driver's 5 Hz: every
#: re-issue costs the pedestrian backend a path recomputation.
DEFAULT_RATE_HZ = 2.0

#: Metres the robot must have moved from its armed position before the case clock starts.
#: Below nav2's goal tolerance, so a robot that has genuinely set off counts at once.
ROBOT_DEPART_M = 0.3
#: A displacement of this much between two ticks (2 Hz: 3 m/s, beyond any of our robots) is a
#: teleport - the reset putting the robot on its start after the plans were armed - and moves
#: the origin rather than starting the clock.
ROBOT_TELEPORT_M = 1.5
#: With a known start, seconds to wait for the robot to be seen on it (the reset's teleport)
#: before falling back to "under way from wherever it is first seen".
ROBOT_SETTLE_S = 10.0


@attrs.define
class Armed:
    """One plan under execution."""

    plan: Plan
    trigger: Trigger | None = None
    reference: Point | None = None
    #: Sim time at the first tick after arming, i.e. when the episode's clock started.
    armed_at: float | None = None
    #: When this plan's own clock started: `armed_at`, or the trigger's firing.
    t0: float | None = None
    last: float | None = None
    issues: int = 0
    released: bool = False
    done: bool = False
    outcome: WaypointOutcome = attrs.Factory(WaypointOutcome)


class WaypointDriver(rclpy.node.Node):
    """Ticks every armed plan and rewrites the routes it owns."""

    def __init__(
        self,
        namespace: str,
        *,
        reroute: Callable[[Mapping[str, Sequence[Point]]], Work],
        robot_pose: Callable[[], Point | None],
        submit: Callable[[Work], None],
        rate_hz: float = DEFAULT_RATE_HZ,
        retune: Callable[[Mapping[str, Mapping[str, Any]]], Work] | None = None,
        depart_after_m: float | None = None,
    ) -> None:
        super().__init__(
            "edge_case_waypoint_driver",
            namespace=namespace,
            use_global_arguments=False,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self._reroute = reroute
        self._retune = retune
        self._robot_pose = robot_pose
        self._submit = submit
        self._armed: list[Armed] = []
        self._broken = False
        #: None: clocks start on the first tick after arming (tests, no robot to wait for).
        self._depart_after_m = depart_after_m
        self._origin: Point | None = None
        self._origin_given = False
        #: With a known start: the robot has been seen on it (the reset placed it there).
        self._settled = False
        self._last_seen: Point | None = None
        self._armed_at: float | None = None
        self.clock_started_at: float | None = None
        self._timer = self.create_timer(1.0 / max(rate_hz, 1e-3), self._tick)

    # Lifecycle
    # ---------

    def arm(self, entries: Sequence[tuple[Plan, Trigger | None, Point | None]], origin: Point | None = None) -> None:
        """Load plans, starting each clock at the next tick.

        The clock starts on the first *tick* rather than here, because `arm` is called at the
        end of a reset while the simulator may still be paused. Arming with nothing disarms.
        `origin` is where the robot starts this episode (abstract frame); with it the robot is
        under way once `depart_after_m` from there, without it from where it is first seen -
        and a jump of `ROBOT_TELEPORT_M` between ticks (the reset placing it) moves that origin.
        """
        self._armed = [
            Armed(
                plan=plan, trigger=trigger, reference=reference,
                outcome=WaypointOutcome(
                    label=plan.label, mode=str(plan.mode), agents=len(plan.homes),
                    trigger=trigger.describe(reference) if trigger else "",
                ),
            )
            for plan, trigger, reference in entries
        ]
        self._broken = False
        self._origin = origin
        self._origin_given = origin is not None
        self._settled = False
        self._last_seen = None
        self._armed_at = None
        self.clock_started_at = None

    def outcomes(self) -> list[WaypointOutcome]:
        return [a.outcome for a in self._armed]

    def shutdown(self) -> None:
        self._timer.cancel()
        self.destroy_node()

    # Ticking
    # -------

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _tick(self) -> None:
        if self._broken or not self._armed:
            return
        now = self._now()
        if now <= 0.0:
            return  # sim clock has not started publishing yet
        robot = self._robot_pose()
        if not self._robot_under_way(now, robot):
            return
        for armed in self._armed:
            if not armed.done:
                self._tick_one(armed, now, robot)

    def _robot_under_way(self, now: float, robot: Point | None) -> bool:
        """Whether the case clock runs: at once without `depart_after_m`, else from the tick
        the robot has moved that far from where it stood when the plans were armed."""
        if self.clock_started_at is not None:
            return True
        if self._depart_after_m is None:
            self.clock_started_at = now
            return True
        if self._armed_at is None:
            self._armed_at = now
        if robot is None:
            return False
        jumped = self._last_seen is not None and (distance_to(robot, self._last_seen) or 0.0) >= ROBOT_TELEPORT_M
        self._last_seen = robot
        if self._origin_given and not self._settled:
            # Plans are armed before the reset has placed the robot: wait until it is seen on
            # its start, or give up on the start after a while and take it from where it is.
            near = distance_to(robot, self._origin)
            if near is not None and near < self._depart_after_m:
                self._settled = True
            elif now - self._armed_at >= ROBOT_SETTLE_S:
                self.get_logger().warn(f"edge_case: robot never seen on its start {self._origin} ({near:.2f} m off after {ROBOT_SETTLE_S:.0f}s); taking its departure from here")
                self._origin_given = False
                self._origin = robot
            return False
        if not self._origin_given and (self._origin is None or jumped):
            self._origin = robot  # first sighting, or the reset just put the robot on its start
            return False
        moved = distance_to(robot, self._origin)
        if moved is None or moved < self._depart_after_m:
            return False
        self.clock_started_at = now
        self.get_logger().warn(
            f"edge_case: robot under way ({moved:.2f} m from its start, {now - self._armed_at:.1f}s after the plans were armed) "
            f"at sim t={now:.1f}s - the case clock starts now"
        )
        return True

    def _tick_one(self, armed: Armed, now: float, robot: Point | None) -> None:
        plan = armed.plan
        if armed.armed_at is None:
            armed.armed_at = now

        if armed.t0 is None:
            if armed.trigger is not None:
                distance = distance_to(robot, armed.reference)
                if not armed.trigger.satisfied(distance):
                    return
                armed.outcome.t_trigger = now - armed.armed_at
                self.get_logger().warn(
                    f"edge_case: {plan.label or plan.mode} trigger fired at t={armed.outcome.t_trigger:.1f}s - "
                    f"{armed.trigger.describe(armed.reference)}, robot at {distance:.2f}m"
                )
            armed.t0 = now
            return

        elapsed = now - armed.t0
        if elapsed < plan.at:
            return

        # Release: a hold or a tracking plan that has run its duration.
        if plan.mode in (Mode.HOLD, Mode.TRACK_ROBOT) and armed.issues and plan.duration > 0 and elapsed >= plan.at + plan.duration:
            armed.done = True
            armed.outcome.t_release = elapsed
            routes = plan.resume_routes()
            if routes:
                self._send(armed, routes, elapsed)
            return

        if armed.last is not None and elapsed - armed.last < plan.period:
            return

        if plan.mode is Mode.RETUNE:
            armed.done = True
            armed.issues += 1
            armed.outcome.t_first = elapsed
            self._send_retune(armed, elapsed)
            return

        routes = self._routes(armed, robot)
        if routes is None:
            return

        armed.last = elapsed
        armed.issues += 1
        if armed.outcome.t_first is None:
            armed.outcome.t_first = elapsed
        if plan.mode is Mode.RALLY or (plan.mode is Mode.SHUFFLE and plan.once):
            armed.done = True
        if plan.mode is Mode.HOLD and plan.duration <= 0:
            armed.done = True  # pinned for the rest of the episode
        self._send(armed, routes, elapsed)

    def _routes(self, armed: Armed, robot: Point | None) -> dict[str, list[Point]] | None:
        plan = armed.plan
        if plan.mode is Mode.RALLY:
            return {name: [plan.targets[name]] for name in plan.homes}
        if plan.mode is Mode.TRACK_ROBOT:
            if robot is None:
                return None  # respawn window; not an error
            return {name: [robot] for name in plan.homes}
        if plan.mode is Mode.HOLD:
            if armed.issues:
                return None  # already pinned; waiting for release
            return plan.hold_routes()
        return plan.shuffle_routes(armed.issues)

    def _send_retune(self, armed: Armed, elapsed: float) -> None:
        if self._retune is None:
            armed.outcome.failures += 1
            armed.outcome.reason = "no retune hook"
            return
        work = self._run_retune(armed, elapsed)
        try:
            self._submit(work)
        except Exception as exc:
            work.close()
            self._fail(f"could not schedule: {exc!r}")

    async def _run_retune(self, armed: Armed, elapsed: float) -> None:
        assert self._retune is not None
        try:
            ok = await self._retune(armed.plan.retune)
        except Exception as exc:
            armed.outcome.failures += 1
            self.get_logger().warn(f"edge_case: retune at t={elapsed:.1f}s failed ({exc!r})")
            return
        n = int(ok) if isinstance(ok, int) and not isinstance(ok, bool) else (len(armed.plan.retune) if ok else 0)
        armed.outcome.issued += n
        if n < len(armed.plan.retune):
            armed.outcome.failures += len(armed.plan.retune) - n
        self.get_logger().warn(f"edge_case: {armed.plan.label or 'retune'} at t={elapsed:.1f}s - {n} of {len(armed.plan.retune)} agent(s) retuned")

    def _send(self, armed: Armed, routes: Mapping[str, Sequence[Point]], elapsed: float) -> None:
        work = self._run(armed, routes, elapsed)
        try:
            self._submit(work)
        except Exception as exc:
            work.close()
            self._fail(f"could not schedule: {exc!r}")

    async def _run(self, armed: Armed, routes: Mapping[str, Sequence[Point]], elapsed: float) -> None:
        try:
            ok = await self._reroute(routes)
        except Exception as exc:
            armed.outcome.failures += 1
            self.get_logger().warn(f"edge_case: reroute at t={elapsed:.1f}s failed ({exc!r})")
            return
        if not ok:
            armed.outcome.failures += 1
            self.get_logger().warn(f"edge_case: reroute at t={elapsed:.1f}s reached no agent - {len(routes)} name(s) did not resolve")
            return
        armed.outcome.issued += len(routes)
        if armed.outcome.issued == len(routes):
            self.get_logger().warn(
                f"edge_case: {armed.plan.label or armed.plan.mode} fired at t={elapsed:.1f}s - {len(routes)} agent(s) rerouted"
            )

    def _fail(self, reason: str) -> None:
        self._broken = True
        for armed in self._armed:
            if not armed.done:
                armed.outcome.reason = reason
        self.get_logger().warn(f"edge_case: waypoint driver disabled for this episode ({reason})")
