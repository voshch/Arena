"""``tm_robots:=edge_case`` — routed traversals with obstruction as an outcome.

Built to pair with ``tm_obstacles:=edge_case``. It inherits its route from the robots
scenario mode, then adds the two things no existing robot mode provides: repeated
traversals through the population, and a terminal *blocked* outcome.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable

import attrs
from arena_rclpy_mixins.ROSParamServer import ROSParamT

from task_generator.manager.robot_manager.robot_manager import RobotManager
from task_generator.shared import Pose
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, default_scenario
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.request import GoToPhase, TaskPhase, TaskRequest
from task_generator.tasks.robots.scenario.impl import TM_Scenario

#: Sentinel meaning "no progress seen yet" for the per-robot watchdog.
_NEVER = -1.0


@attrs.define
class _Progress:
    """Per-robot progress watchdog state.

    Tracks the closest the robot has ever come to its current goal, and when that last
    improved. Distance-to-goal rather than raw speed: a robot legitimately pauses, rotates
    in place, or backs out of a doorway, and none of that is obstruction. Failing to get
    any closer for a sustained window is.
    """

    best_distance: float = math.inf
    last_progress_at: float = _NEVER
    phase_key: tuple[float, float] | None = None

    def reset(self, now: float) -> None:
        self.best_distance = math.inf
        self.last_progress_at = now
        self.phase_key = None


class TM_EdgeCase(TM_Scenario):
    """Robot counterpart to the edge-case obstacle mode.

    Three terminal states, each mapping to a real episode outcome:

    * every traversal completed          -> episode SUCCESS
    * robot obstructed past the timeout  -> episode FAILED, reason "blocked"
    * overall timeout with work left     -> episode FAILED, reason "timed out"

    The base class reports a timeout as plain completion, which reads as SUCCESS. For
    edge-case work a robot that ran out of clock without finishing is a failure, so this
    mode distinguishes the two.
    """

    _traversals: ROSParamT[int]
    _blocked_distance: ROSParamT[float]
    _blocked_timeout: ROSParamT[float]
    _on_blocked: ROSParamT[str]

    # Route
    # -----

    @staticmethod
    def holding(now_s: float, since_s: float, hold_s: float) -> bool:
        """Whether the robot is still in its hold: `hold_s` sim seconds from the reset."""
        return hold_s > 0.0 and (now_s - since_s) < hold_s

    @staticmethod
    def traversal_phases(start: Pose, goal: Pose, traversals: int, retries: int = 1) -> list[TaskPhase]:
        """Alternating goal/start legs for ``traversals`` round trips.

        ``traversals`` counts arrivals at the goal, so N traversals is ``2N - 1`` legs:
        out, back, out, ... The robot ends at the goal rather than the start, which keeps a
        single traversal identical to what the scenario mode already does. Each leg is
        submitted ``retries`` times in a row: a phase nav2 gives up on is retried rather than
        skipped, since the phase policy is `continue`.
        """
        if traversals < 1:
            raise ValueError(f"traversals must be >= 1, got {traversals}")
        if retries < 1:
            raise ValueError(f"retries must be >= 1, got {retries}")
        phases: list[TaskPhase] = []
        for i in range(2 * traversals - 1):
            phases.extend(GoToPhase(pose=goal if i % 2 == 0 else start) for _ in range(retries))
        return phases

    # Obstruction watchdog
    # --------------------

    @staticmethod
    def is_blocked(
        progress: _Progress,
        distance: float,
        now: float,
        *,
        min_improvement: float,
        timeout: float,
    ) -> bool:
        """Advance the watchdog and report whether the robot counts as obstructed.

        Pure apart from mutating ``progress``, so it is testable without a robot.
        """
        if progress.last_progress_at == _NEVER:
            progress.last_progress_at = now

        if distance < progress.best_distance - min_improvement:
            progress.best_distance = distance
            progress.last_progress_at = now
            return False

        return (now - progress.last_progress_at) >= timeout

    @staticmethod
    def near_goal_accepted(
        progress: _Progress,
        distance: float,
        now: float,
        *,
        within: float,
        min_improvement: float,
        after: float,
    ) -> bool:
        """Whether a leg counts as done although nav2 has not said so: the robot is within
        `within` of the goal and has not got `min_improvement` closer for `after` seconds.
        That is a goal cell the planner refuses - inside the robot's inflation next to a
        desk - and the robot is as good as there. Shares the watchdog's progress record."""
        if within <= 0.0 or distance > within:
            return False
        return TM_EdgeCase.is_blocked(progress, distance, now, min_improvement=min_improvement, timeout=after)

    def _goal_for(self, robot: RobotManager) -> Pose | None:
        """Current leg's target, or None when the robot has no active GoTo phase."""
        request = robot._current_request
        if request is None or robot._phase_index >= len(request.phases):
            return None
        phase = request.phases[robot._phase_index]
        return phase.pose if isinstance(phase, GoToPhase) else None

    #: Metres from its start at which a robot counts as under way (the obstacles mode's case
    #: clock uses the same figure, `waypoint_driver.ROBOT_DEPART_M`).
    DEPART_M = 0.3

    def _log_departures(self) -> None:
        """Log once per episode when each robot is under way: the recording tools start the
        capture on that line, and a scenario without runtime effects has no waypoint driver
        to say so."""
        now = float(self.node.sim_time.sec) + self.node.sim_time.nanosec * 1e-9
        for name, robot in self._ctx.robots.items():
            if name in self._departed or robot.pose is None or not getattr(robot, "start_assigned", True):
                continue
            start = robot.start_pos
            here = self._ctx.environment_manager.ezilear(robot.pose)
            moved = math.dist((here.position.x, here.position.y), (start.position.x, start.position.y))
            if moved >= self.DEPART_M:
                self._departed.add(name)
                self._logger.warn(f"edge_case robots: {name!r} under way ({moved:.2f} m from its start) {now - self._last_reset:.1f}s after the reset")

    def _leg_progress(self, name: str, pose: Pose, goal: Pose, now: float) -> tuple[_Progress, float]:
        """The robot's progress record for its current leg (reset when the leg changes) and
        its distance to the leg's goal."""
        progress = self._progress.setdefault(name, _Progress())
        key = (goal.position.x, goal.position.y)
        if progress.phase_key != key:
            # New leg: the previous leg's best distance says nothing about this one.
            progress.reset(now)
            progress.phase_key = key
        return progress, math.dist((pose.position.x, pose.position.y), (goal.position.x, goal.position.y))

    def _check_blocked(self) -> str | None:
        """Reason string if any robot is obstructed, else None."""
        now = float(self.node.sim_time.sec) + self.node.sim_time.nanosec * 1e-9
        for name, robot in self._ctx.robots.items():
            pose = robot.pose
            goal = self._goal_for(robot)
            if pose is None or goal is None:
                continue

            progress, distance = self._leg_progress(name, pose, goal, now)
            if self.is_blocked(
                progress,
                distance,
                now,
                min_improvement=self._blocked_distance.value,
                timeout=self._blocked_timeout.value,
            ):
                return (
                    f"robot {name!r} blocked: no progress toward ({goal.position.x:.2f}, {goal.position.y:.2f}) "
                    f"for {self._blocked_timeout.value:.0f}s, still {distance:.2f}m away"
                )
        return None

    # Lifecycle
    # ---------

    async def reset(self, **kwargs: object) -> None:
        # Sets _start_poses, forbids the endpoints so the crowd cannot spawn on them, and
        # submits the scenario's own phases. We keep all of that and re-submit with the
        # traversal expansion on top.
        await super().reset(**kwargs)
        # `done` and the under-way warning measure from the reset time.
        self._last_reset = self.node.sim_time.sec

        self._progress = {}
        self._departed: set[str] = set()
        traversals = max(1, int(self._traversals.value))
        retries = max(1, int(self._retries.value))
        hold = max(0.0, float(self._hold.value))
        self._hold_s = hold
        self._hold_since = float(self.node.sim_time.sec)
        for name, robot in self._ctx.robots.items():
            start = self._start_poses.get(name)
            # `route_goal`, not `goal_pos`: these phases go straight back into submit_task,
            # which realizes whatever it is handed. `goal_pos` is already map-frame, so
            # re-submitting it would realize it a second time and send the robot out of the
            # world. Both `_start_poses` and `route_goal` are abstract, as submit_task expects.
            goal = robot.route_goal
            if start is None:
                continue
            phases = self.traversal_phases(start, goal, traversals, retries)
            hold_phase: TaskPhase | None = None
            if hold > 0.0:
                # Stay put first. The hold is phase 0 of the *same* request, gated by a done
                # predicate on sim time: the robot's route goal stays the real goal (the
                # obstacles mode designs against it at reset - a separate hold request made it
                # look like start == goal), and nothing has to be re-submitted later. The case
                # clock waits for the robot to set off, so the hold shifts no onset.
                hold_phase = GoToPhase(pose=start)
                phases = [hold_phase, *phases]
            await robot.submit_task(TaskRequest(phases=phases, done_predicate=self._phase_gate(name, hold_phase, hold, self._hold_since)))
            held = f"holds {hold:.0f}s at its start, then " if hold_phase is not None else "-> "
            self._logger.info(f"edge_case robots: {name!r} {held}{traversals} traversal(s) x {retries} attempt(s), {len(phases) - (1 if hold_phase else 0)} leg(s)")

    def _phase_gate(self, name: str, hold_phase: TaskPhase | None, hold: float, since: float) -> Callable[[object, TaskPhase], bool | None]:
        """Done predicate: the hold phase is not done until `hold` sim seconds have passed; a
        GoTo leg is done early when the robot is within `accept_within` of its goal and has
        stopped getting closer (`near_goal_accepted`); everything else is left to its adapter
        (None)."""
        released = False
        accepted: set[tuple[float, float]] = set()

        def gate(robot_manager: object, phase: TaskPhase) -> bool | None:
            nonlocal released
            if hold_phase is not None and phase is hold_phase:
                if self.holding(float(self.node.sim_time.sec), since, hold):
                    return False
                if not released:
                    released = True
                    self._progress = {}
                    self._logger.info(f"edge_case robots: {name!r} released after {hold:.0f}s")
                return True
            if not isinstance(phase, GoToPhase):
                return None
            robot = self._ctx.robots.get(name)
            pose = robot.pose if robot is not None else None
            if pose is None:
                return None
            now = float(self.node.sim_time.sec) + self.node.sim_time.nanosec * 1e-9
            progress, distance = self._leg_progress(name, pose, phase.pose, now)
            if progress.phase_key in accepted:
                return True
            if self.near_goal_accepted(progress, distance, now, within=float(self._accept_within.value), min_improvement=float(self._blocked_distance.value), after=float(self._accept_after.value)):
                accepted.add(progress.phase_key)  # type: ignore[arg-type]
                self._logger.warn(f"edge_case robots: {name!r} accepted its leg {distance:.2f} m short of ({phase.pose.position.x:.2f}, {phase.pose.position.y:.2f}): no closer for {float(self._accept_after.value):.0f}s, the goal cell is unplannable")
                return True
            return None

        return gate

    def _log_outcomes(self) -> None:
        """One line per robot when its request is spent: reached the goal, or ran out of attempts short of it.

        A phase also ends when nav2 gives the goal up (an aborted action result is a finished phase), so
        "every phase done" is not "arrived"; the analysis tools read this line for the arrival rate."""
        for name, robot in self._ctx.robots.items():
            request = robot._current_request  # noqa: SLF001 - the mode owns the request it submitted
            goals = [ph.pose for ph in (request.phases if request is not None else []) if isinstance(ph, GoToPhase)]
            pose = robot.pose
            if not goals or pose is None:
                self._logger.warn(f"edge_case robots: {name!r} request spent with no goal to measure against (request={'none' if request is None else len(request.phases)}, phase index {robot._phase_index}, pose {'known' if pose is not None else 'unknown'})")  # noqa: SLF001
                continue
            goal = goals[-1]
            distance = math.hypot(pose.position.x - goal.position.x, pose.position.y - goal.position.y)
            tolerance = max(float(self._accept_within.value), float(self.node.conf.Robot.GOAL_TOLERANCE_RADIUS.value))
            elapsed = self.node.sim_time.sec - self._last_reset
            if distance <= tolerance:
                self._logger.warn(f"edge_case robots: {name!r} reached its goal ({distance:.2f} m off) after {elapsed:.0f}s")
            else:
                self._logger.warn(f"edge_case robots: {name!r} gave up {distance:.2f} m short of ({goal.position.x:.2f}, {goal.position.y:.2f}) after {elapsed:.0f}s: every attempt ended without arriving")

    @property
    async def done(self) -> bool:
        """Terminate on completion, obstruction, or timeout — distinguishing all three."""
        self._log_departures()
        if self.holding(float(self.node.sim_time.sec), self._hold_since, self._hold_s):
            return False  # standing still on purpose: neither blocked nor finished
        reason = self._check_blocked()
        if reason is not None:
            if self._on_blocked.value == "abort":
                self._logger.warn(f"edge_case robots: {reason}")
                self._ctx.abort_episode(reason)
                return True
            self._logger.warn(f"edge_case robots: {reason} (on_blocked=continue, not ending the episode)")

        if not self._ctx.robots:
            return False

        finished = all(await asyncio.gather(*(robot.is_done for robot in self._ctx.robots.values())))
        if finished:
            self._log_outcomes()
            return True

        elapsed = self.node.sim_time.sec - self._last_reset
        if elapsed > self.node.conf.Robot.TIMEOUT.value:
            # The base class returns True here with no abort, which the node reports as
            # SUCCESS. Unfinished work that ran out of clock is a failure.
            self._ctx.abort_episode(f"timed out after {elapsed}s with traversals incomplete")
            return True

        return False

    def __init__(self, **kwargs: object) -> None:
        # Deliberately skips TM_Scenario.__init__: it would bind the scenario file to
        # `self.namespace('file')`, i.e. `task.edge_case_robot.file`, and this mode would
        # then read a different scenario from the one `tm_obstacles:=edge_case` is
        # populating. Bind to the shared `task.scenario.file` instead so the route and the
        # crowd always come from the same scenario.
        TM_Robots.__init__(self, **kwargs)
        self._config = self.node.ROSParam[str](
            _REGISTRY_NAMESPACE("scenario")("file"),
            default_scenario(self._ctx.world_manager.loaded_world),
        )

        self._progress: dict[str, _Progress] = {}
        self._departed: set[str] = set()
        self._traversals = self.node.ROSParam[int](self.namespace("traversals"), 1)
        self._retries = self.node.ROSParam[int](self.namespace("retries"), 1)
        self._hold = self.node.ROSParam[float](self.namespace("hold"), 0.0)
        self._hold_s = 0.0
        self._hold_since = 0.0
        self._accept_within = self.node.ROSParam[float](self.namespace("accept_within"), 0.6)
        self._accept_after = self.node.ROSParam[float](self.namespace("accept_after"), 5.0)
        self._blocked_distance = self.node.ROSParam[float](self.namespace("blocked_distance"), 0.25)
        self._blocked_timeout = self.node.ROSParam[float](self.namespace("blocked_timeout"), 15.0)
        self._on_blocked = self.node.ROSParam[str](self.namespace("on_blocked"), "abort")
