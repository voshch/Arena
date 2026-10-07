from __future__ import annotations

import asyncio
import json
import math
import os
import time
import typing

import ament_index_python
import arena_bringup.extensions.NodeLogLevelExtension as NodeLogLevelExtension
import attrs
import geometry_msgs.msg
import launch
import launch.launch_description_sources
import launch_ros
import rclpy
import rclpy.logging
import rclpy.node
import rclpy.publisher
import rclpy.timer
import shapely
import tf2_ros
from arena_rclpy_mixins.Async import LaunchHandle
from arena_rclpy_mixins.shared import Namespace
from arena_robots.Robot import RobotView
from arena_runtime._node import NodeInterface
from arena_runtime.sim._interface import SimUnavailable

from task_generator.manager.environment_manager import EnvironmentManager
from task_generator.manager.robot_manager.controller_manager_client import ControllerManagerClient
from task_generator.manager.robot_manager.controller_transitions import next_transition
from task_generator.manager.robot_manager.task_runner import PhaseState, TaskRunner, Transition
from task_generator.shared import Orientation, Pose, Position, Robot

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from arena_simulation_setup.shared.conditions import EpisodeCondition
    from arena_simulation_setup.shared.judge import Sample, Zones

    from task_generator.tasks.robots.adapters import Adapter, ResetContext
    from task_generator.tasks.robots.request import GoToPhase, TaskKind, TaskPhase, TaskRequest


_NAV2_QUIET_NODES = (
    'behavior_server',
    'bt_navigator',
    'collision_monitor',
    'controller_server',
    'lifecycle_manager_navigation',
    'nav2_container',
    'planner_server',
    'smoother_server',
    'velocity_smoother',
    'waypoint_follower',
)
_NAV2_QUIET_RULES = '+[' + ', '.join(f'**/{n}:error' for n in _NAV2_QUIET_NODES) + ']'

_MOVEIT_QUIET_RULES = '+[**/moveit/**:error]'

_CONTROLLER_POLL = 0.2
_CM_CALL_TIMEOUT = 10.0
_CM_REFUSAL_GRACE_S = 20.0
_CM_SWITCH_TIMEOUT = 60.0
_CM_SWITCH_TIMED_OUT = "timed out"

_TELEPORT_TOLERANCE = 0.3
_TELEPORT_SETTLE_TIMEOUT = 2.0
_TELEPORT_POLL = 0.02
_TELEPORT_ATTEMPTS = 3


class RobotManager(NodeInterface):
    """Manages the goal and start position of a robot for all task modes."""

    _namespace: Namespace
    _environment_manager: EnvironmentManager
    _start_pos: Pose
    _goal_pos: Pose
    _robot_radius: float
    _robot: Robot
    _move_base_pub: rclpy.publisher.Publisher
    _goal_pub: rclpy.publisher.Publisher
    _rate_setup: rclpy.timer.Rate
    _config: RobotView
    _adapters: dict[TaskKind, Adapter]
    _adapter_instances: list[Adapter]
    _cap_adapters: dict[str, str]
    _runner: TaskRunner
    _unsupported_kinds_logged: set[TaskKind]
    _abort_episode: Callable[[str], None] | None
    _launch_handle: LaunchHandle | None

    @property
    def robot(self) -> Robot:
        return self._robot

    @property
    def robot_view(self) -> RobotView:
        return self._config

    @property
    def tf_buffer(self) -> tf2_ros.Buffer:
        return self.node.tf_buffer

    @property
    def start_pos(self) -> Pose:
        return self._start_pos

    @property
    def goal_pos(self) -> Pose:
        return self._goal_pos

    @property
    def controls_orientation(self) -> bool:
        return self._adapter.controls_orientation if self._adapter is not None else True

    @property
    def pose_stamped(self) -> tuple[Pose, rclpy.time.Time] | None:
        """Current map-frame pose with its TF stamp (None during reset/respawn windows)."""
        base = self.frame(self._config.model_params.base_frame).raw()
        try:
            t = self.node.tf_buffer.lookup_transform('map', base, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None
        tr = t.transform.translation
        pose = Pose(
            Position(tr.x, tr.y),
            Orientation.from_msg(t.transform.rotation),
        )
        return pose, rclpy.time.Time.from_msg(t.header.stamp)

    @property
    def pose(self) -> Pose | None:
        """Current robot pose in the map frame (None during reset/respawn windows)."""
        stamped = self.pose_stamped
        return None if stamped is None else stamped[0]

    @property
    def goal(self) -> Pose | None:
        """Map-frame pose of the active GoToPhase, or None."""
        from task_generator.tasks.robots.request import GoToPhase

        state = self._runner.active_state
        if state is None or not isinstance(state.phase, GoToPhase) or state.phase.pose is None:
            return None
        return self._environment_manager.realize(state.phase.pose)

    @property
    def runner(self) -> TaskRunner:
        return self._runner

    @property
    def waiting(self) -> bool:
        """The active phase has arrived and is holding for its park time or `until` atom."""
        return self._runner.waiting

    def __init__(
        self,
        *args: object,
        namespace: Namespace,
        environment_manager: EnvironmentManager,
        robot: Robot,
        **kwargs: object,
    ):
        super().__init__(*args, **kwargs)
        self._rate_setup = self.node.create_rate(0.1)

        self._config = robot.model.resolve_sync()

        self._namespace = namespace
        self._environment_manager = environment_manager

        self._start_pos = Pose()
        self._goal_pos = Pose()
        self._robot_radius = 0.25

        self._robot = robot
        self._robot.sim_path = self._environment_manager.realize(robot.name)
        self._robot.extra.setdefault('namespace', self.namespace)

        self._publish_goal_task: asyncio.Task | None = None
        self._watchers: list[Callable[[RobotManager], None]] = []
        self._launch_handle: LaunchHandle | None = None

        # Deferred to break the import cycle between this module and
        # task_generator.tasks (which eagerly loads context.py -> RobotManager).
        from task_generator.tasks.robots.adapters import ADAPTERS
        from task_generator.tasks.robots.request import TaskKind

        caps_available = self._config.effective_caps(self._robot.resolved_request, frames=self._robot.frames).available
        caps_to_kind: dict[str, str] = {}
        for cap in caps_available:
            override = self._robot.adapters.get(cap)
            if override is not None:
                caps_to_kind[cap] = override
                continue
            param_name = f"robot.{cap}_adapter"
            default_kind = self.node.rosparam[str].get(param_name, None)
            if default_kind is None:
                self._logger.warning(f"robot {self._robot.name!r} advertises cap {cap!r} but no default adapter is configured (param {param_name!r}); cap will be unbound.")
                continue
            caps_to_kind[cap] = default_kind

        self._adapters: dict[TaskKind, Adapter] = {}
        self._adapter_instances: list[Adapter] = []
        self._cap_adapters: dict[str, str] = {}

        for cap, kind in caps_to_kind.items():
            if cap not in ADAPTERS:
                raise AssertionError(f"robot {self._robot.name!r}: no adapter registry for cap {cap!r} (registered caps: {sorted(ADAPTERS)})")
            adapter_cls = ADAPTERS[cap].get(kind)
            declared_cap = adapter_cls._adapter_meta.cap
            if declared_cap != cap:
                raise AssertionError(f"adapter {kind!r} declares cap {declared_cap!r} but was selected for cap {cap!r} on robot {self._robot.name!r}")

            adapter_kwargs = self._adapter_kwargs_for(cap, kind)
            try:
                adapter = adapter_cls(robot_manager=self, **adapter_kwargs)
            except TypeError as exc:
                raise AssertionError(f"adapter {kind!r} rejected kwargs {sorted(adapter_kwargs)} for robot {self._robot.name!r}: {exc}") from exc

            missing = adapter.requires - caps_available
            if missing:
                raise AssertionError(f"adapter {kind!r} for robot {self._robot.name!r} missing caps {sorted(missing)}; robot declares {sorted(caps_available)}")

            for tk in adapter.accepts:
                if tk in self._adapters:
                    raise AssertionError(f"robot {self._robot.name!r}: TaskKind {tk!r} claimed by both {self._adapters[tk].kind!r} and {kind!r}")
                self._adapters[tk] = adapter
            self._adapter_instances.append(adapter)
            self._cap_adapters[cap] = kind

        self._adapter = next(
            (a for a in self._adapter_instances if TaskKind.GOTO_POSE in a.accepts),
            self._adapter_instances[0] if self._adapter_instances else None,
        )

        self._runner = TaskRunner(self._robot.name)
        self._placed = False
        self._unsupported_kinds_logged: set[TaskKind] = set()
        self._abort_episode: Callable[[str], None] | None = None

    def bind_abort(self, fn: Callable[[str], None]) -> None:
        self._abort_episode = fn

    def _adapter_kwargs_for(self, cap: str, kind: str) -> dict[str, typing.Any]:
        try:
            cap_raw = self._config.caps._load_cap_file(cap)
        except FileNotFoundError:
            cap_raw = {}  # allocation-derived cap, no static caps/<cap>.yaml
        sub = cap_raw.get(kind, {})
        kwargs: dict[str, typing.Any] = dict(sub) if isinstance(sub, dict) else {}
        # CLI overrides land as `robot.<cap>.<key>` ROS params; they overlay the
        # cap-file YAML so the bound adapter sees flag-level user intent.
        for key, param in self.node.get_parameters_by_prefix(f"robot.{cap}").items():
            kwargs[key] = param.value
        if cap == 'mobile':
            kwargs.setdefault('train_mode', self.node.rosparam[bool].get('robot.train', False))
        return kwargs

    async def set_up_robot(self):
        self._robot.pose.position.z += self._config.model_params.z_offset
        self._robot = (await self._environment_manager.spawn_robot((self._robot,)))[0]

        _gen_goal_topic = self.namespace("goal_pose")

        self._stop_pub = self.node.create_publisher(geometry_msgs.msg.Twist, str(self.namespace("cmd_vel")), 1)
        self._task_pose_pub = self.node.create_publisher(geometry_msgs.msg.PoseStamped, str(self.namespace("task_pose")), 10)
        self._goal_pub = self.node.create_publisher(
            geometry_msgs.msg.PoseStamped,
            _gen_goal_topic,
            10,
        )

    async def launch(self, node_names: set[str]):
        """Bring up the robot's navstack. Split from set_up_robot so callers can sequence the
        LaunchService run after spawn_world_obstacles (which it would otherwise starve)."""
        await self._launch_robot(node_names)

        self._robot_radius = self.node.rosparam[float].get(
            'robot_radius',
            self._robot_radius,
        )

    @property
    def radius(self) -> float:
        return self._robot_radius

    @property
    def safe_distance(self) -> float:
        return self._robot_radius + self.node.conf.Robot.SPAWN_ROBOT_SAFE_DIST.value

    @property
    def model_name(self) -> str:
        return self._robot.model.name

    @property
    def name(self) -> str:
        return self._robot.name

    @property
    def frame(self) -> Namespace:
        return self._robot.frame

    @property
    def base_frame(self) -> str:
        return self.frame(self._config.model_params.base_frame).raw()

    @property
    def accepts(self) -> frozenset[TaskKind]:
        """Task kinds this robot's bound adapters can dispatch."""
        return frozenset(self._adapters.keys())

    @property
    def cap_adapters(self) -> dict[str, str]:
        """Cap -> bound adapter kind, resolved at construction time."""
        return dict(self._cap_adapters)

    @property
    def namespace(self) -> Namespace:
        return self._namespace(self._robot.name)

    def _stop_current_task(self) -> None:
        self._runner.stop()
        if self._publish_goal_task is not None:
            self._publish_goal_task.cancel()
            self._publish_goal_task = None

    def _outcome_for_phase(self, phase: TaskPhase) -> tuple[int | None, str | None]:
        """Return (status, reason) for the phase that just completed."""
        from task_generator.tasks.robots.request import kind_of

        kind = kind_of(phase)
        adapter = self._adapters.get(kind)
        if adapter is None:
            from arena_robots_msgs.action import GotoPose, PlayGesture, ReachPose

            from task_generator.tasks.robots.request import TaskKind

            _UNSUPPORTED_CAP: dict[TaskKind, int] = {
                TaskKind.GOTO_POSE: GotoPose.Result.STATUS_UNSUPPORTED_CAP,
                TaskKind.REACH_POSE: ReachPose.Result.STATUS_UNSUPPORTED_CAP,
                TaskKind.PLAY_GESTURE: PlayGesture.Result.STATUS_UNSUPPORTED_CAP,
            }
            status = _UNSUPPORTED_CAP.get(kind)
            reason = f"no adapter for {kind.name} on robot {self.name}"
            return status, reason
        status = adapter.client_for(kind).status
        reason = adapter.client_for(kind).reason
        return status, reason

    def begin_episode(self) -> None:
        self._runner.begin_episode()
        self._placed = False

    @property
    def placed(self) -> bool:
        """Whether this episode's reset has landed the robot, before that its pose is the previous one."""
        return self._placed

    def publish_task_pose(self, pose: Pose) -> None:
        """Pose the judge sees this tick, in the flattened world frame, recorded so replay judges the same pose."""
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp = self.node.sim_time.to_msg()
        msg.pose = pose.to_msg()
        self._task_pose_pub.publish(msg)

    def set_episode_conditions(self, conditions: Sequence[EpisodeCondition]) -> None:
        self._runner.set_episode_conditions(conditions)

    def finish_episode(self) -> None:
        self._runner.finish_episode()

    async def _dispatch(self, state_index: int) -> None:
        """Dispatch one phase to its adapter, a hold phase first taking the robot's current pose."""
        from task_generator.tasks.robots.adapters.mobile import MobileAdapter
        from task_generator.tasks.robots.request import GoToPhase, kind_of

        state = self._runner.phases[state_index]
        phase = state.phase
        if isinstance(phase, GoToPhase) and phase.hold:
            pose = self.pose
            if pose is None:
                return
            phase = attrs.evolve(phase, pose=self._environment_manager.ezilear(pose))
            state.phase = phase
            state.monitor.phase = phase
            self.node.on_task_submitted(self)
        state.dispatched = True
        kind = kind_of(phase)
        adapter = self._adapters.get(kind)
        if adapter is None:
            if kind not in self._unsupported_kinds_logged:
                self._unsupported_kinds_logged.add(kind)
                self._logger.warning(f"robot {self.name!r} has no adapter for phase kind {kind.name!r}; synthesizing UNSUPPORTED_CAP failure on next tick")
            return
        dispatched = attrs.evolve(phase, pose=self._environment_manager.realize(phase.pose)) if isinstance(phase, GoToPhase) and phase.pose is not None else phase
        await adapter.dispatch_phase(dispatched, self)
        if self._publish_goal_task is not None:
            self._publish_goal_task.cancel()
            self._publish_goal_task = None
        self._note_goal_inputs(state)
        if isinstance(adapter, MobileAdapter):
            self._publish_goal_task = asyncio.create_task(adapter.publish_goal_loop())

    def _note_goal_inputs(self, state: PhaseState) -> None:
        """Record what the mobile adapter gave its planner for `state` and republish the episode record."""
        from task_generator.tasks.robots.adapters.mobile import MobileAdapter
        from task_generator.tasks.robots.request import kind_of

        adapter = self._adapters.get(kind_of(state.phase))
        if isinstance(adapter, MobileAdapter):
            self._runner.goal_inputs = adapter.goal_inputs
            state.instruction = adapter.instruction
            self.node.on_task_submitted(self)

    async def tick(self, sample: Sample, zones: Zones) -> Transition:
        """Judge the active phase on one tick, applying its outcome and dispatching the next phase."""
        from task_generator.tasks.robots.request import GoToPhase, kind_of

        state = self._runner.active_state
        if not self._placed:
            return Transition()
        if state is None:
            return self._runner.step(sample, zones, None, (None, None))
        if not state.dispatched:
            await self._dispatch(state.index)
            state = self._runner.active_state
            if state is None or not state.dispatched:
                return Transition()
        kind = kind_of(state.phase)
        adapter = self._adapters.get(kind)
        if adapter is None:
            transition = self._runner.fail_active(self._outcome_for_phase(state.phase)[1] or "")
        else:
            action_done = None if isinstance(state.phase, GoToPhase) else bool(adapter.client_for(kind).is_done())
            dispatched = self.active_dispatch_phase()
            if dispatched is not None:
                adapter.on_phase_tick(dispatched, self)
            transition = self._runner.step(sample, zones, action_done, self._outcome_for_phase(state.phase), adapter.signal)
        return await self._settle(transition)

    async def _settle(self, transition: Transition) -> Transition:
        ended = transition.ended
        if ended is not None and ended.outcome == "failed":
            self._logger.info(f"robot {self.name!r} phase {ended.index} ({ended.phase.kind}) failed ({ended.reason})")
        for scoped in transition.violated:
            self._logger.info(f"robot {self.name!r} condition {scoped.id} violated: {scoped.condition.p}")
        if transition.stop:
            self._stop_current_task()
            if transition.abort is not None and self._abort_episode is not None:
                self._abort_episode(transition.abort)
            return transition
        state = self._runner.active_state
        if state is not None and not state.dispatched:
            await self._dispatch(state.index)
        return transition

    @property
    async def is_done(self) -> bool:
        """Whether every submitted phase has ended."""
        return self._runner.done

    def active_dispatch_phase(self) -> TaskPhase | None:
        """The active phase as its adapter received it, goto poses in the map frame."""
        from task_generator.tasks.robots.request import GoToPhase

        state = self._runner.active_state
        if state is None or not state.dispatched:
            return None
        phase = state.phase
        if isinstance(phase, GoToPhase) and phase.pose is not None:
            return attrs.evolve(phase, pose=self._environment_manager.realize(phase.pose))
        return phase

    def _resolve_target(self, phase: GoToPhase) -> GoToPhase:
        """Give a named zone, door or elevator target its dispatch pose unless it carries one, a ped target keeps following the ped."""
        world = self.node._world_manager.world_compacted()
        corners = world.lookup_zone_polygon(phase.target) if phase.target is not None and phase.pose is None else None
        if corners is None:
            return phase
        if any(zone.name == phase.target for zone in world.zones):
            try:
                position = self.node._world_manager.get_position_on_map(self.safe_distance, forbid=False, polygon=shapely.Polygon([(corner.x, corner.y) for corner in corners]))
            except RuntimeError as e:
                raise ValueError(f"robot {self.name!r}: goto target {phase.target!r} has no free cell with {self.safe_distance:.2f} m clearance") from e
        else:
            position = world.point_resolver(self.node.conf.General.RNG.stream("robots", "target", self.name)).resolve(phase.target)
        return attrs.evolve(phase, pose=Pose(position=position, orientation=Orientation.identity()))

    async def submit_task(self, request: TaskRequest) -> None:
        """Resolve a typed TaskRequest and append it to this episode's phases. Phase poses are abstract until dispatch."""
        from task_generator.tasks.robots.request import GoToPhase, kind_of

        if not request.phases:
            raise ValueError(f"TaskRequest has no phases; nothing to dispatch (robot={self.name!r})")

        world_manager = self.node._world_manager
        pose = self.pose
        anchor = self._environment_manager.ezilear(pose) if pose is not None else self._start_pos
        current_level = world_manager.level_of_point(anchor.position.x, anchor.position.y)
        routed_phases: list[TaskPhase] = []
        for phase in request.phases:
            if isinstance(phase, GoToPhase):
                phase = self._resolve_target(phase)
                if phase.pose is not None:
                    goal_level = world_manager.level_of_point(phase.pose.position.x, phase.pose.position.y)
                    if current_level and goal_level and goal_level != current_level:
                        for elevator_position in world_manager.elevator_route(current_level, goal_level):
                            routed_phases.append(GoToPhase(pose=Pose(position=elevator_position, orientation=phase.pose.orientation), tolerance_angle=math.pi))
                        current_level = goal_level
            routed_phases.append(phase)

        phases = [self._with_defaults(phase) for phase in routed_phases]
        for phase in phases:
            if isinstance(phase, GoToPhase) and phase.signal:
                adapter = self._adapters.get(kind_of(phase))
                signals = frozenset() if adapter is None else adapter.signals
                if phase.signal not in signals:
                    raise ValueError(f"robot {self.name!r}: goto phase expects signal {phase.signal!r}, its {phase.kind} adapter can send {sorted(signals)}")
            if isinstance(phase, GoToPhase) and phase.target is not None:
                known = set(world_manager.world_compacted().zone_ref_names()) | set(self.node.ped_names())
                if phase.target not in known:
                    raise ValueError(f"robot {self.name!r}: goto target {phase.target!r} is neither a zone, door, elevator nor a pedestrian")
            if kind_of(phase) not in self._adapters and kind_of(phase) not in self._unsupported_kinds_logged:
                self._logger.warning(f"robot {self.name!r} has no adapter for phase kind {phase.kind!r}, the phase will fail as UNSUPPORTED_CAP")

        states = self._runner.submit(phases, request.conditions)
        self.node.on_task_submitted(self)
        if states:
            await self._dispatch(states[0].index)

    def _with_defaults(self, phase: TaskPhase) -> TaskPhase:
        """Fill unset goto tolerances, hold time and signal from the launch parameters, zero angle for robots without heading control."""
        from task_generator.tasks.robots.request import GoToPhase

        if not isinstance(phase, GoToPhase):
            return phase
        conf = self.node.conf.Robot
        if phase.tolerance_angle is not None:
            angle = float(phase.tolerance_angle)
        elif self.controls_orientation:
            angle = float(conf.GOAL_TOLERANCE_ANGLE.value)
        else:
            angle = 0.0
        return attrs.evolve(
            phase,
            tolerance_radius=float(conf.GOAL_TOLERANCE_RADIUS.value if phase.tolerance_radius is None else phase.tolerance_radius),
            tolerance_angle=angle,
            hold_time=float(conf.GOAL_HOLD_TIME.value if phase.hold_time is None else phase.hold_time),
            signal=(str(conf.GOAL_SIGNAL.value) or None) if phase.signal is None else (phase.signal or None),
        )

    async def reset(self, ctx: ResetContext) -> dict[str, BaseException | None]:
        """Fan out adapter on_reset hooks concurrently, return per-kind outcomes."""
        results = await asyncio.gather(
            *(a.on_reset(self, ctx) for a in self._adapter_instances),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, SimUnavailable):
                raise result
        outcomes: dict[str, BaseException | None] = {}
        for adapter, result in zip(self._adapter_instances, results, strict=True):
            if isinstance(result, BaseException):
                self._logger.error(f"adapter {adapter.kind!r} on_reset failed: {result!r}")
                outcomes[adapter.kind] = result
            else:
                outcomes[adapter.kind] = None
        self._placed = True
        state = self._runner.active_state
        if state is not None and state.dispatched:
            self._note_goal_inputs(state)
        return outcomes

    def _pose_stamp(self) -> rclpy.time.Time | None:
        """TF stamp of the robot's latest observed map pose, or None if TF has none."""
        stamped = self.pose_stamped
        return None if stamped is None else stamped[1]

    def _pose_error(self, target: Pose) -> tuple[float, rclpy.time.Time] | None:
        """Planar distance from the robot's observed map pose to ``target``, with its TF stamp."""
        stamped = self.pose_stamped
        if stamped is None:
            return None
        observed, stamp = stamped
        return math.hypot(observed.position.x - target.position.x, observed.position.y - target.position.y), stamp

    async def _await_teleport_landed(self, target: Pose, since: rclpy.time.Time | None) -> tuple[bool, float | None]:
        """Poll TF for a sample stamped after ``since`` within tolerance of ``target``, as ``(landed, last_error)``."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _TELEPORT_SETTLE_TIMEOUT
        last_error: float | None = None
        while True:
            sample = self._pose_error(target)
            if sample is not None:
                error, stamp = sample
                if since is None or stamp > since:
                    last_error = error
                    if error <= _TELEPORT_TOLERANCE:
                        return True, error
            if loop.time() >= deadline:
                return False, last_error
            await asyncio.sleep(_TELEPORT_POLL)

    async def _apply_pose(self, pose: Pose):
        pose.position.z += self._config.model_params.z_offset
        self.robot.pose = pose

        # TF reports the realized map frame, pose is env-local
        target = self._environment_manager.realize(pose)
        await asyncio.gather(*(a.before_move(pose, self) for a in self._adapter_instances))
        self._stop_pub.publish(geometry_msgs.msg.Twist())
        # set-pose only applies on a sim update and TF only republishes while stepping
        async with self.node.unpause_window():
            since = self._pose_stamp()
            last_error: float | None = None
            for attempt in range(1, _TELEPORT_ATTEMPTS + 1):
                results = await self._environment_manager.move_robot((self.robot,))
                if not results or not all(results):
                    raise RuntimeError(f"simulator rejected teleport of robot {self.name!r} (move_robot -> {tuple(results)})")

                await self.node.await_sim_step(_TELEPORT_SETTLE_TIMEOUT)
                landed, last_error = await self._await_teleport_landed(target, since)
                if landed:
                    break
                if last_error is None:
                    base = self.frame(self._config.model_params.base_frame).raw()
                    detail = f"no map -> {base} transform" if self.pose_stamped is None else f"no map -> {base} sample newer than the teleport"
                    self._logger.warning(f"teleport of robot {self.name!r} is unverifiable within {_TELEPORT_SETTLE_TIMEOUT}s: {detail}")
                    break
                self._logger.warning(f"teleport of robot {self.name!r} did not land (attempt {attempt}/{_TELEPORT_ATTEMPTS}): observed {last_error:.2f}m from target ({target.position.x:.2f}, {target.position.y:.2f}), tolerance {_TELEPORT_TOLERANCE}m; retrying")
                since = self._pose_stamp()
            else:
                observed = "unknown" if last_error is None else f"{last_error:.2f}m"
                raise RuntimeError(f"teleport of robot {self.name!r} never landed: after {_TELEPORT_ATTEMPTS} attempts the robot is {observed} from target ({target.position.x:.2f}, {target.position.y:.2f}), tolerance {_TELEPORT_TOLERANCE}m")

        await asyncio.gather(*(a.on_move(pose, self) for a in self._adapter_instances))

    async def move(self, pose: Pose) -> None:
        """Teleport the robot to ``pose``. Positioning only, no task dispatch."""
        self._start_pos = pose
        await self._apply_pose(pose)
        self._notify()

    def watch(self, fn: Callable[[RobotManager], None]) -> None:
        """Calls fn after every teleport and every dispatched task."""
        self._watchers.append(fn)

    def _notify(self) -> None:
        for fn in self._watchers:
            fn(self)

    async def _launch_robot(self, node_paths: set[str]):
        """Launch the robot's navstack via the bound adapters."""
        await asyncio.gather(*(a.ensure_services() for a in self._adapter_instances))
        launch_description = launch.LaunchDescription()
        current_log_level = rclpy.logging.get_logger_effective_level(self.node.get_logger().name).name.lower()
        launch_description.add_action(NodeLogLevelExtension.SetGlobalLogLevelAction(current_log_level))
        launch_description.add_action(NodeLogLevelExtension.SetGlobalLogLevelAction(_NAV2_QUIET_RULES))
        launch_description.add_action(NodeLogLevelExtension.SetGlobalLogLevelAction(_MOVEIT_QUIET_RULES))

        launch_arguments = {
            'robot': self.model_name,
            'task_generator_node': os.path.join(self.node.get_namespace(), self.node.get_name()),
            'namespace': self.namespace,
            'frame': self._robot.frame.tf(),
            'use_sim_time': 'True',
        }

        launch_description.add_action(
            launch.actions.IncludeLaunchDescription(
                launch.launch_description_sources.PythonLaunchDescriptionSource(os.path.join(ament_index_python.packages.get_package_share_directory('arena_simulation_setup'), 'launch/robot.launch.py')),
                launch_arguments=launch_arguments.items(),
            )
        )

        from task_generator.tasks.robots.adapters import AdapterCtx

        adapter_ctx = AdapterCtx(
            namespace=self.namespace,
            robot_name=self.model_name,
            frame=self._robot.frame.tf(),
            task_generator_node=os.path.join(self.node.get_namespace(), self.node.get_name()),
            env_namespace=self.node.get_namespace(),
            use_sim_time=True,
            base_frame=self._config.model_params.base_frame,
            odom_frame=self._config.model_params.odom_frame,
            sensors=self._config.effective_sensors(self._robot.resolved_request, frames=self._robot.frames),
            tf_buffer=None,
            node_handle=self.node,
        )
        adapter_actions: list[object] = [
            launch_ros.actions.PushRosNamespace(str(self.namespace)),
            *(a.launch_description(adapter_ctx) for a in self._adapter_instances),
        ]
        if self._adapter_instances:
            bringup_caps = [a.bringup.cap for a in self._adapter_instances]
            bringup_kinds = [a.bringup.kind for a in self._adapter_instances]
            adapter_actions.append(
                launch_ros.actions.Node(
                    package="arena_robots",
                    executable="task_server",
                    name="task_server",
                    parameters=[
                        {
                            "robot_name": self._robot.model.name,
                            "bringup_caps": bringup_caps,
                            "bringup_kinds": bringup_kinds,
                            "parts_json": json.dumps({t: [{"variant": p.variant, "mount": p.mount} for p in ps] for t, ps in self._robot.resolved_request.items()}),
                            "frame": self._robot.frame.tf(),
                            "use_sim_time": adapter_ctx.use_sim_time,
                        }
                    ],
                )
            )
        bringup = os.path.join(
            ament_index_python.packages.get_package_share_directory('arena_robots'),
            'robots',
            self.model_name,
            'launch',
            'bringup.launch.py',
        )
        if os.path.isfile(bringup):
            adapter_actions.append(
                launch.actions.IncludeLaunchDescription(
                    launch.launch_description_sources.PythonLaunchDescriptionSource(bringup),
                    launch_arguments={
                        'base_frame': self._robot.frame.tf(self._config.model_params.base_frame),
                        'use_sim_time': 'True',
                    }.items(),
                )
            )

        if self._adapter_instances:
            try:
                adapter_actions.extend(self._adapter_instances[0].bringup.telemetry_actions())
            except (OSError, ValueError, RuntimeError) as e:
                self.node.get_logger().error(f"Failed to add telemetry actions for robot {self.model_name!r}: {e!r}")

        launch_description.add_action(launch.actions.GroupAction(adapter_actions))

        async with self.node.unpause_window():
            await self.node.await_sim_step()
            self._launch_handle = await self.node.do_launch_tracked(launch_description)
            ready_timeout = self.node.conf.Robot.READY_TIMEOUT.value
            await asyncio.gather(*(a.await_ready(self, node_paths, ready_timeout) for a in self._adapter_instances))
            await self.bring_up_controllers()
        for adapter in self._adapter_instances:
            await adapter.on_controllers_active(self)

    async def bring_up_controllers(self) -> None:
        """Drive this robot's controllers to active inside the caller's unpause window, raising only on a refusal that outlasts the manager's startup grace."""
        expected = self._environment_manager.robot_controllers(self._robot)
        if not expected:
            return
        cm = ControllerManagerClient(self.node, str(self.namespace("controller_manager")), call_timeout=_CM_CALL_TIMEOUT)
        started = time.monotonic()
        try:
            await cm.ensure()
            while True:
                states = await cm.states()
                if states is None:
                    await asyncio.sleep(_CONTROLLER_POLL)
                    continue
                step = next_transition(expected, states)
                if step is None:
                    return
                action, names = step
                match action:
                    case "fail":
                        raise RuntimeError(f"robot {self.name!r}: controllers finalized, cannot recover: {names}")
                    case "load":
                        ok = await cm.load(names[0])
                        detail = ""
                    case "configure":
                        ok = await cm.configure(names[0])
                        detail = ""
                    case "activate":
                        result = await cm.activate(names, switch_timeout_sec=_CM_SWITCH_TIMEOUT, timeout_sec=_CM_SWITCH_TIMEOUT + 1.0)
                        ok = None if result is None else result[0]
                        detail = "" if result is None else result[1]
                        if ok is False and _CM_SWITCH_TIMED_OUT in detail:
                            self._logger.warning(f"switch_controller {names} on {self.name!r}: {detail}, retrying")
                            ok = None
                    case _:
                        ok = None
                        detail = ""
                if ok is False and action in ("load", "configure") and time.monotonic() - started < _CM_REFUSAL_GRACE_S:
                    self._logger.warning(f"controller_manager refused {action} {names} on {self.name!r} while still starting, retrying")
                    ok = None
                if ok is False:
                    raise RuntimeError(f"robot {self.name!r}: controller_manager refused {action} {names}: {detail}")
                if ok is None:
                    self._logger.info(f"controller transition {action} {names} on {self.name!r} pending, re-reading")
                    await asyncio.sleep(_CONTROLLER_POLL)
        finally:
            cm.close()

    async def destroy(self):
        results = await asyncio.gather(
            *(a.teardown() for a in self._adapter_instances),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, SimUnavailable):
                raise result
        for adapter, result in zip(self._adapter_instances, results, strict=True):
            if isinstance(result, Exception):
                self._logger.warning(f"adapter {adapter.kind!r} teardown failed: {result!r}")
        if self._launch_handle is not None:
            await self._launch_handle.shutdown()
        self._launch_handle = None
        await self._environment_manager.remove_robot((self.robot,))
