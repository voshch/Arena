import array
import asyncio
import hashlib
import json
import random
import traceback
import typing
import uuid
from collections.abc import Mapping, Sequence

import arena_cam_msgs.srv
import arena_runtime_msgs.action
import arena_runtime_msgs.msg
import arena_runtime_msgs.srv
import arena_simulation_setup.tree.assets.Human
import arena_simulation_setup.tree.assets.Object
import arena_simulation_setup.tree.configs.environment
import arena_simulation_setup.tree.configs.parametrized
import arena_simulation_setup.tree.World as World
import attrs
import geometry_msgs.msg
import rclpy
import rclpy.lifecycle
import rclpy.qos
import yaml
from arena_robots.Robot import RobotIdentifier
from arena_robots.SetupFile import Config as RobotSetupConfig
from arena_runtime_msgs.msg import AdapterDisplay
from arena_simulation_setup.shared.conditions import Node
from arena_simulation_setup.shared.judge import NodeMonitor, zone_polygons
from arena_simulation_setup.shared.task import GoToPhase, LoopPhase, TaskPhase, TaskRequest
from arena_simulation_setup.tree.assets.Camera import collect, pick_camera
from arena_simulation_setup.tree.World import point_scenario_assets
from arena_simulation_setup.tree.World.Scenario import EpisodeCondition, TimelineEntry
from arena_viz.kinds import DisplayKind
from arena_viz.style import StyleSpec
from rcl_interfaces.msg import IntegerRange, ParameterDescriptor, ParameterValue
from rcl_interfaces.msg import Parameter as RclParameter
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Empty, Int16, String

from arena_runtime.env.adapters import ResetContext
from arena_runtime.env.constants import Constants
from arena_runtime.env.constants.runtime import RUN_EPISODE_BUSY, RUN_EPISODE_SHUTTING_DOWN
from arena_runtime.env.goal_progress import GoalProgressTracker
from arena_runtime.env.interactive import MarkerHub
from arena_runtime.env.interactive.entities import EntityHandles
from arena_runtime.env.node import _LATCHED, EnvNode, _available
from arena_runtime.env.placement import random_obstacle_pose, random_robot_pose
from arena_runtime.env.robot_manager.robot_manager import footprint_radius
from arena_runtime.env.run_task import RunningTask
from arena_runtime.env.task_layer import (
    EpisodeContext,
    EpisodeRecord,
    EpisodeTask,
    InteractiveContext,
    TaskConfiguration,
    TaskContext,
    TaskModeOverrides,
    TaskModes,
    declare_mode_schemas,
    decode_modes,
    encode_modes,
    encode_namespaces,
    registered_modes,
    task_layer,
    task_layer_name,
)
from arena_runtime.shared import DynamicObstacle, Obstacle, Orientation, Pose, Position, Robot

if typing.TYPE_CHECKING:
    from arena_runtime.env.robot_manager import RobotManager
    from arena_runtime.env.robot_manager.task_runner import PhaseState

_MAX_CONSECUTIVE_RESET_FAILURES = 3

_CAM_PLAY_TIMEOUT = 10.0

# Deeper KeepLast for state/episode: terminal-then-next-RUNNING bursts.
_EPISODE_QOS = rclpy.qos.QoSProfile(
    depth=20,
    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
)


class _TimelineState(typing.TypedDict):
    fired: bool
    next: float
    prev: bool
    rng: random.Random
    when: Node | None
    monitors: dict[str, NodeMonitor]


@attrs.define
class EpisodeRuntime:
    current: EpisodeRecord = attrs.Factory(EpisodeRecord)
    run_seed: str = attrs.field(factory=lambda: uuid.uuid4().hex)
    pending_outcomes: dict = attrs.Factory(dict)
    pending: TaskModeOverrides = attrs.Factory(TaskModeOverrides)
    decided: bool = False
    action_in_flight: bool = False
    respawn_requested: bool = False


def _is_map_frame(frame_id: str) -> bool:
    """Empty, map, or an env's namespaced map frame (env_0/map), all carrying realized map coordinates."""
    return frame_id in ("", "map") or frame_id.endswith("/map")


def _extension_name(model: str) -> str:
    return f"ext_{model}_{uuid.uuid4().hex[:6]}"


def _derive_seed(run_seed: str, world: str, episode_id: int) -> int:
    digest = hashlib.blake2b(
        f"{run_seed}|{world}|{episode_id}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF


class EpisodeNode(EnvNode):
    """The env node with its episode loop: record, queue, staged params, reset cycle, termination, timeline and the spawn services."""

    _episodes: EpisodeRuntime
    _goal_progress: GoalProgressTracker
    _entities: EntityHandles

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(Constants.ENV_NODE_NAMESPACE, automatically_declare_parameters_from_overrides=True, **kwargs)

        layer_name = task_layer_name(self.rosparam[str].get("task_layer", ""))
        self.rosparam[str].set("task_layer", layer_name)
        self._task_layer = task_layer(layer_name)
        self.ROSParam[bool]("resetting", True)
        self.task_params = self._task_layer.declare_task_params(self)
        declare_mode_schemas(self._task_layer, self)

        run_seed = self.rosparam[str].get("run_seed", "")
        queue_depth = self.rosparam[int].get("episode_queue_depth", 10)

        self._declare_mutable_param(
            "task.episode.auto_reset",
            True,
            ParameterDescriptor(
                description=("true = standalone: node auto-advances episodes. false = managed: external controller drives resets."),
            ),
        )
        self._declare_mutable_param(
            "task.episode.fail_on_collision",
            False,
            ParameterDescriptor(
                description=("true = abort the episode as FAILED when the robot footprint contacts a wall, static obstacle, or pedestrian."),
            ),
        )
        self._declare_mutable_param(
            "run_seed",
            run_seed,
            ParameterDescriptor(
                description=("Hex string seeding per-episode blake2b derivation. Empty = random uuid at startup."),
            ),
        )
        self._declare_mutable_param(
            "episode_queue_depth",
            queue_depth,
            ParameterDescriptor(
                description="Publisher depth for state/queue topic.",
                integer_range=[IntegerRange(from_value=1, to_value=100, step=1)],
            ),
        )

        self._episodes = EpisodeRuntime(
            run_seed=run_seed or uuid.uuid4().hex,
        )
        self._goal_progress = GoalProgressTracker()

        self.markers = MarkerHub(self, self.service_namespace("markers"), self._reset_lock)
        self._start_time = self.time
        self._task: EpisodeTask | None = None
        self._force_reset = False
        self._abort_reason: str | None = None
        self._consecutive_reset_failures = 0
        self._reset_sim_time = 0

        self._staged_params: dict[str, ParameterValue] = {}
        self._staged_human_params: dict[str, ParameterValue] = {}
        self._human_params: dict[str, ParameterValue] = {}

        # The scenario timeline evaluated on sim time.
        self._timeline: list[TimelineEntry] = []
        self._timeline_state: list[_TimelineState] = []
        self._timeline_seed: int = 0
        self._timeline_t0: float | None = None
        self._stimulus_tasks: set[asyncio.Task] = set()
        self._stimuli_fired: dict[str, float] = {}
        self._episode_conditions: list[EpisodeCondition] = []

        self._pub_task_reset = self.create_publisher(
            Int16,
            self.service_namespace("task_reset"),
            1,
        )

        self._cam_setup = self.ROSParam[str]("cam.setup", "")
        self._cam_client = self.create_client_wrapper(arena_cam_msgs.srv.Play, f"{self.get_namespace().rstrip('/')}/cam/play", timeout=_CAM_PLAY_TIMEOUT)
        self._cam_absent = False
        self._cam_task: asyncio.Future[tuple[bool, str]] | None = None

        self._pub_state_world = self.create_publisher(
            String,
            self.service_namespace("state", "world"),
            _LATCHED,
        )

        self._pub_state_resetting = self.create_publisher(
            Bool,
            self.service_namespace("state", "resetting"),
            _LATCHED,
        )
        self._pub_state_resetting.publish(Bool(data=False))

        self._pub_state_episode = self.create_publisher(
            arena_runtime_msgs.msg.EpisodeRecord,
            self.service_namespace("state", "episode"),
            _EPISODE_QOS,
        )

        self._pub_state_queue = self.create_publisher(
            arena_runtime_msgs.msg.EpisodeRecord,
            self.service_namespace("state", "queue"),
            rclpy.qos.QoSProfile(
                depth=queue_depth,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self._check_status_task: asyncio.Task | None = None
        self._episode_task: asyncio.Task | None = None
        self._shutting_down = False

        self._run_tasks: dict[str, RunningTask] = {}
        self._assigned: set[str] = set()
        self._pending_lock = asyncio.Lock()

        self._publish_queue_state()

    @property
    def episode_id(self) -> int:
        return self._episodes.current.episode_id

    async def _set_up_task(self) -> None:
        assert self._robots_manager is not None and self._simulator is not None
        self._robots_manager.bind_abort(self.abort_episode)
        self._robots_manager.bind_collision(self._on_collision)
        self._entities = EntityHandles(self.markers, self._realizer, self._environment_manager)
        self._reset_sim_time = self.sim_time.sec
        self._pub_reset_start = self.create_publisher(Empty, "reset_start", 1)
        self._pub_reset_end = self.create_publisher(Empty, "reset_end", 1)

        self._task = await self._task_layer.create_task(
            node=self,
            params=self.task_params,
            ctx=TaskContext(
                environment_manager=self._environment_manager,
                robots_manager=self._robots_manager,
                world_manager=self._world_manager,
                realizer=self._realizer,
                simulator=self._simulator,
                acoustics_simulator=self._acoustics_simulator,
                reset_lock=self._reset_lock,
                conf=TaskConfiguration.of(self.conf),
                episode=EpisodeContext(
                    compromise=self._compromise,
                    abort_episode=self.abort_episode,
                    force_reset=self.force_reset,
                    register_timeline=self.register_timeline,
                    register_conditions=self.register_conditions,
                    register_sound_levels=self.register_sound_levels,
                ),
                interactive=InteractiveContext(markers=self.markers, robot_colors=self.robot_colors, tf_buffer=self.tf_buffer),
            ),
        )

    def _after_setup(self) -> None:
        self._publish_queue_state()

    async def _on_tick(self, ended: Mapping[str, "PhaseState"]) -> None:
        self._evaluate_timeline(self._sample.t)
        self._advance_run_tasks()
        if self._task is not None and self._episodes.current.outcome_state == arena_runtime_msgs.action.RunEpisode.Result.RUNNING and not self._reset_lock.locked():
            assigned, self._assigned = self._assigned, set()
            await self._task.tick(ended, assigned)

    def _extra_semantic_states(self) -> list[arena_runtime_msgs.msg.SemanticEntityState]:
        return [self._timeline_semantic_state()]

    def _interaction_displays(self) -> list[AdapterDisplay]:
        return [
            AdapterDisplay(
                name="Handles",
                topic=str(self.markers.namespace),
                topic_type="visualization_msgs/InteractiveMarkerUpdate",
                kind=DisplayKind.INTERACTIVE_MARKERS,
                style_json=StyleSpec().to_json(),
                topic_must_exist=False,
                group="Interaction",
            )
        ]

    def on_activate(self, state: rclpy.lifecycle.State) -> TransitionCallbackReturn:
        super().on_activate(state)

        def _start() -> None:
            if self._shutting_down:
                return
            self._check_status_task = asyncio.create_task(self._termination_watcher())
            if self.rosparam[bool].get_unsafe("task.episode.auto_reset"):
                self._spawn_episode()

        self.event_loop.call_soon_threadsafe(_start)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: rclpy.lifecycle.State) -> TransitionCallbackReturn:
        super().on_deactivate(state)
        if self._episode_task is not None and not self._episode_task.done():
            self._episode_task.cancel()
        if self._check_status_task is not None and not self._check_status_task.done():
            self._check_status_task.cancel()
        return TransitionCallbackReturn.SUCCESS

    async def teardown(self) -> None:
        self._shutting_down = True
        for task in self._episode_tasks():
            task.cancel()
        await super().teardown()

    def _episode_tasks(self) -> list[asyncio.Task]:
        return [task for task in (self._check_status_task, self._episode_task) if task is not None and not task.done()]

    async def _teardown_episode(self) -> None:
        await asyncio.gather(*self._episode_tasks(), return_exceptions=True)
        if self._task is not None:
            await self._task.teardown()

    # RUNTIME

    def _compromise(self, reason: str) -> None:
        """Record why the current episode is compromised and republish state."""
        if reason not in self._episodes.current.integrity:
            self._episodes.current.integrity.append(reason)
        self._publish_episode_state()

    async def _play_camera(self, request: str, params: str = "") -> tuple[bool, str]:
        """Pick the camera asset `request` selects in this episode and send it to the env's camera server."""
        log = self.get_logger()
        try:
            identifier, warning = await pick_camera(request)
            if warning is not None:
                log.warning(f"cam: {warning}")
            if identifier is None:
                return False, str(warning)
            if not self._cam_client.client.service_is_ready():
                if not self._cam_absent:
                    log.warning(f"cam: no camera server on {self._cam_client.client.srv_name}, start one with viz.cam")
                self._cam_absent = True
                return False, "no camera server"
            self._cam_absent = False
            spec = (await identifier.resolve()).spec
            bundle = await collect(identifier.name, spec)
            play = arena_cam_msgs.srv.Play.Request(root=identifier.name, bundle=yaml.safe_dump(bundle, sort_keys=False), params=params)
            response = await self._cam_client.call_timeout(play)
        except (OSError, ValueError, LookupError, yaml.YAMLError) as e:
            log.warning(f"cam: {request!r} not played: {e}")
            return False, str(e)
        if response is None or not response.success:
            message = "camera server timed out" if response is None else response.message
            log.warning(f"cam: {identifier.name!r} refused: {message}")
            return False, message
        return True, identifier.name

    async def _cb_cam_play(self, request: arena_cam_msgs.srv.Play.Request, response: arena_cam_msgs.srv.Play.Response) -> arena_cam_msgs.srv.Play.Response:
        if self._episodes.current.outcome_state != arena_runtime_msgs.action.RunEpisode.Result.RUNNING:
            response.message = "no episode is running, cam.setup plays at the next episode start"
            return response
        response.success, response.message = await self._play_camera(request.root, request.params)
        return response

    def _mode_namespaces(self, modes: TaskModes) -> dict[tuple[str, str], str]:
        """The param namespace of every selected mode the layer registers one for, by (axis, mode)."""
        spaces = self._task_layer.mode_namespaces()
        return {(axis, mode): str(spaces[axis][mode]) for axis, names in modes.items() for mode in names if mode in spaces.get(axis, {})}

    def _mode_params(self, modes: TaskModes) -> dict[str, ParameterValue]:
        """Live value of every parameter under the namespaces of the selected modes, by full name."""
        prefixes = sorted(set(self._mode_namespaces(modes).values()))
        names = self.list_parameters(prefixes=prefixes, depth=10).names if prefixes else []
        return {p.name: p.get_parameter_value() for p in self.get_parameters(list(names))}

    def _record_to_msg(self, record: EpisodeRecord, params: Mapping[str, ParameterValue]) -> arena_runtime_msgs.msg.EpisodeRecord:
        msg = arena_runtime_msgs.msg.EpisodeRecord()
        msg.episode_id = record.episode_id
        msg.world = record.world
        msg.seed = record.seed
        msg.modes = encode_modes(record.modes)
        msg.task_layer = self.rosparam[str].get("task_layer", "")
        msg.params = [RclParameter(name=name, value=value) for name, value in params.items()]
        msg.robots = list(record.robots)
        msg.outcome_state = record.outcome_state
        msg.outcome_info = record.outcome_info
        msg.goal_uuid = record.goal_uuid
        msg.goal_dist_start = record.goal_dist_start
        msg.goal_dist_min = record.goal_dist_min
        msg.path_length = record.path_length
        msg.integrity = list(record.integrity)
        msg.start_time = record.start_time.to_msg()
        msg.conditions = json.dumps([c.serialize() for c in self._episode_conditions])
        msg.phases = self._phases_json()
        return msg

    def _publish_episode_state(self) -> None:
        record = self._episodes.current
        msg = self._record_to_msg(record, self._mode_params(record.modes))
        msg.human_params = [RclParameter(name=k, value=v) for k, v in self._human_params.items()]
        self._pub_state_episode.publish(msg)

    @staticmethod
    def _norm_token(value: object) -> str:
        """Stringify a literal to the event convention: booleans render 'true'/'false'."""
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    # SCENARIO TIMELINE

    def register_timeline(self, entries: "Sequence[TimelineEntry]", seed: int) -> None:
        """Arm the scenario timeline for the current episode under one seed."""
        self._timeline = list(entries)
        self._timeline_seed = seed
        self._timeline_t0 = None
        self._timeline_state = []
        for idx, entry in enumerate(self._timeline):
            if entry.every is not None and entry.every > 0.0:
                nxt = entry.offset if entry.offset > 0.0 else entry.every
            else:
                nxt = float("inf")
            self._timeline_state.append(
                {"fired": False, "next": nxt, "prev": False, "rng": random.Random(f"{seed}:{idx}"), "when": entry.when_node(), "monitors": {}},
            )

    def reset_timeline(self) -> None:
        """Clear timeline arm state, episode conditions and inert-zone overrides so the next reset replays identically."""
        self._timeline = []
        self._timeline_state = []
        self._timeline_t0 = None
        for task in self._stimulus_tasks:
            task.cancel()
        self._stimulus_tasks.clear()
        self._stimuli_fired.clear()
        self._episode_conditions = []
        super().reset_timeline()

    def register_conditions(self, conditions: "Sequence[EpisodeCondition]") -> None:
        """Episode conditions judged on every robot and carried through to the episode record."""
        self._episode_conditions = list(conditions)
        if self._robots_manager is not None:
            for manager in self._robots_manager.managers.values():
                manager.set_episode_conditions(self._episode_conditions)

    def _resolve_timeline_value(self, raw: str, idx: int) -> str:
        """Draw a seeded value for 'random' or a 'lo..hi' range, else return the literal verbatim."""
        rng = self._timeline_state[idx]["rng"]
        if raw == "random":
            return str(rng.random())
        if ".." in raw:
            lo, _, hi = raw.partition("..")
            try:
                lo_f = float(lo)
                hi_f = float(hi)
            except ValueError:
                return raw
            return str(rng.uniform(lo_f, hi_f))
        return raw

    def _when_true(self, node: Node, monitors: dict[str, NodeMonitor]) -> bool:
        """Whether a `when` trigger holds for some robot of the env."""
        holds = False
        for robot in self._sample.robots or ("robot",):
            monitor = monitors.setdefault(robot, NodeMonitor(node, robot))
            holds = monitor.step(self._sample, self._judge_zones) is True or holds
        return holds

    def _fire_timeline_entry(self, idx: int, entry: "TimelineEntry") -> None:
        for action in entry.set:
            entity = str(action.get("entity", ""))
            field = str(action.get("field", ""))
            value = self._resolve_timeline_value(self._norm_token(action.get("value")), idx)
            self._set_semantic(entity, field, value)
        for stimulus in entry.stimulus:
            for agent in stimulus.agents:
                self._submit_stimulus(agent, stimulus.need, stimulus.intensity, self._sample.t)

    def _submit_stimulus(self, agent: str, need: str, intensity: float, fired_at: float) -> None:
        """Send one timeline stimulus without blocking the tick, recording `fired_at` once the engine takes it."""
        task = asyncio.create_task(self._environment_manager.notify_stimulus(agent, need, intensity))
        self._stimulus_tasks.add(task)

        def _on_done(t: asyncio.Task) -> None:
            if t not in self._stimulus_tasks:
                return
            self._stimulus_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                self.get_logger().warning(f"timeline stimulus {need!r} for {agent!r} failed: {exc!r}")
            elif not t.result():
                self.get_logger().warning(f"timeline stimulus {need!r} for {agent!r} was not taken")
            else:
                self._stimuli_fired.setdefault(f"stimulus.{agent}.{need}", fired_at)
                self._semantics_dirty = True

        task.add_done_callback(_on_done)

    def _timeline_semantic_state(self) -> arena_runtime_msgs.msg.SemanticEntityState:
        """The `timeline` entity: sim time of the tick each stimulus fired on, per `stimulus.<agent>.<need>`, grow-only."""
        msg = arena_runtime_msgs.msg.SemanticEntityState()
        msg.entity = self._realizer.prefix("timeline")
        msg.kind = "timeline"
        msg.discrete_names = list(self._stimuli_fired.keys())
        msg.discrete_values = [self._stringify_float(t) for t in self._stimuli_fired.values()]
        return msg

    def _evaluate_timeline(self, now: float) -> None:
        """One sim-time tick of timeline evaluation. Captures t0 on the first post-unpause tick."""
        if not self._timeline:
            return
        if self._timeline_t0 is None:
            self._timeline_t0 = now
        rel = now - self._timeline_t0
        for idx, entry in enumerate(self._timeline):
            state = self._timeline_state[idx]
            fired = False
            if entry.at is not None:
                if not state["fired"] and rel >= entry.at:
                    state["fired"] = True
                    fired = True
            elif entry.every is not None:
                nxt = state["next"]
                if rel >= nxt and (entry.until is None or nxt <= entry.until):
                    fired = True
                    while state["next"] <= rel:
                        state["next"] += entry.every
            elif (when := state["when"]) is not None:
                current = self._when_true(when, state["monitors"])
                if current and not state["prev"]:
                    fired = True
                state["prev"] = current
            if fired:
                self._fire_timeline_entry(idx, entry)

    # TASK JUDGE

    def on_task_submitted(self, manager: "RobotManager") -> None:
        """Republish the episode record with the robot's phases and flush its semantic state."""
        super().on_task_submitted(manager)
        self._publish_episode_state()

    def _phases_json(self) -> str:
        if self._robots_manager is None:
            return "{}"
        return json.dumps({manager.name: self._robot_phases(manager) for manager in self._robots_manager.managers.values()})

    def _robot_phases(self, manager: "RobotManager") -> dict:
        """The runner's phases and condition spans plus `map_poses`, the realized goto pose per phase (null without one)."""
        data = manager.runner.serialize()
        data["map_poses"] = [list(self._realizer.realize(s.phase.pose).to_2d()) if isinstance(s.phase, GoToPhase) and s.phase.pose is not None else None for s in manager.runner.phases]
        return data

    def set_episode_info(self, info: str) -> None:
        self._episodes.current.outcome_info = info
        self._publish_episode_state()

    @property
    def _loaded_world(self) -> str:
        return "" if self._robots_manager is None else self._world_manager.loaded_world

    def _next_record(self, world: str = "", seed: int = -1) -> EpisodeRecord:
        """The record the next episode runs as: pending overrides over what is live, an explicit world or seed over both."""
        pending = self._episodes.pending
        episode_id = self._episodes.current.episode_id + 1
        resolved_world = world or pending.world or self._loaded_world or self._episodes.current.world
        run_seed = self.rosparam[str].get("run_seed", "") or self._episodes.run_seed
        if seed < 0:
            seed = pending.seed if pending.seed >= 0 else _derive_seed(run_seed, resolved_world, episode_id)
        return EpisodeRecord(episode_id=episode_id, world=resolved_world, seed=seed, modes=self.task_params.resolve(pending.modes))

    def _publish_queue_state(self) -> None:
        record = self._next_record()
        if self._robots_manager is not None:
            record.robots = [m.model_name for m in self._robots_manager.managers.values()]
        record.outcome_state = arena_runtime_msgs.msg.EpisodeRecord.QUEUED
        msg = self._record_to_msg(record, self._mode_params(record.modes) | self._staged_params)
        msg.human_params = [RclParameter(name=k, value=v) for k, v in (self._human_params | self._staged_human_params).items()]
        self._pub_state_queue.publish(msg)

    async def _decide_next(self) -> None:
        """Ask the task layer once per episode what the next one changes, keeping the answer only when it came back."""
        async with self._pending_lock:
            if self._episodes.decided:
                return
            self._episodes.pending = await self._task_layer.next_episode(previous=self._episodes.current, queued=self._episodes.pending)
            self._episodes.decided = True

    async def _decide_after_episode(self) -> None:
        try:
            await self._decide_next()
        except Exception as e:
            self.get_logger().error(f"next_episode failed, asking again when the next episode builds: {e!r}")
        self._publish_queue_state()

    async def _build_next_record(self, world: str, seed: int) -> EpisodeRecord:
        await self._decide_next()
        record = self._next_record(world, seed)
        self.task_params.select(record.modes)
        self._episodes.current = record
        self._episodes.pending = TaskModeOverrides()
        self._episodes.decided = False
        self._goal_progress.reset()

        self._publish_queue_state()
        return record

    async def _run_reset_cycle(self) -> None:
        if self._task is None or self._robots_manager is None:
            raise RuntimeError("task generator not initialized")
        async with self._reset_lock:
            self._assigned.clear()
            self.markers.clear()
            self._entities.clear()
            self._start_time = self.sim_time
            self.get_logger().info("resetting")
            self._pub_state_resetting.publish(Bool(data=True))

            record = self._episodes.current
            await self.hold("reset")
            for manager in self._robots_manager.managers.values():
                manager.begin_episode()
            self._known_subjects.clear()
            try:
                if record.world:
                    await self._world_manager.apply_world(record.world)
                await self._reset_episode(record)
            finally:
                compacted = self._world_manager.world_compacted()
                self._judge_zones = zone_polygons(compacted)
                self._judge_walls = self._wall_grid(compacted)
                for manager in self._robots_manager.managers.values():
                    manager.set_episode_conditions(self._episode_conditions)
                await self.release("reset")
                self._pub_state_resetting.publish(Bool(data=False))
            record.robots = [m.model_name for m in self._robots_manager.managers.values()]

            self._pub_task_reset.publish(Int16(data=record.episode_id - 1))

            self._pub_state_world.publish(String(data=record.world))

            # This is the first instant at which the reset world, robot and
            # pedestrians are all committed. Dataset export clips audio to it.
            record.start_time = self.sim_time
            record.outcome_state = arena_runtime_msgs.action.RunEpisode.Result.RUNNING
            self._publish_episode_state()
            if self._cam_setup.value:
                self._cam_task = asyncio.ensure_future(self._play_camera(self._cam_setup.value))

            log = self.get_logger()
            log.warn("=============")
            log.warn(f"EPISODE STARTED #{record.episode_id}")
            log.info(f"  world:        {record.world}")
            log.info(f"  seed:         {record.seed}")
            for axis, names in record.modes.items():
                log.info(f"  {axis}: {names}")
            log.info(f"  robots:       {record.robots}")

            def _fmt(v: object) -> object:
                # rclpy returns int/float-array params as array.array, which renders as "array('q', [...])".
                return list(v) if isinstance(v, array.array) else v

            params = self._mode_params(record.modes)
            if params:
                log.info("  params:")
                for name, value in params.items():
                    log.info(f"    {name}: {_fmt(Parameter.from_parameter_msg(RclParameter(name=name, value=value)).value)}")

    def _sample_goal_progress(self) -> None:
        assert self._robots_manager is not None
        for manager in self._robots_manager.managers.values():
            pose = manager.pose
            goal = manager.goal
            if pose is None or goal is None:
                continue
            self._goal_progress.sample(
                manager.name,
                (pose.position.x, pose.position.y),
                (goal.position.x, goal.position.y),
                self.sim_time.to_seconds(),
                phase=manager.runner.active,
                waiting=manager.waiting,
            )
        best = self._goal_progress.least_progress()
        if best is None:
            return
        record = self._episodes.current
        record.goal_dist_start = best.start_dist
        record.goal_dist_min = best.min_dist
        record.path_length = best.path_length

    async def _termination_watcher(self) -> None:
        assert self._task is not None
        try:
            while True:
                await asyncio.sleep(0.5)
                episode_id = self._episodes.current.episode_id
                fut = self._episodes.pending_outcomes.get(episode_id)
                if fut is None or fut.done():
                    continue
                self._sample_goal_progress()
                stall_limit = self.conf.Robot.NO_PROGRESS_TIMEOUT.value
                if self._goal_progress.longest_stall(self.sim_time.to_seconds(), self.conf.Robot.GOAL_TOLERANCE_RADIUS.value) > stall_limit:
                    self.fail_episode("no progress")
                    continue
                if not await self._episode_done():
                    continue
                self._close_condition_scopes()
                if self._abort_reason is not None:
                    fut.set_result((arena_runtime_msgs.action.RunEpisode.Result.FAILED, self._abort_reason))
                else:
                    fut.set_result((arena_runtime_msgs.action.RunEpisode.Result.SUCCESS, ""))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.get_logger().error(f"Error in termination watcher: {e}\n{traceback.format_exc()}")
            raise

    def _send_end_message_on_end(self) -> bool:
        desired = self.conf.General.DESIRED_EPISODES.value
        if desired < 0 or self._episodes.current.episode_id < desired:
            return False

        self.get_logger().info(f"Shutting down. All {int(desired)} tasks completed")
        self.request_shutdown("all episodes completed")
        return True

    def _close_condition_scopes(self) -> None:
        """Judge every robot's still open condition scope at episode end."""
        assert self._robots_manager is not None
        for manager in self._robots_manager.managers.values():
            manager.finish_episode()
        self._publish_semantics_snapshot()

    def abort_episode(self, reason: str) -> None:
        self._abort_reason = reason
        self._force_reset = True

    def force_reset(self) -> None:
        self._force_reset = True

    async def _episode_done(self) -> bool:
        assert self._task is not None
        if self._force_reset:
            return True
        if (self.sim_time.sec - self._reset_sim_time) > self.conf.Robot.TIMEOUT.value:
            self.abort_episode("timeout")
            return True
        return await self._task.done

    async def _reset_episode(self, record: EpisodeRecord) -> None:
        assert self._task is not None and self._robots_manager is not None
        self._force_reset = False
        self._abort_reason = None
        self._reset_sim_time = self.sim_time.sec
        environment = self._environment_manager
        try:
            self._pub_reset_start.publish(Empty())

            self.conf.General.RNG.reseed(record.seed)
            point_scenario_assets(None)

            await self._robots_manager.set_up()
            await self._robots_manager.launch_pending()

            await environment.before_reset_episode()

            try:
                self._apply_staged_params()
                await self._apply_staged_human_params()

                plan = await self._task.reset_robots(record)
                for name, entry in plan.robots.items():
                    if entry.task is not None:
                        await self._robots_manager.managers[name].submit_task(entry.task, entry.start_pose, pedestrians=entry.humans)
                self._world_manager.forbid(plan.forbidden_zones)
                await self._task.reset_obstacles(plan, record)

                async def respawn() -> None:
                    await asyncio.gather(
                        environment.spawn_dynamic_obstacles(plan.dynamic_obstacles),
                        environment.spawn_obstacles(plan.obstacles),
                    )
                    await environment.spawn_mode_lights(plan.lights)

                await environment.respawn(respawn)

                robot_outcomes = await asyncio.gather(
                    *(
                        manager.reset(
                            ResetContext(
                                rng=self.conf.General.RNG.stream("robot-adapter", name),
                                start_pose=entry.start_pose if (entry := plan.robots.get(name)) else None,
                                episode_index=self.episode_id,
                            )
                        )
                        for name, manager in self._robots_manager.managers.items()
                    ),
                    return_exceptions=True,
                )
                reasons: list[str] = []
                for name, outcome in zip(self._robots_manager.managers, robot_outcomes, strict=True):
                    if isinstance(outcome, BaseException):
                        failures: dict[str, BaseException] = {"reset": outcome}
                    else:
                        failures = {kind: exc for kind, exc in outcome.items() if exc is not None}
                    if not failures:
                        continue
                    detail = "; ".join(f"{kind}: {exc!r}" for kind, exc in failures.items())
                    reasons.append(f"robot {name!r} adapter reset failed: {detail}")

                if reasons:
                    self._consecutive_reset_failures += 1
                    for reason in reasons:
                        self.get_logger().error(reason)
                    if self._consecutive_reset_failures >= _MAX_CONSECUTIVE_RESET_FAILURES:
                        raise RuntimeError(f"robot reset failed on {self._consecutive_reset_failures} consecutive episodes; treating the environment as unusable: {'; '.join(reasons)}")
                    self.fail_episode(reasons[0] if len(reasons) == 1 else "; ".join(reasons))
                else:
                    self._consecutive_reset_failures = 0

                self._task.after_reset()
            finally:
                await environment.after_reset_episode()

        except Exception as e:
            self.get_logger().error(repr(e))
            raise

        finally:
            self._pub_reset_end.publish(Empty())

    def _on_collision(self) -> None:
        if self.rosparam[bool].get_unsafe("task.episode.fail_on_collision"):
            self.fail_episode("collision")

    def fail_episode(self, reason: str) -> None:
        """Abort the running episode as FAILED with ``reason``, unless it is already aborting."""
        if self._task is None or self._abort_reason is not None:
            return
        self.get_logger().warn(f"failing episode: {reason}")
        self.abort_episode(reason)
        if self._robots_manager is not None:
            self._close_condition_scopes()
        fut = self._episodes.pending_outcomes.get(self._episodes.current.episode_id)
        if fut is not None and not fut.done():
            fut.set_result((arena_runtime_msgs.action.RunEpisode.Result.FAILED, reason))

    # SERVICE CALLBACKS

    async def _cb_reset_episode(
        self,
        request: arena_runtime_msgs.srv.ResetEpisode.Request,
        response: arena_runtime_msgs.srv.ResetEpisode.Response,
    ) -> arena_runtime_msgs.srv.ResetEpisode.Response:
        if self._task is None:
            response.success = False
            response.error_msg = "task generator not initialized"
            return response
        async with self._pending_lock:
            self._episodes.pending.merge(TaskModeOverrides(world=request.world, seed=request.seed))
        if self._episodes.action_in_flight:
            self._episodes.respawn_requested = True
            fut = self._episodes.pending_outcomes.get(self._episodes.current.episode_id)
            if fut is not None and not fut.done():
                fut.set_result((arena_runtime_msgs.action.RunEpisode.Result.SKIPPED, "reset"))
        else:
            self._spawn_episode()
        response.success = True
        return response

    async def _cb_query_scenarios(
        self,
        request: arena_runtime_msgs.srv.QueryScenarios.Request,
        response: arena_runtime_msgs.srv.QueryScenarios.Response,
    ) -> arena_runtime_msgs.srv.QueryScenarios.Response:
        world_name = request.world or self._episodes.current.world
        try:
            world_view = World.WorldIdentifier(world_name).resolve_sync()
        except FileNotFoundError:
            self.get_logger().warning(f'no scenarios: world {world_name!r} not found')
            return response
        response.ids = _available(world_view.scenario)
        return response

    async def _cb_queue_episode(
        self,
        request: arena_runtime_msgs.srv.QueueEpisode.Request,
        response: arena_runtime_msgs.srv.QueueEpisode.Response,
    ) -> arena_runtime_msgs.srv.QueueEpisode.Response:
        Req = arena_runtime_msgs.srv.QueueEpisode.Request
        if request.action != Req.MERGE:
            valid = ", ".join(f"{name}={val}" for name, val in (("MERGE", Req.MERGE),))
            response.success = False
            response.error_msg = f"action: unknown value {request.action!r}. Valid: {valid}"
            return response

        try:
            requested = self._task_layer.validate_modes(decode_modes(request.modes))
        except ValueError as e:
            response.success = False
            response.error_msg = str(e)
            return response

        namespaces = sorted(Constants.HUMAN_PARAM_NAMESPACES.values())
        for p in request.human_params:
            namespace, _, leaf = p.name.partition(".")
            if namespace == Constants.HUMAN_PARAM_RESERVED:
                response.success = False
                response.error_msg = f"human_params: {p.name!r} is reserved, no universal params are defined. Namespaces: {', '.join(namespaces)}"
                return response
            if namespace not in namespaces or not leaf:
                response.success = False
                response.error_msg = f"human_params: {p.name!r} has no known namespace. Namespaces: {', '.join(namespaces)}"
                return response

        async with self._pending_lock:
            pending = self._episodes.pending
            before = self._mode_namespaces(self.task_params.resolve(pending.modes))
            after = self._mode_namespaces(self.task_params.resolve({**pending.modes, **requested}))
            params: dict[str, ParameterValue] = {p.name: p.value for p in request.params}

            kept = set(after.values())
            stale = tuple(f"{namespace}." for key, namespace in before.items() if key not in after and namespace not in kept)
            for name in [name for name in self._staged_params if name.startswith(stale)]:
                del self._staged_params[name]

            pending.merge(TaskModeOverrides(modes=requested, world=request.world))
            self._staged_params.update(params)
            for p in request.human_params:
                self._staged_human_params[p.name] = p.value

        mid_episode = self._episodes.action_in_flight and self._episodes.current.episode_id > 0
        if mid_episode:
            self._compromise("task parameters changed mid-episode")

        self._publish_queue_state()

        response.success = True
        return response

    def _apply_staged_params(self) -> None:
        log = self.get_logger()
        batch: list[Parameter] = []
        for name, value in self._staged_params.items():
            if self.has_parameter(name):
                batch.append(Parameter.from_parameter_msg(RclParameter(name=name, value=value)))
            else:
                log.warning(f"staged param {name!r} is not declared, dropping")
        self._staged_params.clear()
        if batch:
            result = self.set_parameters_atomically(batch)
            if not result.successful:
                log.warning(f"staged params {[p.name for p in batch]} rejected: {result.reason}")

    async def _apply_staged_human_params(self) -> None:
        staged = [RclParameter(name=k, value=v) for k, v in self._staged_human_params.items()]
        self._staged_human_params.clear()
        for p in await self._environment_manager.configure_humans(staged):
            self._human_params[p.name] = p.value

    async def _cb_get_task_modes(
        self,
        request: arena_runtime_msgs.srv.GetTaskModes.Request,
        response: arena_runtime_msgs.srv.GetTaskModes.Response,
    ) -> arena_runtime_msgs.srv.GetTaskModes.Response:
        response.modes = encode_modes(self.task_params.resolve({}))
        return response

    async def _cb_query_task_modes(
        self,
        request: arena_runtime_msgs.srv.QueryTaskModes.Request,
        response: arena_runtime_msgs.srv.QueryTaskModes.Response,
    ) -> arena_runtime_msgs.srv.QueryTaskModes.Response:
        modes = registered_modes(self._task_layer)
        response.modes = encode_modes(modes)
        response.namespaces = encode_namespaces(self._task_layer.mode_namespaces())
        return response

    def _pose_from_request(self, stamped: geometry_msgs.msg.PoseStamped) -> Pose:
        """A map-frame request pose, shifted back into the world frame the task modes place entities in."""
        if not _is_map_frame(stamped.header.frame_id):
            raise ValueError(f"pose must be in the map frame, got {stamped.header.frame_id!r}")
        p = stamped.pose.position
        return self._realizer.ezilear(Pose(Position(p.x, p.y), orientation=Orientation.from_msg(stamped.pose.orientation)))

    async def _cb_spawn_static(
        self,
        request: arena_runtime_msgs.srv.SpawnStatic.Request,
        response: arena_runtime_msgs.srv.SpawnStatic.Response,
    ) -> arena_runtime_msgs.srv.SpawnStatic.Response:
        if self._task is None:
            response.success = False
            response.error_msg = "task generator not initialized"
            return response
        try:
            await arena_simulation_setup.tree.assets.Object.ObjectIdentifier.parse(request.model).resolve_path()
            pose = self._pose_from_request(request.pose) if request.use_pose else None
            obstacle = Obstacle(name=_extension_name(request.model), model=request.model, pose=pose if pose is not None else random_obstacle_pose(self._world_manager), level="")
            await self._environment_manager.spawn_obstacles([obstacle])
            self._entities.add(obstacle)
            self._compromise(f"static {request.model} spawned by hand")
            response.id = obstacle.sim_path
            response.success = True
        except Exception as e:
            response.success = False
            response.error_msg = str(e)
        return response

    async def _cb_spawn_dynamic(
        self,
        request: arena_runtime_msgs.srv.SpawnDynamic.Request,
        response: arena_runtime_msgs.srv.SpawnDynamic.Response,
    ) -> arena_runtime_msgs.srv.SpawnDynamic.Response:
        if self._task is None:
            response.success = False
            response.error_msg = "task generator not initialized"
            return response
        try:
            await arena_simulation_setup.tree.assets.Human.HumanIdentifier.parse(request.model).resolve_path()
            pose = self._pose_from_request(request.pose) if request.use_pose else None
            resolved_pose = pose if pose is not None else random_obstacle_pose(self._world_manager)
            waypoints = self._world_manager.get_positions_on_map(n=2, safe_dist=0, forbid=False, level_id="")
            obstacle = DynamicObstacle(name=_extension_name(request.model), model=request.model, waypoints=waypoints, pose=resolved_pose, level="")
            await self._environment_manager.spawn_dynamic_obstacles([obstacle])
            self._compromise(f"dynamic {request.model} spawned by hand")
            response.id = obstacle.sim_path
            response.success = True
        except Exception as e:
            response.success = False
            response.error_msg = str(e)
        return response

    def _add_pending_robot(self, model: str, name: str | None, pose: Pose | None, args: dict[str, str]) -> str:
        assert self._robots_manager is not None
        assigned_name = name or self._robots_manager.next_name(model)
        (config,) = RobotSetupConfig.parse({"robot": model, "name": assigned_name, **args})
        if "pos" not in config.extra:
            if pose is None:
                clearance = footprint_radius(RobotIdentifier.parse(config.robot).resolve_sync()) + self.conf.Robot.SPAWN_ROBOT_SAFE_DIST.value
                pose = random_robot_pose(self._world_manager, self._robots_manager, safe_dist=clearance)
            config.extra["pos"] = pose.to_2d()
        self._robots_manager.add_pending(assigned_name, Robot.from_setup(config, node=self))
        return assigned_name

    async def _cb_despawn_robot(
        self,
        request: arena_runtime_msgs.srv.DespawnRobot.Request,
        response: arena_runtime_msgs.srv.DespawnRobot.Response,
    ) -> arena_runtime_msgs.srv.DespawnRobot.Response:
        if not self.rosparam[bool].get("initialized", False) or self._robots_manager is None:
            response.success = False
            response.error_msg = "task generator not initialized"
            return response
        try:
            self._robots_manager.remove_pending(request.name)
            self._robots_manager.publish_queue()
            self._compromise(f"robot {request.name} despawned by hand")
            response.success = True
        except Exception as e:
            response.success = False
            response.error_msg = str(e)
        return response

    async def _cb_spawn_robot(
        self,
        request: arena_runtime_msgs.srv.SpawnRobot.Request,
        response: arena_runtime_msgs.srv.SpawnRobot.Response,
    ) -> arena_runtime_msgs.srv.SpawnRobot.Response:
        if not self.rosparam[bool].get("initialized", False) or self._task is None or self._robots_manager is None:
            response.success = False
            response.error_msg = "task generator not initialized"
            return response
        try:
            pose = self._pose_from_request(request.pose) if request.use_pose else None
            args = {kv.key: kv.value for kv in request.args}
            name_out = self._add_pending_robot(request.model, request.name or None, pose, args)
            if request.immediate:
                async with self._reset_lock:
                    await self._robots_manager.spawn_now(name_out, pose, self.episode_id)
            self._robots_manager.publish_queue()
            self._compromise(f"robot {name_out} spawned by hand")
            response.name = name_out
            response.success = True
        except Exception as e:
            response.success = False
            response.error_msg = str(e)
        return response

    async def _cb_move_entity(
        self,
        request: arena_runtime_msgs.srv.MoveEntity.Request,
        response: arena_runtime_msgs.srv.MoveEntity.Response,
    ) -> arena_runtime_msgs.srv.MoveEntity.Response:
        if not self.rosparam[bool].get("initialized", False) or self._robots_manager is None:
            response.success = False
            response.error_msg = "task generator not initialized"
            return response
        if not _is_map_frame(request.pose.header.frame_id):
            response.success = False
            response.error_msg = f"pose must be in the map frame, got {request.pose.header.frame_id!r}"
            return response
        try:
            manager = self._robots_manager.managers.get(request.entity)
            if manager is not None:
                local = self._realizer.ezilear(Pose.from_msg(request.pose.pose))
                async with self._reset_lock:
                    await manager.move(local)
            else:
                try:
                    await self.markers.move(request.entity, request.pose.pose)
                except KeyError:
                    response.success = False
                    response.error_msg = f"unknown entity {request.entity!r}"
                    return response
            self._compromise(f"{request.entity} moved by hand")
            response.success = True
        except Exception as e:
            response.success = False
            response.error_msg = str(e)
        return response

    async def _assign_task(self, robot: str, task_yaml: str) -> tuple["RobotManager", list[int]]:
        """Hand a robot the parsed task, replacing its unfinished phases, an empty one stopping it, and return its manager and the episode-wide indices of the new phases."""
        if not self.rosparam[bool].get("initialized", False) or self._robots_manager is None:
            raise ValueError("task generator not initialized")
        manager = self._robots_manager.managers.get(robot)
        if manager is None:
            raise ValueError(f"unknown robot {robot!r}, robots: {sorted(self._robots_manager.managers)}")
        task = TaskRequest.parse(yaml.safe_load(task_yaml) or [])
        self._ezilear_phases(task.phases)
        async with self._reset_lock:
            first = len(manager.runner.phases)
            if task.phases:
                await manager.submit_task(task)
            else:
                manager.stop_task()
            self._assigned.add(robot)
            return manager, list(range(first, len(manager.runner.phases)))

    async def _cb_submit_task(
        self,
        request: arena_runtime_msgs.srv.SubmitTask.Request,
        response: arena_runtime_msgs.srv.SubmitTask.Response,
    ) -> arena_runtime_msgs.srv.SubmitTask.Response:
        try:
            _, response.phase_indices = await self._assign_task(request.robot, request.task_yaml)
            response.success = True
        except Exception as e:
            response.success = False
            response.error_msg = str(e)
        return response

    def _ezilear_phases(self, phases: Sequence[TaskPhase]) -> None:
        """Shift map-frame goto poses into scenario coordinates in place."""
        for phase in phases:
            if isinstance(phase, GoToPhase) and phase.pose is not None:
                phase.pose = self._realizer.ezilear(phase.pose)
            elif isinstance(phase, LoopPhase):
                self._ezilear_phases(phase.phases)

    # RUN TASK ACTION

    def _run_task_goal_callback(self, goal_request: object) -> GoalResponse:
        return GoalResponse.REJECT if self._shutting_down else GoalResponse.ACCEPT

    def _run_task_cancel_callback(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        self.event_loop.call_soon_threadsafe(self._stop_run_task, bytes(goal_handle.goal_id.uuid))
        return CancelResponse.ACCEPT

    def _stop_run_task(self, goal_id: bytes) -> None:
        for robot, run in list(self._run_tasks.items()):
            if run.goal_id == goal_id:
                run.cancelled = True
                manager = self._robots_manager.managers.get(robot) if self._robots_manager is not None and run.episode_id == self.episode_id else None
                if manager is not None and manager.runner.request_outcome(run.request) is None:
                    manager.stop_task()
                self._advance_run_task(robot, run)
                return

    async def _run_task_execute_callback(self, goal_handle: ServerGoalHandle) -> arena_runtime_msgs.action.RunTask.Result:
        result = await self.loop_future(self._run_task(goal_handle))
        assert isinstance(result, arena_runtime_msgs.action.RunTask.Result)
        return result

    async def _run_task(self, goal_handle: ServerGoalHandle) -> arena_runtime_msgs.action.RunTask.Result:
        result = arena_runtime_msgs.action.RunTask.Result()
        robot = goal_handle.request.robot
        try:
            manager, indices = await self._assign_task(robot, goal_handle.request.task_yaml)
        except Exception as e:
            result.error_msg = str(e)
            goal_handle.abort()
            return result
        result.phase_indices = indices
        if not indices:
            result.outcome = "met"
            goal_handle.succeed()
            return result
        run = RunningTask(handle=goal_handle, episode_id=self.episode_id, request=manager.runner.phases[indices[0]].request, phases=indices, done=self.event_loop.create_future())
        replaced = self._run_tasks.get(robot)
        self._run_tasks[robot] = run
        if replaced is not None:
            self._advance_run_task(robot, replaced)
        if goal_handle.is_cancel_requested:
            self._stop_run_task(run.goal_id)
        result.outcome = await run.done
        result.phase_indices = run.phases
        if run.cancelled:
            await self._until_cancel_requested(goal_handle)
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif result.outcome in ("met", "failed"):
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    async def _until_cancel_requested(self, goal_handle: ServerGoalHandle, timeout: float = 1.0) -> None:
        """Wait for rclpy to move an accepted cancel to CANCELING, which it does only after the cancel callback returned."""
        deadline = self.event_loop.time() + timeout
        while not goal_handle.is_cancel_requested and self.event_loop.time() < deadline:
            await asyncio.sleep(0.005)

    def _advance_run_tasks(self) -> None:
        for robot, run in list(self._run_tasks.items()):
            self._advance_run_task(robot, run)

    def _advance_run_task(self, robot: str, run: RunningTask) -> None:
        manager = None if self._robots_manager is None else self._robots_manager.managers.get(robot)
        runner = manager.runner if manager is not None and run.episode_id == self.episode_id else None
        for index, phase_outcome in run.ended(runner):
            run.handle.publish_feedback(arena_runtime_msgs.action.RunTask.Feedback(phase_indices=run.phases, phase_index=index, outcome=phase_outcome))
        outcome = "dropped" if runner is None else runner.request_outcome(run.request)
        if outcome is not None:
            if self._run_tasks.get(robot) is run:
                del self._run_tasks[robot]
            run.done.set_result(outcome)

    # ACTION SERVER

    def _goal_callback(self, goal_request: object) -> GoalResponse:
        return GoalResponse.REJECT if self._shutting_down or self._episodes.action_in_flight else GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle: object) -> CancelResponse:
        episode_id = self._episodes.current.episode_id
        fut = self._episodes.pending_outcomes.get(episode_id)
        if fut is not None and not fut.done():
            self.event_loop.call_soon_threadsafe(
                fut.set_result,
                (arena_runtime_msgs.action.RunEpisode.Result.SKIPPED, "cancelled"),
            )
        return CancelResponse.ACCEPT

    async def _execute_callback(self, goal_handle: ServerGoalHandle) -> arena_runtime_msgs.action.RunEpisode.Result:
        result = await self.loop_future(
            self._run_episode(
                world=goal_handle.request.world,
                seed=goal_handle.request.seed,
                goal_handle=goal_handle,
            )
        )
        assert isinstance(result, arena_runtime_msgs.action.RunEpisode.Result)
        return result

    async def _run_episode(
        self,
        *,
        world: str = "",
        seed: int = -1,
        goal_handle: ServerGoalHandle | None = None,
    ) -> arena_runtime_msgs.action.RunEpisode.Result:
        if self._shutting_down or self._episodes.action_in_flight:
            if goal_handle is not None:
                goal_handle.abort()
            return arena_runtime_msgs.action.RunEpisode.Result(
                state=arena_runtime_msgs.action.RunEpisode.Result.SKIPPED,
                info=RUN_EPISODE_SHUTTING_DOWN if self._shutting_down else RUN_EPISODE_BUSY,
            )

        self._episodes.action_in_flight = True
        record: EpisodeRecord | None = None
        episode_id = 0
        respawn = True
        outcome_state = arena_runtime_msgs.action.RunEpisode.Result.FAILED
        outcome_info = ""
        try:
            try:
                record = await self._build_next_record(world, seed)
                if goal_handle is not None:
                    record.goal_uuid = bytes(goal_handle.goal_id.uuid).hex()

                    feedback = arena_runtime_msgs.action.RunEpisode.Feedback()
                    feedback.state = arena_runtime_msgs.action.RunEpisode.Feedback.STARTED
                    goal_handle.publish_feedback(feedback)

                await self._run_reset_cycle()
            except Exception as e:
                self.get_logger().error(f"reset_cycle failed: {e!r}\n{traceback.format_exc()}")
                outcome_state = arena_runtime_msgs.action.RunEpisode.Result.FATAL
                outcome_info = repr(e)
                respawn = False
            else:
                episode_id = record.episode_id
                fut = self.event_loop.create_future()
                self._episodes.pending_outcomes[episode_id] = fut

                try:
                    outcome_state, outcome_info = await fut
                except asyncio.CancelledError:
                    outcome_state = arena_runtime_msgs.action.RunEpisode.Result.SKIPPED
                    outcome_info = "cancelled"

            if not outcome_info:
                elapsed = (self.sim_time - self._start_time).to_seconds()
                verb = {
                    arena_runtime_msgs.action.RunEpisode.Result.SUCCESS: "finished",
                    arena_runtime_msgs.action.RunEpisode.Result.FAILED: "failed",
                    arena_runtime_msgs.action.RunEpisode.Result.FATAL: "failed",
                    arena_runtime_msgs.action.RunEpisode.Result.SKIPPED: "skipped",
                }.get(outcome_state, "ended")
                outcome_info = f"{verb} after {elapsed:.1f}s"

            if record is not None:
                record.outcome_state = outcome_state
                record.outcome_info = outcome_info
                self._publish_episode_state()

            if episode_id > 0:
                state_label = {
                    arena_runtime_msgs.action.RunEpisode.Result.SUCCESS: "SUCCESS",
                    arena_runtime_msgs.action.RunEpisode.Result.FAILED: "FAILED",
                    arena_runtime_msgs.action.RunEpisode.Result.SKIPPED: "SKIPPED",
                    arena_runtime_msgs.action.RunEpisode.Result.FATAL: "FATAL",
                }.get(outcome_state, str(outcome_state))
                duration = self.sim_time.to_seconds() - self._start_time.to_seconds()
                log = self.get_logger()
                log.info(f"  state:    {state_label}")
                if outcome_info:
                    log.info(f"  info:     {outcome_info}")
                log.info(f"  duration: {duration:.2f}s")
                log.warn(f"EPISODE FINISHED #{episode_id}")
                log.warn("=============")
                if self._send_end_message_on_end():
                    respawn = False
                else:
                    await self._decide_after_episode()
        finally:
            self._episodes.action_in_flight = False
            self._episodes.pending_outcomes.pop(episode_id, None)
            respawn_requested, self._episodes.respawn_requested = self._episodes.respawn_requested, False

        result = arena_runtime_msgs.action.RunEpisode.Result()
        result.state = outcome_state
        result.info = outcome_info
        result.episode_id = episode_id

        if goal_handle is not None:
            if outcome_state == arena_runtime_msgs.action.RunEpisode.Result.SKIPPED:
                goal_handle.canceled()
            elif outcome_state in (
                arena_runtime_msgs.action.RunEpisode.Result.FAILED,
                arena_runtime_msgs.action.RunEpisode.Result.FATAL,
            ):
                goal_handle.abort()
            else:
                goal_handle.succeed()

        fatal = outcome_state == arena_runtime_msgs.action.RunEpisode.Result.FATAL
        if respawn and rclpy.ok() and not fatal and (respawn_requested or self.rosparam[bool].get_unsafe("task.episode.auto_reset")):
            self._spawn_episode()

        return result

    def _spawn_episode(self, *, world: str = "", seed: int = -1) -> None:
        async def _wrap() -> None:
            await self._run_episode(world=world, seed=seed)

        task = asyncio.create_task(_wrap())
        self._episode_task = task

        def _on_done(t: asyncio.Task) -> None:
            if t.cancelled() or not rclpy.ok():
                return
            exc = t.exception()
            if exc is not None:
                self.get_logger().error(f"_run_episode task failed: {exc!r}\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

        task.add_done_callback(_on_done)

    async def _set_up_services(self):
        await super()._set_up_services()

        self.create_service(
            arena_runtime_msgs.srv.ResetEpisode,
            self.service_namespace("lifecycle", "reset_episode"),
            self._cb_reset_episode,
        )

        self.create_service(
            arena_cam_msgs.srv.Play,
            self.service_namespace("cam", "play"),
            self._cb_cam_play,
        )

        self._run_episode_action_server = ActionServer(
            self,
            arena_runtime_msgs.action.RunEpisode,
            self.service_namespace("lifecycle", "run_episode"),
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        self.create_service(
            arena_runtime_msgs.srv.QueryScenarios,
            self.service_namespace("query", "scenarios"),
            self._cb_query_scenarios,
        )

        self.create_service(
            arena_runtime_msgs.srv.QueueEpisode,
            self.service_namespace("config", "queue_episode"),
            self._cb_queue_episode,
        )

        self.create_service(
            arena_runtime_msgs.srv.GetTaskModes,
            self.service_namespace("config", "get_task_modes"),
            self._cb_get_task_modes,
        )

        self.create_service(
            arena_runtime_msgs.srv.QueryTaskModes,
            self.service_namespace("query", "task_modes"),
            self._cb_query_task_modes,
        )

        self.create_service(
            arena_runtime_msgs.srv.SpawnStatic,
            self.service_namespace("runtime", "spawn_static"),
            self._cb_spawn_static,
        )

        self.create_service(
            arena_runtime_msgs.srv.SpawnDynamic,
            self.service_namespace("runtime", "spawn_dynamic"),
            self._cb_spawn_dynamic,
        )

        self.create_service(
            arena_runtime_msgs.srv.SpawnRobot,
            self.service_namespace("runtime", "spawn_robot"),
            self._cb_spawn_robot,
        )

        self.create_service(
            arena_runtime_msgs.srv.DespawnRobot,
            self.service_namespace("runtime", "despawn_robot"),
            self._cb_despawn_robot,
        )

        self.create_service(
            arena_runtime_msgs.srv.MoveEntity,
            self.service_namespace("runtime", "move"),
            self._cb_move_entity,
        )

        self.create_service(
            arena_runtime_msgs.srv.SubmitTask,
            self.service_namespace("runtime", "submit_task"),
            self._cb_submit_task,
        )

        self._run_task_action_server = ActionServer(
            self,
            arena_runtime_msgs.action.RunTask,
            self.service_namespace("runtime", "run_task"),
            execute_callback=self._run_task_execute_callback,
            goal_callback=self._run_task_goal_callback,
            cancel_callback=self._run_task_cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
