import array
import asyncio
import hashlib
import json
import random
import traceback
import typing
import uuid
from collections.abc import Sequence

import arena_cam_msgs.srv
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
import task_generator_msgs.action
import task_generator_msgs.msg
import task_generator_msgs.srv
import yaml
from arena_rclpy_mixins.Time import Time
from arena_robots.SetupFile import Config as RobotSetupConfig
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
from task_generator_msgs.msg import AdapterDisplay

from arena_runtime.env.adapters import ResetContext
from arena_runtime.env.constants import Constants
from arena_runtime.env.constants.runtime import RUN_EPISODE_BUSY, RUN_EPISODE_SHUTTING_DOWN
from arena_runtime.env.goal_progress import GoalProgressTracker
from arena_runtime.env.interactive import MarkerHub
from arena_runtime.env.interactive.entities import EntityHandles
from arena_runtime.env.node import _LATCHED, EnvNode, _available
from arena_runtime.env.placement import random_obstacle_pose, random_robot_pose
from arena_runtime.env.task_layer import EpisodeTask, TaskContext, task_layer
from arena_runtime.shared import DynamicObstacle, Obstacle, Orientation, Pose, Position, Robot

if typing.TYPE_CHECKING:
    from arena_runtime.env.robot_manager import RobotManager

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
class EpisodeRecord:
    episode_id: int = 0
    world: str = ""
    seed: int = -1
    tm_robots: str = ""
    tm_obstacles: str = ""
    tm_modules: list[str] = attrs.Factory(list)
    robots: list[str] = attrs.Factory(list)
    outcome_state: int = 0
    outcome_info: str = ""
    goal_uuid: str = ""
    goal_dist_start: float = 0.0
    goal_dist_min: float = 0.0
    path_length: float = 0.0
    integrity: list[str] = attrs.Factory(list)
    start_time: Time = attrs.Factory(Time)


@attrs.define
class TaskModeOverrides:
    tm_robots: str = ""
    tm_obstacles: str = ""
    tm_modules: list[str] = attrs.Factory(list)
    keep_modules: bool = False
    world: str = ""


@attrs.define
class EpisodeRuntime:
    current: EpisodeRecord = attrs.Factory(EpisodeRecord)
    run_seed: str = attrs.field(factory=lambda: uuid.uuid4().hex)
    pending_outcomes: dict = attrs.Factory(dict)
    pending_overrides: TaskModeOverrides | None = None
    pending_world: str = ""
    pending_seed: int = -1
    action_in_flight: bool = False


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
        self._task_layer = task_layer()
        super().__init__("task_generator", automatically_declare_parameters_from_overrides=True, **kwargs)

        self.ROSParam[bool]("resetting", True)
        self.task_params = self._task_layer.declare_task_params(self)
        self._task_layer.declare_mode_schemas(self)

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

        self._staged_obstacles_params: dict[str, ParameterValue] = {}
        self._staged_robots_params: dict[str, ParameterValue] = {}
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
            task_generator_msgs.msg.EpisodeRecord,
            self.service_namespace("state", "episode"),
            _EPISODE_QOS,
        )

        self._pub_state_queue = self.create_publisher(
            task_generator_msgs.msg.EpisodeRecord,
            self.service_namespace("state", "queue"),
            rclpy.qos.QoSProfile(
                depth=queue_depth,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self._check_status_task: asyncio.Task | None = None
        self._episode_task: asyncio.Task | None = None
        self._shutting_down = False

        self._publish_bootstrap_queue_state()

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
            ctx=TaskContext(
                environment_manager=self._environment_manager,
                robots_manager=self._robots_manager,
                world_manager=self._world_manager,
                realizer=self._realizer,
                simulator=self._simulator,
                acoustics_simulator=self._acoustics_simulator,
                reset_lock=self._reset_lock,
                compromise=self._compromise,
                abort_episode=self.abort_episode,
            ),
        )

    def _after_setup(self) -> None:
        self._publish_queue_state()

    def _on_tick(self) -> None:
        self._evaluate_timeline(self._sample.t)

    def _extra_semantic_states(self) -> list[task_generator_msgs.msg.SemanticEntityState]:
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

    def _publish_bootstrap_queue_state(self) -> None:
        # Latch a minimal state/queue at construction so the rviz panel can
        # build its mode comboboxes and param tree before managers exist,
        # instead of waiting for _build_next_record at first reset.
        current_robots = self.task_params.robots
        current_obstacles = self.task_params.obstacles
        current_modules = self.task_params.modules

        msg = task_generator_msgs.msg.EpisodeRecord()
        msg.episode_id = 0
        msg.world = ""
        msg.seed = -1
        msg.tm_robots = current_robots
        msg.tm_obstacles = current_obstacles
        msg.tm_modules = list(current_modules)
        msg.robots = []
        msg.outcome_state = task_generator_msgs.msg.EpisodeRecord.QUEUED
        msg.outcome_info = ""
        msg.goal_uuid = ""
        msg.integrity = []
        msg.obstacles_params = self._params_for_mode(current_obstacles)
        msg.robots_params = self._params_for_mode(current_robots)
        self._pub_state_queue.publish(msg)

    def on_activate(self, state: rclpy.lifecycle.State) -> TransitionCallbackReturn:
        super().on_activate(state)

        def _start() -> None:
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
        if self._episodes.current.outcome_state != task_generator_msgs.action.RunEpisode.Result.RUNNING:
            response.message = "no episode is running, cam.setup plays at the next episode start"
            return response
        response.success, response.message = await self._play_camera(request.root, request.params)
        return response

    def _params_for_mode(self, mode: str) -> list[RclParameter]:
        if not mode:
            return []
        prefix = f"task.{mode}"
        names = self.list_parameters(prefixes=[prefix], depth=10).names
        if not names:
            return []
        params = self.get_parameters(list(names))
        result: list[RclParameter] = []
        for p in params:
            leaf = p.name[len(prefix) + 1 :]
            result.append(RclParameter(name=leaf, value=p.get_parameter_value()))
        return result

    def _record_to_msg(self, record: EpisodeRecord) -> task_generator_msgs.msg.EpisodeRecord:
        msg = task_generator_msgs.msg.EpisodeRecord()
        msg.episode_id = record.episode_id
        msg.world = record.world
        msg.seed = record.seed
        msg.tm_robots = record.tm_robots
        msg.tm_obstacles = record.tm_obstacles
        msg.tm_modules = list(record.tm_modules)
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
        msg = self._record_to_msg(self._episodes.current)
        msg.obstacles_params = self._params_for_mode(self._episodes.current.tm_obstacles)
        msg.robots_params = self._params_for_mode(self._episodes.current.tm_robots)
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

    def _timeline_semantic_state(self) -> task_generator_msgs.msg.SemanticEntityState:
        """The `timeline` entity: sim time of the tick each stimulus fired on, per `stimulus.<agent>.<need>`, grow-only."""
        msg = task_generator_msgs.msg.SemanticEntityState()
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

    def _publish_queue_state(self) -> None:
        overrides = self._episodes.pending_overrides
        current_robots = self.task_params.robots
        current_obstacles = self.task_params.obstacles
        current_modules = self.task_params.modules
        record_world = self._episodes.current.world
        loaded_world = self._world_manager.loaded_world
        queued_robots = [m.model_name for m in self._robots_manager.managers.values()] if self._robots_manager is not None else []

        if overrides is None:
            queued_tm_robots = current_robots
            queued_tm_obstacles = current_obstacles
            queued_tm_modules = current_modules
            queued_world = record_world or loaded_world
        else:
            queued_tm_robots = overrides.tm_robots or current_robots
            queued_tm_obstacles = overrides.tm_obstacles or current_obstacles
            queued_tm_modules = current_modules if overrides.keep_modules else overrides.tm_modules
            queued_world = overrides.world or record_world or loaded_world

        obstacles_live = self._params_for_mode(queued_tm_obstacles)
        robots_live = self._params_for_mode(queued_tm_robots)

        obstacles_map = {p.name: p.value for p in obstacles_live}
        for leaf, pv in self._staged_obstacles_params.items():
            obstacles_map[leaf] = pv
        robots_map = {p.name: p.value for p in robots_live}
        for leaf, pv in self._staged_robots_params.items():
            robots_map[leaf] = pv

        # When the two pools share the same mode (e.g. both 'scenario'), they
        # also share the underlying ROS param namespace. Mirror staged edits
        # across pools so the UI shows a single coherent value instead of one
        # side stuck on the live (unstaged) value.
        if queued_tm_obstacles and queued_tm_obstacles == queued_tm_robots:
            for leaf, pv in self._staged_obstacles_params.items():
                if leaf not in self._staged_robots_params:
                    robots_map[leaf] = pv
            for leaf, pv in self._staged_robots_params.items():
                if leaf not in self._staged_obstacles_params:
                    obstacles_map[leaf] = pv

        msg = task_generator_msgs.msg.EpisodeRecord()
        msg.episode_id = self._episodes.current.episode_id
        msg.world = queued_world
        msg.seed = -1
        msg.tm_robots = queued_tm_robots
        msg.tm_obstacles = queued_tm_obstacles
        msg.tm_modules = list(queued_tm_modules)
        msg.robots = list(queued_robots)
        msg.outcome_state = task_generator_msgs.msg.EpisodeRecord.QUEUED
        msg.outcome_info = ""
        msg.goal_uuid = ""
        msg.integrity = []
        msg.obstacles_params = [RclParameter(name=k, value=v) for k, v in obstacles_map.items()]
        msg.robots_params = [RclParameter(name=k, value=v) for k, v in robots_map.items()]
        msg.human_params = [RclParameter(name=k, value=v) for k, v in (self._human_params | self._staged_human_params).items()]
        self._pub_state_queue.publish(msg)

    async def _build_next_record(self, world: str, seed: int) -> None:
        new_id = self._episodes.current.episode_id + 1

        overrides = self._episodes.pending_overrides
        self._episodes.pending_overrides = None

        if overrides is not None and overrides.world:
            world = world or overrides.world
            # World swap itself happens inside `_run_reset_cycle`'s hold window.

        resolved_world = world or self._world_manager.loaded_world or self._episodes.current.world
        run_seed = self.rosparam[str].get("run_seed", "") or self._episodes.run_seed
        resolved_seed = seed if seed >= 0 else _derive_seed(run_seed, resolved_world, new_id)

        current_robots = self.task_params.robots
        current_obstacles = self.task_params.obstacles
        current_modules = self.task_params.modules

        if overrides is None:
            tm_robots, tm_obstacles, tm_modules = current_robots, current_obstacles, current_modules
        else:
            tm_robots = overrides.tm_robots or current_robots
            tm_obstacles = overrides.tm_obstacles or current_obstacles
            tm_modules = current_modules if overrides.keep_modules else overrides.tm_modules

        if tm_robots and tm_robots != current_robots:
            self.rosparam[str].set("task.robots", tm_robots)
        if tm_obstacles and tm_obstacles != current_obstacles:
            self.rosparam[str].set("task.obstacles", tm_obstacles)

        self._episodes.current = EpisodeRecord(
            episode_id=new_id,
            world=resolved_world,
            seed=resolved_seed,
            tm_robots=tm_robots,
            tm_obstacles=tm_obstacles,
            tm_modules=tm_modules,
        )
        self._goal_progress.reset()

        self._publish_queue_state()

    async def _run_reset_cycle(self) -> None:
        if self._task is None or self._robots_manager is None:
            raise RuntimeError("task generator not initialized")
        async with self._reset_lock:
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
                await self._reset_episode(record.seed)
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
            record.outcome_state = task_generator_msgs.action.RunEpisode.Result.RUNNING
            self._publish_episode_state()
            if self._cam_setup.value:
                self._cam_task = asyncio.ensure_future(self._play_camera(self._cam_setup.value))

            log = self.get_logger()
            log.warn("=============")
            log.warn(f"EPISODE STARTED #{record.episode_id}")
            log.info(f"  world:        {record.world}")
            log.info(f"  seed:         {record.seed}")
            log.info(f"  tm_robots:    {record.tm_robots}")
            log.info(f"  tm_obstacles: {record.tm_obstacles}")
            log.info(f"  tm_modules:   {record.tm_modules}")
            log.info(f"  robots:       {record.robots}")

            def _fmt(v: object) -> object:
                # rclpy returns int/float-array params as array.array, which renders as "array('q', [...])".
                return list(v) if isinstance(v, array.array) else v

            obstacles_params = self._params_for_mode(record.tm_obstacles)
            robots_params = self._params_for_mode(record.tm_robots)
            if obstacles_params:
                log.info(f"  {record.tm_obstacles} params:")
                for p in obstacles_params:
                    log.info(f"    {p.name}: {_fmt(Parameter.from_parameter_msg(p).value)}")
            if robots_params:
                log.info(f"  {record.tm_robots} params:")
                for p in robots_params:
                    log.info(f"    {p.name}: {_fmt(Parameter.from_parameter_msg(p).value)}")

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
                    fut.set_result((task_generator_msgs.action.RunEpisode.Result.FAILED, self._abort_reason))
                else:
                    fut.set_result((task_generator_msgs.action.RunEpisode.Result.SUCCESS, ""))
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

    async def _reset_episode(self, seed: int) -> None:
        assert self._task is not None and self._robots_manager is not None
        self._force_reset = False
        self._abort_reason = None
        self._reset_sim_time = self.sim_time.sec
        environment = self._environment_manager
        try:
            self._pub_reset_start.publish(Empty())

            self.conf.General.RNG.reseed(seed)
            point_scenario_assets(None)

            await self._robots_manager.set_up()
            await self._robots_manager.launch_pending()

            await environment.before_reset_episode()

            try:
                self._apply_staged_params()
                await self._apply_staged_human_params()

                plan = await self._task.prepare_reset(seed)

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
                                start_pose=self._task.tm_robots.start_poses.get(name),
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
            fut.set_result((task_generator_msgs.action.RunEpisode.Result.FAILED, reason))

    # SERVICE CALLBACKS

    async def _cb_reset_episode(
        self,
        request: task_generator_msgs.srv.ResetEpisode.Request,
        response: task_generator_msgs.srv.ResetEpisode.Response,
    ) -> task_generator_msgs.srv.ResetEpisode.Response:
        if self._task is None:
            response.success = False
            response.error_msg = "task generator not initialized"
            return response
        self._episodes.pending_world = request.world
        self._episodes.pending_seed = request.seed
        if self._episodes.action_in_flight:
            episode_id = self._episodes.current.episode_id
            fut = self._episodes.pending_outcomes.get(episode_id)
            if fut is not None and not fut.done():
                fut.set_result((task_generator_msgs.action.RunEpisode.Result.SKIPPED, "reset"))
            else:
                self.force_reset()
        else:
            self._spawn_episode()
        response.success = True
        return response

    async def _cb_query_scenarios(
        self,
        request: task_generator_msgs.srv.QueryScenarios.Request,
        response: task_generator_msgs.srv.QueryScenarios.Response,
    ) -> task_generator_msgs.srv.QueryScenarios.Response:
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
        request: task_generator_msgs.srv.QueueEpisode.Request,
        response: task_generator_msgs.srv.QueueEpisode.Response,
    ) -> task_generator_msgs.srv.QueueEpisode.Response:
        Req = task_generator_msgs.srv.QueueEpisode.Request
        if request.action != Req.MERGE:
            valid = ", ".join(f"{name}={val}" for name, val in (("MERGE", Req.MERGE),))
            response.success = False
            response.error_msg = f"action: unknown value {request.action!r}. Valid: {valid}"
            return response

        def reject(field: str, value: str, allowed: list[str]) -> None:
            response.success = False
            response.error_msg = f"{field}: unknown value {value!r}. Allowed: {', '.join(allowed)}"

        vocabulary = self._task_layer.mode_vocabulary()
        for field, value, allowed in (
            ("tm_robots", request.tm_robots, vocabulary.robots),
            ("tm_obstacles", request.tm_obstacles, vocabulary.obstacles),
        ):
            if value and value not in allowed:
                reject(field, value, allowed)
                return response

        validated_modules: list[str] = []
        if not request.keep_modules:
            for mod_str in request.tm_modules:
                if mod_str not in vocabulary.modules:
                    reject("tm_modules", mod_str, vocabulary.modules)
                    return response
                validated_modules.append(mod_str)

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

        existing = self._episodes.pending_overrides or TaskModeOverrides(keep_modules=True)
        if request.tm_robots:
            if request.tm_robots != existing.tm_robots:
                # Mode switch: drop stale leaves from the prior mode's namespace.
                self._staged_robots_params.clear()
            existing.tm_robots = request.tm_robots
        if request.tm_obstacles:
            if request.tm_obstacles != existing.tm_obstacles:
                self._staged_obstacles_params.clear()
            existing.tm_obstacles = request.tm_obstacles
        if not request.keep_modules:
            existing.tm_modules = list(validated_modules)
            existing.keep_modules = False
        if request.world:
            existing.world = request.world
        self._episodes.pending_overrides = existing

        for p in request.obstacles_params:
            self._staged_obstacles_params[p.name] = p.value
        for p in request.robots_params:
            self._staged_robots_params[p.name] = p.value
        for p in request.human_params:
            self._staged_human_params[p.name] = p.value

        mid_episode = self._episodes.action_in_flight and self._episodes.current.episode_id > 0
        if mid_episode:
            self._compromise("task parameters changed mid-episode")

        self._publish_queue_state()

        response.success = True
        return response

    def _apply_staged_params(self) -> None:
        pairs: list[tuple[dict[str, ParameterValue], str]] = [
            (self._staged_obstacles_params, self.task_params.obstacles),
            (self._staged_robots_params, self.task_params.robots),
        ]

        log = self.get_logger()

        # Merge by full_name: TM_Robots and TM_Obstacles can share a mode name
        # (e.g. both 'scenario') and therefore the same underlying ROS param.
        # When both pools stage the same full_name, prefer whichever value
        # differs from the live value, so a stale snapshot from the unedited
        # side does not reset the user's actual edit.
        merged: dict[str, tuple[ParameterValue, object]] = {}
        for staged, mode_value in pairs:
            if not staged or not mode_value:
                staged.clear()
                continue
            for leaf, pv in staged.items():
                full_name = f"task.{mode_value}.{leaf}"
                # Staged dicts accumulate across mode switches; a leaf valid for a prior
                # mode may not be declared under the current namespace.
                if not self.has_parameter(full_name):
                    log.warning(f"staged param {full_name!r} not declared under active mode; dropping")
                    continue
                new_value = Parameter.from_parameter_msg(RclParameter(name=full_name, value=pv)).value
                if full_name in merged:
                    _, prev_value = merged[full_name]
                    if prev_value != self.get_parameter(full_name).value:
                        continue
                merged[full_name] = (pv, new_value)
            staged.clear()

        if merged:
            batch = [Parameter.from_parameter_msg(RclParameter(name=n, value=pv)) for n, (pv, _) in merged.items()]
            result = self.set_parameters_atomically(batch)
            if not result.successful:
                log.warning(f"staged params {list(merged)} rejected: {result.reason}")

    async def _apply_staged_human_params(self) -> None:
        staged = [RclParameter(name=k, value=v) for k, v in self._staged_human_params.items()]
        self._staged_human_params.clear()
        for p in await self._environment_manager.configure_humans(staged):
            self._human_params[p.name] = p.value

    async def _cb_get_task_modes(
        self,
        request: task_generator_msgs.srv.GetTaskModes.Request,
        response: task_generator_msgs.srv.GetTaskModes.Response,
    ) -> task_generator_msgs.srv.GetTaskModes.Response:
        response.tm_robots = self.task_params.robots
        response.tm_obstacles = self.task_params.obstacles
        response.tm_modules = self.task_params.modules
        return response

    async def _cb_query_task_modes(
        self,
        request: task_generator_msgs.srv.QueryTaskModes.Request,
        response: task_generator_msgs.srv.QueryTaskModes.Response,
    ) -> task_generator_msgs.srv.QueryTaskModes.Response:
        names = self._task_layer.mode_names()
        response.obstacles = names.obstacles
        response.robots = names.robots
        response.modules = names.modules
        return response

    def _pose_from_request(self, stamped: geometry_msgs.msg.PoseStamped) -> Pose:
        """A map-frame request pose, shifted back into the world frame the task modes place entities in."""
        if not _is_map_frame(stamped.header.frame_id):
            raise ValueError(f"pose must be in the map frame, got {stamped.header.frame_id!r}")
        p = stamped.pose.position
        return self._realizer.ezilear(Pose(Position(p.x, p.y), orientation=Orientation.from_msg(stamped.pose.orientation)))

    async def _cb_spawn_static(
        self,
        request: task_generator_msgs.srv.SpawnStatic.Request,
        response: task_generator_msgs.srv.SpawnStatic.Response,
    ) -> task_generator_msgs.srv.SpawnStatic.Response:
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
        request: task_generator_msgs.srv.SpawnDynamic.Request,
        response: task_generator_msgs.srv.SpawnDynamic.Response,
    ) -> task_generator_msgs.srv.SpawnDynamic.Response:
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
        resolved_pose = pose if pose is not None else random_robot_pose(self._world_manager, self._robots_manager)
        assigned_name = name or self._robots_manager.next_name(model)
        (config,) = RobotSetupConfig.parse({"robot": model, "name": assigned_name, "pos": resolved_pose.to_2d(), **args})
        self._robots_manager.add_pending(assigned_name, Robot.from_setup(config, node=self))
        return assigned_name

    async def _cb_despawn_robot(
        self,
        request: task_generator_msgs.srv.DespawnRobot.Request,
        response: task_generator_msgs.srv.DespawnRobot.Response,
    ) -> task_generator_msgs.srv.DespawnRobot.Response:
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
        request: task_generator_msgs.srv.SpawnRobot.Request,
        response: task_generator_msgs.srv.SpawnRobot.Response,
    ) -> task_generator_msgs.srv.SpawnRobot.Response:
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
        request: task_generator_msgs.srv.MoveEntity.Request,
        response: task_generator_msgs.srv.MoveEntity.Response,
    ) -> task_generator_msgs.srv.MoveEntity.Response:
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

    async def _cb_submit_task(
        self,
        request: task_generator_msgs.srv.SubmitTask.Request,
        response: task_generator_msgs.srv.SubmitTask.Response,
    ) -> task_generator_msgs.srv.SubmitTask.Response:
        if not self.rosparam[bool].get("initialized", False) or self._robots_manager is None:
            response.success = False
            response.error_msg = "task generator not initialized"
            return response
        manager = self._robots_manager.managers.get(request.robot)
        if manager is None:
            response.success = False
            response.error_msg = f"unknown robot {request.robot!r}, robots: {sorted(self._robots_manager.managers)}"
            return response
        try:
            task = TaskRequest.parse(yaml.safe_load(request.task_yaml) or [])
            self._ezilear_phases(task.phases)
            async with self._reset_lock:
                first = len(manager.runner.phases)
                if task.phases:
                    await manager.submit_task(task)
                else:
                    manager.stop_task()
                response.phase_indices = list(range(first, len(manager.runner.phases)))
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

    # ACTION SERVER

    def _goal_callback(self, goal_request: object) -> GoalResponse:
        return GoalResponse.REJECT if self._shutting_down or self._episodes.action_in_flight else GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle: object) -> CancelResponse:
        episode_id = self._episodes.current.episode_id
        fut = self._episodes.pending_outcomes.get(episode_id)
        if fut is not None and not fut.done():
            self.event_loop.call_soon_threadsafe(
                fut.set_result,
                (task_generator_msgs.action.RunEpisode.Result.SKIPPED, "cancelled"),
            )
        return CancelResponse.ACCEPT

    async def _execute_callback(self, goal_handle: ServerGoalHandle) -> task_generator_msgs.action.RunEpisode.Result:
        result = await self.loop_future(
            self._run_episode(
                world=goal_handle.request.world,
                seed=goal_handle.request.seed,
                goal_handle=goal_handle,
            )
        )
        assert isinstance(result, task_generator_msgs.action.RunEpisode.Result)
        return result

    async def _run_episode(
        self,
        *,
        world: str = "",
        seed: int = -1,
        goal_handle: ServerGoalHandle | None = None,
    ) -> task_generator_msgs.action.RunEpisode.Result:
        if self._shutting_down or self._episodes.action_in_flight:
            if goal_handle is not None:
                goal_handle.abort()
            return task_generator_msgs.action.RunEpisode.Result(
                state=task_generator_msgs.action.RunEpisode.Result.SKIPPED,
                info=RUN_EPISODE_SHUTTING_DOWN if self._shutting_down else RUN_EPISODE_BUSY,
            )

        if not world:
            world = self._episodes.pending_world
        if seed < 0 <= self._episodes.pending_seed:
            seed = self._episodes.pending_seed
        self._episodes.pending_world = ""
        self._episodes.pending_seed = -1

        self._episodes.action_in_flight = True
        episode_id = 0
        respawn = True
        outcome_state = task_generator_msgs.action.RunEpisode.Result.FAILED
        outcome_info = ""
        try:
            await self._build_next_record(world, seed)
            if goal_handle is not None:
                self._episodes.current.goal_uuid = bytes(goal_handle.goal_id.uuid).hex()

                feedback = task_generator_msgs.action.RunEpisode.Feedback()
                feedback.state = task_generator_msgs.action.RunEpisode.Feedback.STARTED
                goal_handle.publish_feedback(feedback)

            try:
                await self._run_reset_cycle()
            except Exception as e:
                self.get_logger().error(f"reset_cycle failed: {e!r}\n{traceback.format_exc()}")
                outcome_state = task_generator_msgs.action.RunEpisode.Result.FATAL
                outcome_info = repr(e)
                respawn = False
            else:
                episode_id = self._episodes.current.episode_id
                fut = self.event_loop.create_future()
                self._episodes.pending_outcomes[episode_id] = fut

                try:
                    outcome_state, outcome_info = await fut
                except asyncio.CancelledError:
                    outcome_state = task_generator_msgs.action.RunEpisode.Result.SKIPPED
                    outcome_info = "cancelled"

            if not outcome_info:
                elapsed = (self.sim_time - self._start_time).to_seconds()
                verb = {
                    task_generator_msgs.action.RunEpisode.Result.SUCCESS: "finished",
                    task_generator_msgs.action.RunEpisode.Result.FAILED: "failed",
                    task_generator_msgs.action.RunEpisode.Result.FATAL: "failed",
                    task_generator_msgs.action.RunEpisode.Result.SKIPPED: "skipped",
                }.get(outcome_state, "ended")
                outcome_info = f"{verb} after {elapsed:.1f}s"

            self._episodes.current.outcome_state = outcome_state
            self._episodes.current.outcome_info = outcome_info
            self._publish_episode_state()

            if episode_id > 0:
                state_label = {
                    task_generator_msgs.action.RunEpisode.Result.SUCCESS: "SUCCESS",
                    task_generator_msgs.action.RunEpisode.Result.FAILED: "FAILED",
                    task_generator_msgs.action.RunEpisode.Result.SKIPPED: "SKIPPED",
                    task_generator_msgs.action.RunEpisode.Result.FATAL: "FATAL",
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
        finally:
            self._episodes.action_in_flight = False
            self._episodes.pending_outcomes.pop(episode_id, None)

        result = task_generator_msgs.action.RunEpisode.Result()
        result.state = outcome_state
        result.info = outcome_info
        result.episode_id = episode_id

        if goal_handle is not None:
            if outcome_state == task_generator_msgs.action.RunEpisode.Result.SKIPPED:
                goal_handle.canceled()
            elif outcome_state in (
                task_generator_msgs.action.RunEpisode.Result.FAILED,
                task_generator_msgs.action.RunEpisode.Result.FATAL,
            ):
                goal_handle.abort()
            else:
                goal_handle.succeed()

        service_driven = outcome_state == task_generator_msgs.action.RunEpisode.Result.SKIPPED and outcome_info == "reset"
        fatal = outcome_state == task_generator_msgs.action.RunEpisode.Result.FATAL
        if respawn and rclpy.ok() and not fatal and (service_driven or self.rosparam[bool].get_unsafe("task.episode.auto_reset")):
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
            task_generator_msgs.srv.ResetEpisode,
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
            task_generator_msgs.action.RunEpisode,
            self.service_namespace("lifecycle", "run_episode"),
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        self.create_service(
            task_generator_msgs.srv.QueryScenarios,
            self.service_namespace("query", "scenarios"),
            self._cb_query_scenarios,
        )

        self.create_service(
            task_generator_msgs.srv.QueueEpisode,
            self.service_namespace("config", "queue_episode"),
            self._cb_queue_episode,
        )

        self.create_service(
            task_generator_msgs.srv.GetTaskModes,
            self.service_namespace("config", "get_task_modes"),
            self._cb_get_task_modes,
        )

        self.create_service(
            task_generator_msgs.srv.QueryTaskModes,
            self.service_namespace("query", "task_modes"),
            self._cb_query_task_modes,
        )

        self.create_service(
            task_generator_msgs.srv.SpawnStatic,
            self.service_namespace("runtime", "spawn_static"),
            self._cb_spawn_static,
        )

        self.create_service(
            task_generator_msgs.srv.SpawnDynamic,
            self.service_namespace("runtime", "spawn_dynamic"),
            self._cb_spawn_dynamic,
        )

        self.create_service(
            task_generator_msgs.srv.SpawnRobot,
            self.service_namespace("runtime", "spawn_robot"),
            self._cb_spawn_robot,
        )

        self.create_service(
            task_generator_msgs.srv.DespawnRobot,
            self.service_namespace("runtime", "despawn_robot"),
            self._cb_despawn_robot,
        )

        self.create_service(
            task_generator_msgs.srv.MoveEntity,
            self.service_namespace("runtime", "move"),
            self._cb_move_entity,
        )

        self.create_service(
            task_generator_msgs.srv.SubmitTask,
            self.service_namespace("runtime", "submit_task"),
            self._cb_submit_task,
        )
