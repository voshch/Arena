import array
import asyncio
import contextlib
import hashlib
import json
import random
import traceback
import typing
import uuid
from collections.abc import Sequence

import arena_robots.Robot
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
import task_generator_msgs.action
import task_generator_msgs.msg
import task_generator_msgs.srv
import tf2_ros
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.Async import ClientWrapper
from arena_rclpy_mixins.shared import Namespace
from arena_robots.Sensor import SensorType
from arena_runtime.sim import BaseSim, SimulatorRegistry
from arena_simulation_setup.tree.World.Scenario import EpisodeCondition, TimelineEntry
from arena_viz.kinds import DisplayKind
from arena_viz.style import StyleSpec
from rcl_interfaces.msg import IntegerRange, ParameterDescriptor, ParameterValue
from rcl_interfaces.msg import Parameter as RclParameter
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.parameter import Parameter
from std_msgs.msg import Int16, String
from task_generator_msgs.msg import AdapterDisplay, AdapterEntry, AdapterVizManifest

from task_generator.constants import Constants
from task_generator.constants.runtime import Configuration
from task_generator.manager.environment_manager import EnvironmentManager
from task_generator.manager.realizer import Realizer
from task_generator.manager.robot_manager import RobotsManager
from task_generator.manager.world_manager.world_manager_ros import (
    WorldManagerROS as WorldManager,
)
from task_generator.shared import Orientation, Pose, Position
from task_generator.simulators.human import BaseHumanSimulator, HumanSimulatorRegistry
from task_generator.tasks import identifier_to_available
from task_generator.tasks.obstacles import ObstacleKind
from task_generator.tasks.registry import MODULE_MODES, OBSTACLES_MODES, ROBOTS_MODES
from task_generator.tasks.task import Task
from task_generator.utils.flags import flag_enabled

from . import SafeCallbackNode

if typing.TYPE_CHECKING:
    from arena_runtime.sim._semantics import SemanticChange, SemanticEntitySnapshot

    from task_generator.shared import SemanticCfg

_LATCHED = rclpy.qos.QoSProfile(
    depth=1,
    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
)

# Deeper KeepLast for state/episode: terminal-then-next-RUNNING bursts.
_EPISODE_QOS = rclpy.qos.QoSProfile(
    depth=20,
    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
)

# Node tick cadence, matching the mechanism-shim sim-time tick.
_SIM_TICK_RATE = 30.0


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
    integrity: bool = True


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


def _derive_seed(run_seed: str, world: str, episode_id: int) -> int:
    digest = hashlib.blake2b(
        f"{run_seed}|{world}|{episode_id}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF


class TaskGenerator(ArenaMixinNode, SafeCallbackNode, rclpy.lifecycle.LifecycleNode):
    """
    Task Generator Node
    Will initialize and reset all tasks. The task to use is read from the `/task_mode` param.
    """

    _world_manager: WorldManager
    _human_simulator: BaseHumanSimulator
    _environment_manager: EnvironmentManager
    _robots_manager: RobotsManager | None = None
    _simulator: BaseSim
    _realizer: Realizer
    _arena_hold_client: ClientWrapper
    _arena_unpause_window_client: ClientWrapper

    _episodes: EpisodeRuntime
    _env_id: int
    _reference: tuple[float, float]
    _prespawn_offset: tuple[float, float]

    @property
    def robots_manager(self) -> RobotsManager:
        return self._robots_manager

    def __init__(self):
        super().__init__("task_generator", automatically_declare_parameters_from_overrides=True)
        self.conf = Configuration(self)

        self._namespace = Namespace(self.get_namespace())

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        Task.declare_parameters(self)

        self.rosparam[bool].set("initialized", False)

        run_seed = self.rosparam[str].get("run_seed", "")
        queue_depth = self.rosparam[int].get("episode_queue_depth", 10)
        self._env_id = self.rosparam[int].get("env_id", 0)
        # Reference and prespawn anchor are unknown at boot, populated by the first
        # confirm_world response handled in WorldManagerROS.apply_world.
        self._reference = (0.0, 0.0)
        self._prespawn_offset = (0.0, 0.0)

        self._declare_mutable_param(
            "auto_reset",
            True,
            ParameterDescriptor(
                description=("true = standalone: node auto-advances episodes. false = managed: external controller drives resets."),
            ),
        )
        self._declare_mutable_param(
            "fail_on_collision",
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

        self._reset_lock: asyncio.Lock = asyncio.Lock()
        self._start_time = self.time
        self._task: Task | None = None

        self._staged_obstacles_params: dict[str, ParameterValue] = {}
        self._staged_robots_params: dict[str, ParameterValue] = {}

        # M2 semantics write path: inert-zone field overrides, bare->realized entity
        # name map, and the scenario timeline evaluated on sim time.
        self._zone_overrides: dict[tuple[str, str], object] = {}
        self._semantic_names: dict[str, str] = {}
        self._timeline: list[TimelineEntry] = []
        self._timeline_state: list[dict[str, object]] = []
        self._timeline_seed: int = 0
        self._timeline_t0: float | None = None
        self._episode_conditions: list[EpisodeCondition] = []
        self._tick_loop_task: asyncio.Task | None = None
        self._semantics_dirty = False

        self._pub_task_reset = self.create_publisher(
            Int16,
            self.service_namespace("task_reset"),
            1,
        )

        self._pub_state_world = self.create_publisher(
            String,
            self.service_namespace("state", "world"),
            _LATCHED,
        )

        self._pub_state_episode = self.create_publisher(
            task_generator_msgs.msg.EpisodeRecord,
            self.service_namespace("state", "episode"),
            _EPISODE_QOS,
        )

        self._pub_state_robots = self.create_publisher(
            task_generator_msgs.msg.RobotFleet,
            self.service_namespace("state", "robots"),
            _LATCHED,
        )

        self._pub_state_viz_manifest = self.create_publisher(
            AdapterVizManifest,
            self.service_namespace("state", "viz_manifest"),
            _LATCHED,
        )

        self._pub_state_robots_pending = self.create_publisher(
            task_generator_msgs.msg.RobotQueue,
            self.service_namespace("state", "robots", "pending"),
            _LATCHED,
        )

        self._pub_state_queue = self.create_publisher(
            task_generator_msgs.msg.EpisodeRecord,
            self.service_namespace("state", "queue"),
            rclpy.qos.QoSProfile(
                depth=queue_depth,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self._pub_state_semantics = self.create_publisher(
            task_generator_msgs.msg.SemanticSnapshot,
            self.service_namespace("state", "semantics"),
            _LATCHED,
        )

        self._pub_heartbeat = self.create_publisher(
            arena_runtime_msgs.msg.Heartbeat,
            self.service_namespace("state", "heartbeat"),
            10,
        )

        self._heartbeat_timer = self.wall_timer(1.0, self._cb_heartbeat_tick)

        self._arena_seen = False
        self._arena_watchdog_timer = self.wall_timer(1.0, self._cb_arena_watchdog)

        self._sub_shutdown_request = self.create_subscription(
            arena_runtime_msgs.msg.ShutdownRequest,
            "/arena/shutdown_request",
            self._cb_shutdown_request,
            10,
        )

        self._check_status_task: asyncio.Task | None = None
        self._episode_task: asyncio.Task | None = None

        self._publish_bootstrap_queue_state()
        self._pub_state_robots_pending.publish(task_generator_msgs.msg.RobotQueue())

    def _publish_bootstrap_queue_state(self) -> None:
        # Latch a minimal state/queue at construction so the rviz panel can
        # build its mode comboboxes and param tree before managers exist,
        # instead of waiting for _build_next_record at first reset.
        current_robots = self.conf.TaskMode.TM_ROBOTS.value.value if self.conf.TaskMode.TM_ROBOTS.value else ""
        current_obstacles = self.conf.TaskMode.TM_OBSTACLES.value.value if self.conf.TaskMode.TM_OBSTACLES.value else ""
        current_modules = [m.value for m in self.conf.TaskMode.TM_MODULES.value]

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
        msg.integrity = True
        msg.obstacles_params = self._params_for_mode(current_obstacles)
        msg.robots_params = self._params_for_mode(current_robots)
        self._pub_state_queue.publish(msg)

    def _declare_mutable_param(self, name: str, default: object, descriptor: ParameterDescriptor) -> None:
        self.rosparam[type(default)].declare_safe(name, default, descriptor=descriptor)

    def aiomonitor_config(self) -> dict[str, object] | None:
        # Skip when not in debug mode (no client attaches), and offset ports per
        # env_id so concurrent envs don't collide on the default 20101/2/3.
        if not flag_enabled(self, "debug", "aiomonitor"):
            return None
        offset = max(self._env_id, 0) * 10
        return {
            "port": 20101 + offset,
            "webui_port": 20102 + offset,
            "console_port": 20103 + offset,
        }

    def _cb_heartbeat_tick(self) -> None:
        msg = arena_runtime_msgs.msg.Heartbeat()
        msg.fqn = self.get_fully_qualified_name()
        msg.stamp = self.wall_time.to_msg()
        self._pub_heartbeat.publish(msg)

    def _cb_arena_watchdog(self) -> None:
        info = self.get_publishers_info_by_topic("/arena/state/envs")
        if info:
            self._arena_seen = True
            return
        if not self._arena_seen:
            return
        self.get_logger().error("/arena/state/envs publisher gone, self-shutting down")
        self._arena_watchdog_timer.cancel()
        self._heartbeat_timer.cancel()
        rclpy.try_shutdown()

    def _cb_shutdown_request(self, msg: arena_runtime_msgs.msg.ShutdownRequest) -> None:
        if msg.env_id != self._env_id:
            return
        self.get_logger().info(f"shutdown request received (reason={msg.reason!r}); shutting down")
        self._heartbeat_timer.cancel()
        rclpy.try_shutdown()

    async def setup(self) -> None:
        try:
            await self._set_up_services()
            await self._arena_hold_client.ensure()
            await self._arena_unpause_window_client.ensure()

            await self._set_up_managers()

            tm_modules = self.conf.TaskMode.TM_MODULES.value
            tm_modules.add(Constants.TaskMode.TM_Module.CLEAR_FORBIDDEN_ZONES)
            tm_modules.add(Constants.TaskMode.TM_Module.RVIZ_UI)

            self._task = await Task.create(
                node=self,
                environment_manager=self._environment_manager,
                robots_manager=self._robots_manager,
                world_manager=self._world_manager,
                modules=list(tm_modules),
            )

            await self._world_manager.sync()
            if flag_enabled(self, "debug", "map_server"):
                await self._world_manager.require_map_server()
            await self._robots_manager.launch_pending()
            self._publish_viz_manifest()

            self.rosparam[bool].set("initialized", True)
            self._publish_queue_state()
        except Exception as e:
            self._logger.error(f"configure failed: {e!r}\n{traceback.format_exc()}")
            return
        self.trigger_configure()

    def on_configure(self, state: rclpy.lifecycle.State) -> TransitionCallbackReturn:
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: rclpy.lifecycle.State) -> TransitionCallbackReturn:
        super().on_activate(state)

        def _start() -> None:
            self._check_status_task = asyncio.create_task(self._termination_watcher())
            if self.rosparam[bool].get_unsafe("auto_reset"):
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

    def on_cleanup(self, state: rclpy.lifecycle.State) -> TransitionCallbackReturn:
        self._heartbeat_timer.cancel()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: rclpy.lifecycle.State) -> TransitionCallbackReturn:
        self._heartbeat_timer.cancel()
        return TransitionCallbackReturn.SUCCESS

    async def teardown(self) -> None:
        self._heartbeat_timer.cancel()
        if self._tick_loop_task is not None and not self._tick_loop_task.done():
            self._tick_loop_task.cancel()
        if self._task is not None:
            await self._task.teardown()
        if self._robots_manager is not None:
            await self._robots_manager.teardown()

    async def hold(self, reason: str) -> None:
        req = arena_runtime_msgs.srv.LifecycleHold.Request()
        req.action = arena_runtime_msgs.srv.LifecycleHold.Request.ACQUIRE
        req.caller_id = self.get_fully_qualified_name()
        req.reason = reason
        await self._arena_hold_client.call_timeout(req)

    async def release(self, reason: str) -> None:
        req = arena_runtime_msgs.srv.LifecycleHold.Request()
        req.action = arena_runtime_msgs.srv.LifecycleHold.Request.RELEASE
        req.caller_id = self.get_fully_qualified_name()
        req.reason = reason
        await self._arena_hold_client.call_timeout(req)

    @contextlib.asynccontextmanager
    async def unpause_window(self):
        req = arena_runtime_msgs.srv.LifecycleUnpauseWindow.Request()
        req.action = arena_runtime_msgs.srv.LifecycleUnpauseWindow.Request.ACQUIRE
        req.caller_id = self.get_fully_qualified_name()
        await self._arena_unpause_window_client.call_forever(req)
        try:
            yield
        finally:
            rel = arena_runtime_msgs.srv.LifecycleUnpauseWindow.Request()
            rel.action = arena_runtime_msgs.srv.LifecycleUnpauseWindow.Request.RELEASE
            rel.caller_id = self.get_fully_qualified_name()
            await self._arena_unpause_window_client.call_timeout(rel)

    async def _set_up_managers(self):
        self._logger.info("Setting up managers")

        ref_x, ref_y = self._reference
        prefix = self.rosparam[str].get('prefix', '')
        realizer = Realizer(Realizer._Configuration(x=ref_x, y=ref_y, prefix=prefix))
        self._realizer = realizer

        self._logger.info("Setting up simulator")
        self._simulator = await SimulatorRegistry.get(
            self.conf.Arena.SIM.value,
            node=self,
            namespace=self._namespace,
            realizer=realizer,
            env_id=self._env_id,
        )
        self._simulator.set_semantics_callback(self._on_semantics_changed)
        self._logger.info("Setting up human simulator")
        self._human_simulator = await HumanSimulatorRegistry.get(
            self.conf.Arena.HUMAN.value,
            node=self,
            namespace=self._namespace,
            simulator=self._simulator,
            realizer=realizer,
        )

        self._logger.info("Setting up environment manager")
        self._environment_manager = EnvironmentManager(
            node=self,
            simulator=self._simulator,
            human_simulator=self._human_simulator,
            realizer=realizer,
        )

        self._logger.info("Setting up world manager")
        self._world_manager = WorldManager(node=self, environment_manager=self._environment_manager)

        await self._world_manager.start()

        self._logger.info("Setting up robots manager")
        self._robots_manager = RobotsManager(node=self, environment_manager=self._environment_manager)

        if self._tick_loop_task is None or self._tick_loop_task.done():
            self._tick_loop_task = asyncio.create_task(self._run_tick_loop())

        self._logger.info("Managers set up")

    # RUNTIME

    def _flip_integrity(self) -> None:
        """Mark current episode as externally mutated and republish state."""
        self._episodes.current.integrity = False
        self._publish_episode_state()

    def _params_for_mode(self, mode: str) -> list[RclParameter]:
        if not mode:
            return []
        prefix = f"task.{mode}"
        names = self.list_parameters(prefixes=[prefix], depth=10).names
        if not names:
            return []
        params = self.get_parameters(names)
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
        msg.integrity = record.integrity
        msg.conditions = json.dumps([c.serialize() for c in self._episode_conditions])
        return msg

    def _publish_episode_state(self) -> None:
        msg = self._record_to_msg(self._episodes.current)
        msg.obstacles_params = self._params_for_mode(self._episodes.current.tm_obstacles)
        msg.robots_params = self._params_for_mode(self._episodes.current.tm_robots)
        self._pub_state_episode.publish(msg)

    def _semantic_entity_state_msg(self, snap: "SemanticEntitySnapshot") -> task_generator_msgs.msg.SemanticEntityState:
        msg = task_generator_msgs.msg.SemanticEntityState()
        msg.entity = snap.entity
        msg.kind = snap.kind
        msg.discrete_names = list(snap.discrete.keys())
        msg.discrete_values = list(snap.discrete.values())
        msg.continuous_names = list(snap.continuous.keys())
        msg.continuous_values = list(snap.continuous.values())
        msg.predicate_names = list(snap.predicates.keys())
        msg.predicate_values = list(snap.predicates.values())
        msg.members = list(snap.members)
        return msg

    @staticmethod
    def _zone_field_role(cfg: "SemanticCfg") -> str:
        """Coercion role of one inert-zone field: 'predicate' | 'continuous' | 'discrete'."""
        if cfg.role == "predicate":
            return "predicate"
        if isinstance(cfg.value, str):
            return "discrete"
        return "continuous"

    def _iter_zones(self) -> typing.Iterator[tuple[str, World.LevelDescription.Zone]]:
        """All zones across levels as (realized name, zone)."""
        for level_id, level in self._world_manager.world.levels.items():
            for zone in level.zones:
                yield self._realizer.prefix(zone.name, level_id), zone

    def _zone_semantic_lookup(self) -> dict[str, dict[str, str]]:
        """Realized zone name -> {inert field name: role}, for write-path validation."""
        lookup: dict[str, dict[str, str]] = {}
        for name, zone in self._iter_zones():
            fields = {cfg.name: self._zone_field_role(cfg) for cfg in zone.semantics if cfg.value is not None}
            if fields:
                lookup[name] = fields
        return lookup

    def _zone_field_value(self, entity: str, field: str) -> str | None:
        """Current stringified value of one inert-zone field, honoring overrides."""
        for name, zone in self._iter_zones():
            if name != entity:
                continue
            for cfg in zone.semantics:
                if cfg.name != field or cfg.value is None:
                    continue
                value = self._zone_overrides.get((entity, field), cfg.value)
                if cfg.role == "predicate":
                    return "true" if bool(value) else "false"
                if isinstance(cfg.value, str):
                    return str(value)
                return self._stringify_float(float(value))
        return None

    def _zone_semantic_states(self) -> list[task_generator_msgs.msg.SemanticEntityState]:
        """Inert zone annotations from the loaded world, env-prefixed, appended to every snapshot."""
        states: list[task_generator_msgs.msg.SemanticEntityState] = []
        for name, zone in self._iter_zones():
            if not zone.semantics:
                continue
            msg = task_generator_msgs.msg.SemanticEntityState()
            msg.entity = name
            msg.kind = "zone"
            discrete_names: list[str] = []
            discrete_values: list[str] = []
            continuous_names: list[str] = []
            continuous_values: list[float] = []
            predicate_names: list[str] = []
            predicate_values: list[bool] = []
            for cfg in zone.semantics:
                if cfg.value is None:
                    continue
                value = self._zone_overrides.get((msg.entity, cfg.name), cfg.value)
                if cfg.role == "predicate":
                    predicate_names.append(cfg.name)
                    predicate_values.append(bool(value))
                elif isinstance(cfg.value, str):
                    discrete_names.append(cfg.name)
                    discrete_values.append(str(value))
                else:
                    continuous_names.append(cfg.name)
                    continuous_values.append(float(value))
            msg.discrete_names = discrete_names
            msg.discrete_values = discrete_values
            msg.continuous_names = continuous_names
            msg.continuous_values = continuous_values
            msg.predicate_names = predicate_names
            msg.predicate_values = predicate_values
            states.append(msg)
        return states

    def _publish_semantics_snapshot(self) -> None:
        """Immediate publish, world-boundary only. Change-driven publishes ride the tick flush."""
        self._semantics_dirty = False
        msg = task_generator_msgs.msg.SemanticSnapshot()
        msg.stamp = self.sim_time.to_msg()
        msg.env_id = self._env_id
        msg.world = self._world_manager.loaded_world
        entities = [self._semantic_entity_state_msg(s) for s in self._simulator.semantics_snapshot()]
        entities.extend(self._zone_semantic_states())
        msg.entities = entities
        self._pub_state_semantics.publish(msg)

    def _on_semantics_changed(self, changes: "Sequence[SemanticChange]") -> None:
        self._semantics_dirty = True

    # SEMANTICS WRITE PATH

    @staticmethod
    def _stringify_float(value: float) -> str:
        return str(value)

    @staticmethod
    def _norm_token(value: object) -> str:
        """Stringify a literal to the event convention: booleans render 'true'/'false'."""
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _register_semantic_entity(self, bare: str, realized: str) -> None:
        """Record the bare-authored -> realized name mapping for internal write resolution."""
        self._semantic_names[bare] = realized

    def _clear_semantic_entities(self) -> None:
        """Drop the name map and zone overrides ahead of a world replacement."""
        self._semantic_names.clear()
        self._zone_overrides.clear()

    def _resolve_semantic_entity(self, entity: str) -> str:
        """Map a bare-authored entity name to its realized name, env-prefixing as a fallback."""
        return self._semantic_names.get(entity) or str(self._realizer.prefix(entity))

    @staticmethod
    def _coerce_zone_value(role: str, value: str) -> object:
        if role == "predicate":
            token = value.strip().lower()
            if token in ("true", "false"):
                return token == "true"
            raise ValueError("malformed value")
        if role == "continuous":
            try:
                return float(value)
            except (ValueError, TypeError):
                raise ValueError("malformed value") from None
        return value

    def _apply_zone_override(self, entity: str, field: str, value: str) -> str:
        """Apply an inert-zone override. '' on success, else a structured reason."""
        fields = self._zone_semantic_lookup().get(entity)
        if fields is None:
            return "unknown entity"
        role = fields.get(field)
        if role is None:
            return "field not writable"
        try:
            coerced = self._coerce_zone_value(role, value)
        except ValueError:
            return "malformed value"
        self._zone_overrides[(entity, field)] = coerced
        self._semantics_dirty = True
        return ""

    def _apply_semantic_checked(self, entity: str, field: str, value: str) -> str:
        """Apply one write to a runtime instance or inert zone. '' on success, else a structured reason."""
        try:
            if self._simulator.set_semantic_value(entity, field, value):
                return ""
        except ValueError as exc:
            if str(exc) == "malformed value":
                return "malformed value"
            # read-only / not-writable runtime field: it may still be an inert zone field
            if self._apply_zone_override(entity, field, value) == "":
                return ""
            return "field not writable"
        return self._apply_zone_override(entity, field, value)

    def _set_semantic(self, entity: str, field: str, value: str) -> bool:
        """Internal authority entry (timeline, modules): resolve, coerce, apply, log-and-skip on error."""
        realized = self._resolve_semantic_entity(entity)
        reason = self._apply_semantic_checked(realized, field, value)
        if reason:
            self.get_logger().warning(f"semantics set skipped: {entity!r}.{field} = {value!r} ({reason})")
            return False
        return True

    def _cb_set_semantic(
        self,
        request: task_generator_msgs.srv.SetSemantic.Request,
        response: task_generator_msgs.srv.SetSemantic.Response,
    ) -> task_generator_msgs.srv.SetSemantic.Response:
        """External untrusted write: full validation, structured error_msg, no bare-name resolution."""
        reason = self._apply_semantic_checked(request.entity, request.field, request.value)
        response.success = not reason
        response.error_msg = reason
        return response

    def _semantic_value(self, entity: str, field: str) -> str | None:
        """Current stringified value of one semantic field across runtime and inert-zone entities."""
        realized = self._resolve_semantic_entity(entity)
        for snap in self._simulator.semantics_snapshot():
            if snap.entity != realized:
                continue
            if field in snap.discrete:
                return snap.discrete[field]
            if field in snap.continuous:
                return self._stringify_float(snap.continuous[field])
            if field in snap.predicates:
                return "true" if snap.predicates[field] else "false"
        return self._zone_field_value(realized, field)

    # SCENARIO TIMELINE

    def register_timeline(self, entries: "Sequence[TimelineEntry]", seed: int) -> None:
        """Arm the scenario timeline for the current episode under one seed."""
        self._timeline = list(entries)
        self._timeline_seed = seed
        self._timeline_t0 = None
        self._timeline_state = []
        for idx, entry in enumerate(self._timeline):
            nxt = (entry.offset + entry.every) if (entry.every is not None and entry.every > 0.0) else float("inf")
            self._timeline_state.append(
                {"fired": False, "next": nxt, "prev": False, "rng": random.Random(f"{seed}:{idx}")},
            )

    def reset_timeline(self) -> None:
        """Clear timeline arm state, episode conditions and inert-zone overrides so the next reset replays identically."""
        self._timeline = []
        self._timeline_state = []
        self._timeline_t0 = None
        self._episode_conditions = []
        self._zone_overrides.clear()
        self._semantics_dirty = True

    def register_conditions(self, conditions: "Sequence[EpisodeCondition]") -> None:
        """Carry the active scenario's episode conditions through to the episode record."""
        self._episode_conditions = list(conditions)

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

    def _when_true(self, when: dict) -> bool:
        expected = self._norm_token(when.get("is"))
        current = self._semantic_value(str(when.get("entity", "")), str(when.get("field", "")))
        return current is not None and current == expected

    def _fire_timeline_entry(self, idx: int, entry: "TimelineEntry") -> None:
        for action in entry.set:
            entity = str(action.get("entity", ""))
            field = str(action.get("field", ""))
            value = self._resolve_timeline_value(self._norm_token(action.get("value")), idx)
            self._set_semantic(entity, field, value)

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
            elif entry.when is not None:
                current = self._when_true(entry.when)
                if current and not state["prev"]:
                    fired = True
                state["prev"] = current
            if fired:
                self._fire_timeline_entry(idx, entry)

    async def _run_tick_loop(self) -> None:
        """Sim-paced node tick: timeline evaluation plus semantics flush at the mechanism rate."""
        with self.sim_time_rate(_SIM_TICK_RATE) as (done, rate):
            while not done.is_set():
                try:
                    await rate.get()
                except asyncio.CancelledError:
                    raise
                try:
                    self._evaluate_timeline(self.sim_time.to_seconds())
                    if self._semantics_dirty:
                        self._publish_semantics_snapshot()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.get_logger().warning(f"sim tick failed: {exc!r}")

    def _publish_viz_manifest(self) -> None:
        """Publish the env-level and per-robot display manifest."""
        _SENSOR_KIND: dict[str, DisplayKind] = {
            SensorType.LASERSCAN: DisplayKind.LASER_SCAN,
            SensorType.POINTCLOUD: DisplayKind.POINTS_3D,
            SensorType.IMAGE: DisplayKind.IMAGE,
            SensorType.DEPTH: DisplayKind.IMAGE,
            SensorType.IMU: DisplayKind.IMU,
            SensorType.CONTACT: DisplayKind.FOOT_CONTACT,
        }

        env_ns = self.get_namespace()
        auditory_ns = self.get_fully_qualified_name()

        env_displays: list[AdapterDisplay] = [
            AdapterDisplay(
                name="Map",
                topic=str(self.service_namespace("map")),
                topic_type="nav_msgs/OccupancyGrid",
                kind=DisplayKind.MAP,
                style_json=StyleSpec(alpha=0.7).to_json(),
                topic_must_exist=False,
            ),
            AdapterDisplay(
                name="TF",
                topic="",
                topic_type="",
                kind=DisplayKind.TF,
                style_json=StyleSpec(enabled=False).to_json(),
                topic_must_exist=False,
            ),
        ]
        latched = StyleSpec(extra={"rviz": {"Reliability Policy": "Reliable", "Durability Policy": "Transient Local"}}).to_json()
        env_displays.append(
            AdapterDisplay(
                name="Pedestrians",
                topic=f"{env_ns}/humans",
                topic_type="",
                kind=DisplayKind.PEDESTRIANS,
                style_json=StyleSpec().to_json(),
                topic_must_exist=False,
                group="Pedestrians",
            )
        )
        # Backend-internal debug overlay, off by default.
        env_displays.append(
            AdapterDisplay(
                name="Extra",
                topic=f"{env_ns}/pedestrian_markers/extra",
                topic_type="visualization_msgs/MarkerArray",
                kind=DisplayKind.MARKER_ARRAY,
                style_json=StyleSpec(enabled=False).to_json(),
                topic_must_exist=False,
                group="Pedestrians",
            )
        )
        # Static environment geometry: not pedestrians, own group.
        env_displays.append(
            AdapterDisplay(
                name="Static",
                topic=f"{env_ns}/pedestrian_markers/static",
                topic_type="visualization_msgs/MarkerArray",
                kind=DisplayKind.MARKER_ARRAY,
                style_json=latched,
                topic_must_exist=False,
                group="Static",
            )
        )
        for leaf in ("static_walls", "static_objects"):
            env_displays.append(
                AdapterDisplay(
                    name=leaf.replace("_", " ").title(),
                    topic=f"{env_ns}/pedestrian_markers/{leaf}",
                    topic_type="visualization_msgs/MarkerArray",
                    kind=DisplayKind.MARKER_ARRAY,
                    style_json=latched,
                    topic_must_exist=True,
                    group="Static",
                )
            )
        for name, topic in (
            (
                "Microphones",
                f"{auditory_ns}/microphone_markers",
            ),
            (
                "Environment Audio Sources",
                f"{auditory_ns}/environment_audio_source_markers",
            ),
            (
                "Pedestrian Heard Sound",
                f"{auditory_ns}/pedestrian_sound_propagation_markers",
            ),
            (
                "Robot Heard Sound",
                f"{auditory_ns}/robot_sound_propagation_markers",
            ),
        ):
            env_displays.append(
                AdapterDisplay(
                    name=name,
                    topic=topic,
                    topic_type="visualization_msgs/MarkerArray",
                    kind=DisplayKind.MARKER_ARRAY,
                    style_json=(
                        latched
                        if name == "Microphones"
                        else StyleSpec(enabled=True).to_json()
                    ),
                    topic_must_exist=False,
                    group="Sound Propagation",
                )
            )
        entries: list[AdapterEntry] = []
        for mgr in self._robots_manager.managers.values():
            ns_value = str(mgr.namespace)
            robot_value = mgr.name

            sensor_displays: list[AdapterDisplay] = []
            for sensor in mgr.robot_view.effective_sensors(mgr.robot.resolved_request, frames=mgr.robot.frames):
                kind = _SENSOR_KIND.get(sensor.type)
                if kind is None:
                    continue
                raw_topic = sensor.topic.replace("${namespace}", ns_value)
                sensor_displays.append(
                    AdapterDisplay(
                        name=sensor.name,
                        topic=raw_topic,
                        topic_type="",
                        kind=kind,
                        style_json="",
                        topic_must_exist=True,
                    )
                )
            if sensor_displays:
                entries.append(
                    AdapterEntry(
                        robot_ns=ns_value,
                        adapter_kind="_sensors",
                        displays=sensor_displays,
                    )
                )
            entries.append(
                AdapterEntry(
                    robot_ns=ns_value,
                    adapter_kind="_auditory",
                    displays=[
                        AdapterDisplay(
                            name="Motor Sound",
                            topic=(
                                f"{auditory_ns}/{robot_value}/"
                                "motor_sound_markers"
                            ),
                            topic_type="visualization_msgs/MarkerArray",
                            kind=DisplayKind.MARKER_ARRAY,
                            style_json=StyleSpec(enabled=True).to_json(),
                            topic_must_exist=False,
                        )
                    ],
                )
            )

            for adapter in mgr._adapter_instances:

                def _subst(s: str, ns_value: str = ns_value, robot_value: str = robot_value) -> str:
                    return s.replace("{ns}", ns_value).replace("{robot}", robot_value)

                displays = [
                    AdapterDisplay(
                        name=hint.name,
                        topic=_subst(hint.topic),
                        topic_type=hint.topic_type,
                        kind=hint.kind,
                        style_json=_subst(hint.style_json),
                        topic_must_exist=hint.topic_must_exist,
                    )
                    for hint in adapter.displays
                ]
                entries.append(
                    AdapterEntry(
                        robot_ns=ns_value,
                        adapter_kind=adapter.kind,
                        displays=displays,
                    )
                )
        self._pub_state_viz_manifest.publish(AdapterVizManifest(env_displays=env_displays, entries=entries))

    def set_episode_info(self, info: str) -> None:
        self._episodes.current.outcome_info = info
        self._publish_episode_state()

    def _publish_queue_state(self) -> None:
        overrides = self._episodes.pending_overrides
        current_robots = self.conf.TaskMode.TM_ROBOTS.value.value if self.conf.TaskMode.TM_ROBOTS.value else ""
        current_obstacles = self.conf.TaskMode.TM_OBSTACLES.value.value if self.conf.TaskMode.TM_OBSTACLES.value else ""
        current_modules = [m.value for m in self.conf.TaskMode.TM_MODULES.value]
        record_world = self._episodes.current.world
        loaded_world = self._world_manager.loaded_world
        queued_robots = [m.model_name for m in self._robots_manager.managers.values()]

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
        msg.integrity = True
        msg.obstacles_params = [RclParameter(name=k, value=v) for k, v in obstacles_map.items()]
        msg.robots_params = [RclParameter(name=k, value=v) for k, v in robots_map.items()]
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

        current_robots = self.conf.TaskMode.TM_ROBOTS.value.value if self.conf.TaskMode.TM_ROBOTS.value else ""
        current_obstacles = self.conf.TaskMode.TM_OBSTACLES.value.value if self.conf.TaskMode.TM_OBSTACLES.value else ""
        current_modules = [m.value for m in self.conf.TaskMode.TM_MODULES.value]

        if overrides is None:
            tm_robots, tm_obstacles, tm_modules = current_robots, current_obstacles, current_modules
        else:
            tm_robots = overrides.tm_robots or current_robots
            tm_obstacles = overrides.tm_obstacles or current_obstacles
            tm_modules = current_modules if overrides.keep_modules else overrides.tm_modules

        if tm_robots and tm_robots != current_robots:
            self.rosparam[str].set("tm_robots", tm_robots)
        if tm_obstacles and tm_obstacles != current_obstacles:
            self.rosparam[str].set("tm_obstacles", tm_obstacles)

        self._episodes.current = EpisodeRecord(
            episode_id=new_id,
            world=resolved_world,
            seed=resolved_seed,
            tm_robots=tm_robots,
            tm_obstacles=tm_obstacles,
            tm_modules=tm_modules,
            integrity=True,
        )

        self._publish_queue_state()

    async def _run_reset_cycle(self) -> None:
        async with self._reset_lock:
            self._start_time = self.sim_time
            self.get_logger().info("resetting")

            record = self._episodes.current
            await self.hold("reset")
            try:
                if record.world:
                    await self._world_manager.apply_world(record.world)
                await self._task.reset(world=record.world, seed=record.seed)
            finally:
                await self.release("reset")
            record.robots = [m.model_name for m in self._robots_manager.managers.values()]

            self._pub_task_reset.publish(Int16(data=record.episode_id - 1))
            self._send_end_message_on_end()

            self._pub_state_world.publish(String(data=record.world))

            record.outcome_state = task_generator_msgs.action.RunEpisode.Result.RUNNING
            self._publish_episode_state()

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

    async def _termination_watcher(self) -> None:
        try:
            while True:
                await asyncio.sleep(0.5)
                episode_id = self._episodes.current.episode_id
                fut = self._episodes.pending_outcomes.get(episode_id)
                if fut is None or fut.done():
                    continue
                if not await self._task.is_done:
                    continue
                if self._task.abort_reason is not None:
                    fut.set_result((task_generator_msgs.action.RunEpisode.Result.FAILED, self._task.abort_reason))
                else:
                    fut.set_result((task_generator_msgs.action.RunEpisode.Result.SUCCESS, ""))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.get_logger().error(f"Error in termination watcher: {e}\n{traceback.format_exc()}")
            raise

    def _send_end_message_on_end(self):
        desired = self.conf.General.DESIRED_EPISODES.value
        if desired < 0 or self._episodes.current.episode_id < desired:
            return

        self.get_logger().info(f"Shutting down. All {int(desired)} tasks completed")
        rclpy.shutdown()

    def fail_episode(self, reason: str) -> None:
        """Abort the running episode as FAILED with ``reason``, unless it is already aborting."""
        if self._task.abort_reason is not None:
            return
        self.get_logger().warn(f"failing episode: {reason}")
        self._task.abort_episode(reason)
        fut = self._episodes.pending_outcomes.get(self._episodes.current.episode_id)
        if fut is not None and not fut.done():
            fut.set_result((task_generator_msgs.action.RunEpisode.Result.FAILED, reason))

    # SERVICE CALLBACKS

    async def _cb_reset_episode(
        self,
        request: task_generator_msgs.srv.ResetEpisode.Request,
        response: task_generator_msgs.srv.ResetEpisode.Response,
    ) -> task_generator_msgs.srv.ResetEpisode.Response:
        self._episodes.pending_world = request.world
        self._episodes.pending_seed = request.seed
        if self._episodes.action_in_flight:
            episode_id = self._episodes.current.episode_id
            fut = self._episodes.pending_outcomes.get(episode_id)
            if fut is not None and not fut.done():
                fut.set_result((task_generator_msgs.action.RunEpisode.Result.SKIPPED, "reset"))
            else:
                self._task.force_reset()
        else:
            self._spawn_episode()
        response.success = True
        return response

    async def _cb_wait_for_world(
        self,
        request: object,
        response: object,
    ) -> object:
        await self._world_manager.sync()
        return response

    async def _cb_query_worlds(
        self,
        request: task_generator_msgs.srv.QueryWorlds.Request,
        response: task_generator_msgs.srv.QueryWorlds.Response,
    ) -> task_generator_msgs.srv.QueryWorlds.Response:
        response.ids = list(identifier_to_available(World.WorldIdentifier))
        return response

    async def _cb_query_scenarios(
        self,
        request: task_generator_msgs.srv.QueryScenarios.Request,
        response: task_generator_msgs.srv.QueryScenarios.Response,
    ) -> task_generator_msgs.srv.QueryScenarios.Response:
        world_name = request.world or self._episodes.current.world
        response.ids = list(identifier_to_available(World.WorldIdentifier(world_name).resolve_sync().scenario))
        return response

    async def _cb_query_robots(
        self,
        request: task_generator_msgs.srv.QueryRobots.Request,
        response: task_generator_msgs.srv.QueryRobots.Response,
    ) -> task_generator_msgs.srv.QueryRobots.Response:
        response.ids = list(identifier_to_available(arena_robots.Robot.RobotIdentifier))
        return response

    async def _cb_query_static_obstacles(
        self,
        request: task_generator_msgs.srv.QueryStaticObstacles.Request,
        response: task_generator_msgs.srv.QueryStaticObstacles.Response,
    ) -> task_generator_msgs.srv.QueryStaticObstacles.Response:
        response.ids = list(identifier_to_available(arena_simulation_setup.tree.assets.Object.ObjectIdentifier, network=True))
        return response

    async def _cb_query_dynamic_obstacles(
        self,
        request: task_generator_msgs.srv.QueryDynamicObstacles.Request,
        response: task_generator_msgs.srv.QueryDynamicObstacles.Response,
    ) -> task_generator_msgs.srv.QueryDynamicObstacles.Response:
        response.ids = list(
            identifier_to_available(
                arena_simulation_setup.tree.assets.Human.HumanIdentifier,
                network=True,
            )
        )
        return response

    async def _cb_query_environments(
        self,
        request: task_generator_msgs.srv.QueryEnvironments.Request,
        response: task_generator_msgs.srv.QueryEnvironments.Response,
    ) -> task_generator_msgs.srv.QueryEnvironments.Response:
        response.ids = list(identifier_to_available(arena_simulation_setup.tree.configs.environment.EnvironmentIdentifier))
        return response

    async def _cb_query_parametrizeds(
        self,
        request: task_generator_msgs.srv.QueryParametrizeds.Request,
        response: task_generator_msgs.srv.QueryParametrizeds.Response,
    ) -> task_generator_msgs.srv.QueryParametrizeds.Response:
        response.ids = list(identifier_to_available(arena_simulation_setup.tree.configs.parametrized.ParametrizedIdentifier))
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

        def reject(field: str, value: str, enum_cls: type) -> None:
            allowed = ", ".join(m.value for m in enum_cls)
            response.success = False
            response.error_msg = f"{field}: unknown value {value!r}. Allowed: {allowed}"

        for field, value, enum_cls in (
            ("tm_robots", request.tm_robots, Constants.TaskMode.TM_Robots),
            ("tm_obstacles", request.tm_obstacles, Constants.TaskMode.TM_Obstacles),
        ):
            if not value:
                continue
            try:
                enum_cls(value)
            except ValueError:
                reject(field, value, enum_cls)
                return response

        validated_modules: list[str] = []
        if not request.keep_modules:
            for mod_str in request.tm_modules:
                try:
                    Constants.TaskMode.TM_Module(mod_str)
                except ValueError:
                    reject("tm_modules", mod_str, Constants.TaskMode.TM_Module)
                    return response
                validated_modules.append(mod_str)

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

        mid_episode = self._episodes.action_in_flight and self._episodes.current.episode_id > 0
        if mid_episode:
            self._episodes.current.integrity = False

        self._publish_queue_state()

        response.success = True
        return response

    def _apply_staged_params(self) -> None:
        active_obstacles = self.conf.TaskMode.TM_OBSTACLES.value
        active_robots = self.conf.TaskMode.TM_ROBOTS.value

        pairs: list[tuple[dict[str, ParameterValue], str]] = [
            (self._staged_obstacles_params, active_obstacles.value if active_obstacles else ""),
            (self._staged_robots_params, active_robots.value if active_robots else ""),
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

    async def _cb_get_task_modes(
        self,
        request: task_generator_msgs.srv.GetTaskModes.Request,
        response: task_generator_msgs.srv.GetTaskModes.Response,
    ) -> task_generator_msgs.srv.GetTaskModes.Response:
        response.tm_robots = self.conf.TaskMode.TM_ROBOTS.value.value if self.conf.TaskMode.TM_ROBOTS.value else ""
        response.tm_obstacles = self.conf.TaskMode.TM_OBSTACLES.value.value if self.conf.TaskMode.TM_OBSTACLES.value else ""
        response.tm_modules = [m.value for m in self.conf.TaskMode.TM_MODULES.value]
        return response

    async def _cb_query_task_modes(
        self,
        request: task_generator_msgs.srv.QueryTaskModes.Request,
        response: task_generator_msgs.srv.QueryTaskModes.Response,
    ) -> task_generator_msgs.srv.QueryTaskModes.Response:
        response.obstacles = [k.value for k in OBSTACLES_MODES.keys()]
        response.robots = [k.value for k in ROBOTS_MODES.keys()]
        response.modules = [k.value for k in MODULE_MODES.keys()]
        return response

    def _pose_from_request(self, stamped: geometry_msgs.msg.PoseStamped) -> Pose:
        p = stamped.pose.position
        return Pose(Position(p.x, p.y), orientation=Orientation.from_msg(stamped.pose.orientation))

    async def _cb_spawn_static(
        self,
        request: task_generator_msgs.srv.SpawnStatic.Request,
        response: task_generator_msgs.srv.SpawnStatic.Response,
    ) -> task_generator_msgs.srv.SpawnStatic.Response:
        try:
            pose = self._pose_from_request(request.pose) if request.use_pose else None
            entity_id = await self._task.tm_obstacles.extend(ObstacleKind.STATIC, request.model, pose)
            self._flip_integrity()
            response.id = entity_id
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
        try:
            pose = self._pose_from_request(request.pose) if request.use_pose else None
            entity_id = await self._task.tm_obstacles.extend(ObstacleKind.DYNAMIC, request.model, pose)
            self._flip_integrity()
            response.id = entity_id
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
        if not self.rosparam[bool].get("initialized", False):
            response.success = False
            response.error_msg = "task generator not initialized"
            return response
        try:
            pose = self._pose_from_request(request.pose) if request.use_pose else None
            args = {kv.key: kv.value for kv in request.args}
            name_out = await self._task.tm_robots.extend(request.model, request.name or None, pose, args=args)
            if request.immediate:
                async with self._reset_lock:
                    await self._robots_manager.spawn_now(name_out, pose)
            self._robots_manager.publish_queue()
            self._flip_integrity()
            response.name = name_out
            response.success = True
        except Exception as e:
            response.success = False
            response.error_msg = str(e)
        return response

    async def _cb_despawn_robot(
        self,
        request: task_generator_msgs.srv.DespawnRobot.Request,
        response: task_generator_msgs.srv.DespawnRobot.Response,
    ) -> task_generator_msgs.srv.DespawnRobot.Response:
        if not self.rosparam[bool].get("initialized", False):
            response.success = False
            response.error_msg = "task generator not initialized"
            return response
        try:
            self._robots_manager.remove_pending(request.name)
            self._robots_manager.publish_queue()
            self._flip_integrity()
            response.success = True
        except Exception as e:
            response.success = False
            response.error_msg = str(e)
        return response

    async def _cb_require_map(
        self,
        request: object,
        response: object,
    ) -> object:
        if not self.rosparam[bool].get("initialized", False):
            response.success = False
            response.message = "task generator not initialized"
            return response
        async with self._reset_lock:
            await self._world_manager.require_map_server()
        response.success = True
        response.message = ""
        return response

    # ACTION SERVER

    def _goal_callback(self, goal_request: object) -> GoalResponse:
        return GoalResponse.REJECT if self._episodes.action_in_flight else GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle: object) -> CancelResponse:
        episode_id = self._episodes.current.episode_id
        fut = self._episodes.pending_outcomes.get(episode_id)
        if fut is not None and not fut.done():
            self.event_loop.call_soon_threadsafe(
                fut.set_result,
                (task_generator_msgs.action.RunEpisode.Result.SKIPPED, "cancelled"),
            )
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle: object) -> task_generator_msgs.action.RunEpisode.Result:
        return self.wait_for(
            self._run_episode(
                world=goal_handle.request.world,
                seed=goal_handle.request.seed,
                goal_handle=goal_handle,
            )
        )

    async def _run_episode(
        self,
        *,
        world: str = "",
        seed: int = -1,
        goal_handle: object | None = None,
    ) -> task_generator_msgs.action.RunEpisode.Result:
        if self._episodes.action_in_flight:
            if goal_handle is not None:
                goal_handle.abort()
            return task_generator_msgs.action.RunEpisode.Result(
                state=task_generator_msgs.action.RunEpisode.Result.SKIPPED,
                reason="busy",
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
        if respawn and rclpy.ok() and not fatal and (service_driven or self.rosparam[bool].get_unsafe("auto_reset")):
            self._spawn_episode()

        return result

    def _spawn_episode(self, **kwargs: object) -> None:
        async def _wrap() -> None:
            await self._run_episode(**kwargs)

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
        self._logger.info("Setting up services")

        self._arena_hold_client = self.create_client_wrapper(
            arena_runtime_msgs.srv.LifecycleHold,
            "/arena/sim_lifecycle/hold",
        )

        self._arena_unpause_window_client = self.create_client_wrapper(
            arena_runtime_msgs.srv.LifecycleUnpauseWindow,
            "/arena/sim_lifecycle/unpause_window",
        )

        self.create_service(
            task_generator_msgs.srv.ResetEpisode,
            self.service_namespace("lifecycle", "reset_episode"),
            self._cb_reset_episode,
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

        from std_srvs.srv import Empty as EmptySrv

        self.create_service(
            EmptySrv,
            self.service_namespace("lifecycle", "wait_for_world"),
            self._cb_wait_for_world,
        )

        self.create_service(
            task_generator_msgs.srv.QueryWorlds,
            self.service_namespace("query", "worlds"),
            self._cb_query_worlds,
        )

        self.create_service(
            task_generator_msgs.srv.QueryScenarios,
            self.service_namespace("query", "scenarios"),
            self._cb_query_scenarios,
        )

        self.create_service(
            task_generator_msgs.srv.QueryRobots,
            self.service_namespace("query", "robots"),
            self._cb_query_robots,
        )

        self.create_service(
            task_generator_msgs.srv.QueryStaticObstacles,
            self.service_namespace("query", "static_obstacles"),
            self._cb_query_static_obstacles,
        )

        self.create_service(
            task_generator_msgs.srv.QueryDynamicObstacles,
            self.service_namespace("query", "dynamic_obstacles"),
            self._cb_query_dynamic_obstacles,
        )

        self.create_service(
            task_generator_msgs.srv.QueryEnvironments,
            self.service_namespace("query", "environments"),
            self._cb_query_environments,
        )

        self.create_service(
            task_generator_msgs.srv.QueryParametrizeds,
            self.service_namespace("query", "parametrizeds"),
            self._cb_query_parametrizeds,
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
            task_generator_msgs.srv.SetSemantic,
            self.service_namespace("semantics", "set"),
            self._cb_set_semantic,
        )

        from std_srvs.srv import Trigger

        self.create_service(
            Trigger,
            self.service_namespace("runtime", "require_map"),
            self._cb_require_map,
        )

        self._logger.info("Services set up")
