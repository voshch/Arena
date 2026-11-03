from __future__ import annotations

import abc
import asyncio
import contextlib
import threading
import traceback
import typing
from collections.abc import Mapping, Sequence

import arena_robots.Robot
import arena_runtime_msgs.msg
import arena_runtime_msgs.srv
import arena_simulation_setup.tree.assets.Human
import arena_simulation_setup.tree.assets.Object
import arena_simulation_setup.tree.configs.environment
import arena_simulation_setup.tree.configs.parametrized
import arena_simulation_setup.tree.World as World
import attrs
import rclpy
import rclpy.lifecycle
import rclpy.qos
import tf2_ros
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.Async import ClientWrapper
from arena_rclpy_mixins.shared import Namespace
from arena_robots.Sensor import SensorType
from arena_runtime_msgs.msg import AdapterDisplay, AdapterEntry, AdapterVizManifest, RecordedTopics
from arena_simulation_setup.shared.judge import Sample, WallGrid
from arena_simulation_setup.tree.World import Level
from arena_simulation_setup.utils.articulation import family_members
from arena_viz.kinds import DisplayKind
from arena_viz.style import StyleSpec
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.exceptions import InvalidHandle
from rclpy.impl.implementation_singleton import rclpy_implementation
from rclpy.lifecycle import TransitionCallbackReturn

from arena_runtime.acoustics import AcousticsSimulatorRegistry, BaseAcousticsSimulator
from arena_runtime.env.colors import RobotColors
from arena_runtime.env.constants import Constants
from arena_runtime.env.constants.runtime import Configuration, migrate_deprecated_params
from arena_runtime.env.environment_manager import EnvironmentManager
from arena_runtime.env.phase_clock import PhaseClock
from arena_runtime.env.realizer import Realizer
from arena_runtime.env.robot_manager import RobotsManager
from arena_runtime.env.world_manager.world_manager_ros import WorldManagerROS as WorldManager
from arena_runtime.flags import flag_enabled
from arena_runtime.hearing import BaseHearing, HearingRegistry
from arena_runtime.humans import BaseHumanSimulator, HumanSimulatorRegistry
from arena_runtime.safe_callback import SafeCallbackNode
from arena_runtime.shared import Orientation, Pose, Position
from arena_runtime.sim import BaseSim, SimulatorRegistry
from arena_runtime.static_tf import StaticTransformBroadcaster

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins.node import AiomonitorConfig
    from arena_simulation_setup.tree import Identifier
    from std_srvs.srv import Trigger

    from arena_runtime.env.adapters import AdapterDisplayHint
    from arena_runtime.env.robot_manager import RobotManager
    from arena_runtime.env.robot_manager.task_runner import PhaseState
    from arena_runtime.shared import SemanticCfg
    from arena_runtime.sim._semantics import SemanticChange, SemanticEntitySnapshot

_LATCHED = rclpy.qos.QoSProfile(
    depth=1,
    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
)

# Node tick cadence, matching the mechanism-shim sim-time tick.
_SIM_TICK_RATE = 30.0
_ONE_TICK = 1e-6
_STEP_TIMEOUT = 10.0


def _available(identifier: type[Identifier]) -> list[str]:
    return [item.shortname for item in identifier.listall()]


async def _available_async(identifier: type[Identifier], **kwargs: bool) -> list[str]:
    return [item.shortname for item in await identifier.listall_async(**kwargs)]


class EnvNode(ArenaMixinNode, SafeCallbackNode, rclpy.lifecycle.LifecycleNode, metaclass=abc.ABCMeta):
    """The per-env lifecycle node: simulator, managers, semantics and entity services, driven by an episode loop subclass."""

    conf: type
    _simulator: BaseSim | None = None
    _realizer: Realizer
    _static_tf_broadcaster: StaticTransformBroadcaster
    _env_id: int
    _reference: tuple[float, float]
    _prespawn_offset: tuple[float, float]
    _world_manager: WorldManager
    _human_simulator: BaseHumanSimulator
    _acoustics_simulator: BaseAcousticsSimulator
    _hearing: BaseHearing
    _environment_manager: EnvironmentManager
    _robots_manager: RobotsManager | None = None
    _arena_hold_client: ClientWrapper
    _arena_unpause_window_client: ClientWrapper
    _arena_step_client: ClientWrapper

    @property
    def robots_manager(self) -> RobotsManager:
        assert self._robots_manager is not None
        return self._robots_manager

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)
        migrate_deprecated_params(self)
        self.conf = Configuration(self)

        self._namespace = Namespace(self.get_namespace())

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.rosparam[bool].set("initialized", False)

        self._env_id = self.rosparam[int].get("env_id", 0)
        # Reference and prespawn anchor are unknown at boot, populated by the first
        # confirm_world response handled in WorldManagerROS.apply_world.
        self._reference = (0.0, 0.0)
        self._prespawn_offset = (0.0, 0.0)

        self._reset_lock: asyncio.Lock = asyncio.Lock()
        self.robot_colors = RobotColors()

        # M2 semantics write path: inert-zone field overrides and bare->realized entity name map.
        self._zone_overrides: dict[tuple[str, str], float | bool | str] = {}
        self._sound_levels: dict[str, float] = {}
        self._semantic_names: dict[str, str] = {}
        self._judge_zones: dict = {}
        self._judge_walls: WallGrid | None = None
        self._known_subjects: set[str] = set()
        self._sample = Sample(t=0.0)
        self._tick_loop_task: asyncio.Task | None = None
        self._semantics_dirty = False

        self._pub_state_robots = self.create_publisher(
            arena_runtime_msgs.msg.RobotFleet,
            self.service_namespace("state", "robots"),
            _LATCHED,
        )

        self._pub_state_viz_manifest = self.create_publisher(
            AdapterVizManifest,
            self.service_namespace("state", "viz_manifest"),
            _LATCHED,
        )

        self._pub_state_recorded_topics = self.create_publisher(
            RecordedTopics,
            self.service_namespace("state", "recorded_topics"),
            _LATCHED,
        )

        self._pub_state_robots_pending = self.create_publisher(
            arena_runtime_msgs.msg.RobotQueue,
            self.service_namespace("state", "robots", "pending"),
            _LATCHED,
        )

        self._pub_state_semantics = self.create_publisher(
            arena_runtime_msgs.msg.SemanticSnapshot,
            self.service_namespace("state", "semantics"),
            _LATCHED,
        )

        self._pub_heartbeat = self.create_publisher(
            arena_runtime_msgs.msg.Heartbeat,
            self.service_namespace("state", "heartbeat"),
            10,
        )

        self._heartbeat_stop = threading.Event()
        threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True).start()

        self._arena_seen = False
        self._arena_watchdog_timer = self.wall_timer(1.0, self._cb_arena_watchdog)

        self._sub_shutdown_request = self.create_subscription(
            arena_runtime_msgs.msg.ShutdownRequest,
            "/arena/shutdown_request",
            self._cb_shutdown_request,
            10,
        )

        self._pub_state_robots_pending.publish(arena_runtime_msgs.msg.RobotQueue())

    # EPISODE LOOP HOOKS

    @abc.abstractmethod
    async def _set_up_task(self) -> None:
        """Build the episode loop's task layer on the managers, before the world syncs."""

    def _compromise(self, reason: str) -> None:
        """Runs when a simulator reports an incident."""
        self._logger.warning(reason)

    def _after_setup(self) -> None:
        """Runs once the env is initialized, before it configures."""

    async def _teardown_episode(self) -> None:
        """Runs in teardown after the tick loop stops, before the robots and the simulator go."""

    async def _on_tick(self, ended: Mapping[str, PhaseState]) -> None:
        """Runs on every sim tick after the robots are judged, before the semantics flush, `ended` mapping each robot to the phase that ended on this tick."""
        del ended

    def _extra_semantic_states(self) -> list[arena_runtime_msgs.msg.SemanticEntityState]:
        """Entities the episode loop appends to every semantics snapshot."""
        return []

    def _interaction_displays(self) -> list[AdapterDisplay]:
        """Env displays the episode loop adds to the viz manifest."""
        return []

    def on_task_submitted(self, manager: RobotManager) -> None:
        """Flush the robot's semantic state after a task submission."""
        del manager
        self._semantics_dirty = True

    def reset_timeline(self) -> None:
        """Drop the episode's inert-zone overrides and sound levels so the next reset replays identically."""
        self._zone_overrides.clear()
        self._sound_levels.clear()
        self._semantics_dirty = True

    # LIFECYCLE

    async def setup(self) -> None:
        clock = PhaseClock()
        try:
            await self._set_up_acoustics()
            clock.lap("acoustics")
            await self._set_up_services()
            clock.lap("services")
            await self._arena_hold_client.ensure()
            await self._arena_unpause_window_client.ensure()
            await self._arena_step_client.ensure()
            clock.lap("clients")

            await self.hold("setup")
            clock.lap("hold")
            try:
                await self._set_up_managers()
                clock.lap("managers")
                assert self._robots_manager is not None
                await self._set_up_task()
                clock.lap("task")
                await self._world_manager.sync()
                clock.lap("world")
                if flag_enabled(self, "debug", "map_server") or self._acoustics_simulator.requires_map_server:
                    await self._world_manager.require_map_server()
                    clock.lap("map_server")
                await self._robots_manager.launch_pending()
                clock.lap("launch")
            finally:
                await self.release("setup")
                clock.lap("release")
            self._publish_viz_manifest()
            self._logger.info(f"setup took {clock}")

            self.rosparam[bool].set("initialized", True)
            self._after_setup()
        except Exception as e:
            self._logger.error(f"configure failed: {e!r}\n{traceback.format_exc()}")
            raise
        self.trigger_configure()

    async def teardown(self) -> None:
        self._heartbeat_stop.set()
        if self._tick_loop_task is not None and not self._tick_loop_task.done():
            self._tick_loop_task.cancel()
            await asyncio.gather(self._tick_loop_task, return_exceptions=True)
        await self._teardown_episode()
        if self._robots_manager is not None:
            await self._robots_manager.teardown()
        if self._simulator is not None:
            await self._simulator.shutdown()

    def _declare_mutable_param(self, name: str, default: object, descriptor: ParameterDescriptor) -> None:
        self.rosparam[type(default)].declare_safe(name, default, descriptor=descriptor)

    def aiomonitor_config(self) -> AiomonitorConfig | None:
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

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(1.0) and rclpy.ok():
            with contextlib.suppress(rclpy_implementation.RCLError, InvalidHandle):
                self._cb_heartbeat_tick()

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
        self._heartbeat_stop.set()
        self.request_shutdown("arena gone")

    def _cb_shutdown_request(self, msg: arena_runtime_msgs.msg.ShutdownRequest) -> None:
        if msg.env_id != self._env_id:
            return
        self.get_logger().info(f"shutdown request received (reason={msg.reason!r}); shutting down")
        self._heartbeat_stop.set()
        self.request_shutdown(msg.reason)

    def on_configure(self, state: rclpy.lifecycle.State) -> TransitionCallbackReturn:
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: rclpy.lifecycle.State) -> TransitionCallbackReturn:
        self._heartbeat_stop.set()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: rclpy.lifecycle.State) -> TransitionCallbackReturn:
        self._heartbeat_stop.set()
        return TransitionCallbackReturn.SUCCESS

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

    async def step_tick(self) -> float | None:
        """Advance the held sim by one physics step through arena_runtime_node, None when it refuses or does not answer."""
        req = arena_runtime_msgs.srv.LifecycleStep.Request()
        req.seconds = _ONE_TICK
        res = await self._arena_step_client.call_timeout(req)
        if res is None or not res.success:
            return None
        return res.advanced

    async def _set_up_acoustics(self):
        self._logger.info("Setting up acoustics simulator")
        self._acoustics_simulator = await AcousticsSimulatorRegistry.get(
            self.conf.Arena.ACOUSTICS.value,
            node=self,
            namespace=self._namespace,
        )

        self._logger.info("Setting up robot hearing")
        self._hearing = await HearingRegistry.get(
            self.conf.Robot.HEARING.value,
            node=self,
            namespace=self._namespace,
        )
        self._pub_state_recorded_topics.publish(RecordedTopics(topics=[*self._acoustics_simulator.recorded_topics(), *self._hearing.recorded_topics()]))
        self._pub_state_viz_manifest.publish(AdapterVizManifest(plugins=list(self._acoustics_simulator.plugins())))

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
        self._simulator.set_incident_callback(self._compromise)

        self._logger.info("Setting up human simulator")
        self._human_simulator = await HumanSimulatorRegistry.get(
            self.conf.Arena.HUMAN.value,
            node=self,
            namespace=self._namespace,
            simulator=self._simulator,
            realizer=realizer,
            acoustics=self._acoustics_simulator,
            robots_visible=self.conf.Human.ROBOTS_VISIBLE.value,
        )
        self._simulator.set_semantics_callback(self._on_semantics_changed)
        self._human_simulator.set_incident_callback(self._compromise)

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

    # SEMANTICS

    def _semantic_entity_state_msg(self, snap: SemanticEntitySnapshot) -> arena_runtime_msgs.msg.SemanticEntityState:
        msg = arena_runtime_msgs.msg.SemanticEntityState()
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
    def _zone_field_role(cfg: SemanticCfg) -> str:
        """Coercion role of one inert-zone field: 'predicate' | 'continuous' | 'discrete'."""
        if cfg.role == "predicate":
            return "predicate"
        if isinstance(cfg.value, str):
            return "discrete"
        return "continuous"

    def _iter_zones(self) -> typing.Iterator[tuple[str, World.Zone]]:
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

    def _zone_semantic_states(self) -> list[arena_runtime_msgs.msg.SemanticEntityState]:
        """Inert zone annotations from the loaded world, env-prefixed, appended to every snapshot."""
        states: list[arena_runtime_msgs.msg.SemanticEntityState] = []
        for name, zone in self._iter_zones():
            if not zone.semantics:
                continue
            msg = arena_runtime_msgs.msg.SemanticEntityState()
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
        msg = arena_runtime_msgs.msg.SemanticSnapshot()
        msg.stamp = self.sim_time.to_msg()
        msg.env_id = self._env_id
        msg.world = self._world_manager.loaded_world
        assert self._simulator is not None
        snapshots = self._simulator.semantics_snapshot()
        self._world_manager.publish_world_markers(snapshots)
        entities = [self._semantic_entity_state_msg(s) for s in snapshots]
        entities.extend(self._zone_semantic_states())
        entities.extend(self._robot_semantic_states())
        entities.extend(self._extra_semantic_states())
        msg.entities = entities
        self._pub_state_semantics.publish(msg)

    def _on_semantics_changed(self, changes: Sequence[SemanticChange]) -> None:
        self._semantics_dirty = True
        self._forward_semantic_changes(changes)

    def _forward_semantic_changes(self, changes: Sequence[SemanticChange]) -> None:
        """Hand discrete and predicate semantic changes to the human simulator under authored and env-stripped names."""
        human = self._human_simulator
        if not changes:
            return
        from arena_runtime.sim._semantics import _SEMANTIC_KINDS

        authored = {realized: bare for bare, realized in self._semantic_names.items()}
        prefix = f"{self._realizer.prefix()}/"
        for change in changes:
            kind = _SEMANTIC_KINDS.get(change.kind)
            if change.field_kind == "state" and (kind.is_continuous(change.field) if kind is not None else self._is_number(change.current)):
                continue
            entity = authored.get(change.entity) or human.authored_name(change.entity)
            value = change.current.removeprefix(prefix)
            human.notify_semantic_change(entity, change.field, value)
            stripped = change.entity.removeprefix(prefix)
            if stripped != entity:
                human.notify_semantic_change(stripped, change.field, value)

    @staticmethod
    def _is_number(value: str) -> bool:
        try:
            float(value)
        except ValueError:
            return False
        return True

    @staticmethod
    def _stringify_float(value: float) -> str:
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
    def _coerce_zone_value(role: str, value: str) -> float | bool | str:
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
        if self._simulator is None:
            return "task generator not initialized"
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

    async def _cb_set_semantic(
        self,
        request: arena_runtime_msgs.srv.SetSemantic.Request,
        response: arena_runtime_msgs.srv.SetSemantic.Response,
    ) -> arena_runtime_msgs.srv.SetSemantic.Response:
        """External untrusted write: full validation, structured error_msg, no bare-name resolution."""
        reason = self._apply_semantic_checked(request.entity, request.field, request.value)
        response.success = not reason
        response.error_msg = reason
        return response

    def _semantic_value(self, entity: str, field: str) -> str | None:
        """Current stringified value of one semantic field across runtime and inert-zone entities."""
        realized = self._resolve_semantic_entity(entity)
        assert self._simulator is not None
        for snap in self._simulator.semantics_snapshot():
            if snap.entity != realized:
                continue
            if field in snap.discrete:
                return snap.discrete[field]
            if field in snap.continuous:
                return self._stringify_float(snap.continuous[field])
            if field in snap.predicates:
                return "true" if snap.predicates[field] else "false"
            if field == "volume_db" and realized in self._sound_levels:
                return self._stringify_float(self._sound_levels[realized])
        zone_value = self._zone_field_value(realized, field)
        if zone_value is not None:
            return zone_value
        return self._robot_field_value(realized, field)

    def _semantic_family(self, entity: str, family: str) -> dict[str, str] | None:
        """Current stringified value of each `<family>.<member>` field of one entity by member, None without a snapshot of the entity."""
        realized = self._resolve_semantic_entity(entity)
        assert self._simulator is not None
        members: dict[str, str] | None = None
        for snap in self._simulator.semantics_snapshot():
            if snap.entity != realized:
                continue
            members = members or {}
            members.update(family_members(snap.discrete, family))
            members.update({member: self._stringify_float(value) for member, value in family_members(snap.continuous, family).items()})
        return members

    def register_sound_levels(self, levels: dict[str, float]) -> None:
        """Asset level of each episode sound, read as its volume_db while its semantics carry none."""
        self._sound_levels = dict(levels)

    # SIM TICK

    async def _run_tick_loop(self) -> None:
        """Sim-paced node tick: robot judging, the episode loop's tick and the semantics flush at the mechanism rate."""
        with self.sim_time_rate(_SIM_TICK_RATE) as (done, rate):
            while not done.is_set():
                try:
                    await rate.get()
                except asyncio.CancelledError:
                    raise
                try:
                    self._sample = self._build_sample()
                    await self._on_tick(await self._tick_robots())
                    if self._semantics_dirty:
                        self._publish_semantics_snapshot()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.get_logger().warning(f"sim tick failed: {exc!r}")

    def _build_sample(self) -> Sample:
        """World state of this tick in the flattened abstract frame the judge and the compacted world share."""
        robots: dict[str, tuple[float, float, float]] = {}
        if self._robots_manager is not None:
            for manager in self._robots_manager.managers.values():
                pose = manager.pose
                if pose is None or not manager.placed:
                    continue
                local = self._environment_manager.ezilear(pose)
                robots[manager.name] = (local.position.x, local.position.y, local.orientation.to_yaw())
        peds: dict[str, tuple[float, float]] = {}
        for name, (x, y), _radius in self._human_simulator.pedestrian_discs():
            local = self._environment_manager.ezilear(Pose(Position(x, y), Orientation.identity()))
            peds[self._human_simulator.authored_name(name)] = (local.position.x, local.position.y)
        clips = {self._human_simulator.authored_name(name): playing for name, playing in self._human_simulator.pedestrian_clips().items()}
        self._known_subjects.update(peds)
        self._known_subjects.update(robots)
        return Sample(t=self.sim_time.to_seconds(), robots=robots, peds=peds, field=self._semantic_value, known=frozenset(self._known_subjects), clips=clips, walls=self._judge_walls, family=self._semantic_family)

    def _wall_grid(self, compacted: Level) -> WallGrid:
        """The loaded world's walls rasterized the way the replay judge rasterizes them."""
        return WallGrid.of_level(compacted)

    async def _tick_robots(self) -> dict[str, PhaseState]:
        if self._robots_manager is None:
            return {}
        managers = list(self._robots_manager.managers.values())
        for manager in managers:
            judged = self._sample.robots.get(manager.name)
            if judged is not None and manager.runner.active is not None:
                manager.publish_task_pose(Pose(Position(judged[0], judged[1]), Orientation.from_yaw(judged[2])))
        transitions = await asyncio.gather(*(manager.tick(self._sample, self._judge_zones) for manager in managers))
        if any(manager.runner.changed for manager in managers):
            for manager in managers:
                manager.runner.changed = False
            self._semantics_dirty = True
        return {manager.name: transition.ended for manager, transition in zip(managers, transitions, strict=True) if transition.ended is not None}

    def ped_names(self) -> list[str]:
        """Bare names of the pedestrians the human simulator currently tracks."""
        return [self._human_simulator.authored_name(name) for name, _xy, _radius in self._human_simulator.pedestrian_discs()]

    @staticmethod
    def _robot_fields(manager: RobotManager) -> dict[str, str]:
        runner = manager.runner
        return {
            "phase": "" if runner.active is None else str(runner.active),
            "met": ",".join(map(str, runner.outcomes("met"))),
            "failed": ",".join(map(str, runner.outcomes("failed"))),
            "dropped": ",".join(map(str, runner.outcomes("dropped"))),
            "violated": ",".join(runner.violated),
        }

    def _robot_semantic_states(self) -> list[arena_runtime_msgs.msg.SemanticEntityState]:
        """Task progress of every robot as a `robot` entity, fields grow-only so change-driven snapshots keep every transition."""
        states: list[arena_runtime_msgs.msg.SemanticEntityState] = []
        if self._robots_manager is None:
            return states
        for manager in self._robots_manager.managers.values():
            fields = self._robot_fields(manager)
            msg = arena_runtime_msgs.msg.SemanticEntityState()
            msg.entity = self._realizer.prefix(manager.name)
            msg.kind = "robot"
            msg.discrete_names = list(fields.keys())
            msg.discrete_values = list(fields.values())
            states.append(msg)
        return states

    def _robot_field_value(self, entity: str, field: str) -> str | None:
        if self._robots_manager is None:
            return None
        for manager in self._robots_manager.managers.values():
            if self._realizer.prefix(manager.name) == entity:
                return self._robot_fields(manager).get(field)
        return None

    # VIZ MANIFEST

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
        latched = StyleSpec(latched=True).to_json()
        human_sim = self.conf.Arena.HUMAN.value
        if human_sim not in (Constants.HumanSimulator.DUMMY, Constants.HumanSimulator.NONE):
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
                        topic_must_exist=False,
                        group="Static",
                    )
                )

        env_displays.extend(self._interaction_displays())

        env_displays.append(
            AdapterDisplay(
                name="World",
                topic=f"{env_ns}/world_markers",
                topic_type="visualization_msgs/MarkerArray",
                kind=DisplayKind.MARKER_ARRAY,
                style_json=StyleSpec(latched=True, enabled=False).to_json(),
                topic_must_exist=False,
                group="Static",
            )
        )

        env_displays.append(
            AdapterDisplay(
                name="Affordances",
                topic=f"{env_ns}/affordance_markers",
                topic_type="visualization_msgs/MarkerArray",
                kind=DisplayKind.MARKER_ARRAY,
                style_json=StyleSpec(latched=True, enabled=False).to_json(),
                topic_must_exist=False,
                group="Interaction",
            )
        )

        env_displays.extend(self._acoustics_simulator.displays())
        entries: list[AdapterEntry] = []
        assert self._robots_manager is not None
        self.robot_colors.retain(self._robots_manager.managers)
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
            for adapter_kind, backend_displays in (
                ("_acoustics", self._acoustics_simulator.robot_displays(robot_value)),
                ("_hearing", self._hearing.robot_displays(robot_value)),
            ):
                if backend_displays:
                    entries.append(
                        AdapterEntry(
                            robot_ns=ns_value,
                            adapter_kind=adapter_kind,
                            displays=list(backend_displays),
                        )
                    )

            for adapter in mgr._adapter_instances:

                def _subst(s: str, ns_value: str = ns_value, robot_value: str = robot_value) -> str:
                    return s.replace("{ns}", ns_value).replace("{robot}", robot_value)

                def _style(hint: AdapterDisplayHint, robot_value: str = robot_value) -> str:
                    if not hint.robot_colored:
                        return _subst(hint.style_json)
                    return attrs.evolve(StyleSpec.from_json(_subst(hint.style_json)), color=self.robot_colors.rgb(robot_value)).to_json()

                displays = [
                    AdapterDisplay(
                        name=hint.name,
                        topic=_subst(hint.topic),
                        topic_type=hint.topic_type,
                        kind=hint.kind,
                        style_json=_style(hint),
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
        self._pub_state_viz_manifest.publish(AdapterVizManifest(env_displays=env_displays, entries=entries, plugins=list(self._acoustics_simulator.plugins())))

    # SERVICES

    async def _cb_wait_for_world(
        self,
        request: object,
        response: object,
    ) -> object:
        await self._world_manager.sync()
        return response

    async def _cb_query_worlds(
        self,
        request: arena_runtime_msgs.srv.QueryWorlds.Request,
        response: arena_runtime_msgs.srv.QueryWorlds.Response,
    ) -> arena_runtime_msgs.srv.QueryWorlds.Response:
        response.ids = _available(World.WorldIdentifier)
        return response

    async def _cb_query_robots(
        self,
        request: arena_runtime_msgs.srv.QueryRobots.Request,
        response: arena_runtime_msgs.srv.QueryRobots.Response,
    ) -> arena_runtime_msgs.srv.QueryRobots.Response:
        response.ids = _available(arena_robots.Robot.RobotIdentifier)
        return response

    async def _cb_query_static_obstacles(
        self,
        request: arena_runtime_msgs.srv.QueryStaticObstacles.Request,
        response: arena_runtime_msgs.srv.QueryStaticObstacles.Response,
    ) -> arena_runtime_msgs.srv.QueryStaticObstacles.Response:
        response.ids = await _available_async(arena_simulation_setup.tree.assets.Object.ObjectIdentifier, network=True)
        return response

    async def _cb_query_dynamic_obstacles(
        self,
        request: arena_runtime_msgs.srv.QueryDynamicObstacles.Request,
        response: arena_runtime_msgs.srv.QueryDynamicObstacles.Response,
    ) -> arena_runtime_msgs.srv.QueryDynamicObstacles.Response:
        response.ids = await _available_async(
            arena_simulation_setup.tree.assets.Human.HumanIdentifier,
            network=True,
        )
        return response

    async def _cb_query_environments(
        self,
        request: arena_runtime_msgs.srv.QueryEnvironments.Request,
        response: arena_runtime_msgs.srv.QueryEnvironments.Response,
    ) -> arena_runtime_msgs.srv.QueryEnvironments.Response:
        response.ids = _available(arena_simulation_setup.tree.configs.environment.EnvironmentIdentifier)
        return response

    async def _cb_query_parametrizeds(
        self,
        request: arena_runtime_msgs.srv.QueryParametrizeds.Request,
        response: arena_runtime_msgs.srv.QueryParametrizeds.Response,
    ) -> arena_runtime_msgs.srv.QueryParametrizeds.Response:
        response.ids = _available(arena_simulation_setup.tree.configs.parametrized.ParametrizedIdentifier)
        return response

    async def _cb_require_map(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        if not self.rosparam[bool].get("initialized", False):
            response.success = False
            response.message = "task generator not initialized"
            return response
        async with self._reset_lock:
            await self._world_manager.require_map_server()
        response.success = True
        response.message = ""
        return response

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

        self._arena_step_client = self.create_client_wrapper(
            arena_runtime_msgs.srv.LifecycleStep,
            "/arena/sim_lifecycle/step",
            timeout=_STEP_TIMEOUT,
        )

        from std_srvs.srv import Empty as EmptySrv

        self.create_service(
            EmptySrv,
            self.service_namespace("lifecycle", "wait_for_world"),
            self._cb_wait_for_world,
        )

        self.create_service(
            arena_runtime_msgs.srv.QueryWorlds,
            self.service_namespace("query", "worlds"),
            self._cb_query_worlds,
        )

        self.create_service(
            arena_runtime_msgs.srv.QueryRobots,
            self.service_namespace("query", "robots"),
            self._cb_query_robots,
        )

        self.create_service(
            arena_runtime_msgs.srv.QueryStaticObstacles,
            self.service_namespace("query", "static_obstacles"),
            self._cb_query_static_obstacles,
        )

        self.create_service(
            arena_runtime_msgs.srv.QueryDynamicObstacles,
            self.service_namespace("query", "dynamic_obstacles"),
            self._cb_query_dynamic_obstacles,
        )

        self.create_service(
            arena_runtime_msgs.srv.QueryEnvironments,
            self.service_namespace("query", "environments"),
            self._cb_query_environments,
        )

        self.create_service(
            arena_runtime_msgs.srv.QueryParametrizeds,
            self.service_namespace("query", "parametrizeds"),
            self._cb_query_parametrizeds,
        )

        self.create_service(
            arena_runtime_msgs.srv.SetSemantic,
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
