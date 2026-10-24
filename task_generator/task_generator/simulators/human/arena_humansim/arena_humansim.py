"""Task generator adapter for arena_humansim."""

import array
import asyncio
import math
import traceback
import typing
from collections.abc import Mapping, Sequence

import attrs
import numpy as np
import yaml
from arena_humansim.core.interaction_kinds import InteractionType, MembershipRole
from arena_humansim_msgs.msg import (
    AgentFrame as AgentFrameMsg,
)
from arena_humansim_msgs.msg import (
    AgentGestures as AgentGesturesMsg,
)
from arena_humansim_msgs.msg import (
    AgentMeta as AgentMetaMsg,
)
from arena_humansim_msgs.msg import (
    AgentState as AgentStateMsg,
)
from arena_humansim_msgs.msg import (
    AgentStates as AgentStatesMsg,
)
from arena_humansim_msgs.msg import (
    Gesture as EngineGestureMsg,
)
from arena_humansim_msgs.msg import (
    InteractionFrame as InteractionFrameMsg,
)
from arena_humansim_msgs.msg import (
    ObstacleConfig as ObstacleConfigMsg,
)
from arena_humansim_msgs.msg import (
    RateKeyframe as RateKeyframeMsg,
)
from arena_humansim_msgs.msg import (
    Shape as ShapeMsg,
)
from arena_humansim_msgs.msg import (
    SinkConfig as SinkConfigMsg,
)
from arena_humansim_msgs.msg import (
    SourceConfig as SourceConfigMsg,
)
from arena_humansim_msgs.msg import (
    TeleportZone as TeleportZoneMsg,
)
from arena_humansim_msgs.msg import (
    Waypoint as WaypointMsg,
)
from arena_humansim_msgs.msg import (
    Waypoints as WaypointsMsg,
)
from arena_humansim_msgs.msg import (
    WorldObjectInfo as WorldObjectInfoMsg,
)
from arena_humansim_msgs.srv import (
    AddObstacles,
    AddSink,
    AddSource,
    AddWalls,
    AddWorldObjects,
    Feedback,
    GetProfile,
    NotifyStimulus,
    RemoveAgents,
    RemoveObstacles,
    RemoveSink,
    RemoveSource,
    RemoveWalls,
    RemoveWorldObjects,
    ResetSimulation,
    SetFlow,
    SetTeleportZones,
    SetWaypoints,
    SetZoneAccess,
    SpawnAgents,
    TeleportAgents,
    UpdateRobot,
)
from arena_people_msgs.msg import Gesture as GestureMsg
from arena_people_msgs.msg import Pedestrian, Pedestrians
from arena_rclpy_mixins.Async import ClientWrapper
from arena_rclpy_mixins.lazy import LazySubscription
from arena_runtime_msgs.msg import LockstepChannel
from arena_simulation_setup.shared.world import CABIN_DOOR_INSET, CABIN_DOOR_THRESHOLD, CABIN_DOOR_WIDTH
from arena_simulation_setup.tree.assets.Human import HumanIdentifier
from arena_simulation_setup.tree.World.model import elevator_doorway_pose
from arena_simulation_setup.utils.affordance import AffordanceClaims, AffordanceKind, AffordanceSlot, SlotOccupied, conflicts, has_z, max_occupancy, placed_slots
from geometry_msgs.msg import (
    Point,
    Point32,
    Quaternion,
    Twist,
    Vector3,
)
from geometry_msgs.msg import Pose as PoseMsg
from geometry_msgs.msg import (
    Pose2D as Pose2DMsg,
)
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters, SetParameters, SetParametersAtomically
from rclpy.parameter import Parameter
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from visualization_msgs.msg import MarkerArray

from task_generator.constants import Constants
from task_generator.constants.rng import stable_int
from task_generator.manager.realizer import Realizer
from task_generator.shared import Door, DynamicObstacle, Elevator, Obstacle, Orientation, Pose, Position, Region, Robot, Wall
from task_generator.simulators.human import PED_RADIUS, BaseHumanSimulator, HumanSimulatorOptions, Segment
from task_generator.simulators.human.itinerary import STIMULUS_TELEPORTED, ItineraryDriver, Stop, semantic_atom

from .agent import ArenaHumanDynamicObstacle, agent_type_def, agent_type_stimuli, flow_agent_template, path_agent_type_name

_DEFERRED_ID_BASE = 1 << 24

_QUEUE_FRONT_OFFSET = 0.7
_QUEUE_STEP = 0.8


@attrs.frozen
class PedMechanics:
    """Scenario fields of a `dynamic:` entry that the arena adapter acts on besides the agent block."""

    spawn_delay: float = 0.0
    # alias -> authored entity
    semantic_alias: dict[str, str] = attrs.Factory(dict)
    # aligned with the waypoint list
    itinerary: list[dict] = attrs.Factory(list)

    @classmethod
    def from_dynamic_obstacle(cls, obs: DynamicObstacle) -> "PedMechanics":
        delay = float(obs.spawn_delay)
        if delay < 0.0 or delay != delay:
            raise ValueError(f"{obs.name}: spawn_delay must be >= 0, got {obs.spawn_delay!r}")
        alias = obs.semantic_alias or {}
        if not isinstance(alias, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in alias.items()):
            raise ValueError(f"{obs.name}: semantic_alias must map alias names to entity names, got {alias!r}")
        itinerary = obs.extra.get("itinerary") or []
        if not isinstance(itinerary, list):
            raise ValueError(f"{obs.name}: itinerary must be a list")
        return cls(spawn_delay=delay, semantic_alias=dict(alias), itinerary=list(itinerary))

    def stops(self, points: list[tuple[float, float]]) -> list[Stop] | None:
        """Itinerary stops at the given (engine-frame) waypoint positions, None when nothing holds the ped."""
        if not self.itinerary or len(self.itinerary) != len(points):
            return None
        stops = [Stop.from_dict({**meta, "x": x, "y": y}) for meta, (x, y) in zip(self.itinerary, points, strict=True)]
        return stops if any(stop.holds for stop in stops) else None


def seat_capacity(humansim: Mapping[str, typing.Any], slots: Sequence[AffordanceSlot]) -> int:
    """Queue length at which needs-driven agents pass the object over: the scenario's capacity or seat count, else the most of its sit slots held together, at least 1."""
    if "capacity" in humansim:
        return int(humansim["capacity"])
    if "seats" in humansim:
        return max(1, len(humansim["seats"]))
    return max(1, max_occupancy(slots))


def slot_height(slot: AffordanceSlot) -> float:
    """Height of a sit slot's surface in the object frame, the object standing on its floor, NaN for a slot without a z range."""
    return slot.bbox[2][1] if has_z(slot.bbox) else math.nan


SEAT_SLOT_REACH = 0.15  # m, a scenario seat this close to a sit slot's footprint takes that slot's height


def seat_height(seat: Pose, slots: Sequence[AffordanceSlot]) -> float:
    """Surface height of a scenario seat in the object frame, NaN when unknown.

    Its z when the scenario gives one, else the height of the sit slot it lies in or next to (within SEAT_SLOT_REACH);
    among overlapping slots, the one whose center is nearest.
    """
    if seat.position.z > 0.0:
        return seat.position.z
    xy = (seat.position.x, seat.position.y)
    near = [slot for slot in slots if has_z(slot.bbox) and slot.distance(xy) <= SEAT_SLOT_REACH]
    if not near:
        return math.nan
    return slot_height(min(near, key=lambda slot: (slot.distance(xy), math.dist(slot.center, xy))))


def held_seats(frame: InteractionFrameMsg) -> dict[str, list[int]]:
    """Seat indices the participants of sit and lie interactions hold, by object."""
    held: dict[str, list[int]] = {}
    objects, offsets, seats, roles = list(frame.object_id), list(frame.member_offset), list(frame.member_seat), list(frame.member_role)
    for index, kind in enumerate(frame.interaction_type):
        if kind not in (InteractionType.SIT_ON, InteractionType.LIE_ON) or not objects[index]:
            continue
        held.setdefault(objects[index], []).extend(int(seats[m]) for m in range(offsets[index], offsets[index + 1]) if roles[m] == MembershipRole.PARTICIPANT and seats[m] >= 0)
    return held


def seat_violation(slots_by_object: Mapping[str, Sequence[AffordanceSlot]], held: Mapping[str, Sequence[int]]) -> str | None:
    """The first object whose held seats overlap or fall outside its slots, None when every held set fits."""
    for object_id, seats in sorted(held.items()):
        slots = slots_by_object.get(object_id)
        if slots is None:
            continue
        claims = AffordanceClaims(list(slots))
        for seat in seats:
            if not 0 <= seat < len(slots):
                return f"affordance invariant: {object_id} holds seat {seat} of {len(slots)}"
            try:
                claims.claim(slots[seat])
            except SlotOccupied as e:
                return f"affordance invariant: {object_id}: {e}"
    return None


def conflict_offsets(slots: Sequence[AffordanceSlot]) -> tuple[list[int], list[int]]:
    """Per-slot overlap lists flattened into offsets and indices for WorldObjectInfo."""
    offsets, flat = [0], []
    for overlapping in conflicts(slots):
        flat.extend(overlapping)
        offsets.append(len(flat))
    return offsets, flat


def reach_radius(bounding_box: Sequence[Sequence[float]]) -> float:
    """Distance from the object origin within which an agent standing beside it reaches any point of its footprint."""
    (x_min, x_max), (y_min, y_max) = bounding_box[0], bounding_box[1]
    return max(math.hypot(x, y) for x in (x_min, x_max) for y in (y_min, y_max)) + PED_RADIUS


class ArenaHumanSimulator(BaseHumanSimulator):
    PARAM_NAMESPACE = Constants.HUMAN_PARAM_NAMESPACES[Constants.HumanSimulator.ARENA]

    @classmethod
    def _register_task_modes(cls):
        from task_generator.tasks.obstacles.prompt import NS, declare_schema
        from task_generator.tasks.registry import OBSTACLES_MODES

        @OBSTACLES_MODES.register(Constants.TaskMode.TM_Obstacles.PROMPT, namespace=NS, schema=declare_schema)
        def _prompt() -> type:
            from task_generator.tasks.obstacles.prompt.arena import TM_Prompt

            return TM_Prompt

    SERVICE_SPAWN_AGENTS = "spawn_agents"
    SERVICE_REMOVE_AGENTS = "remove_agents"
    SERVICE_UPDATE_ROBOT = "update_robot"
    SERVICE_SET_WAYPOINTS = "set_waypoints"
    SERVICE_SET_FLOW = "set_flow"
    SERVICE_ADD_SOURCE = "add_source"
    SERVICE_REMOVE_SOURCE = "remove_source"
    SERVICE_ADD_SINK = "add_sink"
    SERVICE_REMOVE_SINK = "remove_sink"
    SERVICE_ADD_WALLS = "add_walls"
    SERVICE_REMOVE_WALLS = "remove_walls"
    SERVICE_ADD_OBSTACLES = "add_obstacles"
    SERVICE_REMOVE_OBSTACLES = "remove_obstacles"
    SERVICE_ADD_WORLD_OBJECTS = "add_world_objects"
    SERVICE_REMOVE_WORLD_OBJECTS = "remove_world_objects"
    SERVICE_RESET = "reset"
    SERVICE_GET_PROFILE = "get_profile"
    SERVICE_FEEDBACK = "feedback"
    SERVICE_NOTIFY_STIMULUS = "notify_stimulus"
    SERVICE_SET_TELEPORT_ZONES = "set_teleport_zones"
    SERVICE_SET_ZONE_ACCESS = "set_zone_access"
    SERVICE_TELEPORT_AGENTS = "teleport_agents"
    TAKES_STIMULI = True

    def __init__(self, **kwargs: typing.Unpack[HumanSimulatorOptions]) -> None:
        super().__init__(**kwargs)

        self._spawn_client: ClientWrapper = self.node.create_client_wrapper(
            SpawnAgents,
            self.node.service_namespace(self.SERVICE_SPAWN_AGENTS),
        )
        self._remove_client: ClientWrapper = self.node.create_client_wrapper(
            RemoveAgents,
            self.node.service_namespace(self.SERVICE_REMOVE_AGENTS),
        )
        self._set_flow_client: ClientWrapper = self.node.create_client_wrapper(
            SetFlow,
            self.node.service_namespace(self.SERVICE_SET_FLOW),
        )
        self._add_source_client: ClientWrapper = self.node.create_client_wrapper(
            AddSource,
            self.node.service_namespace(self.SERVICE_ADD_SOURCE),
        )
        self._remove_source_client: ClientWrapper = self.node.create_client_wrapper(
            RemoveSource,
            self.node.service_namespace(self.SERVICE_REMOVE_SOURCE),
        )
        self._add_sink_client: ClientWrapper = self.node.create_client_wrapper(
            AddSink,
            self.node.service_namespace(self.SERVICE_ADD_SINK),
        )
        self._remove_sink_client: ClientWrapper = self.node.create_client_wrapper(
            RemoveSink,
            self.node.service_namespace(self.SERVICE_REMOVE_SINK),
        )
        self._add_walls_client: ClientWrapper = self.node.create_client_wrapper(
            AddWalls,
            self.node.service_namespace(self.SERVICE_ADD_WALLS),
        )
        self._remove_walls_client: ClientWrapper = self.node.create_client_wrapper(
            RemoveWalls,
            self.node.service_namespace(self.SERVICE_REMOVE_WALLS),
        )
        self._add_obstacles_client: ClientWrapper = self.node.create_client_wrapper(
            AddObstacles,
            self.node.service_namespace(self.SERVICE_ADD_OBSTACLES),
        )
        self._remove_obstacles_client: ClientWrapper = self.node.create_client_wrapper(
            RemoveObstacles,
            self.node.service_namespace(self.SERVICE_REMOVE_OBSTACLES),
        )
        self._add_world_objects_client: ClientWrapper = self.node.create_client_wrapper(
            AddWorldObjects,
            self.node.service_namespace(self.SERVICE_ADD_WORLD_OBJECTS),
        )
        self._remove_world_objects_client: ClientWrapper = self.node.create_client_wrapper(
            RemoveWorldObjects,
            self.node.service_namespace(self.SERVICE_REMOVE_WORLD_OBJECTS),
        )
        self._update_robot_client: ClientWrapper = self.node.create_client_wrapper(
            UpdateRobot,
            self.node.service_namespace(self.SERVICE_UPDATE_ROBOT),
        )
        self._set_waypoints_client: ClientWrapper = self.node.create_client_wrapper(
            SetWaypoints,
            self.node.service_namespace(self.SERVICE_SET_WAYPOINTS),
        )
        self._reset_client: ClientWrapper = self.node.create_client_wrapper(
            ResetSimulation,
            self.node.service_namespace(self.SERVICE_RESET),
        )
        self._get_profile_client: ClientWrapper = self.node.create_client_wrapper(
            GetProfile,
            self.node.service_namespace(self.SERVICE_GET_PROFILE),
        )
        self._feedback_client: ClientWrapper = self.node.create_client_wrapper(
            Feedback,
            self.node.service_namespace(self.SERVICE_FEEDBACK),
        )
        self._set_params_client: ClientWrapper = self.node.create_client_wrapper(
            SetParametersAtomically,
            self.node.service_namespace("arena_humansim", "set_parameters_atomically"),
        )
        self._set_viz_params_client: ClientWrapper = self.node.create_client_wrapper(
            SetParameters,
            self.node.service_namespace("arena_humansim_viz", "set_parameters"),
        )
        self._viz_config: Realizer._Configuration | None = None
        self._viz_config_task: asyncio.Task[None] | None = None
        self._publish_pending = False
        self._notify_stimulus_client: ClientWrapper = self.node.create_client_wrapper(
            NotifyStimulus,
            self.node.service_namespace(self.SERVICE_NOTIFY_STIMULUS),
        )
        self._set_teleport_zones_client: ClientWrapper = self.node.create_client_wrapper(
            SetTeleportZones,
            self.node.service_namespace(self.SERVICE_SET_TELEPORT_ZONES),
        )
        self._set_zone_access_client: ClientWrapper = self.node.create_client_wrapper(
            SetZoneAccess,
            self.node.service_namespace(self.SERVICE_SET_ZONE_ACCESS),
        )
        self._teleport_agents_client: ClientWrapper = self.node.create_client_wrapper(
            TeleportAgents,
            self.node.service_namespace(self.SERVICE_TELEPORT_AGENTS),
        )

        self._spawns_in_flight: int = 0
        self._contact_config: tuple[str, float] = ("enabled", 1.2)
        self._gesture_config: str = "enabled"

        self._agents_lock: asyncio.Lock = asyncio.Lock()
        self._prev_agent_states: AgentFrameMsg | None = None
        self._curr_agent_states: AgentFrameMsg | None = None
        self._engine_stamp_ns: int | None = None
        self._agent_gestures: dict[int, list[EngineGestureMsg]] = {}
        self._agent_types: dict[int, str] = {}
        self._agent_type_paths: dict[str, str] = {}
        self._arena_pedestrians: Pedestrians = Pedestrians()
        self._arena_pedestrians.header.frame_id = "map"

        # IDs managed by the bridge (scenario-defined agents)
        self._bridge_agent_ids: set[int] = set()
        # IDs of flow agents (source/sink) with a live actor in the simulator
        self._flow_agent_ids: set[int] = set()
        self._flow_meta_wait: dict[int, int] = {}
        self._tagged_models: dict[tuple[str, ...], list[HumanIdentifier]] = {}
        self._warned_untagged_types: set[str] = set()
        # agent_id → human-readable name (scenario YAML name or flow source label)
        self._agent_names: dict[int, str] = {}
        self._teleport_stamps: dict[int, int] = {}

        self._tick_loop_task: asyncio.Task | None = None
        self._update_loop_task: asyncio.Task | None = None
        self._feedback_loop_task: asyncio.Task | None = None

        # world mechanisms
        self._deferred_spawns: list[tuple[float, AgentStateMsg, dict[str, str]]] = []
        self._next_deferred_id = _DEFERRED_ID_BASE
        self._itineraries: dict[int, ItineraryDriver] = {}
        self._semantic_aliases: dict[int, dict[str, str]] = {}
        self._mech_walls_desired: dict[str, list[Segment] | None] = {}
        self._mech_walls_applied: dict[str, list[str]] = {}
        self._mech_objects: dict[str, list[str]] = {}
        self._mech_walls_syncing = False

        # Publish robot poses on world_state topic (consumed by arena_humansim)
        self._world_state_pub = self.node.create_publisher(
            AgentStatesMsg,
            self.node.service_namespace("world_state"),
            10,
        )

        # Subscribe to agent_states topic from arena_humansim
        self.node.create_subscription(
            AgentFrameMsg,
            self.node.service_namespace("agent_states"),
            self._agent_states_callback,
            10,
        )

        self._held_seats: dict[str, list[int]] = {}
        self._seat_slots: dict[str, list[AffordanceSlot]] = {}
        self._seat_violation_reported = False
        self.node.create_subscription(
            InteractionFrameMsg,
            self.node.service_namespace("interactions"),
            self._interactions_callback,
            QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
        )

        self._debug_markers_sub = LazySubscription(
            self.node,
            self._marker_publisher,
            MarkerArray,
            self.node.service_namespace("viz"),
            self._forward_debug_markers,
            QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=5,
            ),
        )

        static_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._static_walls_pub = self.node.create_publisher(
            MarkerArray,
            self._namespace("pedestrian_markers", "static_walls"),
            static_qos,
        )
        self._static_objects_pub = self.node.create_publisher(
            MarkerArray,
            self._namespace("pedestrian_markers", "static_objects"),
            static_qos,
        )
        self.node.create_subscription(
            MarkerArray,
            self.node.service_namespace("viz_static", "walls"),
            lambda msg: self._static_walls_pub.publish(self._markers_from_engine(msg)),
            static_qos,
        )
        self.node.create_subscription(
            MarkerArray,
            self.node.service_namespace("viz_static", "objects"),
            lambda msg: self._static_objects_pub.publish(self._markers_from_engine(msg)),
            static_qos,
        )
        self.node.create_subscription(
            AgentGesturesMsg,
            self.node.service_namespace("agent_gestures"),
            self._agent_gestures_callback,
            static_qos,
        )
        self.node.create_subscription(
            AgentMetaMsg,
            self.node.service_namespace("agent_meta"),
            self._agent_meta_callback,
            static_qos,
        )

    def _forward_debug_markers(self, msg: MarkerArray):
        self._marker_publisher.publish(lambda: self._markers_from_engine(msg))

    def _markers_from_engine(self, msg: MarkerArray) -> MarkerArray:
        """Engine markers into the env frame: marker poses shift, points are pose-relative and stay."""
        cfg = self._realizer.get_config()
        for marker in msg.markers:
            marker.pose.position.x += cfg.x
            marker.pose.position.y += cfg.y
        return msg

    def _interactions_callback(self, msg: InteractionFrameMsg) -> None:
        self.node.event_loop.call_soon_threadsafe(self._apply_held_seats, held_seats(msg))

    def _apply_held_seats(self, held: dict[str, list[int]]) -> None:
        if held == self._held_seats:
            return
        self._held_seats = held
        if not self._seat_violation_reported and (violation := seat_violation(self._seat_slots, held)) is not None:
            self._seat_violation_reported = True
            self._report_incident(violation)
        self._notify_held_seats()

    def held_seats(self) -> dict[str, list[int]]:
        return {object_id: seats for object_id, seats in self._held_seats.items() if object_id in self._seat_slots}

    def _agent_states_callback(self, msg: AgentFrameMsg):
        """Cache prev/curr snapshots from arena_humansim for local interpolation."""
        self._prev_agent_states = self._curr_agent_states
        self._curr_agent_states = msg
        self._engine_stamp_ns = msg.header.stamp.sec * int(1e9) + msg.header.stamp.nanosec
        # publish on receipt (the rate loop can race a stepped clock burst), but
        # coalesced: one roster compute per burst, not per engine tick
        if self._publish_pending:
            return
        self._publish_pending = True
        loop = self.node.event_loop
        loop.call_soon_threadsafe(lambda: loop.create_task(self._publish_on_receipt()))

    def _agent_gestures_callback(self, msg: AgentGesturesMsg):
        """Group the latest engine gestures by owner."""
        grouped: dict[int, list[EngineGestureMsg]] = {}
        for agent_id, gesture in zip(msg.agent_id, msg.gestures, strict=True):
            grouped.setdefault(agent_id, []).append(gesture)
        self._agent_gestures = grouped

    def _agent_meta_callback(self, msg: AgentMetaMsg):
        """Latest agent type per engine agent."""
        types = msg.agent_type if len(msg.agent_type) == len(msg.agent_id) else [""] * len(msg.agent_id)
        self._agent_types = dict(zip(msg.agent_id, types, strict=True))

    def _remember_agent_type_path(self, agent_type: str) -> None:
        """File a path-based agent type under the name the engine reports for it."""
        name = path_agent_type_name(agent_type)
        if name is not None:
            self._agent_type_paths[name] = agent_type

    def _agent_type_of(self, agent_id: int) -> str:
        """Agent type of an engine agent, the spawned yaml path when its type came from a file."""
        name = self._agent_types.get(agent_id, "")
        return self._agent_type_paths.get(name, name)

    async def _publish_on_receipt(self) -> None:
        self._publish_pending = False
        cfg = self._realizer.get_config()
        if cfg != self._viz_config and (self._viz_config_task is None or self._viz_config_task.done()):
            self._viz_config_task = asyncio.create_task(self._configure_viz(cfg))
        await self._publish_interpolated()

    async def _configure_viz(self, cfg: Realizer._Configuration) -> None:
        """Point the engine's marker node at this env's marker topic, shifted by the env offset."""
        parameters = [
            Parameter("output_topic", value=self._marker_publisher.publisher.topic_name).to_parameter_msg(),
            Parameter("offset_x", value=float(cfg.x)).to_parameter_msg(),
            Parameter("offset_y", value=float(cfg.y)).to_parameter_msg(),
            Parameter("robot_bodies", value=False).to_parameter_msg(),
        ]
        response = await self._set_viz_params_client.call_timeout(SetParameters.Request(parameters=parameters))
        if response is not None and all(result.successful for result in response.results):
            self._viz_config = cfg

    async def _publish_interpolated(self) -> None:
        """Interpolate at the current sim time, never before the engine's latest frame, and publish the roster."""
        now = self.node.sim_time
        now_ns = now.sec * int(1e9) + now.nanosec
        if self._engine_stamp_ns is not None:
            now_ns = max(now_ns, self._engine_stamp_ns)
        states = self._interpolate_agent_states(now_ns)
        if states is None:
            return
        peds = self._agent_states_to_pedestrians(states)
        async with self._agents_lock:
            self._arena_pedestrians = peds
        self.publish_arena_peds(peds)

    def _interpolate_agent_states(self, now_ns: int) -> AgentFrameMsg | None:
        """Lerp between prev and curr agent states at the given timestamp, an empty frame at the engine's latest stamp once its agents are removed."""
        curr = self._curr_agent_states
        if curr is None:
            if self._engine_stamp_ns is None:
                return None
            empty = AgentFrameMsg()
            empty.header.stamp.sec = self._engine_stamp_ns // int(1e9)
            empty.header.stamp.nanosec = self._engine_stamp_ns % int(1e9)
            empty.header.frame_id = "map"
            return empty
        prev = self._prev_agent_states
        if prev is None:
            return curr

        prev_ns = prev.header.stamp.sec * int(1e9) + prev.header.stamp.nanosec
        curr_ns = curr.header.stamp.sec * int(1e9) + curr.header.stamp.nanosec
        dt_ns = curr_ns - prev_ns
        if dt_ns <= 0:
            return curr

        alpha = max(0.0, min(1.0, (now_ns - prev_ns) / dt_ns))
        inv = 1.0 - alpha
        _, ci, pi = np.intersect1d(np.asarray(curr.agent_id), np.asarray(prev.agent_id), assume_unique=True, return_indices=True)
        if self._teleport_stamps:
            self._teleport_stamps = {agent_id: stamp for agent_id, stamp in self._teleport_stamps.items() if prev_ns < stamp}
            keep = ~np.isin(np.asarray(curr.agent_id)[ci], list(self._teleport_stamps))
            ci, pi = ci[keep], pi[keep]

        # stamp with the time the lerped pose corresponds to, receivers dead-reckon from it
        pose_ns = prev_ns + int(alpha * dt_ns)
        msg = AgentFrameMsg(**{field: getattr(curr, field) for field in curr.get_fields_and_field_types() if field != "header"})
        msg.header.stamp.sec = int(pose_ns // int(1e9))
        msg.header.stamp.nanosec = int(pose_ns % int(1e9))
        msg.header.frame_id = "map"

        fields = ("x", "y", "vx", "vy", "gait_phase") if len(curr.gait_phase) and len(prev.gait_phase) else ("x", "y", "vx", "vy")
        for field in fields:
            values = np.array(getattr(curr, field))
            values[ci] = inv * np.asarray(getattr(prev, field))[pi] + alpha * values[ci]
            setattr(msg, field, array.array("d", values.tobytes()))
        theta = np.array(curr.theta)
        prev_theta = np.asarray(prev.theta)[pi]
        d_theta = np.arctan2(np.sin(theta[ci] - prev_theta), np.cos(theta[ci] - prev_theta))
        theta[ci] = prev_theta + alpha * d_theta
        msg.theta = array.array("d", theta.tobytes())
        return msg

    _ENGINE_DT_DEFAULT = 0.05

    async def _engine_dt(self) -> float:
        """Read the engine's tick length from its dt param."""
        client = self.node.create_client_wrapper(
            GetParameters,
            self.node.service_namespace("arena_humansim", "get_parameters"),
        )
        try:
            response = await client.call_timeout(GetParameters.Request(names=["dt"]))
        except Exception as e:
            self._logger.warning(f"engine dt read failed ({e}), assuming {self._ENGINE_DT_DEFAULT}")
            return self._ENGINE_DT_DEFAULT
        values = response.values if response is not None else []
        if values and values[0].type == ParameterType.PARAMETER_DOUBLE and values[0].double_value > 0.0:
            return values[0].double_value
        self._logger.warning(f"engine dt param unavailable, assuming {self._ENGINE_DT_DEFAULT}")
        return self._ENGINE_DT_DEFAULT

    async def configure_contact(self, mode: str, standing_distance: float) -> None:
        if self._contact_config == (mode, standing_distance):
            return
        request = SetParametersAtomically.Request(
            parameters=[
                Parameter("contact_mode", value=mode).to_parameter_msg(),
                Parameter("locomotion_standing_distance", value=float(standing_distance)).to_parameter_msg(),
            ],
        )
        response = await self._set_params_client.call_timeout(request)
        if response is None or not response.result.successful:
            reason = response.result.reason if response is not None else "timeout"
            raise RuntimeError(f"arena_humansim rejected contact_mode={mode!r} standing_distance={standing_distance}: {reason}")
        self._contact_config = (mode, standing_distance)
        self._logger.info(f"contact_mode={mode} standing_distance={standing_distance}")

    async def configure_gestures(self, mode: str) -> None:
        if self._gesture_config == mode:
            return
        request = SetParametersAtomically.Request(parameters=[Parameter("gesture_mode", value=mode).to_parameter_msg()])
        response = await self._set_params_client.call_timeout(request)
        if response is None or not response.result.successful:
            reason = response.result.reason if response is not None else "timeout"
            raise RuntimeError(f"arena_humansim rejected gesture_mode={mode!r}: {reason}")
        self._gesture_config = mode
        self._logger.info(f"gesture_mode={mode}")

    async def _configure_impl(self, params: Sequence[ParameterMsg]) -> set[str]:
        accepted: set[str] = set()
        if params:
            response = await self._set_params_client.call_timeout(SetParametersAtomically.Request(parameters=list(params)))
            if response is not None and response.result.successful:
                accepted = {p.name for p in params}
            elif response is not None:
                self._logger.error(f"engine rejected human params {[p.name for p in params]}: {response.result.reason}")
        await self._reset_client.call_timeout(ResetSimulation.Request(soft=True))
        return accepted

    async def setup(self):
        await asyncio.gather(
            *(
                client.ensure()
                for client in (
                    self._spawn_client,
                    self._remove_client,
                    self._set_flow_client,
                    self._add_source_client,
                    self._remove_source_client,
                    self._add_sink_client,
                    self._remove_sink_client,
                    self._add_walls_client,
                    self._remove_walls_client,
                    self._add_obstacles_client,
                    self._remove_obstacles_client,
                    self._add_world_objects_client,
                    self._remove_world_objects_client,
                    self._update_robot_client,
                    self._set_waypoints_client,
                    self._reset_client,
                    self._get_profile_client,
                    self._feedback_client,
                    self._notify_stimulus_client,
                    self._set_teleport_zones_client,
                    self._set_zone_access_client,
                    self._teleport_agents_client,
                    self._set_params_client,
                )
            )
        )
        self._logger.info("All arena_humansim services available")

        # gate period = engine dt: one sample per window paces the sim to the
        # engine's true coverage, so the clock cannot outrun the ped simulation
        engine_dt = await self._engine_dt()
        await self._register_lockstep_channels(
            [
                LockstepChannel(
                    name="engine",
                    topic=str(self.node.service_namespace("agent_states")),
                    type="arena_humansim_msgs/msg/AgentFrame",
                    period_s=engine_dt,
                    hard=True,
                ),
                # hard: a starved roster makes consumers dead-reckon and snap, a slow one should pace the sim down
                LockstepChannel(
                    name="peds",
                    topic=str(self._namespace("arena_peds")),
                    type="arena_people_msgs/msg/Pedestrians",
                    period_s=engine_dt,
                    hard=True,
                ),
            ]
        )

        await self.unpause()

    @classmethod
    async def create(cls, **kwargs: typing.Unpack[HumanSimulatorOptions]) -> "ArenaHumanSimulator":
        self = cls(**kwargs)
        await self.setup()
        return self

    async def pause(self):
        tasks = [t for t in (self._tick_loop_task, self._update_loop_task, self._feedback_loop_task) if t is not None]
        self._tick_loop_task = self._update_loop_task = self._feedback_loop_task = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def unpause(self):
        await self.pause()
        self._tick_loop_task = asyncio.create_task(self._interpolation_loop())
        self._update_loop_task = asyncio.create_task(self._pedestrian_update_loop())
        self._feedback_loop_task = asyncio.create_task(self._feedback_loop())

    @property
    def agent_states(self) -> AgentFrameMsg | None:
        return self._curr_agent_states

    TICK_RATE = 50.0  # Hz, local interpolation rate
    FLOW_META_TICKS = 50
    FLOW_MODEL = "default"
    FEEDBACK_RATE = 20.0  # Hz, matches the possession stream rate

    async def _interpolation_loop(self):
        """Interpolate agent states locally at TICK_RATE and publish pedestrians."""
        try:
            with self.node.sim_time_rate(self.TICK_RATE) as (done, rate):
                while not done.is_set():
                    await rate.get()
                    await self._publish_interpolated()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.error(f"Error in interpolation loop: {e}\n{traceback.format_exc()}")

    def _runtime_obstacle(
        self,
        name: str,
        pose: Pose,
        *,
        model: str | HumanIdentifier = FLOW_MODEL,
        velocity: float = 0.0,
    ) -> DynamicObstacle:
        """Build a runtime DynamicObstacle with env-prefixed sim_path."""
        obs = DynamicObstacle(
            name=name,
            pose=pose,
            model=model,
            waypoints=[],
            velocity=velocity,
        )
        obs.sim_path = self._realizer.prefix(name)
        return obs

    def _make_flow_dynamic_obstacle(self, frame: AgentFrameMsg, i: int, model: str | HumanIdentifier) -> DynamicObstacle:
        """Create a DynamicObstacle for the source-spawned agent at index i."""
        return self._runtime_obstacle(
            name=f"flow_{frame.agent_id[i]}",
            pose=Pose(Position(*self._from_engine(frame.x[i], frame.y[i]))),
            model=model,
            velocity=frame.desired_velocity[i],
        )

    async def _flow_model(self, agent_id: int) -> str | HumanIdentifier:
        """A human model on disk carrying every asset tag of the flow agent's type, FLOW_MODEL when the type lists none or nothing matches."""
        agent_type = self._agent_type_of(agent_id)
        definition = agent_type_def(agent_type) if agent_type else None
        if definition is None or not definition.assets:
            return self.FLOW_MODEL
        if definition.assets not in self._tagged_models:
            self._tagged_models[definition.assets] = await HumanIdentifier.tagged(definition.assets)
        matches = self._tagged_models[definition.assets]
        if not matches:
            if agent_type not in self._warned_untagged_types:
                self._warned_untagged_types.add(agent_type)
                self._logger.warning(f"agent type {agent_type!r} asks for a human model tagged {list(definition.assets)}, no bundle on disk carries them; using {self.FLOW_MODEL!r}")
            return self.FLOW_MODEL
        rng = self.node.conf.General.RNG.stream("humansim", "flow-model", agent_id)
        return matches[int(rng.integers(len(matches)))]

    # The engine runs in the world frame (authored coordinates, levels laid out), this adapter owns the env offset:
    # everything sent in is un-shifted, everything coming out is shifted back.
    def _to_engine(self, x: float, y: float) -> tuple[float, float]:
        cfg = self._realizer.get_config()
        return float(x - cfg.x), float(y - cfg.y)

    def _from_engine(self, x: float, y: float) -> tuple[float, float]:
        cfg = self._realizer.get_config()
        return float(x + cfg.x), float(y + cfg.y)

    def _engine_pose(self, pose: Pose) -> Pose2DMsg:
        x, y = self._to_engine(pose.position.x, pose.position.y)
        return Pose2DMsg(x=x, y=y, theta=pose.orientation.to_yaw())

    def _seat_pose(self, object_pose: Pose, seat: Pose) -> Pose2DMsg:
        """An object-local seat pose in the engine frame."""
        placed = object_pose * seat
        x, y = self._to_engine(placed.position.x, placed.position.y)
        return Pose2DMsg(x=x, y=y, theta=placed.orientation.to_yaw())

    def _seat_poses(self, obstacle: Obstacle, slots: Sequence[AffordanceSlot]) -> tuple[list[Pose2DMsg], list[float]]:
        """Seats and the heights of their surfaces above the floor, NaN where unknown (see `seat_height`).

        Scenario seats override the annotation's sit slots, including an empty list. A bad seat costs the seats, not the
        object; bad sit slots cost scenario seats only their heights. Heights add the object's own z (a raised bed) and
        assume it stands upright.
        """
        humansim = obstacle.humansim
        try:
            if "seats" in humansim:
                seats = [Pose.parse(seat) for seat in humansim["seats"]]
                heights = [seat_height(seat, slots) for seat in seats]
            else:
                rng = self.node.conf.General.RNG.stream("humansim", "seats", obstacle.sim_path)
                seats = [slot.pose(rng) for slot in slots]
                heights = [slot_height(slot) for slot in slots]
            floor = obstacle.pose.position.z
            return [self._seat_pose(obstacle.pose, seat) for seat in seats], [floor + height for height in heights]
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            self._logger.warning(f"Ignoring seats of '{obstacle.sim_path}': {e}")
            return [], []

    async def _pedestrian_update_loop(self):
        """Spawn an actor for each flow agent a source creates, delete it when a
        sink removes the agent, and push the latest pedestrian set to the simulator.
        """
        try:
            with self.node.sim_time_rate(self.TICK_RATE) as (done, rate):
                while not done.is_set():
                    await rate.get()
                    async with self._agents_lock:
                        peds = self._arena_pedestrians
                    if not peds.pedestrians:
                        continue

                    states = self._curr_agent_states
                    if states is not None and not self._spawns_in_flight:
                        current_ids = set(states.agent_id)
                        flow_ids = current_ids - self._bridge_agent_ids
                        new_ids = flow_ids - self._flow_agent_ids
                        gone_ids = self._flow_agent_ids - current_ids

                        if new_ids:
                            index_by_id = {aid: i for i, aid in enumerate(states.agent_id)}
                            to_spawn: list[DynamicObstacle] = []
                            for aid in sorted(new_ids):
                                waited = self._flow_meta_wait.get(aid, 0)
                                if aid not in self._agent_types and waited < self.FLOW_META_TICKS:
                                    self._flow_meta_wait[aid] = waited + 1
                                    continue
                                self._flow_meta_wait.pop(aid, None)
                                self._flow_agent_ids.add(aid)
                                obs = self._make_flow_dynamic_obstacle(states, index_by_id[aid], await self._flow_model(aid))
                                self._agent_names[aid] = obs.sim_path
                                to_spawn.append(obs)
                            if to_spawn:
                                await self._simulator.pedestrian_spawn(await self._ensure_spawnable(to_spawn))

                        for aid in self._flow_meta_wait.keys() - current_ids:
                            del self._flow_meta_wait[aid]

                        if gone_ids:
                            to_delete: list[DynamicObstacle] = []
                            for aid in sorted(gone_ids):
                                self._flow_agent_ids.discard(aid)
                                self._agent_names.pop(aid, None)
                                to_delete.append(self._runtime_obstacle(name=f"flow_{aid}", pose=Pose(Position(0.0, 0.0))))
                            await self._simulator.pedestrian_delete(to_delete)

                    await self.relay_pedestrian_update(peds)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.error(f"Error in pedestrian update loop: {e}\n{traceback.format_exc()}")

    async def _feedback_loop(self):
        """Publish dirty robot and possessed pedestrian poses on world_state topic, step delayed spawns and itineraries."""
        try:
            with self.node.sim_time_rate(self.FEEDBACK_RATE) as (done, rate):
                while not done.is_set():
                    await rate.get()
                    self._publish_world_state()
                    try:
                        await self._step_peds(self._now_s())
                    except Exception as e:
                        self._logger.error(f"ped mechanics step failed: {e!r}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.error(f"Error in feedback loop: {e}\n{traceback.format_exc()}")

    def _publish_world_state(self):
        """Publish robot and possessed pedestrian poses and velocities as AgentStates on world_state topic, each robot also as `robot_<i>` by spawn order, no robots when they are not visible."""
        robots = self.tracked_robots() if self._robots_visible else []
        possessed = self.possessed_peds()
        if not robots and not possessed:
            return
        msg = AgentStatesMsg()
        msg.header.stamp = self.node.sim_time.to_msg()
        msg.header.frame_id = "map"
        named = [(tracked.robot.name, tracked) for tracked in robots]
        own_names = {name for name, _ in named}
        named += [(alias, tracked) for i, tracked in enumerate(robots) if (alias := f"robot_{i}") not in own_names]
        for name, tracked in named:
            a = AgentStateMsg()
            a.agent_id = stable_int(tracked.robot.name) & 0x7FFFFFFF
            a.name = name
            a.pose = self._engine_pose(tracked.pose)
            a.velocity.x, a.velocity.y = tracked.velocity
            a.radius = 0.3
            msg.agents.append(a)  # pyright: ignore[reportAttributeAccessIssue]
        ids = self._agent_ids()
        for name, ped in possessed.items():
            a = AgentStateMsg()
            aid = self._agent_id(name, ids)
            a.agent_id = aid if aid is not None else stable_int(name) & 0x7FFFFFFF
            a.name = name
            a.pose = self._engine_pose(Pose.from_msg(ped.pose))
            a.velocity = ped.twist.linear
            a.radius = 0.35
            msg.agents.append(a)  # pyright: ignore[reportAttributeAccessIssue]
        self._world_state_pub.publish(msg)

    def _agent_states_to_pedestrians(self, msg: AgentFrameMsg) -> Pedestrians:
        peds = Pedestrians()
        peds.header = msg.header
        gestures_by_id = self._agent_gestures
        n = len(msg.agent_id)
        gait_phase = msg.gait_phase if len(msg.gait_phase) == n else [0.0] * n
        gait_cadence = msg.gait_cadence if len(msg.gait_cadence) == n else [0.0] * n
        interaction_ids = msg.interaction_id if len(msg.interaction_id) == n else [-1] * n
        interaction_types = msg.interaction_type if len(msg.interaction_type) == n else [0] * n
        support_heights = msg.support_height if len(msg.support_height) == n else [Pedestrian.NO_SUPPORT] * n
        for agent_id, ex, ey, yaw, vx, vy, animation_state, phase, cadence, interaction_id, interaction_type, support_height in zip(msg.agent_id, msg.x, msg.y, msg.theta, msg.vx, msg.vy, msg.animation_state, gait_phase, gait_cadence, interaction_ids, interaction_types, support_heights, strict=True):
            ped = Pedestrian()
            ped.id = agent_id
            ped.name = self._agent_names.get(agent_id, str(agent_id))
            ped.gait_phase = float(phase)
            ped.gait_cadence = float(cadence)
            ped.agent_type = self._agent_type_of(agent_id)

            x, y = self._from_engine(ex, ey)

            gestures = [GestureMsg(slot=g.slot, at=Point(x=gx, y=gy, z=g.at.z), clip=g.clip, hand=g.hand, render_pose_override=g.render_pose_override) for g in gestures_by_id.get(agent_id, ()) for gx, gy in (self._from_engine(g.at.x, g.at.y),)]

            # contact kinds (hug, handshake) draw on the formation slot, physics stays put
            override = next((g for g in gestures if g.slot == "body" and g.render_pose_override), None)
            if override is not None:
                x, y = override.at.x, override.at.y

            ped.pose = PoseMsg(
                position=Point(x=x, y=y, z=0.0),
                orientation=Quaternion(
                    z=math.sin(yaw / 2.0),
                    w=math.cos(yaw / 2.0),
                ),
            )
            ped.twist = Twist(linear=Vector3(x=vx, y=vy, z=0.0))

            ped.animation_state = animation_state
            ped.gestures = gestures
            ped.interaction_id = int(interaction_id)
            ped.interaction_type = int(interaction_type)
            ped.support_height = float(support_height) if support_height >= 0.0 else Pedestrian.NO_SUPPORT

            peds.pedestrians.append(ped)  # pyright: ignore[reportAttributeAccessIssue]
        return peds

    def _polygon_to_shape_msg(self, polygon: Sequence[Position], centroid: tuple[float, float] | None = None) -> ShapeMsg:
        """Convert a polygon (list of Position) to a Shape msg with POLYGON type.

        Vertices are stored relative to *centroid* (the pose sent alongside
        the shape), because the arena_humansim manager and visualisation
        interpret shape vertices as local offsets from the pose.
        """
        cx, cy = centroid if centroid is not None else self._polygon_centroid(polygon)
        shape = ShapeMsg()
        shape.type = ShapeMsg.POLYGON  # = 2
        shape.vertices = [Point32(x=float(p.x - cx), y=float(p.y - cy), z=0.0) for p in polygon]
        return shape

    def _polygon_centroid(self, polygon: Sequence[Position]) -> tuple[float, float]:
        """Compute centroid of a polygon."""
        if not polygon:
            return 0.0, 0.0
        cx = sum(p.x for p in polygon) / len(polygon)
        cy = sum(p.y for p in polygon) / len(polygon)
        return cx, cy

    async def _add_regions_impl(self, regions: Sequence[Region]) -> bool:
        results = await asyncio.gather(
            *(self._add_region_single(r) for r in regions),
        )
        return all(results)

    async def _add_region_single(self, region: Region) -> bool:
        if region.type == "source":
            return await self._add_source_region(region)
        elif region.type == "sink":
            return await self._add_sink_region(region)
        else:
            self._logger.error(f"Unknown region type: {region.type}")
            return False

    async def _add_source_region(self, region: Region) -> bool:
        request = AddSource.Request()
        src_msg = SourceConfigMsg()
        src_msg.name = region.name
        centroid = self._polygon_centroid(region.polygon)
        cx, cy = self._to_engine(*centroid)
        src_msg.pose = Pose2DMsg(x=cx, y=cy, theta=0.0)
        src_msg.shape = self._polygon_to_shape_msg(region.polygon, centroid)

        cfg = region.config
        for kf in cfg.get("rate_profile", []):
            src_msg.rate_profile.append(RateKeyframeMsg(t=kf.get("t", 0.0), rate=kf.get("rate", 0.0)))  # pyright: ignore[reportAttributeAccessIssue]
        src_msg.max_concurrent = cfg.get("max_concurrent", -1)
        src_msg.max_total = cfg.get("max_total", -1)

        tmpl = flow_agent_template(cfg.get("agent", {}), region.included_from)
        self._remember_agent_type_path(tmpl.agent_type)
        src_msg.agent = tmpl

        request.source = src_msg
        try:
            response = await self._add_source_client.call_timeout(request)
            if response is None:
                self._logger.error("AddSource timed out")
                return False
            if response.success:
                return True
            self._logger.error(f"AddSource failed: {response.message}")
            return False
        except Exception as e:
            self._logger.error(f"AddSource call failed: {e}")
            return False

    async def _add_sink_region(self, region: Region) -> bool:
        request = AddSink.Request()
        sink_msg = SinkConfigMsg()
        sink_msg.name = region.name
        centroid = self._polygon_centroid(region.polygon)
        cx, cy = self._to_engine(*centroid)
        sink_msg.pose = Pose2DMsg(x=cx, y=cy, theta=0.0)
        sink_msg.shape = self._polygon_to_shape_msg(region.polygon, centroid)

        cfg = region.config
        sink_msg.absorption_radius = cfg.get("absorption_radius", 0.5)
        sink_msg.capacity = cfg.get("capacity", -1)

        request.sink = sink_msg
        try:
            response = await self._add_sink_client.call_timeout(request)
            if response is None:
                self._logger.error("AddSink timed out")
                return False
            if response.success:
                return True
            self._logger.error(f"AddSink failed: {response.message}")
            return False
        except Exception as e:
            self._logger.error(f"AddSink call failed: {e}")
            return False

    async def _remove_regions_impl(self, regions: Sequence[Region]) -> bool:
        sources = [r.name for r in regions if r.type == "source"]
        sinks = [r.name for r in regions if r.type == "sink"]
        results = await asyncio.gather(
            *(self._remove_source_by_name(name) for name in sources),
            *(self._remove_sink_by_name(name) for name in sinks),
        )
        return all(results) if results else True

    async def _remove_source_by_name(self, name: str) -> bool:
        request = RemoveSource.Request()
        request.name = name
        try:
            response = await self._remove_source_client.call_timeout(request)
            if response is None:
                self._logger.error("RemoveSource timed out")
                return False
            if response.success:
                return True
            self._logger.error(f"RemoveSource failed: {response.message}")
            return False
        except Exception as e:
            self._logger.error(f"RemoveSource call failed: {e}")
            return False

    async def _remove_sink_by_name(self, name: str) -> bool:
        request = RemoveSink.Request()
        request.name = name
        try:
            response = await self._remove_sink_client.call_timeout(request)
            if response is None:
                self._logger.error("RemoveSink timed out")
                return False
            if response.success:
                return True
            self._logger.error(f"RemoveSink failed: {response.message}")
            return False
        except Exception as e:
            self._logger.error(f"RemoveSink call failed: {e}")
            return False

    async def _spawn_obstacles_impl(self, obstacles: Sequence[Obstacle]) -> Sequence[Obstacle | None]:
        """Send obstacle configs (pose + bounding box + metadata) to arena_humansim.

        Each obstacle is also registered as a WorldObject with optional overrides
        (capacity / satisfies / interaction_radius / formation / type) read from
        ``obstacle.humansim``.
        """
        if not obstacles:
            return obstacles

        # Resolve all asset paths concurrently so the NetResolver can batch them
        resolved_paths = await asyncio.gather(
            *(obstacle.model.resolve_path() for obstacle in obstacles),
            return_exceptions=True,
        )

        obstacles_request = AddObstacles.Request()
        world_objects_request = AddWorldObjects.Request()
        for obstacle, resolved in zip(obstacles, resolved_paths, strict=False):
            try:
                pose = self._engine_pose(obstacle.pose)

                obstacle_type = ""
                raw_slots: list = []
                bounding_box = None
                try:
                    if isinstance(resolved, BaseException):
                        raise resolved
                    with open(resolved / "annotation.yaml") as f:
                        annotation: dict = yaml.safe_load(f.read())
                    raw_slots = list((annotation.get("affordances") or {}).get("sit") or [])
                    bounding_box = annotation["bounding_box"]
                    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounding_box
                    obstacle_type = annotation.get("name", "") or annotation.get("desc", "")
                    msg = ObstacleConfigMsg()
                    msg.name = obstacle.sim_path
                    msg.pose = pose
                    msg.bb_x_min = float(x_min)
                    msg.bb_x_max = float(x_max)
                    msg.bb_y_min = float(y_min)
                    msg.bb_y_max = float(y_max)
                    msg.bb_z_min = float(z_min)
                    msg.bb_z_max = float(z_max)
                    msg.interaction_types = [str(h) for h in annotation.get("hoi", [])]
                    msg.obstacle_type = obstacle_type
                    obstacles_request.obstacles.append(msg)  # pyright: ignore[reportAttributeAccessIssue]
                except Exception as e:
                    self._logger.warning(f"Failed to build obstacle bbox for '{obstacle.sim_path}': {e}")

                try:
                    slots = placed_slots({AffordanceKind.SIT: raw_slots})
                except (TypeError, ValueError) as e:
                    self._logger.warning(f"Ignoring the sit slots of '{obstacle.sim_path}': {e}")
                    slots = []
                humansim = obstacle.humansim
                info = WorldObjectInfoMsg()
                info.object_id = obstacle.sim_path
                info.type = str(humansim.get("type", obstacle_type))
                info.pose = pose
                info.capacity = seat_capacity(humansim, slots)
                satisfies: dict = humansim.get("satisfies") or {}
                info.satisfies_keys = list(satisfies.keys())
                info.satisfies_values = [float(v) for v in satisfies.values()]
                info.seats, info.seat_heights = self._seat_poses(obstacle, slots)
                if "seats" not in humansim and slots and len(info.seats) == len(slots):
                    info.seat_conflict_offset, info.seat_conflicts = conflict_offsets(slots)
                    self._seat_slots[obstacle.sim_path] = slots
                else:
                    self._seat_slots.pop(obstacle.sim_path, None)
                interaction_radius = humansim.get("interaction_radius")
                if interaction_radius is None and info.seats and bounding_box is not None:
                    interaction_radius = reach_radius(bounding_box)
                info.interaction_radius = float(interaction_radius) if interaction_radius is not None else 0.0
                formation: dict = humansim.get("formation") or {}
                formation_type = formation.get("type", "")
                if formation_type:
                    formation_params: dict = formation.get("params") or {}
                    info.formation_type = str(formation_type)
                    info.formation_param_keys = list(formation_params.keys())
                    info.formation_param_values = [float(v) for v in formation_params.values()]
                world_objects_request.objects.append(info)  # pyright: ignore[reportAttributeAccessIssue]
            except Exception as e:
                self._logger.warning(f"Failed to build world object for '{obstacle.sim_path}': {e}")

        if obstacles_request.obstacles:
            try:
                response = await self._add_obstacles_client.call_timeout(obstacles_request)
                if response is None:
                    self._logger.error("AddObstacles timed out")
                elif response.success:
                    self._logger.info(response.message)
                else:
                    self._logger.error(f"AddObstacles failed: {response.message}")
            except Exception as e:
                self._logger.error(f"AddObstacles call failed: {e}")

        if world_objects_request.objects:
            try:
                response = await self._add_world_objects_client.call_timeout(world_objects_request)
                if response is None:
                    self._logger.error("AddWorldObjects timed out")
                elif not response.success:
                    self._logger.error(f"AddWorldObjects failed: {response.message}")
            except Exception as e:
                self._logger.error(f"AddWorldObjects call failed: {e}")

        return obstacles

    async def _spawn_dynamic_obstacles_impl(self, obstacles: Sequence[DynamicObstacle]) -> Sequence[DynamicObstacle | None]:
        """Forward dynamic obstacles to arena_humansim AgentManager via SpawnAgents."""
        if not obstacles:
            return obstacles

        request = SpawnAgents.Request()
        immediate: list[tuple[DynamicObstacle, ItineraryDriver | None, dict[str, str]]] = []
        now = self._now_s()
        for obstacle in obstacles:
            try:
                mechanics = PedMechanics.from_dynamic_obstacle(obstacle)
            except ValueError as e:
                self._logger.error(f"ignoring spawn_delay/semantic_alias/itinerary: {e}")
                mechanics = PedMechanics()
            agent_msg = AgentStateMsg()
            if mechanics.spawn_delay > 0.0:
                agent_msg.agent_id = self._next_deferred_id
                self._next_deferred_id += 1
                self._bridge_agent_ids.add(agent_msg.agent_id)
                self._agent_names[agent_msg.agent_id] = obstacle.sim_path
            agent_msg.name = obstacle.sim_path

            agent_msg.pose = self._engine_pose(obstacle.pose)
            agent_msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)

            parsed = ArenaHumanDynamicObstacle.from_dynamic_obstacle(obstacle)
            params = parsed.sample_params(self.node.conf.General.RNG.stream("humansim", obstacle.sim_path)) if parsed is not None else None

            if parsed is not None and params is not None:
                agent_msg.desired_velocity = params.desired_velocity
                agent_msg.radius = params.agent_radius
                agent_msg.vision_range = params.perception.vision_range
                agent_msg.vision_fov = params.perception.vision_fov
                lp = params.local_planner_params
                agent_msg.relaxation_time = lp.get("relaxation_time", 0.0)
                agent_msg.repulsion_strength = lp.get("repulsion_strength", 0.0)
                agent_msg.repulsion_range = lp.get("repulsion_range", 0.0)
                agent_msg.agent_type = parsed.agent_type
                agent_msg.handedness = params.handedness
                self._remember_agent_type_path(parsed.agent_type)
            else:
                agent_msg.desired_velocity = float(obstacle.velocity)
                agent_msg.radius = 0.35
                agent_msg.vision_range = 0.0
                agent_msg.vision_fov = 0.0
                agent_msg.relaxation_time = 0.0
                agent_msg.repulsion_strength = 0.0
                agent_msg.repulsion_range = 0.0
                agent_msg.agent_type = ""
                agent_msg.handedness = ""

            wp_msg = WaypointsMsg()
            wp_msg.mode = parsed.waypoint_mode if parsed is not None else WaypointsMsg.MODE_REPEAT
            points = [self._to_engine(wp.x, wp.y) for wp in obstacle.waypoints]
            stops = mechanics.stops(points)
            if mechanics.itinerary and stops is None and len(mechanics.itinerary) != len(points):
                self._logger.warning(f"{obstacle.name}: itinerary does not match its {len(points)} waypoints, ignoring dwell/rides")
            driver: ItineraryDriver | None = None
            if stops is not None:
                mode = obstacle.waypoint_mode
                driver = ItineraryDriver(stops, mode)
                points = driver.start()
                wp_msg.mode = WaypointsMsg.MODE_ONCE
            for wx, wy in points:
                wp_msg.points.append(WaypointMsg(pose=Pose2DMsg(x=wx, y=wy, theta=0.0)))  # pyright: ignore[reportAttributeAccessIssue]
            agent_msg.waypoints = wp_msg
            if mechanics.spawn_delay > 0.0:
                if driver is not None:
                    self._itineraries[agent_msg.agent_id] = driver
                self._deferred_spawns.append((now + mechanics.spawn_delay, agent_msg, mechanics.semantic_alias))
                continue
            immediate.append((obstacle, driver, mechanics.semantic_alias))
            request.agents.append(agent_msg)  # pyright: ignore[reportAttributeAccessIssue]

        if not request.agents:
            return obstacles
        self._spawns_in_flight += 1
        try:
            response = await self._spawn_client.call_timeout(request)
            if response is None:
                self._logger.error("SpawnAgents timed out")
                return [None] * len(obstacles)
            if response.success:
                aliases: dict[int, dict[str, str]] = {}
                for agent_id, (obstacle, driver, alias) in zip(response.spawned_ids, immediate, strict=True):
                    self._bridge_agent_ids.add(agent_id)
                    self._agent_names[agent_id] = obstacle.sim_path
                    if driver is not None:
                        self._itineraries[agent_id] = driver
                    if alias:
                        aliases[agent_id] = alias
                self._logger.info(f"Spawned {len(response.spawned_ids)} agents")
                await self._bind_aliases(aliases)
                return obstacles
            else:
                self._logger.error(f"SpawnAgents failed: {response.message}")
                return [None] * len(obstacles)
        except Exception as e:
            self._logger.error(f"SpawnAgents call failed: {e}")
            return [None] * len(obstacles)
        finally:
            self._spawns_in_flight -= 1

    def _agent_ids(self) -> dict[str, int]:
        """Engine id of every spawned agent by ``sim_path``."""
        return {name: aid for aid, name in self._agent_names.items()}

    def _agent_id(self, agent: str, ids: Mapping[str, int] | None = None) -> int | None:
        """Engine id of a spawned agent by name or ``sim_path``, looked up in `ids` (an `_agent_ids()` result) when given."""
        known = self._known_obstacles.get(agent)
        sim_path = known.obstacle.sim_path if known is not None else agent
        return (self._agent_ids() if ids is None else ids).get(sim_path)

    async def notify_stimulus(self, agent: int | str, stimulus: str, intensity: float) -> bool:
        if isinstance(agent, str):
            agent_id = self._agent_id(agent)
            if agent_id is None:
                self._logger.warning(f"NotifyStimulus: {agent!r} is not a known agent")
                return False
            agent = agent_id
        request = NotifyStimulus.Request()
        request.agent_id = agent
        request.stimulus = stimulus
        request.intensity = intensity
        try:
            response = await self._notify_stimulus_client.call_timeout(request)
        except Exception as e:
            self._logger.error(f"NotifyStimulus call failed: {e}")
            return False
        if response is None:
            self._logger.error("NotifyStimulus timed out")
            return False
        if not response.success:
            self._logger.error(f"NotifyStimulus failed: {response.message}")
        return bool(response.success)

    def stimuli_taken(self, obstacle: DynamicObstacle) -> tuple[str, frozenset[str]] | None:
        parsed = ArenaHumanDynamicObstacle.from_dynamic_obstacle(obstacle)
        definition = agent_type_def(parsed.agent_type) if parsed is not None else None
        if parsed is None or definition is None:
            return None
        return parsed.agent_type, agent_type_stimuli(definition)

    async def pedestrian_teleport(self, destinations: Mapping[str, tuple[float, float]]) -> bool:
        """Teleport pedestrians by published name to (x, y) in the engine, keeping their facing."""
        if not destinations:
            return True
        agent_ids = {name: agent_id for agent_id, name in self._agent_names.items()}
        request = TeleportAgents.Request()
        unknown: list[str] = []
        for name, (x, y) in destinations.items():
            agent_id = agent_ids.get(name, int(name) if name.isdigit() else None)
            if agent_id is None:
                unknown.append(name)
                continue
            orientation = self._ped_orientations.get(name)
            ex, ey = self._to_engine(x, y)
            request.agent_id.append(agent_id)  # pyright: ignore[reportAttributeAccessIssue]
            request.pose.append(Pose2DMsg(x=ex, y=ey, theta=Orientation.from_msg(orientation).to_yaw() if orientation is not None else 0.0))  # pyright: ignore[reportAttributeAccessIssue]
        if unknown:
            self._logger.error(f"TeleportAgents: no engine agent for {unknown}")
        if not request.agent_id:
            return False
        try:
            response = await self._teleport_agents_client.call_timeout(request)
        except Exception as e:
            self._logger.error(f"TeleportAgents call failed: {e}")
            return False
        if response is None:
            self._logger.error("TeleportAgents timed out")
            return False
        if not response.success:
            self._logger.error(f"TeleportAgents failed: {response.message}")
            return False
        now = self.node.sim_time
        stamp = now.sec * int(1e9) + now.nanosec
        for agent_id in request.agent_id:
            self._teleport_stamps[agent_id] = stamp
        for agent_id in request.agent_id:
            driver = self._itineraries.get(agent_id)
            if driver is not None:
                step = driver.on_teleport()
                if step.waypoints is not None:
                    await self._send_leg(agent_id, step.waypoints)
            await self.notify_stimulus(agent_id, STIMULUS_TELEPORTED, 1.0)
        return not unknown

    def _zone_msg(self, elevator: Elevator) -> TeleportZoneMsg:
        """A realized cabin as a teleport zone in the engine frame, riders waiting clear of the door threshold and holding inside facing the door, clear of the band where a body blocks the closing door, teleported by Arena only."""
        doorway = elevator_doorway_pose(elevator, standoff=PED_RADIUS / 2.0)
        queue = elevator_doorway_pose(elevator, standoff=CABIN_DOOR_THRESHOLD + PED_RADIUS)
        facing_in = doorway.orientation.to_yaw()
        cx, cy = self._to_engine(elevator.position.x, elevator.position.y)
        dx, dy = self._to_engine(doorway.position.x, doorway.position.y)
        qx, qy = self._to_engine(queue.position.x, queue.position.y)
        reach = math.hypot(doorway.position.x - elevator.position.x, doorway.position.y - elevator.position.y)
        ox, oy = (doorway.position.x - elevator.position.x) / reach, (doorway.position.y - elevator.position.y) / reach
        band = CABIN_DOOR_INSET + CABIN_DOOR_WIDTH / 2.0 + PED_RADIUS
        footprint = []
        for corner in elevator.cabin_corners():
            inset = band if (corner.x - elevator.position.x) * ox + (corner.y - elevator.position.y) * oy > 0.0 else 0.0
            x, y = self._to_engine(corner.x - ox * inset, corner.y - oy * inset)
            footprint.append(Point32(x=x, y=y, z=0.0))
        return TeleportZoneMsg(
            name=elevator.name,
            destinations=list(elevator.destinations),
            entry=Pose2DMsg(x=dx, y=dy, theta=facing_in),
            wait=Pose2DMsg(x=qx, y=qy, theta=facing_in),
            hold=Pose2DMsg(x=cx, y=cy, theta=math.remainder(facing_in + math.pi, math.tau)),
            footprint=footprint,
            open=False,
            delay=-1.0,
        )

    async def _set_elevators_impl(self, elevators: Sequence[Elevator]) -> bool:
        request = SetTeleportZones.Request(zones=[self._zone_msg(elevator) for elevator in elevators])
        try:
            response = await self._set_teleport_zones_client.call_timeout(request)
        except Exception as e:
            self._logger.error(f"SetTeleportZones call failed: {e}")
            return False
        if response is None:
            self._logger.error("SetTeleportZones timed out")
            return False
        if not response.success:
            self._logger.error(f"SetTeleportZones failed: {response.message}")
        return response.success

    async def _set_elevator_doors_impl(self, doors: Mapping[str, bool]) -> bool:
        request = SetZoneAccess.Request(name=list(doors), open=list(doors.values()))
        try:
            response = await self._set_zone_access_client.call_timeout(request)
        except Exception as e:
            self._logger.error(f"SetZoneAccess call failed: {e}")
            return False
        if response is None:
            self._logger.error("SetZoneAccess timed out")
            return False
        if not response.success:
            self._logger.error(f"SetZoneAccess failed: {response.message}")
        return response.success

    async def _remove_obstacles_impl(self, obstacles: Sequence[Obstacle]) -> bool:
        """Remove static obstacles and their world objects from arena_humansim, which keys both by sim_path."""
        if not obstacles:
            return True
        names_list = [obstacle.sim_path for obstacle in obstacles]
        ok = True
        try:
            request = RemoveObstacles.Request()
            request.names = names_list
            response = await self._remove_obstacles_client.call_timeout(request)
            if response is None:
                self._logger.error("RemoveObstacles timed out")
                ok = False
            elif not response.success:
                self._logger.error(f"RemoveObstacles failed: {response.message}")
                ok = False
        except Exception as e:
            self._logger.warning(f"RemoveObstacles call failed: {e}")
            ok = False

        try:
            wo_request = RemoveWorldObjects.Request()
            wo_request.object_ids = names_list
            wo_response = await self._remove_world_objects_client.call_timeout(wo_request)
            if wo_response is None:
                self._logger.error("RemoveWorldObjects timed out")
                ok = False
            elif not wo_response.success:
                self._logger.error(f"RemoveWorldObjects failed: {wo_response.message}")
                ok = False
        except Exception as e:
            self._logger.warning(f"RemoveWorldObjects call failed: {e}")
            ok = False

        return ok

    async def _remove_pedestrians_impl(self) -> bool:
        """Remove all dynamic agents from arena_humansim."""
        request = RemoveAgents.Request()
        request.agent_ids = []  # empty = remove all

        try:
            response = await self._remove_client.call_timeout(request)
            self._next_deferred_id = _DEFERRED_ID_BASE
            self._deferred_spawns.clear()
            self._itineraries.clear()
            self._semantic_aliases.clear()
            self._bridge_agent_ids.clear()
            self._flow_agent_ids.clear()
            self._flow_meta_wait.clear()
            self._tagged_models.clear()
            self._agent_names.clear()
            self._teleport_stamps.clear()
            self._ped_model_uris.clear()
            self._prev_agent_states = None
            self._curr_agent_states = None
            self._agent_gestures = {}
            self._agent_types = {}
            self._seat_violation_reported = False
            self._apply_held_seats({})
            self._arena_pedestrians = Pedestrians()
            self._arena_pedestrians.header.frame_id = "map"
            await self._publish_interpolated()
            if response is None:
                self._logger.error("RemoveAgents timed out")
                return False
            if response.success:
                self._logger.info(response.message)
                return True
            else:
                self._logger.error(f"RemoveAgents failed: {response.message}")
                return False
        except Exception as e:
            self._logger.error(f"RemoveAgents call failed: {e}")
            return False

    async def _spawn_walls_impl(self, walls: Mapping[str, Wall]) -> bool:
        return await self._add_walls(walls)

    async def _spawn_doors_impl(self, doors: Mapping[str, Door]) -> bool:
        objects = [self._door_object(door) for door in doors.values()]
        await self._remove_world_objects([obj.object_id for obj in objects])
        for door, obj in zip(doors.values(), objects, strict=True):
            self._mech_objects[f"door_object:{door.name}"] = [obj.object_id]
        return await self._add_world_objects(objects)

    async def _remove_walls_impl(self, names: Sequence[str]) -> bool:
        return await self._remove_walls(names)

    async def _remove_doors_impl(self, names: Sequence[str]) -> bool:
        ids = [oid for name in names for oid in self._mech_objects.pop(f"door_object:{name}", [])]
        return await self._remove_world_objects(ids)

    async def _add_walls(self, walls: Mapping[str, Wall]) -> bool:
        """Send wall/door segments to arena_humansim via AddWalls."""
        if not walls:
            return True

        request = AddWalls.Request()
        for name, wall in walls.items():
            request.names.append(name)  # pyright: ignore[reportAttributeAccessIssue]
            sx, sy = self._to_engine(wall.start.x, wall.start.y)
            ex, ey = self._to_engine(wall.end.x, wall.end.y)
            request.starts.append(Point32(x=sx, y=sy, z=0.0))  # pyright: ignore[reportAttributeAccessIssue]
            request.ends.append(Point32(x=ex, y=ey, z=0.0))  # pyright: ignore[reportAttributeAccessIssue]

        try:
            response = await self._add_walls_client.call_timeout(request)
            if response is None:
                self._logger.error("AddWalls timed out")
                return False
            if response.success:
                return True
            self._logger.error(f"AddWalls failed: {response.message}")
            return False
        except Exception as e:
            self._logger.error(f"AddWalls call failed: {e}")
            return False

    async def _remove_walls(self, names: Sequence[str]) -> bool:
        """Remove wall/door segments from arena_humansim via RemoveWalls."""
        if not names:
            return True

        request = RemoveWalls.Request()
        request.names = list(names)

        try:
            response = await self._remove_walls_client.call_timeout(request)
            if response is None:
                self._logger.error("RemoveWalls timed out")
                return False
            if response.success:
                return True
            self._logger.error(f"RemoveWalls failed: {response.message}")
            return False
        except Exception as e:
            self._logger.error(f"RemoveWalls call failed: {e}")
            return False

    async def _spawn_robot_impl(self, robots: Sequence[Robot]) -> Sequence[bool]:
        """Register robot poses: published to arena_humansim via world_state topic."""
        self._publish_world_state()
        return (True,) * len(robots)

    async def _remove_robot_impl(self, robots: Sequence[Robot]) -> Sequence[bool]:
        return (True,) * len(robots)

    async def _move_robot_impl(self, robots: Sequence[Robot]) -> Sequence[bool]:
        return (True,) * len(robots)

    # world mechanisms

    async def _step_peds(self, now: float) -> None:
        """Release delayed scenario peds that are due and advance itineraries on the latest engine states."""
        due = [entry for entry in self._deferred_spawns if entry[0] <= now]
        if due:
            self._deferred_spawns = [entry for entry in self._deferred_spawns if entry[0] > now]
            request = SpawnAgents.Request()
            request.agents = [msg for _, msg, _ in due]
            response = await self._spawn_client.call_timeout(request)
            if response is not None and response.success:
                await self._bind_aliases({msg.agent_id: alias for _, msg, alias in due if alias})
            else:
                self._logger.error(f"delayed SpawnAgents failed: {response.message if response is not None else 'no response'}")
        states = self._curr_agent_states
        if not self._itineraries or states is None:
            return
        for aid, x, y in zip(states.agent_id, states.x, states.y, strict=True):
            driver = self._itineraries.get(aid)
            if driver is None:
                continue
            step = driver.update((x, y), now)
            if step.boarded is not None and step.boarded.target:
                self._write_elevator_target(step.boarded)
            if step.waypoints is not None:
                await self._send_leg(aid, step.waypoints)

    def _write_elevator_target(self, stop: Stop) -> None:
        """Press the button: a ped boarding a multi-destination elevator selects its destination."""
        if stop.elevator:
            self.node._set_semantic(stop.elevator, "target", stop.target)

    async def _send_leg(self, aid: int, points: Sequence[tuple[float, float]]) -> None:
        request = SetWaypoints.Request()
        request.agent_id = aid
        request.waypoints = WaypointsMsg(mode=WaypointsMsg.MODE_ONCE, points=[WaypointMsg(pose=Pose2DMsg(x=float(x), y=float(y), theta=0.0)) for x, y in points])
        try:
            response = await self._set_waypoints_client.call_timeout(request)
        except Exception as e:
            self._logger.error(f"SetWaypoints call failed: {e}")
            return
        if response is not None and not response.success:
            self._logger.error(f"SetWaypoints failed: {response.message}")

    async def _bind_aliases(self, aliases: Mapping[int, Mapping[str, str]]) -> None:
        """Register per-ped semantic aliases and replay the current value of each aliased entity's fields."""
        for aid, alias in aliases.items():
            self._semantic_aliases[aid] = dict(alias)
            for name, entity in alias.items():
                for (known, field), value in list(self._semantic_values.items()):
                    if known == entity:
                        await self.notify_stimulus(aid, semantic_atom(name, field, value), 1.0)

    async def _apply_semantic_change(self, entity: str, field: str, previous: str | None, value: str) -> None:
        await super()._apply_semantic_change(entity, field, previous, value)
        for aid, aliases in list(self._semantic_aliases.items()):
            for name, target in aliases.items():
                if target != entity:
                    continue
                if previous is not None and previous != value:
                    await self.notify_stimulus(aid, semantic_atom(name, field, previous), 0.0)
                await self.notify_stimulus(aid, semantic_atom(name, field, value), 1.0)

    def set_mechanism_walls(self, key: str, segments: Sequence[Segment] | None) -> None:
        """Named engine wall set (`__mech/<key>/<i>`), applied in the background, latest call wins."""
        self._mech_walls_desired[key] = None if segments is None else [((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for a, b in segments]
        if not self._mech_walls_syncing:
            self._mech_walls_syncing = True
            self._in_background(self._sync_mechanism_walls())

    async def _sync_mechanism_walls(self) -> None:
        try:
            while self._mech_walls_desired:
                key, segments = next(iter(self._mech_walls_desired.items()))
                del self._mech_walls_desired[key]
                await self._apply_mechanism_walls(key, segments)
        except Exception as e:
            self._logger.error(f"mechanism walls sync failed: {e!r}")
        finally:
            self._mech_walls_syncing = False

    async def _apply_mechanism_walls(self, key: str, segments: Sequence[Segment] | None) -> None:
        old = self._mech_walls_applied.pop(key, [])
        if old:
            await self._remove_walls(old)
        if key.startswith("elevator:"):
            await self._remove_world_objects(self._mech_objects.pop(key, []))
        if not segments:
            return
        walls = {f"__mech/{key}/{i}": Wall(start=Position(a[0], a[1]), end=Position(b[0], b[1])) for i, (a, b) in enumerate(segments)}
        if await self._add_walls(walls):
            self._mech_walls_applied[key] = list(walls)
        if key.startswith("elevator:") and len(segments) == 3:
            objects = self._elevator_objects(key.removeprefix("elevator:"), segments)
            self._mech_objects[key] = [obj.object_id for obj in objects]
            await self._add_world_objects(objects)

    async def _clear_mechanism_walls(self) -> None:
        self._mech_walls_desired.clear()
        names = [name for names in self._mech_walls_applied.values() for name in names]
        self._mech_walls_applied.clear()
        ids = [oid for key, oids in self._mech_objects.items() if key.startswith("elevator:") for oid in oids]
        self._mech_objects = {k: v for k, v in self._mech_objects.items() if not k.startswith("elevator:")}
        await asyncio.gather(self._remove_walls(names), self._remove_world_objects(ids))

    def _elevator_objects(self, realized: str, segments: Sequence[Segment]) -> list[WorldObjectInfoMsg]:
        """`elevator` object at the cabin centre and `elevator_queue` object in the doorway, from the [back, side, side] cabin walls."""
        back, side_a, side_b = segments
        bx, by = self._to_engine((back[0][0] + back[1][0]) / 2.0, (back[0][1] + back[1][1]) / 2.0)
        ax, ay = self._to_engine((side_a[0][0] + side_a[1][0]) / 2.0, (side_a[0][1] + side_a[1][1]) / 2.0)
        sx, sy = self._to_engine((side_b[0][0] + side_b[1][0]) / 2.0, (side_b[0][1] + side_b[1][1]) / 2.0)
        cx, cy = (ax + sx) / 2.0, (ay + sy) / 2.0
        outward = math.atan2(cy - by, cx - bx)
        half_depth = math.hypot(cx - bx, cy - by)
        name = self.authored_name(realized)

        cabin = WorldObjectInfoMsg()
        cabin.object_id = name
        cabin.type = "elevator"
        cabin.pose = Pose2DMsg(x=cx, y=cy, theta=outward)
        cabin.capacity = 0

        queue = WorldObjectInfoMsg()
        queue.object_id = f"{name}/queue"
        queue.type = "elevator_queue"
        queue.pose = Pose2DMsg(x=cx + half_depth * math.cos(outward), y=cy + half_depth * math.sin(outward), theta=outward + math.pi)
        queue.capacity = 0
        queue.formation_type = "line"
        queue.formation_param_keys = ["front_offset", "base_step"]
        queue.formation_param_values = [_QUEUE_FRONT_OFFSET, _QUEUE_STEP]
        return [cabin, queue]

    def _door_object(self, door: Door) -> WorldObjectInfoMsg:
        sx, sy = self._to_engine(door.start.x, door.start.y)
        ex, ey = self._to_engine(door.end.x, door.end.y)
        info = WorldObjectInfoMsg()
        info.object_id = self.authored_name(door.name)
        info.type = "door"
        info.pose = Pose2DMsg(x=(sx + ex) / 2.0, y=(sy + ey) / 2.0, theta=math.atan2(ey - sy, ex - sx) + math.pi / 2.0)
        info.capacity = 0
        return info

    async def _add_world_objects(self, objects: Sequence[WorldObjectInfoMsg]) -> bool:
        if not objects:
            return True
        request = AddWorldObjects.Request()
        request.objects = list(objects)
        try:
            response = await self._add_world_objects_client.call_timeout(request)
        except Exception as e:
            self._logger.error(f"AddWorldObjects call failed: {e}")
            return False
        if response is None or not response.success:
            self._logger.error(f"AddWorldObjects failed: {response.message if response is not None else 'no response'}")
            return False
        return True

    async def _remove_world_objects(self, ids: Sequence[str]) -> bool:
        if not ids:
            return True
        request = RemoveWorldObjects.Request()
        request.object_ids = list(ids)
        try:
            response = await self._remove_world_objects_client.call_timeout(request)
        except Exception as e:
            self._logger.warning(f"RemoveWorldObjects call failed: {e}")
            return False
        return response is not None and response.success
