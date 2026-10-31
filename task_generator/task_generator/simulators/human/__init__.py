from __future__ import annotations

import abc
import asyncio
import collections
import copy
import itertools
import math
import typing
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import attrs
import numpy as np
import rclpy.node
import rclpy.publisher
import rclpy.qos
import rclpy.time
import tf2_ros
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from arena_people_msgs.msg import AnimationSlot, AnimationState, AnimationStates, Pedestrian, Pedestrians
from arena_people_msgs.srv import MovePedestrians
from arena_rclpy_mixins.lazy import LazyPublisher
from arena_rclpy_mixins.registry import AsyncFactoryRegistry as Registry
from arena_rclpy_mixins.shared import Namespace
from arena_runtime._node import NodeInterface
from arena_runtime.lockstep import register_channels
from arena_runtime.sim import BaseSim
from arena_runtime_msgs.msg import LockstepChannel
from arena_simulation_setup.shared import Placement, Stairs
from arena_simulation_setup.tree.assets.Human import HumanIdentifier
from arena_simulation_setup.utils.models import ModelType
from geometry_msgs.msg import Point as PointMsg
from geometry_msgs.msg import Pose as PoseMsg
from geometry_msgs.msg import Quaternion as QuaternionMsg
from rcl_interfaces.msg import Parameter as ParameterMsg
from visualization_msgs.msg import MarkerArray

from task_generator.constants import Constants
from task_generator.manager.realizer import Realizer
from task_generator.shared import Door, DynamicObstacle, Elevator, Obstacle, Orientation, Pose, Position, Region, Robot, Wall
from task_generator.simulators.human.animation_manager import AnimationManager
from task_generator.simulators.human.drivers import DrivenJoints, Roller, load_drivers, with_driven
from task_generator.simulators.human.gestures import Channel, GestureLayer, GestureRequest
from task_generator.simulators.human.itinerary import semantic_atom
from task_generator.simulators.human.possession import PossessionTable
from task_generator.simulators.human.profile import PoseProfile, agent_type_pose_section, default_profile, resolve_pose_profile
from task_generator.simulators.human.support import LIE_S, SETTLE_S, SIT_CLIP, SQUAT_CLIP, TYPICAL_BODY, BodyMeasurements, SeatPlan, Support, blend_pose, load_body, plan_seat, rise_seconds, seat_depth, supported_pose, tilted_pose
from task_generator.simulators.human.utils import (
    KnownObstacle,
    KnownObstacles,
    ObstacleLayer,
)

if typing.TYPE_CHECKING:
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    from task_generator.simulators.acoustics import BaseAcousticsSimulator, PedestrianHearing

PED_RADIUS = 0.3

Segment = tuple[tuple[float, float], tuple[float, float]]

_STREAM_QOS = rclpy.qos.QoSProfile(
    reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
    durability=rclpy.qos.DurabilityPolicy.VOLATILE,
    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
    depth=1,
)


def placed_peds(msg: Pedestrians, placement: Placement) -> Pedestrians:
    """The pedestrians of `msg` in the simulator frame, `msg` itself while the levels are not stacked."""
    if not placement.stacked:
        return msg
    placed = copy.deepcopy(msg)
    for ped in placed.pedestrians:
        position = placement.place(Position(ped.pose.position.x, ped.pose.position.y, ped.pose.position.z))
        ped.pose.position.x, ped.pose.position.y, ped.pose.position.z = position.x, position.y, position.z
    return placed


def stair_height(stairs: Iterable[Stairs], x: float, y: float) -> float:
    """Walking surface height at a map point on the first stair half holding it, 0 off every stair."""
    return next((stair.height(stair.along(x, y)) for stair in stairs if stair.contains(x, y, stair.reach)), 0.0)


def stair_running(stair: Stairs, stairs: Iterable[Stairs], running: Mapping[str, bool]) -> bool:
    """Whether a stair half's belt moves: not once it, the half it boards or a half boarding it is reported stopped."""
    linked = {stair.name, stair.to} | {other.name for other in stairs if other.to == stair.name}
    return all(running.get(name, True) for name in linked if name)


@attrs.define
class TrackedRobot:
    """A robot's map-frame pose and velocity, followed through its base frame in TF."""

    robot: Robot
    tf_frame: str | None
    pose: Pose
    velocity: tuple[float, float] = (0.0, 0.0)
    stamp: float | None = None

    def place(self, pose: Pose) -> None:
        """Teleport to pose, at rest until TF reports the next one."""
        self.pose = copy.deepcopy(pose)
        self.velocity = (0.0, 0.0)
        self.stamp = None

    def refresh(self, tf_buffer: tf2_ros.Buffer) -> None:
        """Update pose and finite-difference velocity from the map->base TF."""
        if self.tf_frame is None:
            return
        try:
            t = tf_buffer.lookup_transform('map', self.tf_frame, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return
        stamp = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
        if self.stamp is not None and stamp <= self.stamp:
            return
        position = Position(x=t.transform.translation.x, y=t.transform.translation.y)
        if self.stamp is not None:
            dt = stamp - self.stamp
            self.velocity = (
                (position.x - self.pose.position.x) / dt,
                (position.y - self.pose.position.y) / dt,
            )
        self.pose = Pose(position, Orientation.from_msg(t.transform.rotation))
        self.stamp = stamp


class HumanSimulatorOptions(typing.TypedDict):
    node: rclpy.node.Node
    namespace: Namespace
    simulator: BaseSim
    realizer: Realizer
    acoustics: BaseAcousticsSimulator
    robots_visible: typing.NotRequired[bool]


@attrs.define
class _Rest:
    """A ped on furniture or rising from it: where it rests, since when, and the poses its body blends from and to."""

    support: Support
    x: float
    y: float
    yaw: float
    since: float
    start: tuple[np.ndarray, np.ndarray]
    released: float | None = None
    end: tuple[np.ndarray, np.ndarray] | None = None


class BaseHumanSimulator(NodeInterface, abc.ABC):
    PARAM_NAMESPACE: typing.ClassVar[str | None] = None
    TAKES_STIMULI: typing.ClassVar[bool] = False

    _arena_peds_publisher: rclpy.publisher.Publisher
    _marker_publisher: LazyPublisher[MarkerArray]
    _static_marker_publisher: rclpy.publisher.Publisher
    _known_obstacles: KnownObstacles
    _known_walls: KnownObstacles[Wall]
    _known_doors: KnownObstacles[Door]

    @classmethod
    def _register_task_modes(cls) -> None:
        """Register simulator-specific obstacle task modes.

        Called during __init__. Subclasses override this to register
        task modes specific to their simulator (e.g. TM_Prompt).
        """

    def __init__(self, *args: object, namespace: Namespace, simulator: BaseSim, realizer: Realizer, acoustics: BaseAcousticsSimulator, robots_visible: bool = True, **kwargs: object) -> None:
        """
        Initialize human simulator.

        Args:
            namespace: global namespace
            simulator: Simulator instance
            realizer: per-env Realizer (used to stamp sim_path on runtime-spawned obstacles)
            acoustics: acoustics simulator, the source of pedestrian hearing
            robots_visible: whether the backend tells its pedestrians where the robots are
        """
        super().__init__(*args, **kwargs)
        self._register_task_modes()
        self._simulator = simulator
        self._namespace = namespace
        self._realizer = realizer
        self._robots_visible = robots_visible

        self._known_obstacles = KnownObstacles[Obstacle]()
        self._known_walls = KnownObstacles[Wall]()
        self._known_doors = KnownObstacles[Door]()
        self._wall_counter = itertools.count()
        self._known_regions: dict[str, Region] = {}
        self._known_elevators: tuple[Elevator, ...] = ()
        self._known_stairs: tuple[Stairs, ...] = ()
        self._stair_running: dict[str, bool] = {}
        self._robots: dict[str, TrackedRobot] = {}
        self._warned_unresolved_models: set[str] = set()
        self._ped_model_uris: dict[str, str] = {}
        self._arena_peds_publisher = self.node.create_publisher(Pedestrians, self._namespace("arena_peds"), 10)
        self._sim_peds_publisher = self.node.create_publisher(Pedestrians, self._namespace("sim_peds"), 10)
        self._animation_states_publisher = self.node.create_publisher(AnimationStates, self._namespace("animation_states"), 10)
        self._marker_publisher = LazyPublisher(
            self.node.create_publisher(
                MarkerArray,
                self._namespace("pedestrian_markers", "extra"),
                rclpy.qos.QoSProfile(
                    reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                    durability=rclpy.qos.DurabilityPolicy.VOLATILE,
                    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                    depth=10,
                ),
            )
        )
        self._static_marker_publisher = self.node.create_publisher(
            MarkerArray,
            self._namespace("pedestrian_markers", "static"),
            rclpy.qos.QoSProfile(
                reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
        self._ped_positions_xy: dict[str, tuple[float, float]] = {}
        self._held_seats_listeners: list[typing.Callable[[], None]] = []
        self._incident_callback: typing.Callable[[str], None] | None = None
        self._ped_orientations: dict[str, QuaternionMsg] = {}
        self._ped_clips: dict[str, frozenset[str]] = {}
        self._gait = AnimationManager(logger=self._logger, fps=20.0)
        self._gait_prev_stamp: dict[int, float] = {}
        self._pose_profiles: dict[str, PoseProfile] = {}
        self._driven = DrivenJoints()
        self._drivers: dict[str, tuple[Roller, ...]] = {}
        self._bodies: dict[str, BodyMeasurements] = {}
        self._seat_plans: dict[int, SeatPlan] = {}
        self._rests: dict[int, _Rest] = {}
        self._free_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._warned_no_squat = False
        self._gestures = GestureLayer(self._gait, self._logger)
        self._gait.gesture_hook = self._gestures
        self.node.create_subscription(
            Pedestrians,
            self._namespace("arena_peds"),
            self._on_arena_peds,
            10,
        )

        self._possession: PossessionTable[Pedestrian] = PossessionTable(phase=self._gait.phase)
        self._stream_gate = False
        self._gate_open_stamp = 0.0
        self._ped_bus_index: dict[str, DynamicObstacle] = {}
        self._warned_unknown: set[str] = set()
        self._warned_bad_joints: set[str] = set()
        self._warned_stale = False
        self.node.create_subscription(
            Pedestrians,
            self._namespace("human", "stream"),
            self._on_human_stream,
            _STREAM_QOS,
        )
        self.node.create_service(
            MovePedestrians,
            self._namespace("human", "move"),
            self._cb_human_move,
        )

        self._hearing: PedestrianHearing | None = acoustics.pedestrian_hearing(self)

        self._semantic_values: dict[tuple[str, str], str] = {}
        self._semantic_backlog: collections.deque[tuple[str, str, str | None, str]] = collections.deque()
        self._semantic_draining = False

        self._simulator.attach_human_simulator(self)

    async def _register_lockstep_channels(self, channels: Sequence[LockstepChannel]) -> None:
        """Fire-once, best-effort registration of this adapter's lockstep channels in its env."""
        await register_channels(self.node, channels, env=str(self._namespace))

    def publish_arena_peds(self, msg: Pedestrians) -> None:
        """Stamp the header, substitute possessed peds, fill bare-name joint_state for peds missing one, then publish."""
        now = self.node.get_clock().now()
        now_sec = now.nanoseconds * 1e-9
        stamp = now.to_msg()
        for name, last_state in self._possession.expire(now_sec):
            self._on_ped_released(name, last_state)
        out = Pedestrians()
        out.header = msg.header
        out.pedestrians = self._possession.substitute(msg.pedestrians, now_sec)
        if out.header.stamp.sec == 0 and out.header.stamp.nanosec == 0:
            out.header.stamp = stamp

        current_ids: set[int] = {ped.id for ped in out.pedestrians}
        for stale in sorted(set(self._gait_prev_stamp) - current_ids):
            self._gestures.forget(stale)
            self._gait.forget(stale)
            self._seat_plans.pop(stale, None)
            self._rests.pop(stale, None)
            self._free_poses.pop(stale, None)
            del self._gait_prev_stamp[stale]
        for stale in sorted(self._driven.known() - current_ids):
            self._driven.forget(stale)

        animation = AnimationStates()
        animation.header = out.header
        tilts: dict[int, tuple[float, float, float, float]] = {}
        for ped in out.pedestrians:
            ped.model_uri = self._ped_model_uris.get(ped.name, "")
            if ped.joint_state.name:
                continue
            prev = self._gait_prev_stamp.get(ped.id)
            if prev is None:
                dt = 0.05
            else:
                dt = max(0.0, now_sec - prev)
            self._gait_prev_stamp[ped.id] = now_sec
            yaw = Orientation.from_msg(ped.pose.orientation).to_yaw()
            speed = ped.twist.linear.x * math.cos(yaw) + ped.twist.linear.y * math.sin(yaw)
            support = self._support(ped)
            channels = [g for g in ped.gestures if g.clip != SIT_CLIP or (support is not None and support.kind == "sit")]
            gesture = GestureRequest(
                channels=tuple(Channel(slot=g.slot, at=(g.at.x, g.at.y, g.at.z), clip=self._seat_clip(ped, support) if g.clip == SIT_CLIP and support is not None else g.clip, hand=g.hand, lock=g.render_pose_override) for g in channels),
                pose=(ped.pose.position.x, ped.pose.position.y, yaw),
                moving=ped.animation_state in (Pedestrian.WALKING, Pedestrian.RUNNING),
            )
            phase = float(ped.gait_phase) if ped.gait_phase != 0.0 else None
            angles = self._gait.compute(ped.id, ped.animation_state, speed, dt, gesture=gesture, phase=phase, profile=self._pose_profile_for(ped.agent_type))
            ped.joint_state = self._gait.joint_state(angles, stamp=stamp)
            ped.gait_phase = self._gait.phase(ped.id)
            if (tilt := self._gait.root_tilt(ped.id)) is not None:
                tilts[ped.id] = tilt
            animation.peds.append(self._animation_state(ped))  # pyright: ignore[reportAttributeAccessIssue]

        for ped in out.pedestrians:
            ped.pose.position.z = stair_height(self._known_stairs, ped.pose.position.x, ped.pose.position.y)
            drivers = self._drivers_for(ped.model_uri)
            if not drivers:
                continue
            yaw = Orientation.from_msg(ped.pose.orientation).to_yaw()
            angles = self._driven.advance(ped.id, drivers, ped.pose.position.x, ped.pose.position.y, yaw)
            ped.joint_state.name, ped.joint_state.position = with_driven(ped.joint_state.name, ped.joint_state.position, angles)

        possessed = self._possession.possessed(now_sec)
        for ped in out.pedestrians:
            if ped.name in possessed:
                self._rests.pop(ped.id, None)
                continue
            if ped.id in tilts and self._support(ped) is None:
                self._tilt_body(ped, tilts[ped.id])
            self._place_resting(ped, now_sec)

        self._arena_peds_publisher.publish(out)
        self._sim_peds_publisher.publish(placed_peds(out, self._realizer.placement()))
        self._animation_states_publisher.publish(animation)

    def _animation_state(self, ped: Pedestrian) -> AnimationState:
        state = AnimationState(id=ped.id, name=ped.name, base_phase=float(ped.gait_phase))
        base = self._gait.get_current_ped_animation(ped.id)
        state.base = base.name if base is not None else ""
        phases = self._gestures.phases(ped.id)
        for slot, overlay, playhead, weight in self._gait.overlays(ped.id):
            kind, channel, phase, clip = phases.get(slot, ("", "", "", ""))
            if not phase and overlay.releasing:
                phase = "release"
            state.slots.append(  # pyright: ignore[reportAttributeAccessIssue]
                AnimationSlot(
                    slot=slot,
                    kind=kind,
                    channel=channel,
                    phase=phase,
                    clip=clip,
                    animation=overlay.anim.name,
                    playhead=float(playhead),
                    duration=float(overlay.anim.duration),
                    weight=float(weight),
                    loop=overlay.loop,
                ),
            )
        return state

    def _support(self, ped: Pedestrian) -> Support | None:
        """Seat or bed the ped rests on, None while it is on its feet."""
        return None

    def _seat_clip(self, ped: Pedestrian, seat: Support) -> str:
        """The ped's sit-down onto its seat (`plan_seat`), its rise as the release, the sit clip without a squat clip."""
        squat = self._gait.clip(SQUAT_CLIP)
        if squat is None:
            if not self._warned_no_squat:
                self._warned_no_squat = True
                self._logger.warning(f"no {SQUAT_CLIP!r} clip, sitters play {SIT_CLIP!r} at any seat height")
            return SIT_CLIP
        name = f"seat:{ped.id}:{seat.height:.3f}:{seat.front}:{seat.back}"
        if name not in self._gait.animations:
            plan = plan_seat(squat.frames, self._body_for(ped.model_uri), seat.height, seat.front, seat.back, squat.fps)
            self._gait.register_transient(name, plan.frames, fps=squat.fps, owner=ped.id, release=plan.release)
            self._seat_plans[ped.id] = plan
        return name

    def _place_resting(self, ped: Pedestrian, now: float) -> None:
        """Set a ped on its furniture, its body blending in from where it stood and, once it has risen, out to where it walks on."""
        p, o = ped.pose.position, ped.pose.orientation
        free = (np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w]))
        support = self._support(ped)
        rest = self._rests.get(ped.id)
        if support is not None:
            yaw = Orientation.from_msg(o).to_yaw()
            if rest is None or rest.released is not None:
                start = rest.end if rest is not None and rest.end is not None else self._free_poses.get(ped.id, free)
                rest = self._rests[ped.id] = _Rest(support=support, x=p.x, y=p.y, yaw=yaw, since=now, start=start)
            else:
                rest.support, rest.x, rest.y, rest.yaw = support, p.x, p.y, yaw
            plan = self._seat_plans.get(ped.id)
            settle = LIE_S if rest.support.kind == "lie" else max(SETTLE_S, plan.down) if plan is not None else SETTLE_S
            pose = blend_pose(rest.start, self._rest_pose(ped, rest), (now - rest.since) / settle)
        elif rest is not None:
            if rest.released is None:
                rest.released = now
            plan = self._seat_plans.get(ped.id)
            hold, settle = (0.0, LIE_S) if rest.support.kind == "lie" else (rise_seconds(plan) if plan is not None else SETTLE_S, SETTLE_S)
            if now - rest.released < hold or rest.end is None:
                pose = rest.end = self._rest_pose(ped, rest)
            elif now - rest.released < hold + settle:
                pose = blend_pose(rest.end, free, (now - rest.released - hold) / settle)
            else:
                del self._rests[ped.id]
                self._free_poses[ped.id] = free
                return
        else:
            self._free_poses[ped.id] = free
            return
        (x, y, z), (qx, qy, qz, qw) = pose
        ped.pose = PoseMsg(position=PointMsg(x=float(x), y=float(y), z=float(z)), orientation=QuaternionMsg(x=float(qx), y=float(qy), z=float(qz), w=float(qw)))

    def _tilt_body(self, ped: Pedestrian, tilt: tuple[float, float, float, float]) -> None:
        """Turn the ped's body by its clip's pelvis tilt (a fall, lying on the ground), the pelvis kept over its position."""
        p = ped.pose.position
        angles = dict(zip(ped.joint_state.name, ped.joint_state.position, strict=True))
        yaw = Orientation.from_msg(ped.pose.orientation).to_yaw()
        (x, y, z), (qx, qy, qz, qw) = tilted_pose(self._body_for(ped.model_uri), angles, p.x, p.y, p.z, yaw, tilt)
        ped.pose = PoseMsg(position=PointMsg(x=float(x), y=float(y), z=float(z)), orientation=QuaternionMsg(x=float(qx), y=float(qy), z=float(qz), w=float(qw)))

    def _rest_pose(self, ped: Pedestrian, rest: _Rest) -> tuple[np.ndarray, np.ndarray]:
        angles = dict(zip(ped.joint_state.name, ped.joint_state.position, strict=True))
        body = self._body_for(ped.model_uri)
        plan = self._seat_plans.get(ped.id)
        depth = seat_depth(plan, angles, body.legs) if rest.support.kind == "sit" and plan is not None else 0.0
        return supported_pose(rest.support, body, angles, rest.x, rest.y, rest.yaw, depth)

    def _body_for(self, model_uri: str) -> BodyMeasurements:
        """Measurements of a ped model, read once per model, typical ones when neither its annotation nor its skin gives them."""
        body = self._bodies.get(model_uri)
        if body is None:
            try:
                body = load_body(model_uri)
            except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError) as e:
                self._logger.warning(f"body measurements of {model_uri!r} rejected ({e})")
                body = None
            if body is None:
                self._logger.warning(f"{model_uri!r} has no body measurements and no skin to measure: seated and lying bodies use typical ones")
                body = TYPICAL_BODY
            self._bodies[model_uri] = body
        return body

    def _drivers_for(self, model_uri: str) -> tuple[Roller, ...]:
        """Drivers of a ped model's rig.yaml, read once per model, none when the file is rejected."""
        drivers = self._drivers.get(model_uri)
        if drivers is None:
            try:
                drivers = load_drivers(model_uri)
            except (ValueError, OSError, yaml.YAMLError) as e:
                self._logger.warning(f"rig drivers of {model_uri!r} rejected ({e}), its driven joints stay still")
                drivers = ()
            self._drivers[model_uri] = drivers
        return drivers

    def _pose_profile_for(self, agent_type: str) -> PoseProfile:
        """Pose profile of an agent type (a yaml path or a builtin name), resolved once per type, default when empty or unknown."""
        profile = self._pose_profiles.get(agent_type)
        if profile is None:
            profile = self._load_pose_profile(agent_type)
            self._pose_profiles[agent_type] = profile
        return profile

    def _load_pose_profile(self, agent_type: str) -> PoseProfile:
        if not agent_type:
            return default_profile()
        try:
            builtin_dir: Path | None = Path(get_package_share_directory("arena_humansim")) / "config" / "agent_types"
        except PackageNotFoundError:
            builtin_dir = None
        try:
            section = agent_type_pose_section(agent_type, builtin_dir)
            if section is None:
                self._logger.warning(f"agent type {agent_type!r} not found, using the default pose profile")
                return default_profile()
            return resolve_pose_profile(section)
        except (ValueError, OSError, yaml.YAMLError) as e:
            self._logger.warning(f"agent type {agent_type!r} pose profile rejected ({e}), using the default")
            return default_profile()

    def held_seats(self) -> dict[str, list[int]]:
        """Indices into each object's annotated sit slots that agents hold, by object sim path."""
        return {}

    def on_held_seats_change(self, callback: typing.Callable[[], None]) -> None:
        self._held_seats_listeners.append(callback)

    def _notify_held_seats(self) -> None:
        for callback in self._held_seats_listeners:
            callback()

    def set_incident_callback(self, callback: typing.Callable[[str], None]) -> None:
        """Receives the reason whenever the backend finds the episode compromised."""
        self._incident_callback = callback

    def _report_incident(self, reason: str) -> None:
        self._logger.error(reason)
        if self._incident_callback is not None:
            self._incident_callback(reason)

    def publish_markers(self, markers: MarkerArray) -> None:
        """Publish a transient debug-overlay MarkerArray on `pedestrian_markers/extra`."""
        self._marker_publisher.publish(lambda: markers)

    def publish_static_markers(self, markers: MarkerArray) -> None:
        """Publish a latched MarkerArray on `pedestrian_markers/static` (TRANSIENT_LOCAL, late subscribers catch up)."""
        self._static_marker_publisher.publish(markers)

    def _on_arena_peds(self, msg: Pedestrians) -> None:
        self._ped_positions_xy = {p.name: (p.pose.position.x, p.pose.position.y) for p in msg.pedestrians}
        self._ped_orientations = {p.name: p.pose.orientation for p in msg.pedestrians}
        self._ped_clips = {p.name: frozenset(g.clip for g in p.gestures) for p in msg.pedestrians}

    def pedestrian_discs(self) -> Iterable[tuple[str, tuple[float, float], float]]:
        return [(name, xy, PED_RADIUS) for name, xy in self._ped_positions_xy.items()]

    def pedestrian_clips(self) -> Mapping[str, frozenset[str]]:
        """Gesture clips each tracked pedestrian is playing, under the name it is published with."""
        return self._ped_clips

    async def pedestrian_teleport(self, destinations: Mapping[str, tuple[float, float]]) -> bool:
        """Teleport tracked pedestrians to given (x, y), keeping their facing. Default impl asks the sim to move them."""
        if not destinations:
            return True
        peds_msg = Pedestrians()
        for name, (x, y) in destinations.items():
            ped = Pedestrian()
            ped.name = name
            cur = self._ped_positions_xy.get(name)
            ped.pose.position.x = x
            ped.pose.position.y = y
            ped.pose.position.z = 0.0
            if (orientation := self._ped_orientations.get(name)) is not None:
                ped.pose.orientation = orientation
            peds_msg.pedestrians.append(ped)  # pyright: ignore[reportAttributeAccessIssue]
            if cur is not None:
                self._ped_positions_xy[name] = (x, y)
        results = await self.relay_pedestrian_update(peds_msg)
        return all(results)

    # world mechanisms

    def authored_name(self, realized: str) -> str:
        """Strip the env prefix and a trailing `/<level_id>` from a realized entity name."""
        prefix = self._realizer.prefix()
        name = realized[len(prefix) + 1 :] if prefix and realized.startswith(f"{prefix}/") else realized
        head, sep, tail = name.rpartition("/")
        if sep and head and tail in self._realizer._level_origins:
            return head
        return name

    def set_mechanism_walls(self, key: str, segments: Sequence[Segment] | None) -> None:
        """Replace the named wall set seen by pedestrian navigation; None removes it. Map frame."""

    async def _clear_mechanism_walls(self) -> None:
        """Drop every mechanism wall set (world teardown)."""

    def notify_semantic_change(self, entity: str, field: str, value: str) -> None:
        """Broadcast `<entity>.<field>==<previous>` at intensity 0 and `<entity>.<field>==<value>` at intensity 1, in order."""
        previous = self._semantic_values.get((entity, field))
        self._semantic_values[(entity, field)] = value
        self._semantic_backlog.append((entity, field, previous, value))
        if not self._semantic_draining:
            self._semantic_draining = True
            self._in_background(self._drain_semantic_backlog())

    def _in_background(self, coro: typing.Coroutine[typing.Any, typing.Any, None]) -> None:
        """Run a coroutine on the node loop from sync code, inside or outside the loop."""
        try:
            asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            loop = self.node.event_loop
            loop.call_soon_threadsafe(lambda: loop.create_task(coro))

    async def _drain_semantic_backlog(self) -> None:
        try:
            while self._semantic_backlog:
                entity, field, previous, value = self._semantic_backlog.popleft()
                try:
                    await self._apply_semantic_change(entity, field, previous, value)
                except Exception as e:
                    self._logger.warning(f"semantic change {semantic_atom(entity, field, value)!r} not delivered: {e!r}")
        finally:
            self._semantic_draining = False

    async def _apply_semantic_change(self, entity: str, field: str, previous: str | None, value: str) -> None:
        if previous is not None and previous != value:
            await self.notify_stimulus(-1, semantic_atom(entity, field, previous), 0.0)
        await self.notify_stimulus(-1, semantic_atom(entity, field, value), 1.0)

    # possession

    def _now_s(self) -> float:
        """Sim-clock seconds from the node clock."""
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _warn_new(self, names: frozenset[str], warned: set[str], reason: str) -> None:
        for name in sorted(names - warned):
            self._logger.warning(f"human/stream: {reason} {name!r}, dropping")
        warned |= names

    async def _on_human_stream(self, msg: Pedestrians) -> None:
        if not self._stream_gate:
            return
        if msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 < self._gate_open_stamp:
            if not self._warned_stale:
                self._warned_stale = True
                self._logger.warning("human/stream: dropping batch stamped before gate open, publishers must stamp headers from the sim clock")
            return
        accepted, unknown, bad_joints, released = self._possession.merge(
            msg.pedestrians,
            known_names=set(self._ped_bus_index),
            gate_open=True,
            now=self._now_s(),
        )
        self._warn_new(unknown, self._warned_unknown, "unknown pedestrian")
        self._warn_new(bad_joints, self._warned_bad_joints, "invalid joint_state for")
        for name, last_state in released:
            self._on_ped_released(name, last_state)
        if accepted:
            relay = Pedestrians()
            relay.header.frame_id = "map"
            relay.header.stamp = msg.header.stamp
            relay.pedestrians = accepted
            await self.relay_pedestrian_update(relay)
        self._on_stream_merged([ped.name for ped in accepted])

    async def _cb_human_move(
        self,
        request: MovePedestrians.Request,
        response: MovePedestrians.Response,
    ) -> MovePedestrians.Response:
        if not self._stream_gate:
            response.results = [MovePedestrians.Response.NOT_FOUND] * len(request.pedestrians)
            return response
        results: list[int] = []
        for ped in request.pedestrians:
            tracked = self._ped_bus_index.get(ped.name)
            if tracked is None:
                results.append(MovePedestrians.Response.NOT_FOUND)
                continue
            tracked.pose = Pose.from_msg(ped.pose)
            await self._simulator.pedestrian_move([tracked])
            results.append(MovePedestrians.Response.SUCCESS)
            self._on_ped_moved(ped.name, ped.pose)
        response.results = results
        return response

    async def relay_pedestrian_update(self, msg: Pedestrians) -> Sequence[bool]:
        """Sim-side funnel: expire and substitute possessed peds, then forward to the physics sim."""
        now = self._now_s()
        for name, last_state in self._possession.expire(now):
            self._on_ped_released(name, last_state)
        out = Pedestrians()
        out.header = msg.header
        out.pedestrians = self._possession.substitute(msg.pedestrians, now)
        return await self._simulator.pedestrian_update(out)

    def possessed_peds(self) -> dict[str, Pedestrian]:
        """Deep-copied name to state mapping of currently possessed peds."""
        return self._possession.states(self._now_s())

    def _close_stream_gate(self) -> None:
        self._stream_gate = False
        self._possession.clear()
        self._warned_unknown.clear()
        self._warned_bad_joints.clear()
        self._warned_stale = False

    def _on_stream_merged(self, names: Sequence[str]) -> None:
        """Hook fired after a gate-open stream batch merges, names are the accepted entries."""

    def _on_ped_released(self, name: str, last_state: Pedestrian) -> None:
        """Hook fired when possession of a ped is released."""

    def _on_ped_moved(self, name: str, pose: PoseMsg) -> None:
        """Hook fired after a human/move teleport."""

    async def spawn_obstacles(self, obstacles: Sequence[Obstacle], layer: ObstacleLayer = ObstacleLayer.INUSE):
        """Spawns static obstacles.

        Args:
            obstacles (Sequence[Obstacle]): Static obstacles to spawn.
            layer (ObstacleLayer, optional): Layer to assign to spawned obstacles. Defaults to ObstacleLayer.INUSE.
        """
        self._logger.debug(f"spawning {len(obstacles)} static obstacles")

        futures: list[typing.Awaitable] = []
        to_register: list[KnownObstacle[Obstacle]] = []
        to_move: list[Obstacle] = []

        for obstacle in obstacles:
            if (known := self._known_obstacles.get(obstacle.name)) is not None:
                known.obstacle = obstacle
                to_move.append(known.obstacle)
                known.layer = layer
            else:
                known = self._known_obstacles.create_or_get(
                    name=obstacle.name,
                    obstacle=obstacle,
                )
            if not known.spawned:
                to_register.append(known)
        if to_move:
            futures.append(self._simulator.move_objects(to_move))

        to_spawn: list[Obstacle] = []
        for known, obstacle in zip(
            to_register,
            await self._spawn_obstacles_impl([known.obstacle for known in to_register]),
            strict=False,
        ):
            if not obstacle:
                continue
            known.obstacle = obstacle
            known.spawned = True

            if known.layer == ObstacleLayer.UNUSED:
                to_spawn.append(known.obstacle)
            known.layer = layer

        if to_spawn:
            futures.append(self._simulator.spawn_objects(to_spawn))
        await asyncio.gather(*futures)

    async def spawn_dynamic_obstacles(self, obstacles: typing.Sequence[DynamicObstacle]):
        """Spawns dynamic obstacles.

        Args:
            obstacles (typing.Sequence[DynamicObstacle]): Dynamic obstacles to spawn.
        """
        self._logger.debug(f"spawning {len(obstacles)} dynamic obstacles")

        futures: list[typing.Awaitable] = []
        to_register: list[KnownObstacle[DynamicObstacle]] = []
        to_move: list[DynamicObstacle] = []
        to_replace: list[DynamicObstacle] = []
        all_known: list[KnownObstacle[DynamicObstacle]] = []

        for obstacle in obstacles:
            if (known := self._known_obstacles.get(obstacle.name)) is not None:
                # renderers key visuals by ped name and a move carries no model,
                # so a model change must go through delete + respawn
                replace = known.obstacle.model.name != obstacle.model.name
                known.obstacle = obstacle
                known.layer = ObstacleLayer.INUSE
                if replace:
                    to_replace.append(known.obstacle)
                else:
                    to_move.append(known.obstacle)
            else:
                known = self._known_obstacles.create_or_get(name=obstacle.name, obstacle=obstacle)
            all_known.append(known)
            if not known.spawned:
                to_register.append(known)
        if to_replace:
            await self._simulator.pedestrian_delete(to_replace)
        if to_move:
            futures.append(self._simulator.pedestrian_move(to_move))

        to_spawn: list[DynamicObstacle] = []
        for known, obstacle in zip(
            to_register,
            await self._spawn_dynamic_obstacles_impl([known.obstacle for known in to_register]),
            strict=False,
        ):
            self._logger.debug(f"Spawned dynamic obstacle: {obstacle}")
            if not obstacle:
                continue

            known.obstacle = obstacle
            known.spawned = True

            if known.layer == ObstacleLayer.UNUSED:
                to_spawn.append(known.obstacle)
            known.layer = ObstacleLayer.INUSE

        # every requested obstacle goes through _ensure_spawnable so reused
        # names keep a fresh model_uri, but only the to_spawn subset is
        # actually handed to the simulator
        to_spawn_names = {obstacle.name for obstacle in to_spawn} | {obstacle.name for obstacle in to_replace}
        resolved = await self._ensure_spawnable([known.obstacle for known in all_known])
        if to_spawn_names:
            futures.append(
                self._simulator.pedestrian_spawn(
                    [obstacle for obstacle in resolved if obstacle.name in to_spawn_names],
                ),
            )
        await asyncio.gather(*futures)

        for known in all_known:
            self._ped_bus_index[known.obstacle.name] = known.obstacle
            self._ped_bus_index[known.obstacle.sim_path] = known.obstacle
        self._stream_gate = True
        self._gate_open_stamp = self._now_s()

    _PEDESTRIAN_FALLBACK: typing.ClassVar[str] = "arenian"

    async def _ensure_spawnable(self, obstacles: Sequence[DynamicObstacle]) -> Sequence[DynamicObstacle]:
        """Swap unresolvable ped models for _PEDESTRIAN_FALLBACK and record each
        ped's resolved SDF path, which publish_arena_peds stamps as model_uri so
        PedSkeletonPlugin can create the actor."""

        async def _sdf_path(obs: DynamicObstacle) -> str | None:
            view = await obs.model.resolve()
            model = await view.model.get(ModelType.SDF)
            if model.type is ModelType.UNKNOWN or model.path is None:
                return None
            return str(model.path)

        async def _resolve(obs: DynamicObstacle) -> DynamicObstacle:
            try:
                path = await _sdf_path(obs)
                if path is not None:
                    self._record_ped_model(obs, path)
                    return obs
            except Exception as e:
                if obs.model.name not in self._warned_unresolved_models:
                    self._warned_unresolved_models.add(obs.model.name)
                    self._logger.warning(
                        f"pedestrian model {obs.model.name!r} unresolved ({e}); using fallback",
                    )
            else:
                if obs.model.name not in self._warned_unresolved_models:
                    self._warned_unresolved_models.add(obs.model.name)
                    self._logger.warning(
                        f"pedestrian model {obs.model.name!r} has no SDF; using fallback",
                    )
            fallback = attrs.evolve(obs, model=HumanIdentifier.parse(self._PEDESTRIAN_FALLBACK))
            try:
                fallback_path = await _sdf_path(fallback)
            except Exception:
                fallback_path = None
            self._record_ped_model(fallback, fallback_path or "")
            return fallback

        return await asyncio.gather(*(_resolve(o) for o in obstacles))

    def _record_ped_model(self, obs: DynamicObstacle, model_uri: str) -> None:
        """Map both a ped's name and sim_path to its SDF path, since the published
        ped name is one or the other depending on the simulator."""
        self._ped_model_uris[obs.name] = model_uri
        self._ped_model_uris[obs.sim_path] = model_uri

    async def spawn_world(
        self,
        walls: Sequence[Wall],
        doors: Sequence[Door],
        *,
        collision_walls: Sequence[Wall] = (),
    ):
        """Spawns world elements.

        Args:
            walls (Sequence[Wall]): Walls to spawn.
            doors (Sequence[Door]): Doors to spawn.
            collision_walls (Sequence[Wall]): Walls registered for obstacle avoidance only, not spawned visually.
        """
        self._logger.debug(f"spawning {len(walls)} walls and {len(doors)} doors")

        wall_map: dict[str, Wall] = {}
        for wall in (*walls, *collision_walls):
            name = f"wall_{next(self._wall_counter)}"
            self._known_walls.create_or_get(
                name=name,
                obstacle=wall,
                layer=ObstacleLayer.WORLD,
            )
            wall_map[name] = wall

        door_map: dict[str, Door] = {}
        for door in doors:
            self._known_doors.create_or_get(
                name=door.name,
                obstacle=door,
                layer=ObstacleLayer.WORLD,
            )
            door_map[door.name] = door

        await asyncio.gather(
            self._simulator.spawn_doors(doors),
            self._simulator.spawn_walls(walls, clear_existing=False),
            self._spawn_walls_impl(wall_map),
            self._spawn_doors_impl(door_map),
        )

    async def spawn_elevators(self, elevators: Sequence[Elevator]) -> bool:
        """Hand the realized elevator cabins of the world to the human simulator, replacing those of the world before."""
        if not elevators and not self._known_elevators and not self._known_stairs:
            return True
        self._known_elevators = tuple(elevators)
        return await self._set_elevators_impl(self._known_elevators)

    def spawn_stairs(self, stairs: Sequence[Stairs]) -> None:
        """Remember the realized stair halves of the world, sent along with the next elevators, every belt running."""
        self._known_stairs = tuple(stairs)
        self._stair_running = {}

    async def stairs_changed(self, running: Mapping[str, bool]) -> None:
        """Escalator halves by realized name whose belt started (True) or stopped (False), every teleport zone sent again."""
        self._stair_running.update(running)
        await self._set_elevators_impl(self._known_elevators)

    def _stair_runs(self, stair: Stairs) -> bool:
        return stair_running(stair, self._known_stairs, self._stair_running)

    async def elevator_doors_changed(self, doors: Mapping[str, bool]) -> None:
        """Cabin doors by realized cabin name that turned open (True) or stopped being open (False)."""
        await self._set_elevator_doors_impl(doors)

    @staticmethod
    def _door_wall_name(door_name: str) -> str:
        return f"__door_{door_name}"

    async def update_doors(
        self,
        doors: Sequence[tuple[Door, bool]],
    ):
        """Update door states. True = open (no wall), False = closed (wall spawned).

        Args:
            doors (Sequence[tuple[Door, bool]]): Door/state pairs.
        """
        to_spawn: dict[str, Wall] = {}
        to_remove: list[str] = []

        for door, is_open in doors:
            name = self._door_wall_name(door.name)
            if is_open:
                if name in self._known_walls:
                    to_remove.append(name)
            else:
                if name not in self._known_walls:
                    wall = Wall(start=door.start, end=door.end)
                    self._known_walls.create_or_get(
                        name=name,
                        obstacle=wall,
                        layer=ObstacleLayer.WORLD,
                    )
                    to_spawn[name] = wall

        futures: list[typing.Awaitable] = []
        if to_spawn:
            self._logger.debug(f"closing {len(to_spawn)} doors: {list(to_spawn)}")
            futures.append(self._spawn_walls_impl(to_spawn))
        if to_remove:
            self._logger.debug(f"opening {len(to_remove)} doors: {to_remove}")
            for name in to_remove:
                self._known_walls.forget(name)
            futures.append(self._remove_walls_impl(to_remove))
        await asyncio.gather(*futures)

    async def unuse_obstacles(self):
        """
        Prepares obstacles for reuse or removal.
        """
        self._logger.debug("unusing obstacles")
        self._close_stream_gate()
        if self._hearing is not None:
            self._hearing.clear()

        unused = {name: known.obstacle for name, known in self._known_obstacles.items() if not isinstance(known.obstacle, DynamicObstacle) and known.layer == ObstacleLayer.UNUSED}

        await asyncio.gather(
            self._remove_pedestrians_impl(),
            self._remove_obstacles_impl(list(unused.values())),
        )
        self._ped_bus_index.clear()

        for name in unused:
            self._known_obstacles.forget(name)

        for obstacle in self._known_obstacles.values():
            if obstacle.layer == ObstacleLayer.INUSE:
                obstacle.spawned = False
                obstacle.layer = ObstacleLayer.UNUSED

    def unuse_world(self) -> tuple[frozenset[str], frozenset[str]]:
        """Mark world-level items for replacement.

        Downgrades WORLD-layer obstacles to UNUSED so they can be
        reclaimed after new world obstacles are spawned.

        Returns:
            Tuple of (old_wall_names, old_door_names) for stale-entry cleanup.
        """
        self._logger.debug("unusing world")

        old_walls = frozenset(self._known_walls.keys())
        old_doors = frozenset(self._known_doors.keys())

        for obstacle in self._known_obstacles.values():
            if obstacle.layer >= ObstacleLayer.WORLD:
                obstacle.spawned = False
                obstacle.layer = ObstacleLayer.UNUSED

        return old_walls, old_doors

    def remove_stale_world(
        self,
        old_walls: frozenset[str],
        old_doors: frozenset[str],
    ):
        """Forget wall/door tracking entries that were not reused during respawn.

        Args:
            old_walls: Wall names from before the respawn.
            old_doors: Door names from before the respawn.
        """
        for name in old_walls:
            if name in self._known_walls:
                self._known_walls.forget(name)
        for name in old_doors:
            if name in self._known_doors:
                self._known_doors.forget(name)

    async def remove_walls(
        self,
        walls: Sequence[Wall],
    ):
        """Removes specific walls from the simulation.

        Args:
            walls (Sequence[Wall]): Walls to remove (matched by start/end coordinates).
        """
        matched: dict[str, None] = {}
        for wall in walls:
            for name, known in self._known_walls.items():
                if known.obstacle.start == wall.start and known.obstacle.end == wall.end:
                    matched[name] = None
                    break

        if not matched:
            return

        names = tuple(matched)
        self._logger.debug(f"removing {len(names)} walls: {names}")
        for name in names:
            self._known_walls.forget(name)
        await self._remove_walls_impl(names)

    async def remove_doors(
        self,
        doors: Sequence[Door],
    ):
        """Removes specific doors from the simulation.

        Args:
            doors (Sequence[Door]): Doors to remove (matched by name).
        """
        names = [door.name for door in doors if door.name in self._known_doors]
        if not names:
            return

        self._logger.debug(f"removing {len(names)} doors: {names}")
        for name in names:
            self._known_doors.forget(name)
        await self._remove_doors_impl(names)

    async def setup_regions(self, regions: typing.Sequence[Region]) -> bool:
        """Configure regions (sources/sinks) for dynamic agent spawning.

        Args:
            regions: Already-resolved Region objects with concrete polygons.
        """
        for region in regions:
            self._known_regions[region.name] = region
        return await self._add_regions_impl(regions)

    async def configure(self, params: Sequence[ParameterMsg]) -> list[ParameterMsg]:
        """Hand the params under this backend's namespace to it at the episode boundary and return the accepted ones."""
        prefix = f"{self.PARAM_NAMESPACE}."
        own = sorted((p for p in params if self.PARAM_NAMESPACE is not None and p.name.startswith(prefix)), key=lambda p: p.name)
        if len(own) < len(params):
            self._logger.info(f"skipping human params of other backends: {[p.name for p in params if p not in own]}")
        accepted = await self._configure_impl([ParameterMsg(name=p.name.removeprefix(prefix), value=p.value) for p in own])
        return [p for p in own if p.name.removeprefix(prefix) in accepted]

    async def _configure_impl(self, params: Sequence[ParameterMsg]) -> set[str]:
        return set()

    async def remove_all_regions(self) -> bool:
        """Remove all tracked regions."""
        regions = tuple(self._known_regions.values())
        self._known_regions.clear()
        return await self._remove_regions_impl(regions)

    async def remove_obstacles(self, purge: ObstacleLayer = ObstacleLayer.UNUSED):
        """Removes obstacles from simulator.

        Args:
            purge (ObstacleLayer, optional): Level of obstacles to remove. Defaults to ObstacleLayer.UNUSED.
        """
        self._logger.debug(f"removing obstacles (level {purge})")
        futures: list[typing.Awaitable] = []

        stale_walls = [name for name, known in self._known_walls.items() if purge >= known.layer]
        stale_doors = [name for name, known in self._known_doors.items() if purge >= known.layer]

        if purge >= ObstacleLayer.WORLD:
            await self._simulator.remove_mechanisms()
            await self._simulator.remove_lights()
            await self._clear_mechanism_walls()
            self._semantic_values.clear()
            futures.append(self._simulator.remove_world())
            if self._known_elevators or self._known_stairs:
                self._known_elevators = ()
                self._known_stairs = ()
                self._stair_running = {}
                futures.append(self._set_elevators_impl(()))

        if stale_walls:
            futures.append(self._remove_walls_impl(stale_walls))
            for name in stale_walls:
                self._known_walls.forget(name)
        if stale_doors:
            futures.append(self._remove_doors_impl(stale_doors))
            for name in stale_doors:
                self._known_doors.forget(name)

        static: list[Obstacle] = []
        dynamic: list[DynamicObstacle] = []
        for oid, known in list(self._known_obstacles.items()):
            if purge >= known.layer:  # tmp: always respawn all dynamic obstacles
                if isinstance(known.obstacle, DynamicObstacle):
                    dynamic.append(known.obstacle)
                else:
                    static.append(known.obstacle)
                self._known_obstacles.forget(name=oid)

        if dynamic:
            self._close_stream_gate()
            for obstacle in dynamic:
                self._ped_bus_index.pop(obstacle.name, None)
                self._ped_bus_index.pop(obstacle.sim_path, None)

        if static:
            futures.append(self._remove_obstacles_impl(static))
        futures.append(self._simulator.delete_objects(static))
        futures.append(self._simulator.pedestrian_delete(dynamic))
        await asyncio.gather(*futures)

    async def _sim_then_impl(
        self,
        sim_call: typing.Callable[[Sequence[Robot]], typing.Awaitable[Sequence[bool]]],
        impl_call: typing.Callable[[Sequence[Robot]], typing.Awaitable[Sequence[bool]]],
        robots: Sequence[Robot],
    ) -> tuple[bool, ...]:
        sim_success = await sim_call(robots)
        succeeded = tuple(r for r, s in zip(robots, sim_success, strict=False) if s)
        human_success = await impl_call(succeeded)
        human_iter = iter(human_success)
        return tuple(s and next(human_iter) for s in sim_success)

    async def spawn_robot(
        self,
        robots: Sequence[Robot],
    ) -> Sequence[bool]:
        """Spawns robots.

        Args:
            robots (Sequence[Robot]): Robots to spawn.

        Returns:
            Sequence[bool]: Success of each robot spawn.
        """
        self._logger.debug(f"spawning {len(robots)} robots")
        return await self._sim_then_impl(
            self._simulator.robot_spawn,
            self._track_robots,
            robots,
        )

    async def remove_robot(
        self,
        robots: Sequence[Robot],
    ) -> Sequence[bool]:
        """Removes robots from the simulation.

        Args:
            robots (Sequence[Robot]): Robots to remove.

        Returns:
            Sequence[bool]: Success of each robot removal.
        """
        self._logger.debug(f"removing {len(robots)} robots")
        return await self._sim_then_impl(
            self._simulator.robot_delete,
            self._untrack_robots,
            robots,
        )

    async def move_robot(
        self,
        robots: Sequence[Robot],
    ) -> Sequence[bool]:
        """Moves robots.

        Args:
            robots (Sequence[Robot]): Robots to move.

        Returns:
            Sequence[bool]: Success of each robot move.
        """
        self._logger.debug(f"moving {len(robots)} robots")
        return await self._sim_then_impl(
            self._simulator.robot_move,
            self._place_robots,
            robots,
        )

    async def _track_robots(self, robots: Sequence[Robot]) -> Sequence[bool]:
        for robot in robots:
            tf_frame: str | None = None
            try:
                view = await robot.model.resolve()
                tf_frame = robot.frame(view.model_params.base_frame).raw()
            except Exception as e:
                self._logger.warning(f"robot {robot.name!r}: base frame unresolved ({e}), using spawn pose only")
            self._robots[robot.name] = TrackedRobot(robot=robot, tf_frame=tf_frame, pose=copy.deepcopy(robot.pose))
        return await self._spawn_robot_impl(robots)

    async def _untrack_robots(self, robots: Sequence[Robot]) -> Sequence[bool]:
        for robot in robots:
            self._robots.pop(robot.name, None)
        return await self._remove_robot_impl(robots)

    async def _place_robots(self, robots: Sequence[Robot]) -> Sequence[bool]:
        for robot in robots:
            tracked = self._robots.get(robot.name)
            if tracked is not None:
                tracked.place(robot.pose)
        return await self._move_robot_impl(robots)

    def tracked_robots(self) -> list[TrackedRobot]:
        """Spawned robots in spawn order, pose and velocity refreshed from TF."""
        for tracked in self._robots.values():
            tracked.refresh(self.node.tf_buffer)
        return list(self._robots.values())

    # impl

    async def pause(self):
        pass

    async def unpause(self):
        pass

    async def notify_stimulus(self, agent: int | str, stimulus: str, intensity: float) -> bool:
        """True when the engine took the stimulus for `agent`, an engine agent id or a spawned agent's name or ``sim_path``."""
        return False

    def stimuli_taken(self, obstacle: DynamicObstacle) -> tuple[str, frozenset[str]] | None:
        """The agent type of `obstacle` and the stimulus names it acts on, None when the backend cannot tell."""
        del obstacle
        return None

    def check_stimuli(self, scenario: str, humans: Sequence[DynamicObstacle], timeline: Sequence[TimelineEntry]) -> None:
        """Raise ValueError naming `scenario` for a timeline stimulus this backend drops or the agent type of its target ignores."""
        stimuli = [stimulus for entry in timeline for stimulus in entry.stimulus]
        if stimuli and not self.TAKES_STIMULI:
            raise ValueError(f"scenario {scenario!r} has timeline stimulus entries, which only the arena human backend acts on, run it with human:=arena")
        by_name = {human.name: human for human in humans}
        for stimulus in stimuli:
            for agent in stimulus.agents:
                taken = self.stimuli_taken(by_name[agent])
                if taken is not None and stimulus.need not in taken[1]:
                    raise ValueError(f"scenario {scenario!r}: stimulus {stimulus.need!r} for {agent!r} is neither a need nor a step `until` event of its agent type {taken[0]!r}, which acts on {sorted(taken[1])}")

    async def _set_elevators_impl(self, elevators: Sequence[Elevator]) -> bool:
        return True

    async def _set_elevator_doors_impl(self, doors: Mapping[str, bool]) -> bool:
        return True

    async def configure_contact(self, mode: str, standing_distance: float) -> None:
        """Contact interactions touch (`enabled`) or hold at `standing_distance` (`locomotion_only`)."""
        del standing_distance
        if mode != "enabled":
            raise NotImplementedError(f"{type(self).__name__} has no contact interactions, contact_mode={mode!r} would be a silent no-op")

    async def configure_gestures(self, mode: str) -> None:
        """Agents publish gestures (`enabled`) or behave the same and publish none (`disabled`)."""
        if mode != "enabled":
            raise NotImplementedError(f"{type(self).__name__} publishes no gestures, gesture_mode={mode!r} would be a silent no-op")

    @abc.abstractmethod
    async def _spawn_obstacles_impl(
        self,
        obstacles: Sequence[Obstacle],
    ) -> Sequence[Obstacle | None]: ...

    @abc.abstractmethod
    async def _spawn_dynamic_obstacles_impl(
        self,
        obstacles: Sequence[DynamicObstacle],
    ) -> Sequence[DynamicObstacle | None]: ...

    @abc.abstractmethod
    async def _remove_obstacles_impl(
        self,
        obstacles: Sequence[Obstacle],
    ) -> bool: ...

    @abc.abstractmethod
    async def _remove_pedestrians_impl(
        self,
    ) -> bool: ...

    @abc.abstractmethod
    async def _spawn_walls_impl(
        self,
        walls: Mapping[str, Wall],
    ) -> bool: ...

    @abc.abstractmethod
    async def _spawn_doors_impl(
        self,
        doors: Mapping[str, Door],
    ) -> bool: ...

    @abc.abstractmethod
    async def _remove_walls_impl(
        self,
        names: Sequence[str],
    ) -> bool: ...

    @abc.abstractmethod
    async def _remove_doors_impl(
        self,
        names: Sequence[str],
    ) -> bool: ...

    @abc.abstractmethod
    async def _spawn_robot_impl(
        self,
        robots: Sequence[Robot],
    ) -> Sequence[bool]: ...

    @abc.abstractmethod
    async def _remove_robot_impl(
        self,
        robots: Sequence[Robot],
    ) -> Sequence[bool]: ...

    @abc.abstractmethod
    async def _move_robot_impl(
        self,
        robots: Sequence[Robot],
    ) -> Sequence[bool]: ...

    @abc.abstractmethod
    async def _add_regions_impl(self, regions: Sequence[Region]) -> bool: ...

    @abc.abstractmethod
    async def _remove_regions_impl(self, regions: Sequence[Region]) -> bool: ...


HumanSimulatorRegistry = Registry[Constants.HumanSimulator, BaseHumanSimulator]()


@HumanSimulatorRegistry.register(Constants.HumanSimulator.DUMMY)
async def dummy(**kwargs: typing.Unpack[HumanSimulatorOptions]) -> BaseHumanSimulator:
    from .dummy import DummyHumanSimulator

    return DummyHumanSimulator(**kwargs)


@HumanSimulatorRegistry.register(Constants.HumanSimulator.NONE)
async def none(**kwargs: typing.Unpack[HumanSimulatorOptions]) -> BaseHumanSimulator:
    # 'none' launches no human-sim node (empty GroupAction in human.launch.py);
    # the node still needs a no-op backend for the registry lookup.
    from .dummy import DummyHumanSimulator

    return DummyHumanSimulator(**kwargs)


@HumanSimulatorRegistry.register(Constants.HumanSimulator.REPLAY)
async def replay(**kwargs: typing.Unpack[HumanSimulatorOptions]) -> BaseHumanSimulator:
    from .replay import ReplayHumanSimulator

    return ReplayHumanSimulator(**kwargs)


@HumanSimulatorRegistry.register(Constants.HumanSimulator.HUNAV)
async def lazy_hunavsim(**kwargs: typing.Unpack[HumanSimulatorOptions]) -> BaseHumanSimulator:
    # Imported lazily: hunav_msgs / hunav_sim are optional deps, so registering the
    # backend must not require them to be installed. Only selecting human:=hunav does.
    from .hunav.hunav import HunavHumanSimulator

    return await HunavHumanSimulator.create(**kwargs)


@HumanSimulatorRegistry.register(Constants.HumanSimulator.ARENA)
async def arenasim(**kwargs: typing.Unpack[HumanSimulatorOptions]) -> BaseHumanSimulator:
    from .arena_humansim.arena_humansim import ArenaHumanSimulator

    return await ArenaHumanSimulator.create(**kwargs)
