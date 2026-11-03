"""Qt-free motion engine: composes per-ped intents into human/stream.

Headless-importable: construct against a real rclpy node, call tick() on your
own cadence. Subscription callbacks stash only, tick() is the sole reader and
mutator of shared state, so ROS-callback threads never touch Qt-adjacent state.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os
import time
from collections import deque
from typing import TYPE_CHECKING

from rviz_utils.hri.rig import wire_angles

from human_steering import clips, compose, integrate
from human_steering.clips import Clip

if TYPE_CHECKING:
    import rclpy.node
    from arena_people_msgs.msg import Pedestrian, Pedestrians
    from arena_people_msgs.srv import MovePedestrians
    from arena_runtime.humans.gait import LIMITS, GaitGenerator
    from arena_runtime_msgs.msg import EpisodeRecord
    from geometry_msgs.msg import Twist
    from rclpy.qos import QoSProfile
    from rclpy.subscription import Subscription
    from rclpy.task import Future
else:
    try:
        from arena_runtime.humans.gait import LIMITS, GaitGenerator
    except ImportError:  # pragma: no cover - exercised only without a sourced ROS install
        GaitGenerator = None
        LIMITS = ()
    try:
        from arena_people_msgs.msg import Pedestrian, Pedestrians
        from arena_people_msgs.srv import MovePedestrians
        from arena_runtime_msgs.msg import EpisodeRecord
        from geometry_msgs.msg import Twist
    except ImportError:  # pragma: no cover - exercised only without a sourced ROS install
        Pedestrian = Pedestrians = MovePedestrians = Twist = EpisodeRecord = None

STREAM_HZ = 20.0
RUN_THRESHOLD_MPS = 1.8
CMD_VEL_DEPTH = 10
ROSTER_DEPTH = 10

MANIFEST_SUFFIX = "/state/viz_manifest"

_IDLE, _WALKING, _RUNNING = 0, 1, 2
_WALK_THRESHOLD_MPS = 1e-3


def auto_state(speed: float, run_threshold: float = RUN_THRESHOLD_MPS) -> int:
    """Animation state implied by speed, absent an explicit override."""
    if speed >= run_threshold:
        return _RUNNING
    if speed > _WALK_THRESHOLD_MPS:
        return _WALKING
    return _IDLE


def _empty_manifest_due(prev_count: int, current_count: int) -> bool:
    """True exactly on the tick the held set falls from non-empty to empty: one release manifest, not a steady drumbeat."""
    return prev_count > 0 and current_count == 0


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Planar yaw from a quaternion, ignoring roll/pitch."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_to_quat_zw(yaw: float) -> tuple[float, float]:
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


@dataclasses.dataclass
class ClipState:
    """A ped's active clip playback: engine-time (sim dt accumulated) t0 so pausing
    the sim freezes it along with everything else."""

    name: str
    t0_engine: float
    speed_scale: float = 1.0
    loop: bool = True
    blend_from: dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Intent:
    """One ped's current drive state. mode gates which integrator runs, posed/clip/gaze
    layer on top independent of mode (compose.py resolves the per-joint precedence).
    held is the possession claim, dropped only by release/episode clear/despawn."""

    held: bool = False
    mode: str = "idle"  # "idle" | "waypoints" | "teleop"
    waypoints: list[tuple[float, float]] = dataclasses.field(default_factory=list)
    waypoint_cursor: int = 0
    loop: bool = False
    speed: float = 1.0
    waypoint_total: int = 0
    state_override: int | None = None
    posed: dict[str, float] = dataclasses.field(default_factory=dict)
    clip: ClipState | None = None
    clip_release: tuple[dict[str, float], float] | None = None
    gaze: tuple[float, float, float] | None = None
    teleop_twist: tuple[float, float, float] = (0.0, 0.0, 0.0)
    teleop_last_cmd_wall: float = 0.0


class IntentStore:
    """Per-ped intent bookkeeping and pose-integration seed. Pure Python: no ROS, no Qt."""

    def __init__(self) -> None:
        self._intents: dict[str, Intent] = {}
        self._poses: dict[str, tuple[float, float, float]] = {}
        self._episode_id: int | None = None

    def __contains__(self, name: str) -> bool:
        return name in self._intents

    def get(self, name: str) -> Intent:
        return self._intents.setdefault(name, Intent())

    def held(self) -> list[tuple[str, Intent]]:
        """Peds this panel has claimed, idle or moving, until an explicit release."""
        return [(name, intent) for name, intent in self._intents.items() if intent.held]

    def pose(self, name: str) -> tuple[float, float, float] | None:
        return self._poses.get(name)

    def set_pose(self, name: str, x: float, y: float, yaw: float) -> None:
        self._poses[name] = (x, y, yaw)

    def resync(self, name: str, x: float, y: float, yaw: float) -> None:
        """Re-seed pose from the roster mirror, unless the panel holds the ped."""
        intent = self._intents.get(name)
        if intent is not None and intent.held:
            return
        self._poses[name] = (x, y, yaw)

    def teleport(self, name: str, x: float, y: float, yaw: float | None = None) -> None:
        """GUI teleport: reseed the integrator immediately so the next tick doesn't revert it."""
        _, _, old_yaw = self._poses.get(name, (x, y, 0.0))
        self._poses[name] = (x, y, yaw if yaw is not None else old_yaw)

    def forget(self, name: str) -> None:
        self._intents.pop(name, None)
        self._poses.pop(name, None)

    def on_episode(self, episode_id: int) -> bool:
        """Clear every intent when the episode id changes. Returns whether it cleared."""
        changed = self._episode_id is not None and episode_id != self._episode_id
        if changed:
            self._intents.clear()
            self._poses.clear()
        self._episode_id = episode_id
        return changed

    def expire_teleop(self, name: str, now_wall: float, deadman_s: float = integrate.DEADMAN_S) -> bool:
        """Drop a stale teleop back to idle past the deadman window, the claim stays held. Returns whether it dropped."""
        intent = self._intents.get(name)
        if intent is not None and intent.mode == "teleop" and integrate.deadman_expired(intent.teleop_last_cmd_wall, now_wall, deadman_s):
            intent.mode = "idle"
            intent.teleop_twist = (0.0, 0.0, 0.0)
            return True
        return False


@dataclasses.dataclass(frozen=True)
class RosterStatus:
    """One roster row's display summary: state label, speed, waypoint progress."""

    state_label: str
    speed: float
    waypoint_progress: str | None = None


_ANIMATION_STATE_LABELS = {_IDLE: "IDLE", _WALKING: "WALKING", _RUNNING: "RUNNING"}


def _roster_state_label(intent: Intent | None, animation_state: int) -> str:
    """TELEOP (GUI-local mode) overrides the roster-mirror's own animation state."""
    if intent is not None and intent.mode == "teleop":
        return "TELEOP"
    return _ANIMATION_STATE_LABELS.get(animation_state, "IDLE")


def _roster_waypoint_progress(intent: Intent | None) -> str | None:
    """Waypoint-chain progress for a GUI-driven ped, else None."""
    if intent is None or intent.mode != "waypoints" or not intent.waypoint_total:
        return None
    if intent.loop:
        return f"loop ({intent.waypoint_total} wp)"
    index = min(intent.waypoint_total, intent.waypoint_cursor + 1)
    return f"wp {index}/{intent.waypoint_total}"


@dataclasses.dataclass(frozen=True)
class Namespaces:
    node_ns: str
    env_ns: str
    map_topic: str


def _viz_matches(node_ns: str, target: str) -> bool:
    env_ns = os.path.dirname(node_ns)
    return target in (node_ns, env_ns, env_ns.lstrip("/")) or os.path.basename(env_ns) == target


def resolve_namespace(node: rclpy.node.Node, target: str | None = None) -> Namespaces | None:
    """One discovery pass over `<node_ns>/state/viz_manifest` topics (arena viz's own
    convention). None if nothing matches yet, callers retry on a later tick."""
    candidates = [name[: -len(MANIFEST_SUFFIX)] for name, _types in node.get_topic_names_and_types() if name.endswith(MANIFEST_SUFFIX)]
    if not candidates:
        return None
    if target is not None:
        candidates = [c for c in candidates if _viz_matches(c, target)]
        if not candidates:
            return None
    node_ns = candidates[0]
    env_ns = os.path.dirname(node_ns) or "/"
    return Namespaces(node_ns=node_ns, env_ns=env_ns, map_topic=f"{node_ns}/map")


def _stream_qos() -> QoSProfile:
    import rclpy.qos

    return rclpy.qos.QoSProfile(
        reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
        durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        history=rclpy.qos.HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def _episode_qos() -> QoSProfile:
    import rclpy.qos

    # Mirrors node.py's _EPISODE_QOS exactly: deeper KeepLast for terminal-then-next bursts.
    return rclpy.qos.QoSProfile(depth=20, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)


class Driver:
    """Motion engine for one env: composes per-ped intents and streams the held
    claim set onto human/stream. Qt-free, construct against a real node once
    resolve_namespace() succeeds."""

    def __init__(self, node: rclpy.node.Node, namespaces: Namespaces, unlimited: bool = False) -> None:
        self._node = node
        self._env_ns = namespaces.env_ns.rstrip("/")
        self._node_ns = namespaces.node_ns.rstrip("/")
        self._unlimited = unlimited

        self._intents = IntentStore()
        self._gait: dict[str, GaitGenerator] = {}
        self._roster_ids: dict[str, int] = {}
        self._roster_model_uri: dict[str, str] = {}
        self._clip_libraries: dict[str, dict[str, Clip]] = {}
        self._last_composed: dict[str, dict[str, float]] = {}
        self._roster_joint_state: dict[str, dict[str, float]] = {}
        self._roster_speed: dict[str, float] = {}
        self._roster_state: dict[str, int] = {}

        self._pending_peds: Pedestrians | None = None
        self._pending_episode_id: int | None = None
        self._pending_cmd_vel: dict[str, tuple[float, float, float, float]] = {}
        self._cmd_vel_subs: dict[str, Subscription] = {}

        self._engine_time_s = 0.0
        self._last_tick_sim_s: float | None = None
        self._publish_times: deque[float] = deque(maxlen=40)
        self._prev_held_count = 0

        stream_topic = f"{self._env_ns}/human/stream"
        move_topic = f"{self._env_ns}/human/move"

        existing = node.get_publishers_info_by_topic(stream_topic)
        if existing:
            node.get_logger().warning(
                f"{stream_topic} already has {len(existing)} publisher(s), last writer wins, multi-panel driving is not arbitrated",
            )

        self._pub = node.create_publisher(Pedestrians, stream_topic, _stream_qos())
        self._move_client = node.create_client(MovePedestrians, move_topic)
        self._sub_peds = node.create_subscription(Pedestrians, f"{self._env_ns}/arena_peds", self._on_peds, ROSTER_DEPTH)
        self._sub_episode = node.create_subscription(EpisodeRecord, f"{self._node_ns}/state/episode", self._on_episode, _episode_qos())

    # -- ROS callbacks: stash only, never mutate shared state directly --

    def _on_peds(self, msg: Pedestrians) -> None:
        self._pending_peds = msg

    def _on_episode(self, msg: EpisodeRecord) -> None:
        self._pending_episode_id = msg.episode_id

    def _on_cmd_vel(self, name: str, msg: Twist) -> None:
        self._pending_cmd_vel[name] = (msg.linear.x, msg.linear.y, msg.angular.z, time.monotonic())

    # -- drain: tick-thread only --

    def _drain_peds(self) -> None:
        msg = self._pending_peds
        if msg is None:
            return
        self._pending_peds = None
        seen: set[str] = set()
        for ped in msg.pedestrians:
            seen.add(ped.name)
            self._roster_ids[ped.name] = ped.id
            self._roster_model_uri[ped.name] = ped.model_uri
            self._roster_speed[ped.name] = math.hypot(ped.twist.linear.x, ped.twist.linear.y)
            self._roster_state[ped.name] = ped.animation_state
            yaw = _yaw_from_quat(ped.pose.orientation.x, ped.pose.orientation.y, ped.pose.orientation.z, ped.pose.orientation.w)
            self._intents.resync(ped.name, ped.pose.position.x, ped.pose.position.y, yaw)
            if ped.body.joints:
                self._roster_joint_state[ped.name] = wire_angles(ped)
        stale = set(self._roster_ids) - seen
        for name in stale:
            self._roster_ids.pop(name, None)
            self._roster_model_uri.pop(name, None)
            self._gait.pop(name, None)
            self._last_composed.pop(name, None)
            self._roster_joint_state.pop(name, None)
            self._roster_speed.pop(name, None)
            self._roster_state.pop(name, None)
            self._intents.forget(name)
        self._sync_cmd_vel_subs()

    def _drain_episode(self) -> None:
        if self._pending_episode_id is None:
            return
        episode_id = self._pending_episode_id
        self._pending_episode_id = None
        self._intents.on_episode(episode_id)

    def _drain_cmd_vel(self) -> None:
        if not self._pending_cmd_vel:
            return
        pending, self._pending_cmd_vel = self._pending_cmd_vel, {}
        for name, (vx, vy, wz, wall) in pending.items():
            self.teleop_input(name, vx, vy, wz, wall)

    def teleop_input(self, name: str, vx: float, vy: float, wz: float, now_wall: float | None = None) -> None:
        """Shared cmd_vel/teleop-pad input path: sets teleop mode + twist, refreshes the
        deadman timestamp. Callers must keep calling while held, expire_teleop() releases
        after 0.5s idle."""
        intent = self._intents.get(name)
        intent.held = True
        intent.mode = "teleop"
        intent.teleop_twist = (vx, vy, wz)
        intent.teleop_last_cmd_wall = now_wall if now_wall is not None else time.monotonic()

    def _sync_cmd_vel_subs(self) -> None:
        current = set(self._roster_ids)
        existing = set(self._cmd_vel_subs)
        for name in current - existing:
            topic = f"{self._env_ns}/human/{clips.slug(name)}/cmd_vel"
            self._cmd_vel_subs[name] = self._node.create_subscription(
                Twist,
                topic,
                functools.partial(self._on_cmd_vel, name),
                CMD_VEL_DEPTH,
            )
        for name in existing - current:
            self._node.destroy_subscription(self._cmd_vel_subs.pop(name))

    # -- clip library --

    def _clip_library_for(self, name: str) -> dict[str, Clip]:
        model_uri = self._roster_model_uri.get(name, "")
        if model_uri not in self._clip_libraries:
            self._clip_libraries[model_uri] = clips.load_library(model_uri) if model_uri else clips.load_poses_dir()
        return self._clip_libraries[model_uri]

    def clip_inventory(self, name: str) -> list[str]:
        return sorted(self._clip_library_for(name))

    # -- intent mutation (panel/canvas call these) --

    def set_waypoints(self, name: str, points: list[tuple[float, float]], loop: bool, speed: float) -> None:
        intent = self._intents.get(name)
        intent.held = True
        intent.mode = "waypoints"
        intent.waypoints = list(points)
        intent.waypoint_cursor = 0
        intent.loop = loop
        intent.speed = speed
        intent.waypoint_total = len(points)

    def append_waypoint(self, name: str, point: tuple[float, float], speed: float) -> None:
        """Append one stop to the stable route without touching the cursor, starting a new looping route if the ped has none."""
        intent = self._intents.get(name)
        intent.held = True
        if intent.mode != "waypoints":
            intent.mode = "waypoints"
            intent.waypoints = []
            intent.waypoint_cursor = 0
            intent.loop = True
            intent.waypoint_total = 0
        intent.waypoints.append(point)
        intent.waypoint_total += 1
        intent.speed = speed

    def waypoints(self, name: str) -> list[tuple[float, float]]:
        """Current route for one ped in original click order, empty if it is not routing."""
        if name not in self._intents:
            return []
        intent = self._intents.get(name)
        if intent.mode != "waypoints":
            return []
        return list(intent.waypoints)

    def stop(self, name: str) -> None:
        intent = self._intents.get(name)
        intent.mode = "idle"
        intent.waypoints.clear()
        intent.waypoint_cursor = 0
        intent.teleop_twist = (0.0, 0.0, 0.0)

    def engage_joint(self, name: str, joint: str, value: float) -> None:
        intent = self._intents.get(name)
        intent.held = True
        intent.posed[joint] = value

    def disengage_joint(self, name: str, joint: str) -> None:
        self._intents.get(name).posed.pop(joint, None)

    def set_state_override(self, name: str, state: int | None) -> None:
        intent = self._intents.get(name)
        if state is not None:
            intent.held = True
        intent.state_override = state

    def set_gaze(self, name: str, target: tuple[float, float, float] | None) -> None:
        intent = self._intents.get(name)
        if target is not None:
            intent.held = True
        intent.gaze = target

    def start_clip(self, name: str, clip_name: str, speed_scale: float = 1.0, loop: bool = True) -> bool:
        if clip_name not in self._clip_library_for(name):
            return False
        intent = self._intents.get(name)
        intent.held = True
        intent.clip = ClipState(
            name=clip_name,
            t0_engine=self._engine_time_s,
            speed_scale=speed_scale,
            loop=loop,
            blend_from=dict(self._last_composed.get(name, {})),
        )
        return True

    def stop_clip(self, name: str) -> None:
        intent = self._intents.get(name)
        if intent.clip is not None:
            intent.clip_release = (dict(self._last_composed.get(name, {})), self._engine_time_s)
        intent.clip = None

    def teleport(self, name: str, x: float, y: float, yaw: float | None = None) -> None:
        """Reseed locally first (so the next tick doesn't revert it), then ask the backend."""
        self._intents.get(name).held = True
        self._intents.teleport(name, x, y, yaw)
        request = MovePedestrians.Request()
        ped = Pedestrian()
        ped.name = name
        ped.pose.position.x = x
        ped.pose.position.y = y
        if yaw is not None:
            qz, qw = _yaw_to_quat_zw(yaw)
            ped.pose.orientation.z = qz
            ped.pose.orientation.w = qw
        request.pedestrians = [ped]
        future = self._move_client.call_async(request)
        future.add_done_callback(functools.partial(self._on_teleport_response, name))

    def _on_teleport_response(self, name: str, future: Future) -> None:
        exc = future.exception()
        if exc is not None:
            self._node.get_logger().warning(f"human/move failed for {name!r}: {exc!r}")

    def release(self, name: str) -> None:
        """Drop the claim and the whole intent, the ped leaves the stream this tick."""
        self._intents.forget(name)
        self._last_composed.pop(name, None)

    # -- status (panel.py) --

    @property
    def stream_hz(self) -> float:
        if len(self._publish_times) < 2:
            return 0.0
        span = self._publish_times[-1] - self._publish_times[0]
        return (len(self._publish_times) - 1) / span if span > 0.0 else 0.0

    def human_move_available(self) -> bool:
        return self._move_client.service_is_ready()

    def roster(self) -> list[str]:
        return sorted(self._roster_ids)

    def held_names(self) -> set[str]:
        """Local claim set: peds this panel holds until released, no backend round-trip."""
        return {name for name, _intent in self._intents.held()}

    def roster_status(self, name: str) -> RosterStatus:
        """Roster-row display summary for one ped: state pill, speed, waypoint progress."""
        intent = self._intents.get(name) if name in self._intents else None
        return RosterStatus(
            state_label=_roster_state_label(intent, self._roster_state.get(name, _IDLE)),
            speed=self._roster_speed.get(name, 0.0),
            waypoint_progress=_roster_waypoint_progress(intent),
        )

    def pose(self, name: str) -> tuple[float, float, float] | None:
        return self._intents.pose(name)

    def all_poses(self) -> dict[str, tuple[float, float, float]]:
        poses = {name: self._intents.pose(name) for name in self._roster_ids}
        return {name: pose for name, pose in poses.items() if pose is not None}

    def current_joint_state(self, name: str) -> dict[str, float] | None:
        """Best-available composed angles for FK preview: this tick's composed output
        for a held ped, else the last bus snapshot."""
        return self._last_composed.get(name) or self._roster_joint_state.get(name)

    # -- the tick --

    def tick(self) -> None:
        """Advance every held ped by one step and publish the full claim set, stamped
        from the node's own clock: paused sim -> dt 0 -> everything freezes."""
        now = self._node.get_clock().now()
        now_sim = now.nanoseconds * 1e-9
        dt = max(0.0, now_sim - self._last_tick_sim_s) if self._last_tick_sim_s is not None else 0.0
        self._last_tick_sim_s = now_sim
        self._engine_time_s += dt
        now_wall = time.monotonic()

        self._drain_peds()
        self._drain_episode()
        self._drain_cmd_vel()

        held = self._intents.held()
        if _empty_manifest_due(self._prev_held_count, len(held)):
            release = Pedestrians()
            release.header.frame_id = "map"
            release.header.stamp = now.to_msg()
            self._pub.publish(release)
            self._publish_times.append(now_wall)
        self._prev_held_count = len(held)

        if not held:
            return

        out = Pedestrians()
        out.header.frame_id = "map"
        out.header.stamp = now.to_msg()
        peds = []
        for name, intent in held:
            self._intents.expire_teleop(name, now_wall)
            peds.append(self._step_ped(name, intent, dt, now_wall))
        out.pedestrians = peds

        self._pub.publish(out)
        self._publish_times.append(now_wall)

    def _step_ped(self, name: str, intent: Intent, dt: float, now_wall: float) -> Pedestrian:
        x, y, yaw = self._intents.pose(name) or (0.0, 0.0, 0.0)
        speed = 0.0

        if intent.mode == "waypoints" and intent.waypoint_cursor < len(intent.waypoints):
            nx, ny, nyaw, cursor = integrate.advance_waypoints(
                x,
                y,
                yaw,
                intent.waypoints,
                intent.waypoint_cursor,
                intent.speed,
                dt,
                intent.loop,
            )
            intent.waypoint_cursor = cursor
            speed = math.hypot(nx - x, ny - y) / dt if dt > 0.0 else 0.0
            x, y, yaw = nx, ny, nyaw
        elif intent.mode == "teleop":
            vx, vy, wz = intent.teleop_twist
            x, y, yaw = integrate.teleop_step(x, y, yaw, vx, vy, wz, dt)
            speed = math.copysign(math.hypot(vx, vy), vx)
        self._intents.set_pose(name, x, y, yaw)

        state = intent.state_override if intent.state_override is not None else auto_state(abs(speed))

        agent_id = self._roster_ids.get(name, 0)
        gait_gen = self._gait.setdefault(name, GaitGenerator())
        gait_angles = gait_gen.compute(agent_id, state, speed, dt)

        clip_angles: dict[str, float] = {}
        if intent.clip is not None:
            library = self._clip_library_for(name)
            clip = library.get(intent.clip.name)
            if clip is not None:
                t = (self._engine_time_s - intent.clip.t0_engine) * intent.clip.speed_scale
                sampled = clips.sample(clip, t)
                elapsed_in = self._engine_time_s - intent.clip.t0_engine
                clip_angles = clips.blend(intent.clip.blend_from, sampled, elapsed_in)
        elif intent.clip_release is not None:
            # blend-out: ease the last clip pose back toward this tick's gait fallback.
            release_values, stop_time = intent.clip_release
            elapsed_out = self._engine_time_s - stop_time
            if elapsed_out >= clips.BLEND_S:
                intent.clip_release = None
            else:
                clip_angles = clips.blend(release_values, gait_angles, elapsed_out)

        gaze_angles: dict[str, float] = {}
        if intent.gaze is not None:
            tx, ty, tz = intent.gaze
            y_head, p_head = compose.solve_gaze(x, y, yaw, tx, ty, tz)
            gaze_angles = {"y_head": y_head, "p_head": p_head}

        names = GaitGenerator.JOINT_NAMES
        composed = compose.compose(names, slider=intent.posed, gaze=gaze_angles, clip=clip_angles, gait=gait_angles)
        if not self._unlimited:
            composed = {n: _clamp(v, *LIMITS[i]) for i, (n, v) in enumerate(composed.items())}
        self._last_composed[name] = composed
        return self._build_pedestrian(name, x, y, yaw, speed, state, composed)

    def _build_pedestrian(
        self,
        name: str,
        x: float,
        y: float,
        yaw: float,
        speed: float,
        state: int,
        joint_angles: dict[str, float],
    ) -> Pedestrian:
        ped = Pedestrian()
        ped.name = name
        ped.id = self._roster_ids.get(name, 0)
        ped.pose.position.x = x
        ped.pose.position.y = y
        qz, qw = _yaw_to_quat_zw(yaw)
        ped.pose.orientation.z = qz
        ped.pose.orientation.w = qw
        ped.twist.linear.x = speed * math.cos(yaw)
        ped.twist.linear.y = speed * math.sin(yaw)
        ped.animation_state = state
        names = list(joint_angles.keys())
        ped.joint_state.name = names
        ped.joint_state.position = [joint_angles[n] for n in names]
        return ped

    def close(self) -> None:
        """Destroy every publisher/subscription/client. Call on plugin shutdown."""
        self._node.destroy_publisher(self._pub)
        self._node.destroy_client(self._move_client)
        self._node.destroy_subscription(self._sub_peds)
        self._node.destroy_subscription(self._sub_episode)
        for sub in self._cmd_vel_subs.values():
            self._node.destroy_subscription(sub)
        self._cmd_vel_subs.clear()


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
