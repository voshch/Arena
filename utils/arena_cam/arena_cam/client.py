"""ROS 2 client for the Arena viewport cameras.

`CamNode` fans a `Camera` out across one or more viewport surfaces: the sim GUI
camera at `/arena/viewport/*` and any number of per-env rviz cameras at
`/arena/env_<id>/arena_env_node/viewport/*`. Each camera of a take is a
`CameraTake`, one per rig camera or a lone one. The `Camera` facade authors the
shot in world coordinates. Each endpoint subtracts its env's world origin (the
registry `reference`) so the same absolute shot lands at the matching place in every
env. That localization stops once a reference frame is set: the camera pose is then
composed as reference * local, so poses are relative to the reference and only the
reference pose itself is localized. The sim endpoint has a zero offset.

It runs via `run_main`: `setup` discovers the selected endpoints, plays the timeline,
then shuts down. A segment drives three ways. LIVE: stream keyframes on `cmd_view`,
paced by wall-clock. LIVE LOCKSTEP: walk the segment at a fixed fps, stream each
frame once the sim reached it. RECORD: walk it at a fixed fps and `capture` each frame
synchronously, so the output is deterministic. Each camera records to its own file
per surface, and `_frame` gates the sim once for all of them.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import functools
import typing
from pathlib import Path

import rclpy
from arena_people_msgs.msg import FaceView
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.Time import Time
from arena_runtime.lockstep import register_channels
from arena_runtime_msgs.msg import (
    EnvRegistry,
    LockstepChannel,
    LockstepHeartbeat,
    LockstepStatus,
)
from arena_runtime_msgs.srv import LifecycleHold, LifecycleStep
from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import Point, PoseStamped
from hri_msgs.msg import IdsList
from rclpy.duration import Duration
from viewport_control_msgs.msg import ViewportView
from viewport_control_msgs.srv import (
    ViewportCapture,
    ViewportSetProjection,
    ViewportSetReferenceFrame,
    ViewportSetView,
)

from . import context, curves, faces, surfaces
from .context import EnvContext
from .curves import FOV_DEFAULT, Quat, Vec3
from .record import Recorder, claim, take_paths
from .shots import RigCamera
from .surfaces import TargetSelection

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    import rclpy.node
    import rclpy.subscription
    from arena_rclpy_mixins.Async import ClientWrapper
    from rclpy.impl.rcutils_logger import RcutilsLogger

    from .camera import Camera
    from .mounts import Mount

    # A frame sampler: eased progress in [0, 1] -> (position, quat, fov).
    Frame = Callable[[float], tuple[Vec3, Quat, float]]


class Steered(typing.NamedTuple):
    """One frame of live input. `referenced` marks the pose as relative to a set reference frame."""

    position: Vec3
    quat: Quat
    fov: float
    referenced: bool
    clip_near: float = 0.0


# cmd_view publish rate for streamed segments (Hz, wall-clock LIVE mode).
_FRAME_RATE = 60.0

# Keyframes are stamped this far ahead, so the plugin's buffer rides out publish
# stalls up to this long. The cost is this much added view latency, which the
# interactive driver trades away for responsiveness (see drive.DRIVE_LEAD).
LEAD = 0.3

FACE_TIMEOUT = 15.0

CAPTURE_ATTEMPTS = 3

_BEAT_MATCH_TIMEOUT = 5.0

# A fresh node discovers the GUI processes one by one, measured up to 1.5 s apart,
# so the target set counts as complete once it has stood still this long.
_DISCOVERY_QUIET = 2.0


class _Endpoint:
    """One viewport surface: its namespace, its world->local offset, and ROS handles."""

    def __init__(self, node: CamNode, ns: str, offset: tuple[float, float]) -> None:
        self.ns = ns
        self._ox, self._oy = offset
        self.set_view = node.create_client_wrapper(ViewportSetView, f"{ns}/viewport/set_view", timeout=10.0)
        self.set_reference = node.create_client_wrapper(ViewportSetReferenceFrame, f"{ns}/viewport/set_reference_frame", timeout=10.0)
        self.set_projection = node.create_client_wrapper(ViewportSetProjection, f"{ns}/viewport/set_projection", timeout=10.0)
        # Generous timeout: a capture round-trips a full rendered frame.
        self.capture = node.create_client_wrapper(ViewportCapture, f"{ns}/viewport/capture", timeout=30.0)
        self.cmd_view = node.create_publisher(ViewportView, f"{ns}/viewport/cmd_view", surfaces.STREAM_QOS)
        self._pose: tuple[Vec3, Quat] | None = None
        self.reference: ViewportSetReferenceFrame.Request | None = None
        node.create_subscription(PoseStamped, f"{ns}/viewport/camera_pose", self._on_pose, 10)

    def _on_pose(self, msg: PoseStamped) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        self._pose = ((p.x, p.y, p.z), (q.w, q.x, q.y, q.z))

    def localize(self, position: Vec3) -> Vec3:
        """World coords -> this env's local frame (pure planar offset, no rotation)."""
        return surfaces.localize((self._ox, self._oy), position)

    def world_pose(self) -> tuple[Vec3, Quat] | None:
        """Latest camera pose lifted back into world coords, or None if not yet seen."""
        if self._pose is None:
            return None
        (x, y, z), q = self._pose
        return ((x + self._ox, y + self._oy, z), q)


class FaceWatch:
    """A face being looked through, holding the subscriptions that keep its frames published."""

    def __init__(self, node: rclpy.node.Node, face_id: str, view: FaceView, subscriptions: list[rclpy.subscription.Subscription]) -> None:
        self.frame = faces.frame(face_id)
        self.fov = faces.fov(view)
        self.clip_near = view.clip_near
        self._node = node
        self._subscriptions = subscriptions

    def close(self) -> None:
        for sub in self._subscriptions:
            self._node.destroy_subscription(sub)
        self._subscriptions = []


class FaceWatches:
    """The face each surface of a camera looks through, `frames` keyed by surface namespace."""

    def __init__(self, watches: dict[str, FaceWatch]) -> None:
        self._watches = watches
        first = next(iter(watches.values()))
        self.frames = {ns: watch.frame for ns, watch in watches.items()}
        self.fov = first.fov
        self.clip_near = first.clip_near

    def close(self) -> None:
        for watch in self._watches.values():
            watch.close()


def frame_floor(last: Time | None, period: Time) -> Time | None:
    """Earliest sim time of the next lockstep frame: one period after the last beat, where the scheduler next freezes on the cam."""
    return None if last is None else last + period


@dataclasses.dataclass(frozen=True)
class CamFrame:
    """One camera's pose for one frame of a take."""

    position: Vec3
    quat: Quat
    world_orientation: bool
    fov: float
    clip_near: float


def default_reference() -> ViewportSetReferenceFrame.Request:
    """The identity reference an unreferenced camera films in, explicit since an empty request latches."""
    req = ViewportSetReferenceFrame.Request()
    req.has_pose = True
    req.pose = surfaces.ros_pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    req.mode = surfaces.REFERENCE_MODES["full"]
    return req


class CameraTake:
    """One camera of a running take: the verb surface of its timeline, its reference per surface and its recorders."""

    def __init__(self, node: CamNode, camera: RigCamera, endpoints: list[_Endpoint]) -> None:
        self.node = node
        self.camera = camera
        self.endpoints = endpoints
        # 0 keeps each camera's near clip, < 0 restores its default
        self.clip_near = 0.0
        # True once a reference frame is set: from then on poses are relative to it and
        # must not be localized. Until then the reference is identity and the env offset
        # is what maps an absolute shot into each env.
        self.referenced = False
        self.frame: CamFrame | None = None
        self.recorders: dict[str, Recorder] = {}
        self.references = {endpoint.ns: default_reference() for endpoint in endpoints}

    def tag(self, ns: str) -> str:
        """File tag of this camera on surface `ns`: the surface alone for a lone camera, else the camera's name."""
        return surfaces.camera_tag(self.camera.name, ns, len(self.endpoints))

    def ok(self) -> bool:
        return self.node.ok()

    def get_logger(self) -> RcutilsLogger:
        return self.node.get_logger()

    def camera_pose(self) -> tuple[Vec3, Quat] | None:
        """A representative camera world pose to seed the cursor, or None if not yet seen."""
        for endpoint in self.endpoints:
            pose = endpoint.world_pose()
            if pose is not None:
                return pose
        return None

    def local(self, endpoint: _Endpoint, position: Vec3) -> Vec3:
        """World coords -> endpoint-local, unless a reference is set (poses are relative to it)."""
        return position if self.referenced else endpoint.localize(position)

    async def prefill(self, text: str) -> None:
        """Read what `text`'s context tokens hold while the sim still runs, before a hold freezes the rosters."""
        for endpoint in self.endpoints:
            await self.node.fill(text, endpoint)

    async def prepare_mount(self, mount: Mount) -> None:
        position, quat = mount.local_pose()
        await self.set_reference(mount.frame, None, "full")
        self.frame = CamFrame(position, quat, False, mount.fov, -1.0)

    async def look(self, eye: Vec3, target: Vec3, fov: float = 0.0) -> bool:
        if self.node.recording:
            return await self.hand(eye, curves.look_at_quat(eye, target), False, fov)
        ok = True
        for endpoint in self.endpoints:
            req = ViewportSetView.Request()
            eye_local, target_local = self.local(endpoint, eye), self.local(endpoint, target)
            req.eye = Point(x=float(eye_local[0]), y=float(eye_local[1]), z=float(eye_local[2]))
            req.target = Point(x=float(target_local[0]), y=float(target_local[1]), z=float(target_local[2]))
            req.fov = float(fov)
            req.clip_near = float(self.clip_near)
            ok = await self.node.call(endpoint.set_view, req) and ok
        return ok

    async def set_reference(self, entity: str = "", pose: tuple[Vec3, Quat] | None = None, mode: str = "full", frames: Mapping[str, str] | None = None) -> bool:
        """Set this camera's reference, at once on a surface it has alone, at its next capture on a shared one, where a latch falls back to the world frame."""
        ok = True
        for endpoint in self.endpoints:
            req = ViewportSetReferenceFrame.Request()
            req.entity = await self.node.fill(frames[endpoint.ns] if frames is not None else entity, endpoint)
            req.has_pose = pose is not None
            if pose is not None:
                # The reference pose is authored in world coords, so it localizes once here.
                req.pose = surfaces.ros_pose(endpoint.localize(pose[0]), pose[1])
            elif self.node.shared and not req.entity:
                req = default_reference()
            req.mode = surfaces.REFERENCE_MODES[mode]
            self.references[endpoint.ns] = req
            if not self.node.shared:
                endpoint.reference = req
                ok = await self.node.call(endpoint.set_reference, req) and ok
        self.referenced = True
        return ok

    async def set_projection(self, projection: str) -> bool:
        return await self.node.set_projection(projection, self.endpoints)

    def stream(self, position: Vec3, quat: Quat, world_orientation: bool = False, fov: float = 0.0) -> None:
        if not rclpy.ok():
            return
        stamp = self.node.keyframe_stamp()
        for endpoint in self.endpoints:
            msg = ViewportView()
            msg.target_time = stamp
            msg.pose = surfaces.ros_pose(self.local(endpoint, position), quat)
            msg.world_orientation = bool(world_orientation)
            msg.fov = float(fov)
            msg.clip_near = float(self.clip_near)
            endpoint.cmd_view.publish(msg)

    async def hand(self, position: Vec3, quat: Quat, world_orientation: bool, fov: float) -> bool:
        """Hand this camera's next frame to the node and wait until every camera's is filmed."""
        return await self.node.hand(self, CamFrame(position, quat, world_orientation, fov, self.clip_near))

    async def drive(self, duration: float, world_orientation: bool, frame_at: Frame, span: tuple[float, float] = (0.0, 1.0)) -> None:
        """Play the `span` part of one segment by sampling `frame_at(t)` over t in [0, 1].

        A framed take walks a fixed `duration * fps` frames, each one captured or gated
        synchronously, so timing is exact. Otherwise keyframes stream on `cmd_view` paced
        by wall-clock, so a starved step jumps to the right point rather than running
        the whole move in slow motion.
        """
        first, last = span
        width = last - first
        if self.node.framed:
            frames = max(1, round(duration * width * self.node.fps))
            for i in range(frames):
                if not rclpy.ok():
                    return
                pos, quat, fov = frame_at(first + width * (i + 1) / frames)
                if not await self.hand(pos, quat, world_orientation, fov):
                    return
            return
        length = duration * width
        if length <= 0.0:
            pos, quat, fov = frame_at(last)
            self.stream(pos, quat, world_orientation, fov)
            return
        loop = asyncio.get_running_loop()
        start = loop.time()
        period = 1.0 / _FRAME_RATE
        while rclpy.ok():
            t = min(1.0, (loop.time() - start) / length)
            pos, quat, fov = frame_at(first + width * t)
            self.stream(pos, quat, world_orientation, fov)
            if t >= 1.0:
                break
            await asyncio.sleep(period)

    async def steer(self, next_frame: Callable[[float], Steered | None]) -> None:
        """Record an open-ended segment: `next_frame(dt)` advances the live input one frame period, None ends it."""
        while rclpy.ok():
            frame = next_frame(1.0 / self.node.fps)
            if frame is None:
                return
            self.referenced = frame.referenced
            self.clip_near = frame.clip_near
            if not await self.hand(frame.position, frame.quat, False, frame.fov):
                return

    async def watch_faces(self, face: str) -> FaceWatches | None:
        """Watch the face each surface looks through, one watch per distinct face, None when one is not found."""
        names = {endpoint.ns: await self.node.fill(face, endpoint) for endpoint in self.endpoints}
        watches: dict[str, FaceWatch] = {}
        for name in set(names.values()):
            watch = await self.node.watch_face(name)
            if watch is None:
                for found in watches.values():
                    found.close()
                return None
            watches[name] = watch
        return FaceWatches({ns: watches[name] for ns, name in names.items()})


class CamNode(ArenaMixinNode):
    """Standalone node that plays a `Camera` timeline against the selected viewports."""

    channel_env = ""
    ride_only = False

    def __init__(
        self,
        *,
        timeline: Camera | None,
        targets: TargetSelection,
        node_name: str = "arena_cam",
        record: tuple[str, float] | None = None,
        force: bool = False,
        lead: float = LEAD,
        lockstep: bool = False,
        fps: float = 30.0,
        mounts: Sequence[Mount] = (),
        env: int = 0,
    ) -> None:
        super().__init__(node_name)
        self._timeline = timeline
        self._selection = targets
        self.lead = lead
        self._record = Path(record[0]) if record is not None else None
        self.fps = record[1] if record is not None else fps
        self._force = force
        self._lockstep = lockstep
        self._mounts = list(mounts)
        self._env = env
        self._contexts: dict[int, EnvContext] = {}
        self._off_stamp = 0
        self._off_most = 0.0
        self._env_refs: dict[int, tuple[float, float]] = {}
        self._endpoints: list[_Endpoint] = []
        self._takes: list[CameraTake] = []
        self.shared = False
        self._running: set[CameraTake] = set()
        self._arrived: set[CameraTake] = set()
        self._turn: asyncio.Future[bool] | None = None
        self._subscribed = False
        self._settled = False
        self._hold_client: ClientWrapper | None = None
        self._step_client: ClientWrapper | None = None
        self._lockstep_time: Time | None = None
        # Follower mode: an active run owns the stepping, so the cam rides it as a
        # registered hard channel instead of taking its own hold.
        self._scheduler_active = False
        self._run_paused = False
        self._follower = False
        self._beat_pub = None
        self._error: str | None = None
        self.ran = False

    @property
    def recording(self) -> bool:
        return self._record is not None

    @property
    def framed(self) -> bool:
        """True when a take advances frame by frame instead of on the wall clock."""
        return self.recording or self._lockstep

    async def setup(self) -> None:
        self.ran = await self.take()
        rclpy.try_shutdown()

    async def take(self) -> bool:
        """Connect, run the timeline, close the files. True when the shot ran and every recording finished usable, a host with its own context awaits this directly."""
        assert self._timeline is not None
        return await self.connect() and await self.play(self._timeline)

    async def connect(self) -> bool:
        """Find the selected viewport surfaces. True when at least one answers."""
        if not self._subscribed:
            self._subscribed = True
            self.create_subscription(EnvRegistry, surfaces.ENVS_TOPIC, self._on_envs, surfaces.ENVS_QOS)
            self.create_subscription(LockstepStatus, "/arena/state/lockstep", self._on_lockstep_status, surfaces.ENVS_QOS)
        found = await self._await_endpoints()
        if not found:
            self.get_logger().error("no viewport targets found, is the sim GUI / rviz up? (headless sim has none)")
            return False
        known = {endpoint.ns: endpoint for endpoint in self._endpoints}
        reachable: list[_Endpoint] = []
        for ns, offset in found:
            endpoint = known.get(ns) or _Endpoint(self, ns, offset)
            if await endpoint.set_view.ensure(timeout_sec=10.0):
                reachable.append(endpoint)
            else:
                self.get_logger().warning(f"{endpoint.ns}/viewport did not answer, skipping")
        self._endpoints = reachable
        self._settled = bool(reachable)
        if not self._endpoints:
            self.get_logger().error("no reachable viewport targets")
            return False
        return True

    def _surfaces_of(self, camera: RigCamera) -> list[_Endpoint]:
        chosen = surfaces.by_source(camera.source, [endpoint.ns for endpoint in self._endpoints])
        return [endpoint for endpoint in self._endpoints if endpoint.ns in chosen]

    def _make_takes(self, timeline: Camera) -> list[CameraTake] | None:
        cameras = [*timeline.cameras(recording=self.recording), *(RigCamera(mount.tag, source="sim", mount=mount) for mount in self._mounts)]
        rigged = len(cameras) > 1
        if rigged and not self._lockstep:
            self.get_logger().error("mounts and rigs record in lockstep, pass lockstep")
            return None
        takes = [CameraTake(self, camera, self._surfaces_of(camera)) for camera in cameras]
        lost = [take.camera.name for take in takes if not take.endpoints]
        if lost:
            self.get_logger().error(f"cameras {lost} select no surface, mounts record from the sim camera, select it")
            return None
        return takes

    async def play(self, timeline: Camera) -> bool:
        """Run one take on the connected surfaces. True when it ran and every recording finished usable."""
        self._timeline, self._error, self._off_stamp, self._off_most = timeline, None, 0, 0.0
        takes = self._make_takes(timeline)
        if takes is None:
            return False
        self._takes, self.shared = takes, len(takes) > 1
        if self.ride_only:
            self._lockstep = self._scheduler_active
        if self._record is not None and not await self._open_recorders():
            return False
        names = ", ".join(endpoint.ns for endpoint in self._endpoints)
        self.get_logger().info(f"viewport connected ({names}), {'recording' if self._record else 'playing'} shot")
        await self.arrives(f"{self._endpoints[0].ns}/viewport/camera_pose", PoseStamped)  # seed the cursor
        await self._start_state(timeline)
        try:
            if self._lockstep and self._scheduler_active:
                await self._run_follower()
            elif self._lockstep:
                await self._run_lockstep()
            else:
                await self._run_takes()
            if self.shared and self._error is None:
                await self._hand_back()
        finally:
            self._close_recorders()
        if rclpy.ok() and self._record is None:
            self.get_logger().info("shot complete")
        return self._error is None

    async def _start_state(self, timeline: Camera) -> None:
        """Everything a take needs from the running sim before a hold freezes it."""
        if self.ride_only:
            for endpoint in self._endpoints:
                endpoint.reference = default_reference()
                await self.call(endpoint.set_reference, endpoint.reference)
        if timeline.projection:
            await self.set_projection(timeline.projection, self._endpoints)
        for take in self._takes:
            if take.camera.mount is not None:
                await take.prepare_mount(take.camera.mount)
            await timeline.prepare(take)

    async def _hand_back(self) -> None:
        """Leave each surface on the first camera's reference, not on whichever filmed last."""
        first = self._takes[0]
        for endpoint in first.endpoints:
            await self._show(first, endpoint)

    async def _open_recorders(self) -> bool:
        """One ffmpeg per camera and surface. A lone camera keeps the file name, several get tags."""
        assert self._record is not None
        pairs = [(take, endpoint) for take in self._takes for endpoint in take.endpoints]
        paths = take_paths(self._record, [take.tag(endpoint.ns) for take, endpoint in pairs])
        try:
            claim(paths, self._force)
        except FileExistsError as e:
            self.get_logger().error(str(e))
            return False
        for endpoint in {endpoint.ns: endpoint for _take, endpoint in pairs}.values():
            if not await endpoint.capture.ensure(timeout_sec=10.0):
                self.get_logger().error(f"no {endpoint.ns}/viewport/capture service, rebuild the plugin for record mode")
                return False
        for (take, endpoint), path in zip(pairs, paths, strict=True):
            take.recorders[endpoint.ns] = Recorder(str(path), self.fps)
            self.get_logger().info(f"recording {take.tag(endpoint.ns)} to {path}")
        return True

    def _close_recorders(self) -> None:
        for take in self._takes:
            for ns, recorder in take.recorders.items():
                tag = take.tag(ns)
                if recorder.close() and self._error is None:
                    self.get_logger().info(f"saved {tag} recording to {recorder.path} ({recorder.n} frames)")
                else:
                    self.fail(f"{tag} recording failed after {recorder.n} frames, {recorder.path} is not usable")
            take.recorders = {}
        if self.shared and self._record is not None:
            self.get_logger().info(f"camera frames off the first camera's sim time: {self._off_stamp}, by up to {self._off_most * 1e3:.1f} ms")

    def fail(self, message: str) -> None:
        """Log a take failure and stop the take, the first failure is what the take reports."""
        self.get_logger().error(message)
        if self._error is None:
            self._error = message

    def _on_lockstep_status(self, msg: LockstepStatus) -> None:
        if self._follower and self._scheduler_active and not msg.active:
            self.get_logger().warning("lockstep run ended mid-take, remaining frames are not gated")
        self._scheduler_active = bool(msg.active)
        self._run_paused = bool(msg.active and msg.paused)

    async def _await_resumed(self) -> None:
        """Hold the take while the ridden lockstep run is paused."""
        if not self._run_paused:
            return
        self.get_logger().info("lockstep run paused, take holding until resume")
        while self._run_paused and rclpy.ok():
            await asyncio.sleep(0.2)

    async def _run_follower(self) -> None:
        """Ride the active lockstep run instead of driving the sim: register a hard
        cam channel at 1/fps, film the tick the run froze on for the cam, then
        pulse that stamp, so frames are gate-exact with every producer's window data arrived."""
        topic = f"{self.get_fully_qualified_name()}/lockstep"
        channel = LockstepChannel(
            name="cam",
            topic=topic,
            type="arena_runtime_msgs/msg/LockstepHeartbeat",
            period_s=1.0 / self.fps,
            hard=True,
        )
        if not await register_channels(self, [channel], env=self.channel_env, caller=topic, timeout=60.0):
            if self.ride_only:
                self.get_logger().warning("cam channel registration failed, playing on the wall clock")
                self._lockstep = False
                await self._run_takes()
                return
            self.get_logger().warning("cam channel registration failed, falling back to hold-and-step")
            await self._run_lockstep()
            return
        if self._beat_pub is None:
            self._beat_pub = self.create_publisher(LockstepHeartbeat, topic, 10)
        self._follower = True
        self._lockstep_time = await self._first_beat(topic)
        self.get_logger().info(f"riding active lockstep run: cam gated at {self.fps:g} fps")
        try:
            await self._run_takes()
        finally:
            self._follower = False
            await register_channels(self, [], env=self.channel_env, caller=topic, timeout=60.0)

    async def _first_beat(self, topic: str) -> Time | None:
        """Beat the current tick so even the first frame lands on a tick the run froze on, None when the scheduler never subscribed."""
        assert self._beat_pub is not None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _BEAT_MATCH_TIMEOUT
        while self.count_subscribers(topic) == 0:
            if loop.time() > deadline:
                return None
            await asyncio.sleep(0.01)
        stamp = self.sim_time
        beat = LockstepHeartbeat()
        beat.header.stamp = stamp.to_msg()
        self._beat_pub.publish(beat)
        return stamp

    async def _run_lockstep(self) -> None:
        """Acquire a sim hold and run the timeline with physics stepped 1/fps per frame."""
        self._hold_client = self.create_client_wrapper(LifecycleHold, "/arena/sim_lifecycle/hold")
        self._step_client = self.create_client_wrapper(LifecycleStep, "/arena/sim_lifecycle/step")
        req = LifecycleHold.Request()
        req.action = LifecycleHold.Request.ACQUIRE
        req.caller_id = self.get_fully_qualified_name()
        req.reason = "record"
        try:
            if await self._hold_client.call_timeout(req) is None:
                self.fail("sim hold timed out, nothing recorded")
                return
            self._lockstep_time = await self._held_sim_time()
            await self._run_takes()
        finally:
            rel = LifecycleHold.Request()
            rel.action = LifecycleHold.Request.RELEASE
            rel.caller_id = self.get_fully_qualified_name()
            rel.reason = "record"
            await self._hold_client.call_timeout(rel)

    async def _held_sim_time(self) -> Time:
        """The held sim's time once /clock stands still, the sample arena_runtime_node steps from too."""
        seen = self.sim_time
        while rclpy.ok():
            await asyncio.sleep(0.2)
            if self.sim_time == seen:
                break
            seen = self.sim_time
        return seen

    async def _run_takes(self) -> None:
        """Run every timeline camera, each frame waiting for all of them. Mount cameras have no timeline and film every frame."""
        assert self._timeline is not None
        timeline = self._timeline
        running = [take for take in self._takes if take.camera.mount is None]
        self._running, self._arrived, self._turn = set(running), set(), None

        async def run(take: CameraTake) -> None:
            await timeline.run(take)
            await self._finished(take)

        await asyncio.gather(*(run(take) for take in running))

    async def hand(self, take: CameraTake, frame: CamFrame) -> bool:
        """Take `take`'s next frame, the last camera to arrive films them all. True when the frame was filmed."""
        take.frame = frame
        self._arrived.add(take)
        if self._running <= self._arrived:
            return await self._flush()
        if self._turn is None:
            self._turn = asyncio.get_running_loop().create_future()
        return await self._turn

    async def _finished(self, take: CameraTake) -> None:
        """Drop an ended camera from the barrier, it keeps its last frame until the longest camera ends."""
        self._running.discard(take)
        self._arrived.discard(take)
        if self._running and self._running <= self._arrived:
            await self._flush()

    async def _flush(self) -> bool:
        self._arrived = set()
        turn, self._turn = self._turn, None
        ok = await self._frame()
        if turn is not None:
            turn.set_result(ok)
        return ok

    async def _gate(self) -> tuple[bool, RosTime | None]:
        """Advance the sim to this frame where lockstep asks for it. (False, _) stops the take, else the sim time the frame must show."""
        if self._follower:
            await self._await_resumed()
            if self._lockstep_time is None:
                await self._held_sim_time()
            floor = frame_floor(self._lockstep_time, Time.from_float(1.0 / self.fps))
            if not self.recording:
                await self._await_floor(floor)
            return True, None if floor is None else floor.to_msg()
        if not self._lockstep:
            return True, None
        step_req = LifecycleStep.Request()
        step_req.seconds = 1.0 / self.fps
        assert self._step_client is not None and self._lockstep_time is not None
        step_res = await self._step_client.call_timeout(step_req)
        if step_res is None or not step_res.success:
            detail = "service timed out" if step_res is None else step_res.error_msg
            self.fail(f"lockstep step failed ({detail}), stopping take")
            return False, None
        self._lockstep_time = self._lockstep_time + Time.from_float(step_res.advanced)
        return True, self._lockstep_time.to_msg()

    async def _await_floor(self, floor: Time | None) -> None:
        """Wait for the run to reach `floor`, a live frame has no capture to defer. Paced by fps once the run ended."""
        while floor is not None and rclpy.ok() and self._scheduler_active and self.sim_time < floor:
            await self._await_resumed()
            await asyncio.sleep(0.002)
        if not self._scheduler_active:
            await asyncio.sleep(1.0 / self.fps)

    async def _frame(self) -> bool:
        """Gate the sim once, then film every camera's frame at that sim time."""
        if self._error is not None:
            return False
        gated, min_sim_time = await self._gate()
        if not gated:
            return False
        if self.recording:
            stamps = await self._capture_frames(min_sim_time)
            if stamps is None:
                return False
        else:
            self._stream_frames()
            stamps = [self.sim_time.to_msg()]
        if self._follower:
            assert self._beat_pub is not None
            self._lockstep_time = max(Time.from_msg(stamp) for stamp in stamps)
            beat = LockstepHeartbeat()
            beat.header.stamp = self._lockstep_time.to_msg()
            self._beat_pub.publish(beat)
        return True

    def _stream_frames(self) -> None:
        for take in self._takes:
            frame = take.frame
            if frame is not None:
                take.stream(frame.position, frame.quat, frame.world_orientation, frame.fov)

    async def _show(self, take: CameraTake, endpoint: _Endpoint) -> bool:
        """Put `take`'s reference on a shared surface, a no-op when the surface already holds it."""
        req = take.references[endpoint.ns]
        if endpoint.reference is req:
            return True
        endpoint.reference = req
        if await self.call(endpoint.set_reference, req):
            return True
        self.fail(f"{take.tag(endpoint.ns)} not referenced, stopping record")
        return False

    async def _capture_frames(self, min_sim_time: RosTime | None) -> list[RosTime] | None:
        """Capture each camera's frame through its surfaces, later cameras at the sim time the first capture of the surface showed."""
        stamps: list[RosTime] = []
        floors: dict[str, RosTime] = {}
        for take in self._takes:
            frame = take.frame
            if frame is None:
                continue
            if self.shared:
                for endpoint in take.endpoints:
                    if not await self._show(take, endpoint):
                        return None
            if frame.fov <= 0.0 and len(take.endpoints) > 1:
                frame = dataclasses.replace(frame, fov=FOV_DEFAULT)  # left open, each camera would film through its own lens
            shots = await asyncio.gather(*(self._grab(take, endpoint, frame, floors.get(endpoint.ns, min_sim_time)) for endpoint in take.endpoints))
            for endpoint, stamp in zip(take.endpoints, shots, strict=True):
                if stamp is None:
                    return None
                first = floors.setdefault(endpoint.ns, stamp)
                off = abs(stamp.sec - first.sec + (stamp.nanosec - first.nanosec) * 1e-9)
                if off > 0.0:
                    self._off_stamp += 1
                    self._off_most = max(self._off_most, off)
                stamps.append(stamp)
        return stamps

    async def capture(self, take: CameraTake, endpoint: _Endpoint, frame: CamFrame, min_sim_time: RosTime | None = None) -> ViewportCapture.Response | None:
        req = ViewportCapture.Request()
        req.pose = surfaces.ros_pose(take.local(endpoint, frame.position), frame.quat)
        req.world_orientation = bool(frame.world_orientation)
        req.fov = float(frame.fov)
        req.clip_near = float(frame.clip_near)
        if min_sim_time is not None:
            req.min_sim_time = min_sim_time
        try:
            return await endpoint.capture.call_timeout(req)
        except Exception as e:
            self.get_logger().warning(f"capture call failed: {e}")
            return None

    async def _grab(self, take: CameraTake, endpoint: _Endpoint, frame: CamFrame, min_sim_time: RosTime | None) -> RosTime | None:
        attempt = 0
        while True:
            res = await self.capture(take, endpoint, frame, min_sim_time)
            if res is not None and res.success:
                break
            if self._follower and self._run_paused:
                # the run paused mid-frame and the deferred capture hit its
                # deadline, retry the same tick once the run resumes
                await self._await_resumed()
                continue
            attempt += 1
            detail = "service timed out" if res is None else res.message
            if attempt < CAPTURE_ATTEMPTS:
                self.get_logger().warning(f"{endpoint.ns} capture failed ({detail}), retrying the frame ({attempt}/{CAPTURE_ATTEMPTS})")
                continue
            self.fail(f"{endpoint.ns} capture failed {CAPTURE_ATTEMPTS} times ({detail}), stopping record")
            return None
        try:
            take.recorders[endpoint.ns].write(res.image)
        except (ValueError, OSError) as e:
            self.fail(f"{endpoint.ns} frame not encoded ({e}), stopping record")
            return None
        return res.image.header.stamp

    def _on_envs(self, msg: EnvRegistry) -> None:
        self._env_refs = surfaces.env_refs(msg)

    async def _await_endpoints(self) -> list[tuple[str, tuple[float, float]]]:
        """Wait for the selected viewport surfaces, then until discovery and the env table stop adding to them."""
        found: list[tuple[str, tuple[float, float]]] = []
        waited = quiet = 0.0
        while rclpy.ok():
            latest = self._find_targets()
            quiet = quiet + 0.5 if latest == found else 0.0
            found = latest
            if found and (self._settled or quiet >= _DISCOVERY_QUIET):
                return found
            await asyncio.sleep(0.5)
            waited += 0.5
            if not found and waited >= 10.0 and (waited % 10.0) < 0.5:
                self.get_logger().warning(f"arena cam: waiting for viewport targets ({waited:.0f}s elapsed)")
        return []

    def _find_targets(self) -> list[tuple[str, tuple[float, float]]]:
        names = [name for name, _types in self.get_service_names_and_types()]
        return surfaces.find_targets(names, self._selection, self._env_refs)

    async def fill(self, text: str, endpoint: _Endpoint) -> str:
        """`text` with its context tokens filled for `endpoint`'s env, an unfillable token fails the take and leaves an empty string."""
        if not context.tokens(text):
            return text
        env_id = surfaces.env_id_from_ns(endpoint.ns)
        env_id = self._env if env_id is None else env_id
        env = self._contexts.setdefault(env_id, EnvContext(self, env_id))
        try:
            return await env.fill(text)
        except LookupError as e:
            self.fail(f"cam: {e.args[0]}")
            return ""

    async def watch_face(self, face: str, timeout: float = FACE_TIMEOUT) -> FaceWatch | None:
        """Wake the face rosters, resolve `face` on them and subscribe its view, None after `timeout` s."""
        rosters: dict[str, list[str]] = {}

        def on_roster(topic: str, msg: IdsList) -> None:
            rosters[topic] = list(msg.ids)

        subs: dict[str, rclpy.subscription.Subscription] = {}
        views: collections.deque[FaceView] = collections.deque(maxlen=1)

        def on_view(msg: FaceView) -> None:
            views.append(msg)

        target: tuple[str, str] | None = None
        error = f"no face roster answered within {timeout:.0f} s, is hri_producer running?"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            for topic in faces.tracked_topics(name for name, _types in self.get_topic_names_and_types()):
                if topic not in subs:
                    subs[topic] = self.create_subscription(IdsList, topic, functools.partial(on_roster, topic), 10)
            if rosters and target is None:
                try:
                    target = faces.match(rosters, face)
                except LookupError as e:
                    error = str(e)
                else:
                    subs[faces.view_topic(*target)] = self.create_subscription(FaceView, faces.view_topic(*target), on_view, 10)
            if target is not None and views:
                return FaceWatch(self, target[1], views[-1], list(subs.values()))
            await asyncio.sleep(0.2)
        for sub in subs.values():
            self.destroy_subscription(sub)
        self.get_logger().error(f"cam: {error}")
        return None

    def ok(self) -> bool:
        """False once the rclpy context is shutting down or the take failed, so the timeline stops cleanly."""
        return rclpy.ok() and self._error is None

    def keyframe_stamp(self) -> RosTime:
        """The target time of a keyframe sent now. A gated frame shows at once, since lockstep already paces it."""
        lead = 0.0 if self._lockstep else self.lead
        return (self.get_clock().now() + Duration(seconds=lead)).to_msg()

    async def set_projection(self, projection: str, endpoints: Iterable[_Endpoint]) -> bool:
        ok = True
        for endpoint in endpoints:
            req = ViewportSetProjection.Request()
            req.projection = projection
            ok = await self.call(endpoint.set_projection, req) and ok
        return ok

    async def call(self, client: ClientWrapper, req: object) -> bool:
        try:
            res = await client.call_timeout(req)
        except Exception as e:
            self.get_logger().warning(f"viewport call failed: {e}")
            return False
        if res is None:
            self.get_logger().warning("viewport service timed out")
            return False
        if not res.success:
            self.get_logger().warning(f"viewport call rejected: {res.message}")
        return res.success
