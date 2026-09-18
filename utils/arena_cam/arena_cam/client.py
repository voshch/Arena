"""ROS 2 client for the Arena viewport cameras.

`CamNode` fans a single `Camera` timeline out across one or more viewport surfaces:
the sim GUI camera at `/arena/viewport/*` and any number of per-env rviz cameras at
`/arena/env_<id>/task_generator_node/viewport/*`. The `Camera` facade authors the
shot in world coordinates. Each endpoint subtracts its env's world origin (the
registry `reference`) so the same absolute shot lands at the matching place in every
env. That localization stops once a reference frame is set: the camera pose is then
composed as reference * local, so poses are relative to the reference and only the
reference pose itself is localized. The sim endpoint has a zero offset.

It runs via `run_main`: `setup` discovers the selected endpoints, plays the timeline,
then shuts down. A segment drives two ways. LIVE: stream keyframes on `cmd_view`,
paced by wall-clock. RECORD: walk the segment at a fixed fps and `capture` each frame
synchronously, so the output is deterministic. Each endpoint records to its own file.
"""

from __future__ import annotations

import asyncio
import typing
from pathlib import Path

import rclpy
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.Time import Time
from arena_runtime_msgs.msg import (
    EnvRegistry,
    LockstepChannel,
    LockstepHeartbeat,
    LockstepRegistration,
    LockstepStatus,
)
from arena_runtime_msgs.srv import LifecycleHold, LifecycleStep, LockstepRegister
from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import Point, PoseStamped
from rclpy.duration import Duration
from viewport_control_msgs.msg import ViewportView
from viewport_control_msgs.srv import (
    ViewportCapture,
    ViewportSetProjection,
    ViewportSetReferenceFrame,
    ViewportSetView,
)

from . import curves, surfaces
from .curves import Quat, Vec3
from .record import Recorder, claim, tagged
from .surfaces import TargetSelection

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from arena_rclpy_mixins.Async import ClientWrapper

    from .camera import Camera

    # A frame sampler: eased progress in [0, 1] -> (position, quat, fov).
    Frame = Callable[[float], tuple[Vec3, Quat, float]]


class Steered(typing.NamedTuple):
    """One frame of live input. `referenced` marks the pose as relative to a set reference frame."""

    position: Vec3
    quat: Quat
    fov: float
    referenced: bool


# cmd_view publish rate for streamed segments (Hz, wall-clock LIVE mode).
_FRAME_RATE = 60.0

# Keyframes are stamped this far ahead, so the plugin's buffer rides out publish
# stalls up to this long. The cost is this much added view latency, which the
# interactive driver trades away for responsiveness (see drive.DRIVE_LEAD).
LEAD = 0.3

# Horizontal fov (rad) assumed while no verb has set one.
FOV_DEFAULT = 1.047

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
        self.recorder: Recorder | None = None
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


class CamNode(ArenaMixinNode):
    """Standalone node that plays a `Camera` timeline against the selected viewports."""

    def __init__(
        self,
        *,
        timeline: Camera,
        targets: TargetSelection,
        node_name: str = "arena_cam",
        record: tuple[str, float] | None = None,
        force: bool = False,
        lead: float = LEAD,
        lockstep: bool = False,
    ) -> None:
        super().__init__(node_name)
        self._timeline = timeline
        self._selection = targets
        self.lead = lead
        self._record = Path(record[0]) if record is not None else None
        self._fps = record[1] if record is not None else 0.0
        self._force = force
        self._lockstep = lockstep and record is not None
        self._env_refs: dict[int, tuple[float, float]] = {}
        self._endpoints: list[_Endpoint] = []
        # True once a reference frame is set: from then on poses are relative to it and
        # must not be localized. Until then the reference is identity and the env offset
        # is what maps an absolute shot into each env.
        self._referenced = False
        self._hold_client: ClientWrapper | None = None
        self._step_client: ClientWrapper | None = None
        self._lockstep_time: Time | None = None
        # Follower mode: an active run owns the stepping, so the cam rides it as a
        # registered hard channel instead of taking its own hold.
        self._scheduler_active = False
        self._run_paused = False
        self._follower = False
        self._beat_pub = None

    async def setup(self) -> None:
        await self.take()
        rclpy.try_shutdown()

    async def take(self) -> bool:
        """Connect, run the timeline, close the files. True when the shot ran, a host with its own context awaits this directly."""
        self.create_subscription(EnvRegistry, surfaces.ENVS_TOPIC, self._on_envs, surfaces.ENVS_QOS)
        self.create_subscription(LockstepStatus, "/arena/state/lockstep", self._on_lockstep_status, surfaces.ENVS_QOS)

        found = await self._await_endpoints()
        if not found:
            self.get_logger().error("no viewport targets found, is the sim GUI / rviz up? (headless sim has none)")
            return False
        self._endpoints = [_Endpoint(self, ns, offset) for ns, offset in found]
        reachable: list[_Endpoint] = []
        for endpoint in self._endpoints:
            if await endpoint.set_view.ensure(timeout_sec=10.0):
                reachable.append(endpoint)
            else:
                self.get_logger().warning(f"{endpoint.ns}/viewport did not answer, skipping")
        self._endpoints = reachable

        if not self._endpoints:
            self.get_logger().error("no reachable viewport targets")
            return False
        if self._record is None or await self._open_recorders():
            names = ", ".join(endpoint.ns for endpoint in self._endpoints)
            self.get_logger().info(f"viewport connected ({names}), {'recording' if self._record else 'playing'} shot")
            await self.arrives(f"{self._endpoints[0].ns}/viewport/camera_pose", PoseStamped)  # seed the cursor
            try:
                if self._lockstep and self._scheduler_active:
                    await self._run_follower()
                elif self._lockstep:
                    await self._run_lockstep()
                else:
                    await self._timeline.run(self)
            finally:
                for endpoint in self._endpoints:
                    self._close_recorder(endpoint)
            if rclpy.ok() and self._record is None:
                self.get_logger().info("shot complete")
            return True
        return False

    async def _open_recorders(self) -> bool:
        """One ffmpeg per camera. A lone camera keeps the file name, several get `-sim` / `-viz<env>` tags."""
        paths = [self._record] if len(self._endpoints) == 1 else [tagged(self._record, surfaces.tag(endpoint.ns)) for endpoint in self._endpoints]
        try:
            claim(paths, self._force)
        except FileExistsError as e:
            self.get_logger().error(str(e))
            return False
        for endpoint in self._endpoints:
            if not await endpoint.capture.ensure(timeout_sec=10.0):
                self.get_logger().error(f"no {endpoint.ns}/viewport/capture service, rebuild the plugin for record mode")
                return False
        for endpoint, path in zip(self._endpoints, paths, strict=True):
            endpoint.recorder = Recorder(str(path), self._fps)
            self.get_logger().info(f"recording {surfaces.tag(endpoint.ns)} to {path}")
        return True

    def _close_recorder(self, endpoint: _Endpoint) -> None:
        recorder = endpoint.recorder
        if recorder is None:
            return
        tag = surfaces.tag(endpoint.ns)
        if recorder.close():
            self.get_logger().info(f"saved {tag} recording to {recorder.path} ({recorder.n} frames)")
        else:
            self.get_logger().error(f"{tag} recording failed after {recorder.n} frames, {recorder.path} is not usable")

    def _on_lockstep_status(self, msg: LockstepStatus) -> None:
        if self._follower and self._scheduler_active and not msg.active:
            self.get_logger().warning("lockstep run ended mid-take, remaining frames are not gated")
        self._scheduler_active = bool(msg.active)
        self._run_paused = bool(msg.active and msg.paused)

    async def _await_resumed(self) -> None:
        """Hold the take while the ridden lockstep run is paused."""
        if not self._run_paused:
            return
        self.get_logger().info("lockstep run paused, recording holding until resume")
        while self._run_paused and rclpy.ok():
            await asyncio.sleep(0.2)

    async def _run_follower(self) -> None:
        """Ride the active lockstep run instead of driving the sim: register a hard
        cam channel at 1/fps, pulse one window per frame, then capture the frozen
        tick, so frames are gate-exact with every producer's window data arrived."""
        register = self.create_client_wrapper(LockstepRegister, "/arena/sim_lifecycle/lockstep/register")
        topic = f"{self.get_fully_qualified_name()}/lockstep"
        registration = LockstepRegistration(
            caller=topic,
            env="",
            channels=[
                LockstepChannel(
                    name="cam",
                    topic=topic,
                    type="arena_runtime_msgs/msg/LockstepHeartbeat",
                    period_s=1.0 / self._fps,
                    hard=True,
                )
            ],
        )
        response = await register.call_timeout(LockstepRegister.Request(registration=registration))
        if response is None or not response.success:
            detail = "service timed out" if response is None else response.error_msg
            self.get_logger().warning(f"cam channel registration failed ({detail}), falling back to hold-and-step record")
            await self._run_lockstep()
            return
        self._beat_pub = self.create_publisher(LockstepHeartbeat, topic, 10)
        self._follower = True
        self._lockstep_time = self.sim_time
        self.get_logger().info(f"riding active lockstep run: cam gated at {self._fps:g} fps")
        try:
            await self._timeline.run(self)
        finally:
            self._follower = False
            await register.call_timeout(LockstepRegister.Request(registration=LockstepRegistration(caller=topic, env="", channels=[])))

    async def _run_lockstep(self) -> None:
        """Acquire a sim hold and run the timeline with physics stepped 1/fps per recorded frame."""
        self._hold_client = self.create_client_wrapper(LifecycleHold, "/arena/sim_lifecycle/hold")
        self._step_client = self.create_client_wrapper(LifecycleStep, "/arena/sim_lifecycle/step")
        req = LifecycleHold.Request()
        req.action = LifecycleHold.Request.ACQUIRE
        req.caller_id = self.get_fully_qualified_name()
        req.reason = "record"
        try:
            if await self._hold_client.call_timeout(req) is None:
                self.get_logger().error("sim hold timed out, nothing recorded")
                return
            self._lockstep_time = await self._held_sim_time()
            await self._timeline.run(self)
        finally:
            rel = LifecycleHold.Request()
            rel.action = LifecycleHold.Request.RELEASE
            rel.caller_id = self.get_fully_qualified_name()
            rel.reason = "record"
            await self._hold_client.call_timeout(rel)

    async def _held_sim_time(self) -> Time:
        """The held sim's time once /clock stands still, the sample arena_node steps from too."""
        seen = self.sim_time
        while rclpy.ok():
            await asyncio.sleep(0.2)
            if self.sim_time == seen:
                break
            seen = self.sim_time
        return seen

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
            if found and quiet >= _DISCOVERY_QUIET:
                return found
            await asyncio.sleep(0.5)
            waited += 0.5
            if not found and waited >= 10.0 and (waited % 10.0) < 0.5:
                self.get_logger().warning(f"arena cam: waiting for viewport targets ({waited:.0f}s elapsed)")
        return []

    def _find_targets(self) -> list[tuple[str, tuple[float, float]]]:
        names = [name for name, _types in self.get_service_names_and_types()]
        return surfaces.find_targets(names, self._selection, self._env_refs)

    def ok(self) -> bool:
        """False once the rclpy context is shutting down, so streaming stops cleanly."""
        return rclpy.ok()

    def camera_pose(self) -> tuple[Vec3, Quat] | None:
        """A representative camera world pose to seed the cursor, or None if not yet seen."""
        for endpoint in self._endpoints:
            pose = endpoint.world_pose()
            if pose is not None:
                return pose
        return None

    def _local(self, endpoint: _Endpoint, position: Vec3) -> Vec3:
        """World coords -> endpoint-local, unless a reference is set (poses are relative to it)."""
        return position if self._referenced else endpoint.localize(position)

    # low-level verbs ------------------------------------------------------

    async def look(self, eye: Vec3, target: Vec3, fov: float = 0.0) -> bool:
        if self._record is not None:
            return await self._record_frame(eye, curves.look_at_quat(eye, target), False, fov)
        ok = True
        for endpoint in self._endpoints:
            req = ViewportSetView.Request()
            eye_local, target_local = self._local(endpoint, eye), self._local(endpoint, target)
            req.eye = Point(x=float(eye_local[0]), y=float(eye_local[1]), z=float(eye_local[2]))
            req.target = Point(x=float(target_local[0]), y=float(target_local[1]), z=float(target_local[2]))
            req.fov = float(fov)
            ok = await self._call(endpoint.set_view, req) and ok
        return ok

    async def set_reference(self, entity: str = "", pose: tuple[Vec3, Quat] | None = None, mode: str = "full") -> bool:
        ok = True
        for endpoint in self._endpoints:
            req = ViewportSetReferenceFrame.Request()
            req.entity = entity
            req.has_pose = pose is not None
            if pose is not None:
                # The reference pose is authored in world coords, so it localizes once here.
                req.pose = surfaces.ros_pose(endpoint.localize(pose[0]), pose[1])
            req.mode = surfaces.REFERENCE_MODES[mode]
            ok = await self._call(endpoint.set_reference, req) and ok
        self._referenced = True
        return ok

    async def set_projection(self, projection: str) -> bool:
        ok = True
        for endpoint in self._endpoints:
            req = ViewportSetProjection.Request()
            req.projection = projection
            ok = await self._call(endpoint.set_projection, req) and ok
        return ok

    def stream(self, position: Vec3, quat: Quat, world_orientation: bool = False, fov: float = 0.0) -> None:
        if not rclpy.ok():
            return
        stamp = (self.get_clock().now() + Duration(seconds=self.lead)).to_msg()
        for endpoint in self._endpoints:
            msg = ViewportView()
            msg.target_time = stamp
            msg.pose = surfaces.ros_pose(self._local(endpoint, position), quat)
            msg.world_orientation = bool(world_orientation)
            msg.fov = float(fov)
            endpoint.cmd_view.publish(msg)

    async def drive(self, duration: float, world_orientation: bool, frame_at: Frame) -> None:
        """Play one segment by sampling `frame_at(t)` over t in [0, 1].

        Record mode walks a fixed `duration * fps` frames and captures each one
        synchronously, so timing is exact. Live mode streams on `cmd_view` paced by
        wall-clock, so a starved step jumps to the right point rather than running
        the whole move in slow motion.
        """
        if self._record is not None:
            frames = max(1, round(duration * self._fps))
            for i in range(frames):
                if not rclpy.ok():
                    return
                pos, quat, fov = frame_at((i + 1) / frames)
                if not await self._record_frame(pos, quat, world_orientation, fov):
                    return
            return
        if duration <= 0.0:
            pos, quat, fov = frame_at(1.0)
            self.stream(pos, quat, world_orientation, fov)
            return
        loop = asyncio.get_running_loop()
        start = loop.time()
        period = 1.0 / _FRAME_RATE
        while rclpy.ok():
            t = min(1.0, (loop.time() - start) / duration)
            pos, quat, fov = frame_at(t)
            self.stream(pos, quat, world_orientation, fov)
            if t >= 1.0:
                break
            await asyncio.sleep(period)

    async def steer(self, next_frame: Callable[[float], Steered | None]) -> None:
        """Record an open-ended segment: `next_frame(dt)` advances the live input one frame period, None ends it."""
        while rclpy.ok():
            frame = next_frame(1.0 / self._fps)
            if frame is None:
                return
            self._referenced = frame.referenced
            if not await self._record_frame(frame.position, frame.quat, False, frame.fov):
                return

    async def capture(self, endpoint: _Endpoint, position: Vec3, quat: Quat, world_orientation: bool, fov: float, min_sim_time: RosTime | None = None) -> object | None:
        req = ViewportCapture.Request()
        req.pose = surfaces.ros_pose(self._local(endpoint, position), quat)
        req.world_orientation = bool(world_orientation)
        req.fov = float(fov)
        if min_sim_time is not None:
            req.min_sim_time = min_sim_time
        try:
            return await endpoint.capture.call_timeout(req)
        except Exception as e:
            self.get_logger().warning(f"capture call failed: {e}")
            return None

    async def _record_frame(self, position: Vec3, quat: Quat, world_orientation: bool, fov: float) -> bool:
        """Advance the sim one frame period where lockstep asks for it, then grab the frame from every camera."""
        if fov <= 0.0 and len(self._endpoints) > 1:
            fov = FOV_DEFAULT  # left open, each camera would film through its own lens
        min_sim_time = None
        if self._follower:
            # cover the next frame window so the scheduler advances one period and
            # freezes at its gate, then capture that frozen tick
            await self._await_resumed()
            self._lockstep_time = self._lockstep_time + Time.from_float(1.0 / self._fps)
            beat = LockstepHeartbeat()
            beat.header.stamp = self._lockstep_time.to_msg()
            self._beat_pub.publish(beat)
            min_sim_time = self._lockstep_time.to_msg()
        elif self._lockstep:
            step_req = LifecycleStep.Request()
            step_req.seconds = 1.0 / self._fps
            step_res = await self._step_client.call_timeout(step_req)
            if step_res is None or not step_res.success:
                detail = "service timed out" if step_res is None else step_res.error_msg
                self.get_logger().warning(f"lockstep step failed ({detail}), stopping record")
                return False
            self._lockstep_time = self._lockstep_time + Time.from_float(step_res.advanced)
            min_sim_time = self._lockstep_time.to_msg()
        grabbed = await asyncio.gather(*(self._grab(endpoint, position, quat, world_orientation, fov, min_sim_time) for endpoint in self._endpoints))
        return all(grabbed)

    async def _grab(self, endpoint: _Endpoint, position: Vec3, quat: Quat, world_orientation: bool, fov: float, min_sim_time: RosTime | None) -> bool:
        while True:
            res = await self.capture(endpoint, position, quat, world_orientation, fov, min_sim_time)
            if res is not None and res.success:
                break
            if self._follower and self._run_paused:
                # the run paused mid-frame and the deferred capture hit its
                # deadline, retry the same tick once the run resumes
                await self._await_resumed()
                continue
            detail = "service timed out" if res is None else res.message
            self.get_logger().warning(f"{endpoint.ns} capture failed ({detail}), stopping record")
            return False
        try:
            endpoint.recorder.write(res.image)
        except (ValueError, OSError) as e:
            self.get_logger().warning(f"{endpoint.ns} frame not encoded ({e}), stopping record")
            return False
        return True

    async def _call(self, client: ClientWrapper, req: object) -> bool:
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
