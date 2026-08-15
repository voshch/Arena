"""Interactive viewport driving: key intent in, `cmd_view` keyframes out.

Runs on a plain rclpy node owned by whatever hosts the input (the rqt panel), so
there is no asyncio bridge: publishes are synchronous, service calls fire-and-forget.
Streaming is continuous while engaged, including at rest, since the plugin hands the
camera back after ~400 ms of silence. Release is just stopping the stream.

Keyframes lead by `DRIVE_LEAD` instead of the scripted `client.LEAD`: a starved
publisher stutters at the last keyframe rather than trailing every keypress.
"""

from __future__ import annotations

import typing

from arena_runtime_msgs.msg import EnvRegistry
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from task_generator_msgs.msg import RobotFleet
from viewport_control_msgs.msg import ViewportView
from viewport_control_msgs.srv import (
    ViewportCapture,
    ViewportSetProjection,
    ViewportSetReferenceFrame,
    ViewportSetView,
)

from . import record, surfaces
from .fly import Fly, Intent, View

if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import rclpy.node

    from .curves import Quat, Vec3
    from .surfaces import TargetSelection

DRIVE_LEAD = 0.03
TICK_HZ = 60.0
REDISCOVER_S = 2.0
# stop waiting for a fresh camera_pose after a frame change and carry on
RESEED_TIMEOUT = 1.0
ROBOTS_SUFFIX = "/state/robots"

REFERENCE_MODES = ("full", "yaw", "position")


class _Surface:
    """One viewport surface on a host-owned node: stream publisher, clients, latest pose."""

    def __init__(self, node: rclpy.node.Node, ns: str, offset: tuple[float, float], on_pose: Callable[[], None]) -> None:
        self.ns = ns
        self.offset = offset
        self._on_new_pose = on_pose
        self.cmd_view = node.create_publisher(ViewportView, f"{ns}/viewport/cmd_view", surfaces.STREAM_QOS)
        self.set_view = node.create_client(ViewportSetView, f"{ns}/viewport/set_view")
        self.set_reference = node.create_client(ViewportSetReferenceFrame, f"{ns}/viewport/set_reference_frame")
        self.set_projection = node.create_client(ViewportSetProjection, f"{ns}/viewport/set_projection")
        self.capture = node.create_client(ViewportCapture, f"{ns}/viewport/capture")
        self.pose: tuple[Vec3, Quat] | None = None
        self.sub = node.create_subscription(PoseStamped, f"{ns}/viewport/camera_pose", self._on_pose, 10)

    def _on_pose(self, msg: PoseStamped) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        self.pose = ((p.x, p.y, p.z), (q.w, q.x, q.y, q.z))
        self._on_new_pose()

    def world_pose(self) -> tuple[Vec3, Quat] | None:
        """Latest camera pose lifted back into world coords, or None if not yet seen."""
        if self.pose is None:
            return None
        (x, y, z), q = self.pose
        return ((x + self.offset[0], y + self.offset[1], z), q)

    def destroy(self, node: rclpy.node.Node) -> None:
        node.destroy_subscription(self.sub)
        node.destroy_publisher(self.cmd_view)
        for client in (self.set_view, self.set_reference, self.set_projection, self.capture):
            node.destroy_client(client)


class EntityRoster:
    """Live robot frames (`env_0/jackal`), the sim_paths a reference frame can track."""

    def __init__(self, node: rclpy.node.Node) -> None:
        self._node = node
        self._subs: dict[str, object] = {}
        self._names: dict[str, list[str]] = {}

    def refresh(self) -> None:
        for name, _types in self._node.get_topic_names_and_types():
            if name.endswith(ROBOTS_SUFFIX) and name not in self._subs:
                self._subs[name] = self._node.create_subscription(RobotFleet, name, lambda msg, topic=name: self._on_fleet(topic, msg), surfaces.ENVS_QOS)

    def _on_fleet(self, topic: str, msg: RobotFleet) -> None:
        self._names[topic] = [robot.descriptor.frame for robot in msg.robots if robot.descriptor.frame]

    def names(self) -> list[str]:
        return sorted({name for names in self._names.values() for name in names})


class Driver:
    """Drives the selected viewport surfaces from live intent. Qt-free, tick it from anywhere."""

    def __init__(self, node: rclpy.node.Node, selection: TargetSelection, *, lead: float = DRIVE_LEAD) -> None:
        self._node = node
        self._selection = selection
        self.lead = lead
        self.fly = Fly()
        self.engaged = False
        self.entity = ""
        # the entity currently framed, empty while flying the world frame
        self.anchor = ""
        self.reference_mode = REFERENCE_MODES[0]
        self.status = ""
        self._surfaces: dict[str, _Surface] = {}
        self._env_refs: dict[int, tuple[float, float]] = {}
        # set once a reference frame exists: poses are relative to it, so they stop localizing
        self._referenced = False
        self._pose_seq = 0
        self._reseed_at: int | None = None
        self._reseed_wait = 0.0
        self._stills: Path | None = None
        node.create_subscription(EnvRegistry, surfaces.ENVS_TOPIC, self._on_envs, surfaces.ENVS_QOS)
        self.rediscover()

    # discovery ------------------------------------------------------------

    @property
    def namespaces(self) -> list[str]:
        return sorted(self._surfaces)

    def rediscover(self) -> None:
        """Sync the surface set with the live graph, keeping existing handles."""
        names = [name for name, _types in self._node.get_service_names_and_types()]
        found = dict(surfaces.find_targets(names, self._selection, self._env_refs))
        for ns in list(self._surfaces):
            if ns not in found:
                self._surfaces.pop(ns).destroy(self._node)
        for ns, offset in found.items():
            if ns in self._surfaces:
                self._surfaces[ns].offset = offset
            else:
                self._surfaces[ns] = _Surface(self._node, ns, offset, self._bump_pose)

    def _bump_pose(self) -> None:
        self._pose_seq += 1

    def _on_envs(self, msg: EnvRegistry) -> None:
        self._env_refs = surfaces.env_refs(msg)
        for ns, surface in self._surfaces.items():
            env_id = surfaces.env_id_from_ns(ns)
            if env_id is not None:
                surface.offset = self._env_refs.get(env_id, surface.offset)

    # engagement -----------------------------------------------------------

    def engage(self) -> None:
        """Take the camera, adopting wherever the user left it."""
        if self.engaged:
            return
        if not self.anchor:
            pose = self.world_pose()
            if pose is not None:
                self.fly.seed(*pose)
            else:
                # nothing to imitate yet, hold the stream until a camera_pose lands
                self._reseed_at = self._pose_seq + 1
                self._reseed_wait = 0.0
        self.engaged = True

    def release(self) -> None:
        """Stop publishing: the plugin hands the camera back to its own controls."""
        self.engaged = False
        self.fly.stop()

    def mirror(self) -> None:
        """Track the live camera while it is not ours, so the readout and a capture stay true.

        Only in the world frame: `camera_pose` is published in world coords, which cannot
        be mapped back into an entity reference without that entity's pose.
        """
        if self.engaged or self._referenced:
            return
        pose = self.world_pose()
        if pose is not None:
            self.fly.seed(*pose)

    def world_pose(self) -> tuple[Vec3, Quat] | None:
        """A pose to adopt, preferring the sim camera (`/arena` sorts ahead of the envs)."""
        for ns in sorted(self._surfaces):
            pose = self._surfaces[ns].world_pose()
            if pose is not None:
                return pose
        return None

    # driving --------------------------------------------------------------

    def tick(self, dt: float, intent: Intent) -> tuple[Vec3, Quat, float] | None:
        """Integrate one frame and stream it, or mirror the live camera. None when not driving."""
        if not self._surfaces:
            return None
        if not self.engaged:
            self.mirror()
            return None
        if self._reseed_at is not None:
            self._reseed_wait += dt
            if self._pose_seq < self._reseed_at and self._reseed_wait < RESEED_TIMEOUT:
                return None
            pose = self.world_pose()
            if pose is not None:
                self.fly.seed(*pose)
            self._reseed_at = None
        pos, quat, fov = self.fly.tick(dt, intent)
        self._stream(pos, quat, fov)
        return pos, quat, fov

    def _stream(self, pos: Vec3, quat: Quat, fov: float) -> None:
        stamp = (self._node.get_clock().now() + Duration(seconds=self.lead)).to_msg()
        for surface in self._surfaces.values():
            msg = ViewportView()
            msg.target_time = stamp
            msg.pose = surfaces.ros_pose(self._local(surface, pos), quat)
            msg.world_orientation = False
            msg.fov = float(fov)
            surface.cmd_view.publish(msg)

    def _local(self, surface: _Surface, pos: Vec3) -> Vec3:
        return pos if self._referenced else surfaces.localize(surface.offset, pos)

    # commands -------------------------------------------------------------

    def command(self, name: str) -> None:
        """Dispatch one tapped key. Unknown labels are ignored."""
        if name == "tab":
            self.fly.toggle_mode()
        elif name == "f":
            self.frame()
        elif name == "shift+f":
            self.cycle_reference_mode()
        elif name == "h":
            self.to_world()
        elif name == "1":
            self.fly.canonical(View.FRONT)
        elif name == "3":
            self.fly.canonical(View.RIGHT)
        elif name == "7":
            self.fly.canonical(View.TOP)
        elif name == "space":
            self.fly.stop()
        elif name == "p":
            self.grab_still()

    def frame(self) -> None:
        """Orbit the target entity in its own frame, so the camera follows it."""
        if not self.entity:
            self.status = "no target entity selected"
            return
        self._call_reference(self.entity, None, self.reference_mode)
        self.anchor = self.entity
        self.fly.frame((0.0, 0.0, 0.0), self.fly.radius)
        self.status = f"framing {self.entity} ({self.reference_mode})"

    def cycle_reference_mode(self) -> None:
        index = REFERENCE_MODES.index(self.reference_mode)
        self.reference_mode = REFERENCE_MODES[(index + 1) % len(REFERENCE_MODES)]
        if self.anchor:
            self.frame()
        else:
            self.status = f"reference mode {self.reference_mode}"

    def to_world(self) -> None:
        """Drop the entity anchor, holding the camera where it is."""
        # carry the live world pose across the swap: the plugin would otherwise compose the
        # last entity-local keyframe against the world origin and teleport there
        pose = self.world_pose()
        self._call_reference("", ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), "full")
        self.anchor = ""
        if pose is not None:
            self.fly.seed(*pose)
        else:
            self._reseed_at = self._pose_seq + 1
            self._reseed_wait = 0.0
        self.status = "world frame"

    def grab_still(self) -> None:
        """Capture the current view to a PPM under $ARENA_DATA_DIR/recordings/stills."""
        surface = self._surfaces[self.namespaces[0]] if self._surfaces else None
        if surface is None or not surface.capture.service_is_ready():
            self.status = "capture service unavailable"
            return
        pos, quat, fov = self.fly.pose()
        req = ViewportCapture.Request()
        req.pose = surfaces.ros_pose(self._local(surface, pos), quat)
        req.world_orientation = False
        req.fov = float(fov)
        surface.capture.call_async(req).add_done_callback(self._on_still)

    def _on_still(self, future: object) -> None:
        result = future.result()
        if result is None or not result.success:
            self.status = f"capture failed: {result.message if result is not None else 'no response'}"
            return
        if self._stills is None:
            self._stills = record.resolve_dir("stills")
            self._stills.mkdir(parents=True, exist_ok=True)
        path = record.next_still(self._stills)
        path.write_bytes(record.encode_ppm(result.image))
        self.status = f"captured {path}"

    def _call_reference(self, entity: str, pose: tuple[Vec3, Quat] | None, mode: str) -> None:
        for surface in self._surfaces.values():
            if not surface.set_reference.service_is_ready():
                continue
            req = ViewportSetReferenceFrame.Request()
            req.entity = entity
            req.has_pose = pose is not None
            if pose is not None:
                # authored in world coords, so it localizes once here
                req.pose = surfaces.ros_pose(surfaces.localize(surface.offset, pose[0]), pose[1])
            req.mode = surfaces.REFERENCE_MODES[mode]
            surface.set_reference.call_async(req)
        self._referenced = True

    # readout --------------------------------------------------------------

    def summary(self) -> str:
        pos, _quat, fov = self.fly.pose()
        fwd = self.fly.forward_vec()
        where = f"pos {pos[0]:.1f} {pos[1]:.1f} {pos[2]:.1f}"
        aim = f"dir {fwd[0]:.2f} {fwd[1]:.2f} {fwd[2]:.2f}"
        fov_text = "sim" if fov <= 0.0 else f"{fov:.2f}"
        return f"{where} | {aim} | fov {fov_text} | {self.anchor or 'world'} | {len(self._surfaces)} cam"
