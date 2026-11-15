"""Record and draw a robot's contacts with people and the assistive devices they hold (humansim `robot_contacts`).

  record OUT.jsonl  subscribes to the running humansim's robot_contacts, agent_states, agent_devices, world_state and
                    world_geometry, optionally drives the robot straight ahead (--drive m/s on its cmd_vel, stopped at the
                    first contact; with --start-within D, only once a person is within D m of it), and writes one JSON
                    object per line: meta, walls, 10 Hz ticks (robot, people, devices) and every contact edge (START /
                    END: time, part, point, depth, duration, poses, device). A CSV of the edges lands next to it. In
                    Isaac Sim, --capture saves the viewport (/arena/viewport/capture, GUI only) at the first contact, a
                    camera beside the robot looking at the robot and the person, and --props [KIND ...] draws the held devices of those kinds (all without
                    one) as their object model (the agent type's `device.model`) where humansim poses them. The white
                    cane, rolling cane and rollator characters hold theirs already, so only `--props walking_frame` is
                    needed. --camera saves the robot's own camera frame of that moment, in any simulator and headless.
  plot RUN.jsonl OUT.png [--image FRAME.png]
                    top-down view in humansim's frame: the walls, both paths, and the first contact with the device,
                    the robot and the person as they stood, the contact point marked; with --image, that
                    frame (e.g. an Isaac Sim viewport capture of the same moment) on the left.

Scenarios: arena_simulation_setup worlds/map_empty/scenarios/device_contact_<device>, the robot driving head-on at the
person; `_crossing`, the robot crossing the person's path at 90 degrees (record with the --start-within its
scenario.yaml gives); `_overtaking`, the robot catching up from behind with a person walking slower than it. Launched with robot.mobile:=none, e.g.

  arena launch sim:=gazebo world:=map_empty robot:=turtlebot human:=arena tm_robots:=scenario tm_obstacles:=scenario \\
      task.scenario.file:=device_contact_white_cane robot.mobile:=none
  python3 -m arena_runtime.humans.arena_humansim.device_contacts record white_cane.jsonl --drive 0.4 --robot-radius 0.17 \\
      --ros-args -p use_sim_time:=true
  python3 -m arena_runtime.humans.arena_humansim.device_contacts plot white_cane.jsonl white_cane.png

With sim:=isaac, add `--capture white_cane_isaac.png` to record (and `--props walking_frame` for the walking frame), and
`--image white_cane_isaac.png` to plot. humansim publishes in the env's frame `<env>/map`, which Arena anchors in the
simulator's `map` at the env's slot (the static TF `map` -> `env_0/map`), where Isaac Sim places prims and the viewport
camera, so the props and the capture camera are shifted by that anchor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arena_humansim_msgs.msg import AgentDevices, AgentFrame, AgentStates, DeviceState, RobotContacts, WorldGeometry
    from geometry_msgs.msg import Point, Pose, Pose2D, Vector3
    from matplotlib.patches import Patch
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from tf2_msgs.msg import TFMessage

BODY = "body"
TICK_PERIOD = 0.1  # s of sim time between recorded ticks
FACE_YAW = {"x": 0.0, "-x": math.pi, "y": math.pi / 2, "-y": -math.pi / 2}  # an object model's front (annotation `face`) as a yaw
CAPTURE_WAIT = 20.0  # wall seconds to wait for the viewport capture after recording ends
ANCHOR_WAIT = 10.0  # wall seconds to wait for the env's anchor TF before taking the env frame as the simulator's
VIEW_HEIGHT = 0.75  # m, height the capture camera looks at, between the robot and the person


def _quat_toward(dx: float, dy: float, dz: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) turning +x toward (dx, dy, dz) with no roll: a viewport camera's view (+x forward, +z up) or a cane's shaft."""
    n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    yaw, pitch = math.atan2(dy, dx), -math.asin(max(-1.0, min(1.0, dz / n)))
    cp, sp, cy, sy = math.cos(pitch / 2), math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2)
    return cp * cy, -sp * sy, sp * cy, cp * sy


def _model_info(model: str) -> tuple[str, float, float]:
    """(USD file, yaw of the front, length along +x) of an object model, resolved the way Arena resolves obstacles."""
    import asyncio

    import yaml
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier

    path = Path(asyncio.run(ObjectIdentifier.parse(model).resolve_path()))
    annotation = yaml.safe_load((path / "annotation.yaml").read_text())
    usd = next(p for pattern in ("*.usdz", "*.usd", "*.usda", "*.usdc") for p in sorted(path.glob(pattern)))
    return str(usd), FACE_YAW.get(str(annotation.get("face", "x")), 0.0), float(annotation["bounding_box"][0][1])


def _prop_pose(d: DeviceState, face_yaw: float, length: float, shift: Sequence[float]) -> Pose:
    """Simulator pose of a device's object model: a cane's (origin at its top, shaft along +x) ending at the tip, a box's
    origin at the device origin, moved by `shift` from humansim's frame into the simulator's."""
    from geometry_msgs.msg import Pose

    pose = Pose()
    if d.kind == "cane":
        dx, dy, dz = d.end.x - d.start.x, d.end.y - d.start.y, d.end.z - d.start.z
        n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        x0, y0, z0 = d.end.x - length * dx / n, d.end.y - length * dy / n, d.end.z - length * dz / n
        w, x, y, z = _quat_toward(dx, dy, dz)
    else:
        x0, y0, z0 = d.start.x, d.start.y, d.start.z
        yaw = d.yaw - face_yaw
        w, x, y, z = math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
    pose.position.x, pose.position.y, pose.position.z = x0 + shift[0], y0 + shift[1], z0 + shift[2]
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = w, x, y, z
    return pose


def _env_frame(topic: str) -> str:
    """The env frame of a topic under /arena/<env>/..., `map` (no shift) for any other."""
    parts = topic.strip("/").split("/")
    return f"{parts[1]}/map" if len(parts) > 2 and parts[0] == "arena" else "map"


class _Trigger:
    """Starts the robot once a person comes within `within` m of it, after one was seen farther: a person spawned close does
    not start it. `within` 0 starts it at once."""

    def __init__(self, within: float) -> None:
        self.within = within
        self.armed = self.fired = within <= 0.0

    def update(self, robot: Sequence[float] | None, agents: Sequence[dict[str, Any]]) -> float | None:
        """The distance to the nearest person at the update that starts the robot, None at any other."""
        if self.fired or not robot or not agents:
            return None
        near = min(math.hypot(a["x"] - robot[0], a["y"] - robot[1]) for a in agents)
        if near > self.within:
            self.armed = True
        elif self.armed:
            self.fired = True
            return near
        return None


class _Anchor:
    """Where humansim's frame (the env's `<env>/map`) sits in the simulator's `map`: Arena's static anchor TF `map` -> `<env>/map`."""

    def __init__(self, node: Node) -> None:
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from tf2_msgs.msg import TFMessage

        self._static: dict[str, tuple[str, tuple[float, float, float]]] = {}
        node.create_subscription(TFMessage, "/tf_static", self._on_static, QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._log = node.get_logger()
        self.frame = ""
        self._since: float | None = None
        self.shift: tuple[float, float, float] | None = None

    def _on_static(self, m: TFMessage) -> None:
        for t in m.transforms:
            v = t.transform.translation
            self._static[t.child_frame_id.lstrip("/")] = (t.header.frame_id.lstrip("/"), (v.x, v.y, v.z))

    def resolve(self) -> tuple[float, float, float] | None:
        """(dx, dy, dz) from humansim's frame to the simulator's, None while the anchor is unknown."""
        if self.shift is not None or not self.frame:
            return self.shift
        import time

        parent, shift = self._static.get(self.frame, ("", (0.0, 0.0, 0.0)))
        if self.frame != "map" and parent != "map":
            self._since = self._since or time.monotonic()
            if time.monotonic() - self._since < ANCHOR_WAIT:
                return None
            self._log.warning(f"no static TF map -> {self.frame} after {ANCHOR_WAIT:.0f} s, taking humansim's frame as the simulator's")
        self.shift = shift
        self._log.info(f"{self.frame} sits at ({shift[0]:.2f}, {shift[1]:.2f}, {shift[2]:.2f}) in map, the props and the capture camera are shifted by it")
        return self.shift


class _IsaacProps:
    """The held devices' object models in Isaac Sim, spawned once and moved to humansim's pose of each, in the simulator's frame."""

    def __init__(self, node: Node, anchor: _Anchor, kinds: set[str]) -> None:
        from isaacsim_msgs.srv import EditPrims, SpawnPrims

        self._spawn = node.create_client(SpawnPrims, "/isaac/SpawnPrims")
        self._edit = node.create_client(EditPrims, "/isaac/EditPrims")
        self._log = node.get_logger()
        self._anchor, self._kinds = anchor, kinds
        self._models: dict[str, tuple[str, float, float]] = {}
        self._spawned: set[str] = set()
        self._busy: Any = None

    def update(self, devices: Sequence[DeviceState]) -> None:
        from isaacsim_msgs.msg import Prim
        from isaacsim_msgs.srv import EditPrims, SpawnPrims

        shift = self._anchor.resolve()
        if shift is None or not self._spawn.service_is_ready():
            return
        new, moved = [], []
        for d in devices:
            if not d.model or (self._kinds and d.kind not in self._kinds):
                continue
            if d.model not in self._models:
                self._models[d.model] = _model_info(d.model)
            usd, face_yaw, length = self._models[d.model]
            prim = Prim(usd_path=usd, name=f"device_{d.agent_id}", pose=_prop_pose(d, face_yaw, length, shift), is_static=True)
            (moved if prim.name in self._spawned else new).append(prim)
        if new:
            names = [p.name for p in new]
            future = self._spawn.call_async(SpawnPrims.Request(prims=new))
            future.add_done_callback(lambda f: self._log.info(f"Isaac Sim spawned {', '.join(f'{n}: {bool(ok)}' for n, ok in zip(names, getattr(f.result(), 'ret', []), strict=False))}"))
            self._spawned.update(names)
            self._log.info(f"spawning {', '.join(f'{p.name} ({p.usd_path})' for p in new)} in Isaac Sim")
        if moved and (self._busy is None or self._busy.done()):
            self._busy = self._edit.call_async(EditPrims.Request(prims=moved, pose=True))


class _Capture:
    """One viewport capture (/arena/viewport/capture) of a contact, from beside the robot, looking between it and the person."""

    def __init__(self, node: Node, out: Path, view: Sequence[float], fov: float) -> None:
        from viewport_control_msgs.srv import ViewportCapture

        self._client = node.create_client(ViewportCapture, "/arena/viewport/capture")
        self._out, self._view, self._fov, self._log = out, view, fov, node.get_logger()
        self._future: Any = None
        self.saved = False

    @property
    def pending(self) -> bool:
        return self._future is not None and not self.saved

    def request(self, robot: Pose2D, person: Pose2D, shift: Sequence[float]) -> None:
        """Capture from the robot's left, behind and above the midpoint between the robot and the person, in the simulator's frame."""
        from viewport_control_msgs.srv import ViewportCapture

        if self._future is not None:
            return
        side, back, up = self._view
        c, s = math.cos(robot.theta), math.sin(robot.theta)
        mx, my = (robot.x + person.x) / 2 + shift[0], (robot.y + person.y) / 2 + shift[1]
        eye = (mx - back * c - side * s, my - back * s + side * c, up + shift[2])
        target = (mx, my, VIEW_HEIGHT + shift[2])
        req = ViewportCapture.Request(world_orientation=True, fov=self._fov)
        req.pose.position.x, req.pose.position.y, req.pose.position.z = eye
        w, x, y, z = _quat_toward(target[0] - eye[0], target[1] - eye[1], target[2] - eye[2])
        req.pose.orientation.w, req.pose.orientation.x, req.pose.orientation.y, req.pose.orientation.z = w, x, y, z
        self._future = self._client.call_async(req)
        self._log.info(f"capturing the viewport from ({eye[0]:.2f}, {eye[1]:.2f}, {eye[2]:.2f}) in map")

    def poll(self) -> None:
        if self._future is None or self.saved or not self._future.done():
            return
        res = self._future.result()
        self.saved = True
        if res is None or not res.success:
            self._log.error(f"viewport capture failed: {getattr(res, 'message', 'no response')}")
            return
        _save_image(res.image, self._out)
        self._log.info(f"saved the viewport capture to {self._out}")


def _save_image(img: Image, out: Path) -> None:
    """Write an rgb8 / bgr8 / rgba8 / bgra8 image message to a PNG."""
    import numpy as np
    from matplotlib import image

    channels = 4 if img.encoding in ("rgba8", "bgra8") else 3
    a = np.frombuffer(bytes(img.data), dtype=np.uint8).reshape(img.height, img.step)[:, : img.width * channels].reshape(img.height, img.width, channels)[:, :, :3]
    image.imsave(out, a[:, :, ::-1] if img.encoding.startswith("bgr") else a)


def _xyz(p: Point | Vector3) -> list[float]:
    return [round(p.x, 4), round(p.y, 4), round(p.z, 4)]


def _pose(p: Pose2D) -> list[float]:
    return [round(p.x, 4), round(p.y, 4), round(p.theta, 4)]


def _device(d: DeviceState) -> dict[str, Any]:
    return {"agent_id": d.agent_id, "kind": d.kind, "start": _xyz(d.start), "end": _xyz(d.end), "centre": _xyz(d.centre), "yaw": round(d.yaw, 4), "radius": d.radius, "size": _xyz(d.size)}


def record(argv: Sequence[str]) -> int:
    import rclpy
    from arena_humansim_msgs.msg import AgentDevices, AgentFrame, AgentStates, RobotContacts, WorldGeometry
    from geometry_msgs.msg import Twist
    from rcl_interfaces.msg import Parameter as ParameterMsg
    from rcl_interfaces.msg import ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    parser = argparse.ArgumentParser(prog="device_contacts record", description="Record robot contacts (see the module docstring).")
    parser.add_argument("out", type=Path, help="JSON lines file to write, a .csv of the contact edges lands next to it")
    parser.add_argument("--drive", type=float, default=0.0, help="drive the robot straight ahead at this speed (m/s) until its first contact, 0 = leave it alone")
    parser.add_argument("--start-within", type=float, default=0.0, help="hold the robot until a person comes within this distance (m) of it, then drive, 0 = drive at once")
    parser.add_argument("--robot-radius", type=float, default=0.0, help="set humansim's robot_contact.robot_radius (m), 0 = keep it")
    parser.add_argument("--robot-height", type=float, default=0.0, help="set humansim's robot_contact.robot_height (m), 0 = keep it")
    parser.add_argument("--hold", type=float, default=4.0, help="sim seconds to keep recording after the first contact")
    parser.add_argument("--timeout", type=float, default=180.0, help="sim seconds to record at most")
    parser.add_argument("--props", nargs="*", metavar="KIND", help="Isaac Sim: draw the held devices of these kinds (all without one) as their object model where humansim poses them")
    parser.add_argument("--capture", type=Path, help="at the first contact, save the simulator viewport to this PNG (Isaac Sim)")
    parser.add_argument("--view", type=float, nargs=3, default=(3.2, 1.0, 1.5), metavar=("LEFT", "BACK", "UP"), help="capture camera offset from the midpoint between the robot and the person: to the robot's left, behind it, up (m)")
    parser.add_argument("--fov", type=float, default=0.95, help="horizontal field of view of the capture (rad), 0 = the viewport's own (default: %(default)s)")
    parser.add_argument("--env-frame", default="", help="humansim's TF frame, anchored in the simulator's `map` (default: <env>/map of the robot_contacts topic)")
    parser.add_argument("--camera", type=Path, help="save the robot camera's first frame at or after the first device contact to this PNG")
    parser.add_argument("--camera-topic", default="oakd/image", help="suffix of the robot camera's image topic (default: %(default)s)")
    args = parser.parse_args(rclpy.utilities.remove_ros_args(list(argv))[1:])

    rclpy.init(args=list(argv))
    node = Node("device_contacts_recorder")
    log = node.get_logger()
    anchor = _Anchor(node)
    anchor.frame = args.env_frame
    props = _IsaacProps(node, anchor, set(args.props)) if args.props is not None else None
    capture = _Capture(node, args.capture, args.view, args.fov) if args.capture else None
    out = args.out.open("w")
    edges: list[dict[str, Any]] = []
    state: dict[str, Any] = {"robot": None, "agents": [], "devices": [], "next_tick": None, "t0": None, "first": None, "subs": {}, "frame": None, "camera_saved": False}
    trigger = _Trigger(args.start_within)
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    stream = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)

    def write(obj: dict[str, Any]) -> None:
        out.write(json.dumps(obj) + "\n")

    def now() -> float:
        return node.get_clock().now().nanoseconds * 1e-9

    def on_frame(m: AgentFrame) -> None:
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        state["agents"] = [{"id": int(i), "x": round(x, 4), "y": round(y, 4), "theta": round(th, 4), "radius": r} for i, x, y, th, r, k in zip(m.agent_id, m.x, m.y, m.theta, m.radius, m.kind, strict=True) if k == 0]
        if state["t0"] is None:
            state["t0"] = t
        if state["next_tick"] is None or t >= state["next_tick"]:
            state["next_tick"] = t + TICK_PERIOD
            write({"kind": "tick", "t": round(t, 3), "robot": state["robot"], "agents": state["agents"], "devices": state["devices"]})

    def on_world_state(m: AgentStates) -> None:
        robots = [a for a in m.agents if not a.name.startswith("robot_")] or list(m.agents)
        if robots:
            state["robot"] = _pose(robots[0].pose)

    def on_devices(m: AgentDevices) -> None:
        state["devices"] = [_device(d) for d in m.devices]
        if props is not None:
            props.update(m.devices)

    def on_camera(m: Image) -> None:
        state["frame"] = m
        first = state["first"]
        if args.camera and first is not None and not state["camera_saved"] and m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 >= first:
            _save_image(m, args.camera)
            state["camera_saved"] = True
            log.info(f"saved the robot camera frame of the first contact to {args.camera}")

    def on_walls(m: WorldGeometry) -> None:
        write({"kind": "walls", "walls": [[round(a.x, 3), round(a.y, 3), round(b.x, 3), round(b.y, 3)] for a, b in zip(m.wall_starts, m.wall_ends, strict=True)]})

    def on_contacts(m: RobotContacts) -> None:
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        for c in m.contacts:
            edge = {
                "kind": "contact",
                "t": round(t, 3),
                "event": "start" if c.event == c.START else "end",
                "part": c.part,
                "robot_id": c.robot_id,
                "robot_name": c.robot_name,
                "agent_id": c.agent_id,
                "agent_name": c.agent_name,
                "agent_type": c.agent_type,
                "point": _xyz(c.point),
                "depth": round(c.depth, 4),
                "duration": round(c.duration, 3),
                "robot_pose": _pose(c.robot_pose),
                "agent_pose": _pose(c.agent_pose),
                "device": _device(c.device) if c.device.kind else None,
            }
            write(edge)
            edges.append(edge)
            log.info(f"{edge['event']} {edge['part']} of {edge['agent_type'] or edge['agent_id']} at t={t:.2f} ({edge['point'][0]:.2f}, {edge['point'][1]:.2f})")
            if state["first"] is None:
                state["first"] = t
            if capture is not None and c.event == c.START:
                capture.request(c.robot_pose, c.agent_pose, anchor.resolve() or (0.0, 0.0, 0.0))

    handlers = {
        "robot_contacts": (RobotContacts, on_contacts, stream, "arena_humansim_msgs/msg/RobotContacts"),
        "agent_states": (AgentFrame, on_frame, stream, "arena_humansim_msgs/msg/AgentFrame"),
        "agent_devices": (AgentDevices, on_devices, stream, "arena_humansim_msgs/msg/AgentDevices"),
        "world_state": (AgentStates, on_world_state, stream, "arena_humansim_msgs/msg/AgentStates"),
        "world_geometry": (WorldGeometry, on_walls, latched, "arena_humansim_msgs/msg/WorldGeometry"),
    }
    cmd_pub = None
    configured = not (args.robot_radius or args.robot_height)

    def discover() -> None:
        nonlocal cmd_pub, configured
        topics = node.get_topic_names_and_types()
        for name, types in topics:
            suffix = name.rsplit("/", 1)[-1]
            if suffix in handlers and suffix not in state["subs"] and handlers[suffix][3] in types:
                msg_type, cb, qos, _ = handlers[suffix]
                state["subs"][suffix] = node.create_subscription(msg_type, name, cb, qos)
                log.info(f"recording {name}")
                if suffix == "robot_contacts" and not anchor.frame:
                    anchor.frame = _env_frame(name)
        anchor.resolve()
        if args.camera and "camera" not in state["subs"]:
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image

            for name, types in topics:
                if name.endswith(args.camera_topic) and "sensor_msgs/msg/Image" in types:
                    state["subs"]["camera"] = node.create_subscription(Image, name, on_camera, qos_profile_sensor_data)
                    log.info(f"watching the robot camera {name}")
                    break
        if args.drive and cmd_pub is None:
            for name, types in topics:
                if name.endswith("/cmd_vel") and "geometry_msgs/msg/Twist" in types and node.count_subscribers(name) > 0:
                    cmd_pub = node.create_publisher(Twist, name, 10)
                    log.info(f"driving on {name} at {args.drive} m/s")
                    break
        if not configured and "robot_contacts" in state["subs"]:
            infos = node.get_publishers_info_by_topic(state["subs"]["robot_contacts"].topic_name)
            if infos:
                ns = infos[0].node_namespace.rstrip("/")
                client = node.create_client(SetParameters, f"{ns}/{infos[0].node_name}/set_parameters")
                if client.wait_for_service(timeout_sec=2.0):
                    req = SetParameters.Request()
                    for key, value in (("robot_contact.robot_radius", args.robot_radius), ("robot_contact.robot_height", args.robot_height)):
                        if value:
                            req.parameters.append(ParameterMsg(name=key, value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=value)))
                    client.call_async(req)
                    configured = True
                    write({"kind": "meta", "robot_radius": args.robot_radius, "robot_height": args.robot_height, "drive": args.drive})
                    log.info(f"set {', '.join(p.name for p in req.parameters)} on {ns}/{infos[0].node_name}")

    node.create_timer(1.0, discover)
    if configured:
        write({"kind": "meta", "robot_radius": 0.0, "robot_height": 0.0, "drive": args.drive})

    def drive() -> None:
        if cmd_pub is None:
            return
        near = trigger.update(state["robot"], state["agents"])
        if near is not None:
            log.info(f"a person is {near:.2f} m away, driving")
        cmd = Twist()
        cmd.linear.x = args.drive if state["first"] is None and trigger.fired else 0.0
        cmd_pub.publish(cmd)

    node.create_timer(0.05, drive)
    start = None
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if capture is not None:
                capture.poll()
            t = now()
            if t <= 0.0:
                continue
            start = t if start is None else start
            if state["first"] is not None and t > state["first"] + args.hold or t > start + args.timeout:
                break
        if capture is not None and capture.pending:
            import time

            deadline = time.monotonic() + CAPTURE_WAIT
            while capture.pending and time.monotonic() < deadline and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
                capture.poll()
    finally:
        if cmd_pub is not None:
            cmd_pub.publish(Twist())
        if args.camera and not state["camera_saved"] and state["frame"] is not None and state["first"] is not None:
            _save_image(state["frame"], args.camera)
            log.info(f"saved the last robot camera frame to {args.camera}")
        out.close()
        with args.out.with_suffix(".csv").open("w", newline="") as f:
            cols = ["t", "event", "part", "agent_type", "agent_name", "robot_name", "x", "y", "z", "depth", "duration", "robot_x", "robot_y", "agent_x", "agent_y"]
            w = csv.writer(f)
            w.writerow(cols)
            for e in edges:
                w.writerow([e["t"], e["event"], e["part"], e["agent_type"], e["agent_name"], e["robot_name"], *e["point"], e["depth"], e["duration"], *e["robot_pose"][:2], *e["agent_pose"][:2]])
        log.info(f"{len(edges)} contact edges, written to {args.out} and {args.out.with_suffix('.csv')}")
        node.destroy_node()
        rclpy.try_shutdown()
    return 0 if edges else 1


def _load(path: Path) -> tuple[dict[str, Any], list[list[float]], list[dict[str, Any]], list[dict[str, Any]]]:
    meta: dict[str, Any] = {}
    walls: list[list[float]] = []
    ticks: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        kind = row.pop("kind")
        if kind == "meta":
            meta = row
        elif kind == "walls":
            walls = row["walls"]
        elif kind == "tick":
            ticks.append(row)
        elif kind == "contact":
            edges.append(row)
    return meta, walls, ticks, edges


def plot(argv: Sequence[str]) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    parser = argparse.ArgumentParser(prog="device_contacts plot", description="Draw a recorded device contact (see the module docstring).")
    parser.add_argument("run", type=Path, help="JSON lines file written by `record`")
    parser.add_argument("out", type=Path, help="PNG to write")
    parser.add_argument("--image", type=Path, help="a frame of the same moment (e.g. an Isaac Sim viewport capture) drawn on the left")
    parser.add_argument("--image-title", default="(a) the moment of contact", help="caption of --image (default: %(default)s)")
    parser.add_argument("--title", default="Robot contact with {what} ({agent_type})", help="figure title, {what} (the part touched) and {agent_type} filled in (default: %(default)s)")
    args = parser.parse_args(list(argv))

    meta, walls, ticks, edges = _load(args.run)
    starts = [e for e in edges if e["event"] == "start"]
    hit = starts[0] if starts else None
    if hit is None:
        print(f"{args.run}: no contact recorded", file=sys.stderr)
        return 1

    ink, ink_2, grid, surface = "#1f1e1c", "#52514e", "#e6e5e1", "#fcfcfb"
    robot_c, person_c, device_c, hit_c = "#2a6fdb", "#b8860b", "#3d3c39", "#c92a2a"
    until = _since_reset([k for k in ticks if k["t"] <= hit["t"] + 0.05])
    t0 = until[0]["t"] if until else hit["t"]
    robot_path = [k["robot"][:2] for k in until if k["robot"]]
    person_path = [(a["x"], a["y"]) for k in until for a in k["agents"] if a["id"] == hit["agent_id"]]
    radius = meta.get("robot_radius") or 0.0
    if not radius:
        radius = next((a["radius"] for k in until for a in k["agents"] if a["id"] == hit["robot_id"]), 0.3)
    person_r = next((a["radius"] for a in reversed(until[-1]["agents"]) if a["id"] == hit["agent_id"]), 0.3) if until else 0.3

    ncols = 2 if args.image else 1
    fig = plt.figure(figsize=(6.4 * ncols, 5.6))
    fig.patch.set_facecolor(surface)
    if args.image:
        import numpy as np

        ax_img = fig.add_subplot(1, 2, 1)
        img = plt.imread(str(args.image))
        lit = np.flatnonzero((img[..., :3].max(axis=2) > (0.04 if img.dtype.kind == "f" else 10)).mean(axis=1) > 0.5)  # rows mostly lit: Isaac has no sky
        ax_img.imshow(img[lit[0] : lit[-1] + 1] if lit.size else img)
        ax_img.set_axis_off()
        ax_img.set_title(args.image_title, fontsize=10, color=ink)
    ax = fig.add_subplot(1, ncols, ncols)
    for x0, y0, x1, y1 in walls:
        ax.plot([x0, x1], [y0, y1], color=ink_2, linewidth=2.0, solid_capstyle="butt")
    if robot_path:
        ax.plot(*zip(*robot_path, strict=True), color=robot_c, linewidth=2.0, label="robot path")
    if person_path:
        ax.plot(*zip(*person_path, strict=True), color=person_c, linewidth=2.0, linestyle=(0, (4, 2)), label="person path")
    rx, ry, _ = hit["robot_pose"]
    ax.add_patch(Circle((rx, ry), radius, facecolor=robot_c, alpha=0.25, edgecolor=robot_c, linewidth=1.5))
    ax_, ay_, _ = hit["agent_pose"]
    ax.add_patch(Circle((ax_, ay_), person_r, facecolor=person_c, alpha=0.2, edgecolor=person_c, linewidth=1.5))
    dev = hit["device"] or next((d for k in reversed(until) for d in k["devices"] if d["agent_id"] == hit["agent_id"]), None)  # a body contact: the device as last posed
    if dev is not None:
        if dev["kind"] == "cane":
            ax.plot([dev["start"][0], dev["end"][0]], [dev["start"][1], dev["end"][1]], color=device_c, linewidth=2.5, solid_capstyle="round", label="cane")
        else:
            cx, cy, _ = dev["centre"]
            hl, hw, yaw = dev["size"][0] / 2, dev["size"][1] / 2, dev["yaw"]
            c, s = math.cos(yaw), math.sin(yaw)
            corners = [(cx + c * u - s * v, cy + s * u + c * v) for u, v in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
            ax.add_patch(Polygon(corners, closed=True, facecolor="none", edgecolor=device_c, linewidth=2.0, label=dev["kind"].replace("_", " ")))
    px, py, pz = hit["point"]
    ax.plot([px], [py], marker="*", markersize=16, color=hit_c, markeredgecolor=surface, markeredgewidth=1.0, linestyle="none", label="contact")
    label = f"{hit['part'].replace('_', ' ')} contact\nt = {hit['t'] - t0:.2f} s into the episode (sim {hit['t']:.2f} s)\n({px:.2f}, {py:.2f}) m" + (f", {pz:.2f} m up" if pz > 0.01 else "")
    right = not walls or px <= (min(min(w[0], w[2]) for w in walls) + max(max(w[0], w[2]) for w in walls)) / 2  # label on the side with room
    ax.annotate(label, (px, py), xytext=(14 if right else -14, 18), textcoords="offset points", ha="left" if right else "right", fontsize=8.5, color=ink, arrowprops={"arrowstyle": "-", "color": ink_2, "linewidth": 0.8})
    xs = [p[0] for p in robot_path + person_path] + [px]
    ys = [p[1] for p in robot_path + person_path] + [py]
    if walls:
        xs += [w[i] for w in walls for i in (0, 2)]
        ys += [w[i] for w in walls for i in (1, 3)]
    pad = 0.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.grid(color=grid, linewidth=0.6)
    ax.tick_params(labelsize=8, colors=ink_2)
    ax.set_xlabel("x (m)", fontsize=9, color=ink_2)
    ax.set_ylabel("y (m)", fontsize=9, color=ink_2)
    for spine in ax.spines.values():
        spine.set_color("#d4d3cf")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("(b) top-down, humansim frame" if args.image else "top-down, humansim frame", fontsize=10, color=ink)

    zoom = ax.inset_axes((0.56, 0.04, 0.42, 0.42))
    for artist in list(ax.lines):
        zoom.plot(*artist.get_data(), color=artist.get_color(), linewidth=artist.get_linewidth(), linestyle=artist.get_linestyle(), marker=artist.get_marker(), markersize=artist.get_markersize(), markeredgecolor=artist.get_markeredgecolor())
    for patch in list(ax.patches):
        zoom.add_patch(patch.__class__(**_patch_args(patch)))
    cx, cy = (px + ax_) / 2, (py + ay_) / 2  # the contact and the person both in view
    half = max(1.0, math.hypot(px - ax_, py - ay_) / 2 + person_r + 0.3)
    zoom.set_xlim(cx - half, cx + half)
    zoom.set_ylim(cy - half, cy + half)
    zoom.set_aspect("equal")
    zoom.set_xticks([])
    zoom.set_yticks([])
    for spine in zoom.spines.values():
        spine.set_color(ink_2)
    what = "the person's body" if hit["part"] == BODY else f"a {hit['part'].replace('_', ' ')}"
    fig.suptitle(args.title.format(what=what, agent_type=hit["agent_type"]), fontsize=11, color=ink)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, facecolor=surface)
    print(f"saved {args.out}: {hit['part']} contact at t={hit['t'] - t0:.2f} s, ({px:.2f}, {py:.2f})")
    return 0


def _since_reset(ticks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ticks after the robot's last jump of more than a metre between two ticks: the episode reset teleports it."""
    jumps = [i for i in range(len(ticks) - 1) if ticks[i]["robot"] and ticks[i + 1]["robot"] and math.dist(ticks[i]["robot"][:2], ticks[i + 1]["robot"][:2]) > 1.0]
    return ticks[jumps[-1] + 1 :] if jumps else ticks


def _patch_args(patch: Patch) -> dict[str, Any]:
    """Keyword arguments that rebuild a Circle or Polygon patch for the zoom inset."""
    style = {"facecolor": patch.get_facecolor(), "edgecolor": patch.get_edgecolor(), "linewidth": patch.get_linewidth()}
    if hasattr(patch, "get_radius"):
        return {"xy": patch.center, "radius": patch.get_radius(), **style}
    return {"xy": patch.get_xy(), "closed": True, **style}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    commands = {"record": record, "plot": plot}
    if len(argv) < 2 or argv[1] not in commands:
        print(__doc__, file=sys.stderr)
        return 2
    cmd = argv.pop(1)
    return commands[cmd](argv if cmd == "record" else argv[1:])


if __name__ == "__main__":
    sys.exit(main())
