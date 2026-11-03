"""A stand-in runtime for camera tests: the sim viewport, hold and step, and an optional lockstep run that advances on the cam's beats."""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable

from arena_cam.surfaces import STREAM_QOS

import rclpy
import rclpy.callback_groups
import rclpy.executors

WIDTH, HEIGHT = 32, 24


@dataclasses.dataclass
class Capture:
    """A capture the viewport served: the reference it was under and the sim time it showed."""

    reference: str
    fov: float
    clip_near: float
    stamp_ns: int


class Runtime:
    """The /arena surface of a sim. With `run_period_ns` a lockstep run is active and the clock moves one period per cam beat."""

    def __init__(self, run_period_ns: int = 0) -> None:
        from arena_rclpy_mixins.qos import latched
        from arena_people_msgs.msg import Pedestrian, Pedestrians
        from arena_runtime_msgs.msg import EnvRecord, EnvRegistry, LockstepStatus
        from arena_runtime_msgs.srv import LifecycleHold, LifecycleStep, LockstepRegister
        from geometry_msgs.msg import PoseStamped
        from rosgraph_msgs.msg import Clock
        from arena_runtime_msgs.msg import RobotDescriptor, RobotFleet, RobotState
        from viewport_control_msgs.msg import ViewportView
        from viewport_control_msgs.srv import ViewportCapture, ViewportSetProjection, ViewportSetReferenceFrame, ViewportSetView

        self.node = rclpy.create_node("arena_cam_runtime")
        self.run_period_ns = run_period_ns
        self.clock_ns = 1_000_000_000
        self.lock = threading.Lock()
        self.reference = ""
        self.references: list[str] = []
        self.projections: list[str] = []
        self.captures: list[Capture] = []
        self.keyframes: list[int] = []
        self.steps = 0
        self.holds: list[int] = []
        self.channels: list[tuple[str, str]] = []
        self.beats: list[int] = []
        self._beat_subs: dict[str, object] = {}
        self.node.create_service(ViewportSetView, "/arena/viewport/set_view", self._set_view)
        self.node.create_service(ViewportSetReferenceFrame, "/arena/viewport/set_reference_frame", self._set_reference)
        self.node.create_service(ViewportSetProjection, "/arena/viewport/set_projection", self._set_projection)
        self.node.create_service(ViewportCapture, "/arena/viewport/capture", self._capture, callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup())
        self.node.create_service(LifecycleHold, "/arena/sim_lifecycle/hold", self._hold)
        self.node.create_service(LifecycleStep, "/arena/sim_lifecycle/step", self._step)
        self.node.create_service(LockstepRegister, "/arena/sim_lifecycle/lockstep/register", self._register)
        self.node.create_subscription(ViewportView, "/arena/viewport/cmd_view", self._on_view, STREAM_QOS)
        self._clock = self.node.create_publisher(Clock, "/clock", 10)
        self._pose = self.node.create_publisher(PoseStamped, "/arena/viewport/camera_pose", 10)
        self.node.create_publisher(LockstepStatus, "/arena/state/lockstep", latched(1)).publish(LockstepStatus(active=run_period_ns > 0))
        self.node.create_publisher(EnvRegistry, "/arena/state/envs", latched(1)).publish(EnvRegistry(envs=[EnvRecord(env_id=0)]))
        fleet = RobotFleet(robots=[RobotState(descriptor=RobotDescriptor(name="jackal", base_frame="env_0/jackal/base_link"))])
        self.node.create_publisher(RobotFleet, "/arena/env_0/arena_env_node/state/robots", latched(1)).publish(fleet)
        self._peds = self.node.create_publisher(Pedestrians, "/arena/env_0/arena_peds", 10)
        self._roster = Pedestrians(pedestrians=[Pedestrian(id=4, name="env_0/edge_0")])
        self._Clock, self._PoseStamped = Clock, PoseStamped
        self.node.create_timer(0.02, self._tick)
        self._executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self.node)
        self._spin = threading.Thread(target=self._executor.spin, daemon=True)

    def __enter__(self) -> Runtime:
        self._spin.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._executor.shutdown()
        self._spin.join(timeout=10)
        self.node.destroy_node()

    def _stamp(self):
        from builtin_interfaces.msg import Time

        return Time(sec=self.clock_ns // 1_000_000_000, nanosec=self.clock_ns % 1_000_000_000)

    def _tick(self) -> None:
        with self.lock:
            stamp = self._stamp()
        self._clock.publish(self._Clock(clock=stamp))
        self._peds.publish(self._roster)
        pose = self._PoseStamped()
        pose.pose.position.z = 5.0
        pose.pose.orientation.w = 1.0
        self._pose.publish(pose)

    def _set_view(self, _request, response):
        response.success = True
        return response

    def _set_reference(self, request, response):
        with self.lock:
            self.reference = request.entity if request.entity else "world" if request.has_pose else "latched"
            self.references.append(self.reference)
        response.success = True
        return response

    def _set_projection(self, request, response):
        with self.lock:
            self.projections.append(request.projection)
        response.success = True
        return response

    def _capture(self, request, response):
        from sensor_msgs.msg import Image

        floor_ns = request.min_sim_time.sec * 1_000_000_000 + request.min_sim_time.nanosec
        if not self.wait(lambda: self.clock_ns >= floor_ns):
            response.success = False
            response.message = "capture timed out, min_sim_time not reached"
            return response
        with self.lock:
            self.captures.append(Capture(self.reference, request.fov, request.clip_near, self.clock_ns))
            frame = len(self.captures)
            stamp = self._stamp()
        response.image = Image(height=HEIGHT, width=WIDTH, encoding="rgb8", step=WIDTH * 3, data=bytes([frame % 256]) * (WIDTH * HEIGHT * 3))
        response.image.header.stamp = stamp
        response.success = True
        return response

    def _hold(self, request, response):
        with self.lock:
            self.holds.append(request.action)
        response.paused = request.action == request.ACQUIRE
        return response

    def _step(self, request, response):
        with self.lock:
            self.clock_ns += round(request.seconds * 1e9)
            self.steps += 1
        response.success = True
        response.advanced = request.seconds
        return response

    def _register(self, request, response):
        from arena_runtime_msgs.msg import LockstepHeartbeat

        registration = request.registration
        with self.lock:
            for channel in registration.channels:
                self.channels.append((channel.name, registration.env))
                if channel.topic not in self._beat_subs:
                    self._beat_subs[channel.topic] = self.node.create_subscription(LockstepHeartbeat, channel.topic, self._on_beat, 10)
            if not registration.channels:
                self.channels.append(("", registration.env))
        response.success = True
        return response

    def _on_beat(self, msg) -> None:
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        with self.lock:
            self.beats.append(stamp)
            self.clock_ns = stamp + self.run_period_ns

    def _on_view(self, _msg) -> None:
        with self.lock:
            self.keyframes.append(self.clock_ns)

    def wait(self, predicate: Callable[[], bool], timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if predicate():
                    return True
            time.sleep(0.02)
        return False
