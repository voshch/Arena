"""Wait for a requested amount of timestamped audio in ROS simulation time."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import rclpy
from geometry_msgs.msg import Point
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.task import Future
from rosgraph_msgs.msg import Clock
from task_generator_msgs.action import RunEpisode
from task_generator_msgs.msg import AudioFrame, EpisodeRecord, HeardSoundEvent


class CaptureWaiter(Node):
    def __init__(
        self,
        duration: float,
        wall_timeout: float,
        namespace: str,
        robot_name: str,
        *,
        run_episode_action: str | None = None,
        world: str = "",
        inject_reference_sound: bool = False,
    ):
        super().__init__("arena_acoustics_capture_waiter")
        self.duration_ns = round(duration * 1_000_000_000)
        self.wall_deadline = time.monotonic() + wall_timeout
        self.clock_ns: int | None = None
        self.start_ns: int | None = None
        self.raw_end_ns: int | None = None
        self.rendered_end_ns: int | None = None
        self.raw_chunks = 0
        self.rendered_chunks = 0
        self.done = False
        self.error: str | None = None
        self.action_name = run_episode_action
        self.world = world
        self.goal_handle = None
        self.goal_accepted: bool | None = None
        self.cancel_requested = False
        self.action_result_state: int | None = None
        self.action_result_info: str | None = None
        self.action_episode_id: int | None = None
        self.terminal_event_state: int | None = None
        self.terminal_event_info: str | None = None
        self.inject_reference_sound = inject_reference_sound
        self.injected_reference_events = 0
        qos = QoSProfile(depth=100, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        # rclpy.node.Node owns an instance attribute named ``_clock``. Using
        # that name for our callback works only until Node.__init__ replaces
        # the bound method with its Clock object, which Jazzy then rejects as
        # a non-callable subscription callback.
        self.create_subscription(Clock, "/clock", self._on_clock, qos)
        prefix = "/" + namespace.strip("/") if namespace.strip("/") else ""
        robot = robot_name.strip("/")
        if not robot:
            raise ValueError("robot_name must not be empty")
        self.robot_name = robot
        self.raw_topic = f"{prefix}/{robot}/audio/raw_array"
        self.rendered_topic = f"{prefix}/{robot}/audio/headphones/stereo"
        self.episode_topic = f"{prefix}/state/episode"
        self.create_subscription(AudioFrame, self.raw_topic, self._raw, qos)
        self.create_subscription(AudioFrame, self.rendered_topic, self._rendered, qos)
        episode_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(EpisodeRecord, self.episode_topic, self._episode, episode_qos)
        sound_qos = QoSProfile(
            depth=50,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.sound_publisher = self.create_publisher(HeardSoundEvent, f"{prefix}/heard_sound_events", sound_qos) if inject_reference_sound else None
        self.create_timer(2.0, self._publish_reference_sound)
        self.create_timer(0.25, self._watchdog)
        self.action_client = ActionClient(self, RunEpisode, run_episode_action) if run_episode_action else None

    def start_episode(self) -> None:
        if self.action_client is None:
            return
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.error = f"episode action is unavailable: {self.action_name}"
            self.done = True
            return
        goal = RunEpisode.Goal()
        goal.world = self.world
        goal.seed = -1
        self.action_client.send_goal_async(goal).add_done_callback(self._goal_response)

    def _goal_response(self, future: Future[Any]) -> None:
        try:
            self.goal_handle = future.result()
        except Exception as exc:
            self.error = f"episode goal failed: {exc}"
            self.done = True
            return
        self.goal_accepted = bool(self.goal_handle.accepted)
        if not self.goal_accepted:
            self.error = "episode goal was rejected"
            self.done = True
            return
        self.goal_handle.get_result_async().add_done_callback(self._action_result)

    def _action_result(self, future: Future[Any]) -> None:
        try:
            result = future.result().result
            self.action_result_state = int(result.state)
            self.action_result_info = str(result.info)
            self.action_episode_id = int(result.episode_id)
        except Exception as exc:
            self.error = f"episode result failed: {exc}"
        self._maybe_finish()

    def _request_cancel(self) -> None:
        if self.action_client is None or self.cancel_requested or self.goal_handle is None:
            return
        self.cancel_requested = True
        self.goal_handle.cancel_goal_async().add_done_callback(self._cancel_response)

    def _cancel_response(self, future: Future[Any]) -> None:
        try:
            response = future.result()
            if not response.goals_canceling:
                self.error = "episode server did not accept capture-complete cancellation"
                self.done = True
        except Exception as exc:
            self.error = f"episode cancellation failed: {exc}"
            self.done = True

    @staticmethod
    def _stamp(msg: AudioFrame) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _on_clock(self, msg: Clock) -> None:
        self.clock_ns = int(msg.clock.sec) * 1_000_000_000 + int(msg.clock.nanosec)

    def _raw(self, msg: AudioFrame) -> None:
        self.raw_chunks += 1
        chunk_end_ns = self._chunk_end(msg)
        self.raw_end_ns = chunk_end_ns if self.raw_end_ns is None else max(self.raw_end_ns, chunk_end_ns)
        self._maybe_finish()

    def _rendered(self, msg: AudioFrame) -> None:
        self.rendered_chunks += 1
        chunk_end_ns = self._chunk_end(msg)
        self.rendered_end_ns = chunk_end_ns if self.rendered_end_ns is None else max(self.rendered_end_ns, chunk_end_ns)
        self._maybe_finish()

    def _episode(self, msg: EpisodeRecord) -> None:
        state = int(msg.outcome_state)
        if state == int(EpisodeRecord.RUNNING):
            self.start_ns = int(msg.start_time.sec) * 1_000_000_000 + int(msg.start_time.nanosec)
            self._publish_reference_sound()
        elif state in (
            int(EpisodeRecord.SUCCESS),
            int(EpisodeRecord.FAILED),
            int(EpisodeRecord.SKIPPED),
            int(EpisodeRecord.FATAL),
        ):
            self.terminal_event_state = state
            self.terminal_event_info = str(msg.outcome_info)
            if not self.coverage_complete:
                self.error = f"episode ended before audio coverage completed: state={state} info={msg.outcome_info!r}"
                self.done = True
        self._maybe_finish()

    def _publish_reference_sound(self) -> None:
        """Inject an audible, deterministic calibration event during capture.

        Some simulator backends provide pedestrian trajectories without their
        optional footstep event bridge. The reference greeting still traverses
        the normal propagation and microphone mixer, making silent captures a
        hard failure while keeping the stimulus identifiable in diagnostics.
        """
        # RUNNING is a state stream, not a one-shot event, and the timer below
        # deliberately retries while startup settles.  Never turn either into
        # a repeating greeting: one calibration event is sufficient for this
        # single capture waiter.
        if self.sound_publisher is None or self.start_ns is None or self.coverage_complete or self.injected_reference_events >= 1:
            return
        stamp = self.get_clock().now().to_msg()
        event_id = f"dataset-reference:{self.injected_reference_events}:{stamp.sec}:{stamp.nanosec}"
        for index, channel in enumerate(("front_left", "front_right", "rear_left", "rear_right")):
            msg = HeardSoundEvent()
            msg.header.stamp = stamp
            msg.header.frame_id = "map"
            msg.event_id = event_id
            msg.listener_id = f"{self.robot_name}_mic_{channel}"
            msg.source_agent_id = -10_001
            msg.source_agent_name = "dataset_reference_source"
            msg.sound_type = "greeting"
            msg.label = "dataset_reference"
            msg.asset_id = "greeting"
            msg.source_position = Point(x=0.0, y=0.0, z=1.6)
            msg.listener_position = Point(x=1.5, y=4.7, z=0.3)
            msg.distance = 4.93
            msg.bearing_rad = -1.88
            msg.source_volume_db = 85.0
            msg.received_volume_db = 74.0 - 0.25 * index
            msg.hearing_threshold_db = 20.0
            msg.direct_delay_sec = 0.014 + 0.0001 * index
            msg.audible = True
            msg.propagation_level = 3
            msg.propagation_backend = "dataset_reference"
            self.sound_publisher.publish(msg)
        self.injected_reference_events += 1

    def _chunk_end(self, msg: AudioFrame) -> int:
        channels = int(msg.channel_count)
        frames = int(msg.frame_count)
        if str(msg.encoding) != "32FC1" or not bool(msg.interleaved) or channels <= 0 or frames <= 0 or int(msg.sample_rate) <= 0:
            self.error = "audio stream contains an invalid encoding, channel count, or sample rate"
            self.done = True
            return self._stamp(msg)
        if len(msg.data) != frames * channels:
            self.error = "AudioFrame.data length does not match frame_count * channel_count"
            self.done = True
            return self._stamp(msg)
        return self._stamp(msg) + round(frames * 1_000_000_000 / int(msg.sample_rate))

    def _maybe_finish(self) -> None:
        if not self.coverage_complete:
            return
        if self.action_client is None:
            self.done = True
            return
        if self.action_result_state is None and self.terminal_event_state is None:
            self._request_cancel()
        if self.action_result_state is not None and self.terminal_event_state is not None:
            self.done = True

    @property
    def coverage_complete(self) -> bool:
        if self.start_ns is None or self.raw_end_ns is None or self.rendered_end_ns is None:
            return False
        requested_end_ns = self.start_ns + self.duration_ns
        return self.raw_end_ns >= requested_end_ns and self.rendered_end_ns >= requested_end_ns

    def _watchdog(self) -> None:
        if time.monotonic() >= self.wall_deadline:
            self.error = "wall-time watchdog expired before the requested simulation-time audio duration"
            self.done = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--wall-timeout", type=float, required=True)
    parser.add_argument("--namespace", required=True, help="environment namespace, for example /arena/env_0")
    parser.add_argument("--robot-name", default="jackal")
    parser.add_argument("--run-episode-action", help="send and cleanly cancel this RunEpisode action after capture")
    parser.add_argument("--world", default="", help="world passed to --run-episode-action")
    parser.add_argument("--inject-reference-sound", action="store_true", help="publish an audible calibration event every two seconds during the episode")
    args = parser.parse_args(argv)
    rclpy.init()
    node = CaptureWaiter(
        args.duration,
        args.wall_timeout,
        args.namespace,
        args.robot_name,
        run_episode_action=args.run_episode_action,
        world=args.world,
        inject_reference_sound=args.inject_reference_sound,
    )
    try:
        node.start_episode()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.25)
        action_complete = node.action_client is None or (node.action_result_state is not None and node.terminal_event_state is not None and node.action_result_state == node.terminal_event_state)
        result = {
            "valid": node.error is None and node.coverage_complete and action_complete,
            "coverage_complete": node.coverage_complete,
            "start_timestamp_ns": node.start_ns,
            "end_clock_ns": node.clock_ns,
            "raw_end_timestamp_ns": node.raw_end_ns,
            "rendered_end_timestamp_ns": node.rendered_end_ns,
            "raw_chunks_seen": node.raw_chunks,
            "rendered_chunks_seen": node.rendered_chunks,
            "injected_reference_events": node.injected_reference_events,
            "raw_topic": node.raw_topic,
            "rendered_topic": node.rendered_topic,
            "episode_topic": node.episode_topic,
            "goal_accepted": node.goal_accepted,
            "cancel_requested": node.cancel_requested,
            "action_result_state": node.action_result_state,
            "action_result_info": node.action_result_info,
            "action_episode_id": node.action_episode_id,
            "terminal_event_state": node.terminal_event_state,
            "terminal_event_info": node.terminal_event_info,
            "error": node.error,
        }
        print(json.dumps(result))
        return 0 if result["valid"] else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
