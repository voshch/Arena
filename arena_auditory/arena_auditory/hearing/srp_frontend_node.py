"""Untrained hearing front-end: energy-onset detection plus array GCC-PHAT bearing, drop-in sibling of seld_frontend_node."""

from __future__ import annotations

import numpy as np
import rclpy
from arena_runtime_msgs.msg import LockstepHeartbeat
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from task_generator_msgs.msg import AudioFrame, HeardSoundEvent, RobotFleet

from arena_auditory.hearing.belief_node import latched_qos, transient_event_qos
from arena_auditory.hearing.doa import ArrayBearing
from arena_auditory.hearing.fleet import bind_robot
from arena_auditory.hearing.onset import OnsetDetector
from arena_auditory.hearing.timeline import AudioTimeline
from arena_auditory.lockstep import register_hard_channel

NB_CH = 4


class SrpFrontendNode(Node):
    def __init__(self, **kwargs: object) -> None:
        super().__init__("srp_frontend", **kwargs)
        self.declare_parameter("audio_topic", "")
        self.declare_parameter("event_topic", "")
        self.declare_parameter("listener_id", "")
        self.declare_parameter("robot_fleet_topic", "")
        self.declare_parameter("robot", "")
        self.declare_parameter("hop_s", 0.1)
        self.declare_parameter("floor_window_s", 5.0)
        self.declare_parameter("onset_db", 6.0)
        self.declare_parameter("sensitivity_dbfs_at_94_dbspl", -26.0)
        self.declare_parameter("hearing_threshold_db", 0.0)
        self.declare_parameter("audio_reliable", False)

        g = self.get_parameter
        self._hop_s = float(g("hop_s").value)
        self._floor_window_s = float(g("floor_window_s").value)
        self._onset_db = float(g("onset_db").value)
        self._spl_offset = 94.0 - float(g("sensitivity_dbfs_at_94_dbspl").value)
        self._frame_id = ""
        self._fs = 0
        self._hop_samples = 0
        self._consumed = 0
        self._buf = np.zeros((0, NB_CH), dtype=np.float32)
        self._prev_hop: np.ndarray | None = None
        self._onset: OnsetDetector | None = None
        self._doa: ArrayBearing | None = None
        self._timeline: AudioTimeline | None = None
        self._n_hops = 0
        self._n_events = 0
        self._bad_frames = 0
        self._n_no_bearing = 0

        self._audio_qos = QoSProfile(depth=64)
        self._audio_qos.reliability = ReliabilityPolicy.RELIABLE if bool(g("audio_reliable").value) else ReliabilityPolicy.BEST_EFFORT
        self._pub = None
        self._tick_pub = None
        self._listener_id = str(g("listener_id").value)
        self.get_logger().info(f"srp_frontend up: hop {self._hop_s:.3f} s, floor window {self._floor_window_s:.1f} s, onset {self._onset_db:.1f} dB")
        if str(g("audio_topic").value):
            self._bind(str(g("audio_topic").value), str(g("event_topic").value))
        else:
            self.create_subscription(RobotFleet, str(g("robot_fleet_topic").value), self._cb_fleet, latched_qos())

    def _bind(self, audio_topic: str, event_topic: str) -> None:
        self._pub = self.create_publisher(HeardSoundEvent, event_topic, transient_event_qos())
        self.create_subscription(AudioFrame, audio_topic, self._cb_audio, self._audio_qos)
        self.create_timer(5.0, self._diag)
        self.get_logger().info(f"srp_frontend: audio {audio_topic!r} -> events {event_topic!r}")

    def _cb_fleet(self, msg: RobotFleet) -> None:
        if self._pub is not None:
            return
        binding = bind_robot(msg, str(self.get_parameter("robot").value), str(self.get_parameter("robot_fleet_topic").value))
        if binding is None:
            return
        if not self._listener_id:
            self._listener_id = f"robot:{binding.name}"
        self._bind(f"{binding.tg_node}/{binding.name}/audio/raw_array", f"{binding.tg_node}/{binding.name}/heard_sound_srp")
        tick_topic = f"{binding.tg_node}/{binding.name}/lockstep/hearing"
        self._tick_pub = self.create_publisher(LockstepHeartbeat, tick_topic, 10)
        if bool(self.get_parameter("use_sim_time").value):
            register_hard_channel(
                self,
                name=f"hearing/{binding.name}",
                topic=self._tick_pub.topic_name,
                msg_type="arena_runtime_msgs/msg/LockstepHeartbeat",
                period_s=self._hop_s,
                env=self.resolve_topic_name(binding.tg_node),
            )

    def _cb_audio(self, msg: AudioFrame) -> None:
        if int(msg.channel_count) < NB_CH or (self._fs and int(msg.sample_rate) != self._fs):
            self._bad_frames += 1
            if self._bad_frames % 100 == 1:
                self.get_logger().warning(f"audio frame {msg.sample_rate} Hz x {msg.channel_count} ch, need {self._fs or 'any'} Hz x {NB_CH}+ ch ({self._bad_frames} dropped)")
            return
        if self._fs == 0:
            self._fs = int(msg.sample_rate)
            self._hop_samples = max(int(round(self._hop_s * self._fs)), 1)
            self._onset = OnsetDetector(self._fs, hop_s=self._hop_s, floor_window_s=self._floor_window_s, onset_db=self._onset_db)
            self._doa = ArrayBearing(self._fs)
            self._timeline = AudioTimeline(self._fs, max_gap_samples=self._hop_samples * 10)
            self._prev_hop = np.zeros((self._hop_samples, NB_CH), dtype=np.float32)
        data = np.asarray(msg.data, dtype=np.float32)
        ch, n = int(msg.channel_count), int(msg.frame_count)
        if data.size != ch * n:
            self._bad_frames += 1
            return
        block = data.reshape(n, ch) if msg.interleaved else data.reshape(ch, n).T
        block = block[:, :NB_CH]
        self._frame_id = msg.header.frame_id
        fill = self._timeline.observe(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec, n)
        if fill:
            self._buf = np.concatenate([self._buf, np.zeros((fill, NB_CH), dtype=np.float32)])
        self._buf = np.concatenate([self._buf, block])
        while self._buf.shape[0] >= self._hop_samples:
            hop = self._buf[: self._hop_samples]
            self._buf = self._buf[self._hop_samples :]
            self._step(hop)
        if self._tick_pub is not None:
            end_ns = self._timeline.time_ns(self._timeline.samples)
            beat = LockstepHeartbeat()
            beat.header.stamp.sec = end_ns // 1_000_000_000
            beat.header.stamp.nanosec = end_ns % 1_000_000_000
            beat.header.frame_id = self._frame_id
            self._tick_pub.publish(beat)

    def _step(self, hop: np.ndarray) -> None:
        self._n_hops += 1
        fired, peak_db, floor_db = self._onset.step(hop)
        hop_end = self._consumed + hop.shape[0]
        centre_ns = self._timeline.time_ns(hop_end) - self._hop_samples * 500_000_000 // self._fs
        self._consumed = hop_end
        if fired:
            combined = np.concatenate([self._prev_hop, hop])
            fitted, _residual, valid = self._doa.bearing(combined)
            if not valid:
                self._n_no_bearing += 1
            msg = HeardSoundEvent()
            msg.header.stamp.sec = centre_ns // 1_000_000_000
            msg.header.stamp.nanosec = centre_ns % 1_000_000_000
            msg.header.frame_id = self._frame_id
            msg.event_id = f"srp:{self._n_hops}"
            msg.listener_id = self._listener_id
            msg.source_agent_id = -1
            msg.sound_type = "onset"
            msg.label = f"srp:onset:{peak_db - floor_db:.1f}dB"
            msg.distance = -1.0
            if valid:
                msg.bearing_rad = fitted
                msg.propagation_backend = "srp"
            else:
                msg.bearing_rad = float("nan")
                msg.propagation_backend = "srp:nobearing"
            msg.source_volume_db = float("nan")
            msg.received_volume_db = float(peak_db + self._spl_offset)
            msg.hearing_threshold_db = float(self.get_parameter("hearing_threshold_db").value)
            msg.audible = True
            self._pub.publish(msg)
            self._n_events += 1
        self._prev_hop = hop

    def _diag(self) -> None:
        self.get_logger().debug(f"hops {self._n_hops} events {self._n_events} bad frames {self._bad_frames} no-bearing fits {self._n_no_bearing} gaps {self._timeline.gaps if self._timeline is not None else 0}")


def main() -> None:
    rclpy.init()
    node = SrpFrontendNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
