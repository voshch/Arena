"""Live SELDnet front-end: 4-mic ``AudioFrame`` stream -> ``HeardSoundEvent`` at label-frame rate.

Subscribes to the microphone array renderer's ``<robot>/audio/raw_array``
(``task_generator_msgs/AudioFrame``, interleaved float PCM, 16 kHz, 4 channels),
runs SALSA-Lite + SELDnet over a sliding 5 s window once per 100 ms label
frame, and publishes one ``HeardSoundEvent`` per detection so the belief node
stays source-agnostic.  Bearings are robot-frame (CCW from +x), so the consumer
runs with ``bearing_frame: robot``.  ``bearing_source`` picks where the bearing
comes from: ``gcc`` (default) fits the array geometry to GCC-PHAT pair delays of
the detection frame, ``seld`` takes the model's azimuth.  The checkpoint's DOA
head is front-biased (sources beside the robot come back near 0 deg), the fit
is not, and with two simultaneous sources the fit follows the louder one.

Only the fields a front-end can know are filled: ``sound_type`` (class name),
``bearing_rad``, ``received_volume_db`` (frame RMS through the array's MEMS
sensitivity), ``label`` (``seld:<class>:activity=<v>``), ``listener_id``,
``audible``.  Ground-truth fields stay at their defaults.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from task_generator_msgs.msg import AudioFrame, HeardSoundEvent, RobotFleet

from arena_auditory.hearing import weights
from arena_auditory.hearing.belief_node import latched_qos, transient_event_qos
from arena_auditory.hearing.doa import ArrayBearing
from arena_auditory.hearing.fleet import bind_robot
from arena_auditory.hearing.seld import SeldFrontend, SeldStream
from arena_auditory.hearing.timeline import AudioTimeline


class SeldFrontendNode(Node):
    def __init__(self, **kwargs: object) -> None:
        super().__init__("seld_frontend", **kwargs)
        self.declare_parameter("audio_topic", "")  # explicit; empty = <tg_node>/<robot>/audio/raw_array from the fleet
        self.declare_parameter("event_topic", "")  # explicit; empty = <tg_node>/<robot>/heard_sound_seld from the fleet
        self.declare_parameter("listener_id", "")  # explicit; empty = robot:<robot> from the fleet
        self.declare_parameter("robot_fleet_topic", "")
        self.declare_parameter("robot", "")
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("scaler", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("det_thresh", 0.5)
        self.declare_parameter("lookahead_frames", 5)
        self.declare_parameter("bearing_source", "gcc")  # gcc | seld
        self.declare_parameter("sensitivity_dbfs_at_94_dbspl", -26.0)
        self.declare_parameter("hearing_threshold_db", 0.0)
        self.declare_parameter("audio_reliable", False)

        g = self.get_parameter
        files = weights.ensure()
        self._fe = SeldFrontend(
            checkpoint=str(g("checkpoint").value) or files["checkpoint"],
            scaler=str(g("scaler").value) or files["scaler"],
            device=str(g("device").value),
            det_thresh=float(g("det_thresh").value),
        )
        self._stream = SeldStream(self._fe, lookahead=int(g("lookahead_frames").value))
        source = str(g("bearing_source").value)
        if source not in ("gcc", "seld"):
            raise ValueError(f"bearing_source must be gcc or seld, not {source!r}")
        self._doa = ArrayBearing(self._fe.fs) if source == "gcc" else None
        self._spl_offset = 94.0 - float(g("sensitivity_dbfs_at_94_dbspl").value)
        self._frame_id = ""
        self._timeline = AudioTimeline(self._fe.fs, max_gap_samples=self._fe.window_samples)
        self._n_frames = 0
        self._n_events = 0
        self._bad_frames = 0
        self._n_no_bearing = 0

        self._audio_qos = QoSProfile(depth=64)
        self._audio_qos.reliability = ReliabilityPolicy.RELIABLE if bool(g("audio_reliable").value) else ReliabilityPolicy.BEST_EFFORT
        self._pub = None
        self._listener_id = str(g("listener_id").value)
        self.get_logger().info(f"seld_frontend up on {self._fe.device}: window {self._fe.window_samples / self._fe.fs:.1f} s, lookahead {self._stream.lookahead} frames")
        if str(g("audio_topic").value):
            self._bind(str(g("audio_topic").value), str(g("event_topic").value))
        else:
            self.create_subscription(RobotFleet, str(g("robot_fleet_topic").value), self._cb_fleet, latched_qos())

    def _bind(self, audio_topic: str, event_topic: str) -> None:
        self._pub = self.create_publisher(HeardSoundEvent, event_topic, transient_event_qos())
        self.create_subscription(AudioFrame, audio_topic, self._cb_audio, self._audio_qos)
        self.create_timer(self._fe.label_hop_len / self._fe.fs, self._on_timer)
        self.create_timer(5.0, self._diag)
        self.get_logger().info(f"seld_frontend: audio {audio_topic!r} -> events {event_topic!r}")

    def _cb_fleet(self, msg: RobotFleet) -> None:
        if self._pub is not None:
            return
        binding = bind_robot(msg, str(self.get_parameter("robot").value), str(self.get_parameter("robot_fleet_topic").value))
        if binding is None:
            return
        if not self._listener_id:
            self._listener_id = f"robot:{binding.name}"
        self._bind(f"{binding.tg_node}/{binding.name}/audio/raw_array", f"{binding.tg_node}/{binding.name}/heard_sound_seld")

    def _cb_audio(self, msg: AudioFrame) -> None:
        if int(msg.sample_rate) != self._fe.fs or int(msg.channel_count) < self._fe.nb_raw_ch:
            self._bad_frames += 1
            if self._bad_frames % 100 == 1:
                self.get_logger().warning(f"audio frame {msg.sample_rate} Hz x {msg.channel_count} ch, need {self._fe.fs} Hz x {self._fe.nb_raw_ch} ch ({self._bad_frames} dropped)")
            return
        data = np.asarray(msg.data, dtype=np.float32)
        ch, n = int(msg.channel_count), int(msg.frame_count)
        if data.size != ch * n:
            self._bad_frames += 1
            return
        block = data.reshape(n, ch) if msg.interleaved else data.reshape(ch, n).T
        self._frame_id = msg.header.frame_id
        fill = self._timeline.observe(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec, n)
        if fill:
            self._stream.push(np.zeros((fill, ch), dtype=np.float32))
        self._stream.push(block)

    def _on_timer(self) -> None:
        if self._timeline.start_ns is None or not self._stream.ready():
            return
        dets, end, seg = self._stream.step()
        self._n_frames += 1
        if not dets:
            return
        fe = self._fe
        centre_ns = self._timeline.time_ns(end) - fe.label_hop_len * 500_000_000 // fe.fs
        rms = float(np.sqrt(np.mean(seg[-fe.label_hop_len :] ** 2))) if seg.size else 0.0
        level_db = 20.0 * math.log10(max(rms, 1e-9)) + self._spl_offset
        fitted, valid = (None, True)
        if self._doa is not None:
            fitted, _residual, valid = self._doa.bearing(seg)
            if not valid:
                self._n_no_bearing += 1
        listener = self._listener_id
        for d in dets:
            msg = HeardSoundEvent()
            msg.header.stamp.sec = centre_ns // 1_000_000_000
            msg.header.stamp.nanosec = centre_ns % 1_000_000_000
            msg.header.frame_id = self._frame_id
            msg.event_id = f"seld:{self._n_frames}:{d.cls}"
            msg.listener_id = listener
            msg.source_agent_id = -1
            msg.sound_type = d.sound_type
            msg.label = f"seld:{d.sound_type}:activity={d.activity:.3f}"
            msg.distance = -1.0
            if fitted is None:
                msg.bearing_rad = float(d.azimuth_rad)
                msg.propagation_backend = "seldnet"
            elif valid:
                msg.bearing_rad = fitted
                msg.propagation_backend = "seldnet+gcc"
            else:
                msg.bearing_rad = float("nan")
                msg.propagation_backend = "seldnet+gcc:nobearing"
            msg.source_volume_db = float("nan")
            msg.received_volume_db = float(level_db)
            msg.hearing_threshold_db = float(self.get_parameter("hearing_threshold_db").value)
            msg.audible = True
            self._pub.publish(msg)
            self._n_events += 1

    def _diag(self) -> None:
        self.get_logger().debug(f"frames {self._n_frames} events {self._n_events} audio samples {self._stream.samples_seen} gaps {self._timeline.gaps} rewinds {self._timeline.rewinds} bad frames {self._bad_frames} no-bearing fits {self._n_no_bearing}")


def main() -> None:
    rclpy.init()
    node = SeldFrontendNode()
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
