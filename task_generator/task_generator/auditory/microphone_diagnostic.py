"""Console diagnostics for the Jackal four-channel raw array."""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from task_generator_msgs.msg import AudioFrame

from task_generator.auditory.spatial_audio import CHANNEL_NAMES, dbfs_from_rms, gcc_phat, rms


class MicrophoneDiagnostic(Node):
    def __init__(self) -> None:
        super().__init__("microphone_diagnostic")
        self.declare_parameter("topic", "jackal/audio/raw_array")
        self.declare_parameter("report_period", 0.5)
        self.declare_parameter("activity_threshold_dbfs", -70.0)
        self._latest: tuple[np.ndarray, int] | None = None
        self.create_subscription(AudioFrame, self.get_parameter("topic").value, self._on_audio, 10)
        self.create_timer(float(self.get_parameter("report_period").value), self._report)

    def _on_audio(self, msg: AudioFrame) -> None:
        if msg.channel_count != 4 or msg.frame_count == 0:
            self.get_logger().warning("expected a non-empty four-channel AudioFrame")
            return
        data = np.asarray(msg.data, dtype=np.float32)
        expected = int(msg.channel_count * msg.frame_count)
        if data.size != expected:
            self.get_logger().warning(f"malformed AudioFrame: {data.size} values, expected {expected}")
            return
        self._latest = (data.reshape(msg.frame_count, msg.channel_count).T.copy(), int(msg.sample_rate))

    def _report(self) -> None:
        if self._latest is None:
            return
        audio, rate = self._latest
        levels = [dbfs_from_rms(float(value)) for value in rms(audio, axis=1)]
        pairs = ((0, 1, "FL -> FR"), (2, 3, "RL -> RR"), (0, 2, "FL -> RL"), (1, 3, "FR -> RR"))
        estimates = [(label, *gcc_phat(audio[first], audio[second], sample_rate_hz=rate, max_tau_seconds=0.002)) for first, second, label in pairs]
        left_energy = float(rms(audio[[0, 2]]))
        right_energy = float(rms(audio[[1, 3]]))
        front_energy = float(rms(audio[[0, 1]]))
        rear_energy = float(rms(audio[[2, 3]]))

        def evidence(first: float, second: float) -> tuple[float, float]:
            total = first + second
            return (0.5, 0.5) if total <= 1e-12 else (first / total, second / total)

        left, right = evidence(left_energy, right_energy)
        front, rear = evidence(front_energy, rear_energy)
        lines = [
            "Source activity detected" if max(levels) >= float(self.get_parameter("activity_threshold_dbfs").value) else "No source activity",
            *(f"{name.replace('_', ' ').upper()}: {level:.1f} dBFS" for name, level in zip(CHANNEL_NAMES, levels, strict=True)),
            "GCC-PHAT:",
            *(f"  {label}: {delay * 1e6:+.0f} us (confidence {confidence:.3f})" for label, delay, confidence in estimates),
            "coarse spatial evidence (energy heuristic):",
            f"  left: {left:.2f}  right: {right:.2f}",
            f"  front: {front:.2f}  rear: {rear:.2f}",
        ]
        self.get_logger().info("\n" + "\n".join(lines))


def main() -> None:
    rclpy.init()
    node = MicrophoneDiagnostic()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
