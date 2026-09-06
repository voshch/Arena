"""Publish a 4-channel wav as ``AudioFrame`` blocks, the renderer's wire format, for front-end tests.

    ros2 run arena_auditory hearing_audio_replay <wav> [--topic audio/raw_array] [--block 320] [--speed 1.0]

``--speed 0`` publishes as fast as the subscriber side can take (no pacing).
Stamps come from the node clock at the start of each block.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from task_generator_msgs.msg import AudioFrame

from arena_auditory.hearing.seld import load_wav

CHANNEL_NAMES = ("front_left", "front_right", "rear_left", "rear_right")


class AudioReplay(Node):
    def __init__(self, wav: str, topic: str, block: int, speed: float, frame_id: str, loop: bool) -> None:
        super().__init__("audio_replay")
        self.audio, self.fs = load_wav(wav)
        if self.audio.shape[1] < 4:
            raise SystemExit(f"{wav}: {self.audio.shape[1]} channels, need 4")
        self.audio = self.audio[:, :4].astype(np.float32)
        self.block = int(block)
        self.speed = float(speed)
        self.frame_id = frame_id
        self.loop = loop
        self.pub = self.create_publisher(AudioFrame, topic, 64)
        self.cursor = 0
        self.sent = 0

    def run(self) -> None:
        period = self.block / self.fs
        next_at = time.monotonic()
        while rclpy.ok():
            if self.cursor >= len(self.audio):
                if not self.loop:
                    break
                self.cursor = 0
            chunk = self.audio[self.cursor : self.cursor + self.block]
            self.cursor += self.block
            msg = AudioFrame()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.sample_rate = int(self.fs)
            msg.channel_count = 4
            msg.frame_count = int(chunk.shape[0])
            msg.encoding = "32FC1"
            msg.interleaved = True
            msg.channel_names = list(CHANNEL_NAMES)
            msg.data = chunk.reshape(-1).tolist()
            self.pub.publish(msg)
            self.sent += 1
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.speed > 0.0:
                next_at += period / self.speed
                delay = next_at - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
        self.get_logger().info(f"published {self.sent} blocks ({self.sent * self.block / self.fs:.1f} s)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--topic", default="audio/raw_array")
    ap.add_argument("--block", type=int, default=320)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--frame-id", default="base_link")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--settle", type=float, default=1.0, help="seconds to wait for subscribers before the first block")
    args, ros_args = ap.parse_known_args(argv if argv is not None else sys.argv[1:])
    rclpy.init(args=ros_args)
    node = AudioReplay(args.wav, args.topic, args.block, args.speed, args.frame_id, args.loop)
    try:
        time.sleep(args.settle)
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
