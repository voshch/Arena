from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException

from task_generator.auditory.audio_playback_node import SoundPlaybackNode


class HumanSoundPlaybackNode(SoundPlaybackNode):
    def __init__(self, **kwargs: object) -> None:
        super().__init__("human_sound_playback", "human", **kwargs)


def main() -> None:
    rclpy.init()
    node = HumanSoundPlaybackNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
