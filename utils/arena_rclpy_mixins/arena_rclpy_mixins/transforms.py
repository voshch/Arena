"""TF listening on a dedicated node and daemon thread, so /tf never wakes the owning node's executor."""

from __future__ import annotations

import contextlib
import threading
import uuid

import rclpy
import tf2_ros
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor


class ThreadedTransformListener:
    """Feeds buffer from /tf and /tf_static on its own node and executor until close()."""

    def __init__(self, buffer: tf2_ros.Buffer) -> None:
        self._node = rclpy.create_node(f"tf_listener_{uuid.uuid4().hex[:8]}", start_parameter_services=False, enable_rosout=False)
        self._listener = tf2_ros.TransformListener(buffer, self._node)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, name="tf_listener", daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        with contextlib.suppress(ExternalShutdownException):
            self._executor.spin()

    def close(self) -> None:
        self._executor.shutdown()
        self._thread.join()
        self._node.destroy_node()
