"""Subscribe to the env's tf and tf_static, mirror transforms into rerun's transform tree."""

from __future__ import annotations

import rclpy.node
import rclpy.qos
import rerun as rr
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

from rerun_utils.entity_paths import tf_path


class TFMirror:
    """Long-lived `<tf_namespace>/tf` subscriber that logs each transform to rerun under env_<id>/tf/<frame>."""

    def __init__(self, node: rclpy.node.Node, env_id: int, tf_namespace: str = "") -> None:
        self._node = node
        self._env_id = env_id
        self._dyn_sub = node.create_subscription(
            TFMessage,
            f"{tf_namespace}/tf",
            self._on_tf,
            rclpy.qos.QoSProfile(depth=100),
        )
        self._static_sub = node.create_subscription(
            TFMessage,
            f"{tf_namespace}/tf_static",
            self._on_tf_static,
            rclpy.qos.QoSProfile(
                depth=100,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

    def _on_tf(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            self._log(t, static=False)

    def _on_tf_static(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            self._log(t, static=True)

    def _log(self, t: TransformStamped, static: bool) -> None:
        path = tf_path(self._env_id, t.child_frame_id)
        tr = t.transform.translation
        q = t.transform.rotation
        rr.log(
            path,
            rr.Transform3D(
                translation=[tr.x, tr.y, tr.z],
                rotation=rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w]),
            ),
            static=static,
        )
