"""Subscribe to the env's tf and tf_static, mirror them into rerun as named transform frames."""

from __future__ import annotations

import rclpy.node
import rclpy.qos
import rerun as rr
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

from rerun_utils.entity_paths import frame_name, tf_path


class TFMirror:
    """Long-lived `<tf_namespace>/tf` subscriber that logs each transform as a child -> parent frame relation.

    Rerun then chains the frames as tf does, and every renderer puts its data in its message's frame (`in_frame`).
    A frame that is no transform's child roots the tree: it is tied to the env's entity, so views find it there.
    """

    def __init__(self, node: rclpy.node.Node, env_id: int, tf_namespace: str = "") -> None:
        self._node = node
        self._env_id = env_id
        self._children: set[str] = set()
        self._roots: set[str] = set()
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
        child, parent = frame_name(t.child_frame_id), frame_name(t.header.frame_id)
        self._children.add(child)
        if parent not in self._children and parent not in self._roots:
            self._roots.add(parent)
            # identity into the implicit frame of env_<id>/tf, so the root sits at the env's origin; temporal, not
            # static, since static data would shadow a real transform into this frame that arrives later
            rr.log(tf_path(self._env_id, parent), rr.Transform3D(child_frame=parent))
        tr = t.transform.translation
        q = t.transform.rotation
        rr.log(
            tf_path(self._env_id, child),
            rr.Transform3D(
                translation=[tr.x, tr.y, tr.z],
                rotation=rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w]),
                child_frame=child,
                parent_frame=parent,
            ),
            static=static,
        )
