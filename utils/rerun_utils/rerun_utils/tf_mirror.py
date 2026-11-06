"""Subscribe to the env's tf and tf_static, mirror them into rerun as named transform frames."""

from __future__ import annotations

from dataclasses import dataclass

import rclpy.node
import rclpy.qos
import rerun as rr
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

from rerun_utils.entity_paths import ENV_ORIGIN, frame_name, root_path, tf_path


@dataclass(frozen=True)
class Tie:
    """Tie the root `frame` to the env's origin, or untie it once a transform into it arrives."""

    frame: str
    tied: bool


class TFTree:
    """The frames seen so far; the ones that are no transform's child are the roots tied to the env's origin."""

    def __init__(self) -> None:
        self._children: set[str] = set()
        self._roots: set[str] = set()

    @property
    def roots(self) -> frozenset[str]:
        return frozenset(self._roots)

    def ingest(self, child: str, parent: str) -> list[Tie]:
        """Record the transform of `child` into `parent`; the ties to log, whatever order transforms arrive in."""
        ties: list[Tie] = []
        if child in self._roots:
            self._roots.remove(child)
            ties.append(Tie(child, False))
        self._children.add(child)
        if parent not in self._children and parent not in self._roots:
            self._roots.add(parent)
            ties.append(Tie(parent, True))
        return ties


class TFMirror:
    """Long-lived `<tf_namespace>/tf` subscriber that logs each transform as a child -> parent frame relation.

    Rerun then chains the frames as tf does, and every renderer puts its data in its message's frame (`in_frame`).
    A frame that is no transform's child roots the tree: it is tied to the recording root (`ENV_ORIGIN`), so views
    on `/` reach it, and untied again if a transform into it arrives later.
    """

    def __init__(self, node: rclpy.node.Node, env_id: int, tf_namespace: str = "") -> None:
        self._node = node
        self._env_id = env_id
        self._tree = TFTree()
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
        for tie in self._tree.ingest(child, parent):
            if tie.tied:
                rr.log(root_path(self._env_id, tie.frame), rr.Transform3D(child_frame=tie.frame, parent_frame=ENV_ORIGIN))
            else:
                rr.log(root_path(self._env_id, tie.frame), rr.Clear(recursive=False))
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
