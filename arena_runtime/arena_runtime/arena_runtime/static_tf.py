from collections.abc import Iterable

import geometry_msgs.msg
import rclpy.node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from tf2_msgs.msg import TFMessage


class StaticTransformBroadcaster:
    """Latched /tf_static publisher holding one transform per child frame, a resend replaces it."""

    def __init__(self, node: rclpy.node.Node) -> None:
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
        self._publisher = node.create_publisher(TFMessage, "/tf_static", qos)
        self._transforms: dict[str, geometry_msgs.msg.TransformStamped] = {}

    def sendTransform(self, transform: geometry_msgs.msg.TransformStamped | Iterable[geometry_msgs.msg.TransformStamped]) -> None:
        for t in [transform] if isinstance(transform, geometry_msgs.msg.TransformStamped) else transform:
            self._transforms[t.child_frame_id] = t
        self._publisher.publish(TFMessage(transforms=list(self._transforms.values())))
