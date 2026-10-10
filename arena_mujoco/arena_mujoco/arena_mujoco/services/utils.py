"""Service registration helper."""

from collections.abc import Callable

import rclpy.node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

SERVICE_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1000, reliability=ReliabilityPolicy.RELIABLE)


class Service:
    """Descriptor for a single ROS 2 service registration."""

    def __init__(self, srv_type: type, srv_name: str, callback: Callable[..., object]) -> None:
        self.kwargs: dict[str, object] = {
            'srv_type': srv_type,
            'srv_name': srv_name,
            'callback': callback,
        }

    def create(self, node: rclpy.node.Node) -> None:
        """Register this service on node."""
        node.create_service(**self.kwargs, qos_profile=SERVICE_QOS)  # type: ignore[arg-type]
