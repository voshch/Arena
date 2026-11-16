"""Publish a URDF string on `robot_description` and stay alive as a latched source.

Spawned via launch's OnProcessStart *after* the controller_manager so the CM is the
early-joiner from this publisher's perspective. That avoids FastDDS's unreliable
transient_local replay to late-joining readers.

Republishes every second until the CM's ``list_controllers`` service appears, then
stops. The service is created inside ``init_services()`` right after the CM has
successfully consumed the URDF and initialized its ResourceManager.
"""

import rclpy
from arena_rclpy_mixins.params import Param, declare
from controller_manager_msgs.srv import ListControllers
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String


class UrdfPublisher(Node):
    def __init__(self) -> None:
        super().__init__('urdf_publisher')
        urdf = declare(self, Param[str]('robot_description', '', 'URDF string published latched on robot_description.')).value
        if not urdf:
            raise RuntimeError('robot_description parameter is empty')
        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._msg = String(data=urdf)
        self._pub = self.create_publisher(String, 'robot_description', qos)
        self._cm_probe = self.create_client(
            ListControllers,
            'controller_manager/list_controllers',
        )
        self._pub.publish(self._msg)
        self._timer = self.create_timer(1.0, self._tick)

    def _tick(self) -> None:
        if self._cm_probe.service_is_ready():
            self._timer.cancel()
            self.destroy_timer(self._timer)
            return
        self._pub.publish(self._msg)


def main() -> None:
    rclpy.init()
    node = UrdfPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
