"""Forward `/gz/clock` to `/clock` only when the stamp changes (gz republishes
its clock every loop iteration, paused ones included)."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rosgraph_msgs.msg import Clock

_CLOCK_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
)


class ClockRelay(Node):
    def __init__(self) -> None:
        super().__init__('clock_relay')
        self._last: tuple[int, int] | None = None
        self._pub = self.create_publisher(Clock, '/clock', _CLOCK_QOS)
        self.create_subscription(Clock, '/gz/clock', self._on_clock, _CLOCK_QOS)

    def _on_clock(self, msg: Clock) -> None:
        stamp = (msg.clock.sec, msg.clock.nanosec)
        if stamp == self._last:
            return
        self._last = stamp
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = ClockRelay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
