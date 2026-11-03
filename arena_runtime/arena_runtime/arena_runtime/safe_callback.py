import rclpy.callback_groups
import rclpy.node


class SafeCallbackNode(rclpy.node.Node):
    """
    Automatically make clients part of a new MutuallyExclusiveCallbackGroup to avoid deadlocks.
    """

    @property
    def default_callback_group(self) -> rclpy.callback_groups.CallbackGroup:
        return rclpy.callback_groups.ReentrantCallbackGroup()
