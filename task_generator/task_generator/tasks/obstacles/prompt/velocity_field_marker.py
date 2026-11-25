import numpy as np
from arena_rclpy_mixins.Time import TimeNode
from geometry_msgs.msg import Point
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class VelocityFieldVisualizer:
    def __init__(
        self,
        node: TimeNode,
        map_min_x: float = 0.0,
        map_min_y: float = 0.0,
        map_max_x: float = 0.0,
        map_max_y: float = 0.0,
        topic_name: str = "/velocity_field_marker",
    ):
        self.node = node
        self._logger = node.get_logger()

        # Internal state for coordinate conversion
        self.map_min_x = map_min_x
        self.map_min_y = map_min_y
        self.map_max_x = map_max_x
        self.map_max_y = map_max_y

        # RViz Publisher
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.marker_publisher: Publisher = self.node.create_publisher(
            MarkerArray,
            topic_name,
            self.qos_profile,
        )

    def update_world_bounds(self, x_min: float, y_min: float, x_max: float, y_max: float) -> None:
        """Update bounds used for coordinate mapping."""
        self.map_min_x = x_min
        self.map_min_y = y_min
        self.map_max_x = x_max
        self.map_max_y = y_max

    def publish_markers(self, field: np.ndarray):
        """
        field: np.ndarray of shape (G, H, W, 2)
        """
        assert len(field.shape) == 4
        g = field.shape[0]

        marker_array = MarkerArray()

        id_counter = 0

        for group_idx in range(g):
            n_makers = self._add_markers(field[group_idx], f"group_{group_idx}", marker_array, id_counter)
            id_counter += n_makers

        self._logger.debug(f"Publishing {id_counter} markers")
        self.marker_publisher.publish(marker_array)

    def _add_markers(
        self,
        field: np.ndarray,
        namespace: str,
        marker_array: MarkerArray,
        start_id: int,
    ) -> int:
        """
        Converts normalized grid to RViz Markers.
        field: np.ndarray of shape (H, W, 2)
        """
        h, w, _ = field.shape

        # Calculate cell sizes in world coordinates
        world_width = self.map_max_x - self.map_min_x
        world_height = self.map_max_y - self.map_min_y
        assert world_width * world_height > 0, f"Invalid world size: {world_height}x{world_width}"
        dx = world_width / w
        dy = world_height / h

        # Downsample for performance if grid is too dense
        step = 2 if h > 32 else 1

        id_counter = start_id
        for grid_y in range(0, h, step):
            for grid_x in range(0, w, step):
                # Center of the cell in world coordinates
                world_x = self.map_min_x + (grid_x + 1.0) * dx
                world_y = self.map_min_y + (grid_y + 1.0) * dy

                vx = field[grid_y, grid_x, 0]
                vy = field[grid_y, grid_x, 1]

                # Skip near-zero vectors
                if np.hypot(vx, vy) < 0.05:
                    continue

                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.node.sim_time.to_msg()
                marker.ns = namespace
                marker.id = id_counter
                marker.type = Marker.ARROW
                marker.action = Marker.ADD

                # Arrow Start (World position)
                start = Point(x=world_x, y=world_y, z=0.1)
                # Arrow End (Position + scaled velocity)
                end = Point(x=world_x + vx * dx, y=world_y + vy * dy, z=0.1)

                marker.points = [start, end]

                # Aesthetics
                marker.scale.x = 0.08  # Shaft diameter
                marker.scale.y = 0.15  # Head diameter
                marker.scale.z = 0.15  # Head length
                marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)  # Orange

                marker_array.markers.append(marker)  # pyright: ignore[reportAttributeAccessIssue]
                id_counter += 1

        return id_counter


if __name__ == "__main__":
    import os

    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    rclpy.init()

    node = TimeNode("velocity_field_marker_test_node")

    visualizer = VelocityFieldVisualizer(node)

    visualizer.update_world_bounds(0.0, 0.0, 25.0, 34.2)

    # visualizer.publish_markers(np.random.uniform(-1, 1, size=(3, 64, 64, 2)))
    velocity_field = np.load(f"{os.environ['HOME']}/Desktop/velocity_field.npy")
    visualizer.publish_markers(velocity_field)

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
