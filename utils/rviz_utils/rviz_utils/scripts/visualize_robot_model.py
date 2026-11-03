#!/usr/bin/env python3

import functools
import os
import traceback
from typing import TypedDict

import rclpy
import rclpy.qos
import yaml
from arena_runtime_msgs.msg import RobotFleet
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry
from arena_rclpy_mixins.lazy import LazyPublisher
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray


class _Footprint(TypedDict, total=False):
    type: str
    radius: float
    points: list[tuple[float, float]]


class _Body(TypedDict, total=False):
    color: list[float]
    footprints: list[_Footprint]


class VisualizeRobotModel(Node):
    def __init__(self):
        super().__init__('visualize_robot_model')

        self.declare_parameter('arena_env_node', '/arena_env_node')
        tg_node = self.get_parameter('arena_env_node').value
        assert isinstance(tg_node, str)

        self.srv_start_setup = self.create_service(Empty, "start_model_visualization", self.start_model_visualization_callback)

        self.robot_models = {}
        self.publisher_map = {}
        self.subscribers = []
        self._latest_fleet: RobotFleet | None = None

        self._fleet_sub = self.create_subscription(
            RobotFleet,
            os.path.join(tg_node, 'state', 'robots'),
            self._on_fleet,
            qos_profile=rclpy.qos.QoSProfile(
                depth=1,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

    def _on_fleet(self, msg: RobotFleet) -> None:
        self._latest_fleet = msg

    def start_model_visualization_callback(self, request: Empty.Request, response: Empty.Response) -> Empty.Response:
        if self._latest_fleet is None:
            self.get_logger().error("No fleet snapshot received yet on state/robots; cannot visualize.")
            return Empty.Response()

        # Get the appropriate odom topic based on complexity
        robot_odom_topic = self.get_complexity_odom_topic()

        for state in self._latest_fleet.robots:
            robot = state.descriptor
            # Load the model file
            model_file = self.read_robot_model_file(robot.model)

            # Generate markers for the model
            markers_for_model = self.create_marker_array_for_robot(model_file)

            # Store the markers
            self.robot_models[robot.model] = markers_for_model

            # Create publisher for each robot
            self.publisher_map[robot.name] = LazyPublisher(self.create_publisher(MarkerArray, os.path.join(robot.ns, "visualize", "model"), 10))

            # Create subscriber for each robot's odometry
            self.subscribers.append(self.create_subscription(Odometry, os.path.join(robot.ns, robot_odom_topic), functools.partial(self.publish_model, args=(robot.model, robot.name)), 10))

        return Empty.Response()

    def publish_model(self, data: Odometry, args: tuple[str, str]) -> None:
        robot_model, name = args
        if not self.publisher_map[name].wanted:
            return

        try:
            markers = self.robot_models[robot_model]
        except Exception:
            self.get_logger().error(f"Error - Getting markers from dict {robot_model}")
            return

        for marker in markers:
            marker.header = data.header
            marker.header.frame_id = "map"
            marker.pose = data.pose.pose

        try:
            self.publisher_map[name].publish(lambda: MarkerArray(markers=markers))
        except Exception:
            self.get_logger().error(traceback.format_exc())
            self.get_logger().error(f"Error - publishing markers {name}")

    def read_robot_model_file(self, robot_model: str) -> list[_Body]:
        try:
            # In ROS2, use ament_index_python instead of rospkg
            from ament_index_python.packages import get_package_share_directory

            file_path = os.path.join(get_package_share_directory('simulation_setup'), "entities", "robots", robot_model, f"{robot_model}.model.yaml")

            with open(file_path) as file:
                return yaml.safe_load(file)["bodies"]
        except Exception as e:
            self.get_logger().error(f"Error reading robot model file: {e}")
            return self.get_default_model()

    def get_default_model(self) -> list[_Body]:
        # Simple default model if the real one can't be found
        return [{'color': [0, 0, 1, 1], 'footprints': [{'type': 'circle', 'radius': 0.25}]}]

    def create_marker_array_for_robot(self, bodies: list[_Body]) -> list[Marker]:
        markers = []

        for i, body in enumerate(bodies):
            color = body.get("color", [0, 0, 1, 1])

            for j, footprint in enumerate(body.get("footprints", [])):
                marker = self.create_marker_from_footprint(footprint, color, i * 100 + j)
                markers.append(marker)

        return markers

    def create_marker_from_footprint(self, footprint: _Footprint, color: list[float], id: int) -> Marker:
        r, g, b, a = color

        marker = Marker()
        marker.id = id
        marker.action = Marker.ADD
        marker.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))

        if footprint.get("type") == "circle":
            marker.type = Marker.SPHERE
            marker.scale = Vector3(x=footprint.get("radius", 0.25) * 2, y=footprint.get("radius", 0.25) * 2, z=0.1)
        else:
            marker.type = Marker.LINE_STRIP
            marker.scale = Vector3(x=0.03, y=0, z=0)

            points = [Point(x=float(x), y=float(y), z=0.0) for x, y in footprint.get("points", [])]

            # Close the loop by adding the first point again
            if points and len(points) > 0:
                points.append(points[0])
            marker.points = points

        return marker

    def get_complexity_odom_topic(self) -> str:
        self.declare_parameter('complexity', 1)
        complexity = self.get_parameter('complexity').value

        if complexity == 1:
            return "odom"
        elif complexity == 2:
            return "odom_amcl"
        else:
            return "odom"  # Default


def main(args: list[str] | None = None) -> None:
    from arena_rclpy_mixins.spin import spin_node

    rclpy.init(args=args)
    spin_node(VisualizeRobotModel())


if __name__ == "__main__":
    main()
