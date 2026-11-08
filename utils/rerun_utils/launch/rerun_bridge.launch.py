"""Spawn the rerun web-viewer bridge for one arena env.

Args:
  ns          env namespace (e.g. /env_0); passed as positional argv to the node.
  web_port    port for the rerun web viewer (default: 9090).
  grpc_port   port for the rerun gRPC stream (default: 9876).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    ns = LaunchConfiguration('ns')
    web_port = LaunchConfiguration('web_port')
    grpc_port = LaunchConfiguration('grpc_port')

    return LaunchDescription([
        DeclareLaunchArgument('ns', description='env namespace, e.g. /env_0'),
        DeclareLaunchArgument('web_port', default_value='9090', description='Port of the rerun web viewer.'),
        DeclareLaunchArgument('grpc_port', default_value='9876', description='Port of the rerun gRPC server.'),
        Node(
            package='rerun_utils',
            executable='rerun_bridge',
            name='rerun_bridge',
            arguments=[ns],
            parameters=[{
                'web_port': ParameterValue(web_port, value_type=int),
                'grpc_port': ParameterValue(grpc_port, value_type=int),
            }],
            output='screen',
        ),
    ])
