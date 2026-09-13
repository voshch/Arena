"""Spawn rviz for one arena env, with optional view/robot selection.

Args:
  ns      env namespace (e.g. /env_0); passed as positional argv to the node.
  view    map | robot | robot3p (default: map)
  robot   robot index in the fleet (default: 0); negative indexes from the end.
  view_distance  camera distance for the robot views, metres (default: 8.0)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    ns = LaunchConfiguration('ns')
    view = LaunchConfiguration('view')
    robot = LaunchConfiguration('robot')
    view_distance = LaunchConfiguration('view_distance')

    return LaunchDescription([
        DeclareLaunchArgument('ns', description='env namespace, e.g. /env_0'),
        DeclareLaunchArgument('view', default_value='map', choices=['map', 'robot', 'robot3p']),
        DeclareLaunchArgument('robot', default_value='0', description='robot index in the fleet'),
        DeclareLaunchArgument('view_distance', default_value='8.0', description='camera distance for the robot views, metres'),
        Node(
            package='rviz_utils',
            executable='rviz_config',
            name='rviz_config_generator',
            arguments=[ns],
            parameters=[{
                'view': ParameterValue(view, value_type=str),
                'robot': ParameterValue(robot, value_type=int),
                'view_distance': ParameterValue(view_distance, value_type=float),
            }],
            output='screen',
        ),
    ])
