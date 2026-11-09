"""Rviz of one arena env with its view and robot selection."""

from arena_rclpy_mixins.launch_argument import LaunchArgument, forwarded
from launch import LaunchContext, LaunchDescription, LaunchDescriptionEntity
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from rviz_utils.params import GENERATOR_PARAMS


def _generator(context: LaunchContext) -> list[LaunchDescriptionEntity]:
    return [Node(package='rviz_utils', executable='rviz_config', name='rviz_config_generator', arguments=[LaunchConfiguration('ns')], parameters=[forwarded(context, GENERATOR_PARAMS)], output='screen')]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        LaunchArgument('ns', description='Env namespace.'),
        *(LaunchArgument.from_param(param) for param in GENERATOR_PARAMS),
        OpaqueFunction(function=_generator),
    ])
