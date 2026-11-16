"""Rerun web-viewer bridge of one arena env."""

from arena_rclpy_mixins.launch_argument import LaunchArgument, forwarded
from launch import LaunchContext, LaunchDescription, LaunchDescriptionEntity
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from rerun_utils.params import BRIDGE_PARAMS


def _bridge(context: LaunchContext) -> list[LaunchDescriptionEntity]:
    return [Node(package='rerun_utils', executable='rerun_bridge', name='rerun_bridge', arguments=[LaunchConfiguration('ns')], parameters=[forwarded(context, BRIDGE_PARAMS)], output='screen')]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        LaunchArgument('ns', description='Env namespace.'),
        *(LaunchArgument.from_param(param) for param in BRIDGE_PARAMS),
        OpaqueFunction(function=_bridge),
    ])
