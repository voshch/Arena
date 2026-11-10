
import launch
import launch_ros.actions
from arena_rclpy_mixins.launch_argument import LaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    ld = launch.LaunchDescription([
        LaunchArgument(
            name='global_frame_id',
            default_value='map',
            description='Map frame id.',
        ),
        LaunchArgument(
            name='odom_frame_id',
            default_value='odom',
            description='Odometry frame id.',
        ),
        launch_ros.actions.Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_tfpublisher',
            arguments=['0', '0', '0', '0', '0', '0', LaunchConfiguration('global_frame_id'), LaunchConfiguration('odom_frame_id')]
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
