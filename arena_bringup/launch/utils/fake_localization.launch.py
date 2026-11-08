
import launch
import launch_ros.actions
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='global_frame_id',
            default_value='map',
            description='Frame id of the map frame, parent of the static transform.',
        ),
        launch.actions.DeclareLaunchArgument(
            name='odom_frame_id',
            default_value='odom',
            description='Frame id of the odometry frame, child of the static transform.',
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
