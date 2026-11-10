from launch import LaunchDescription
from launch_ros.actions import Node
from arena_rclpy_mixins.launch_argument import LaunchArgument


def generate_launch_description():
    # Launch Arguments
    use_sim_time = LaunchArgument('use_sim_time', default_value='true', description='Use the sim clock.')
    namespace = LaunchArgument('namespace', description='Namespace of the HuNav nodes.')

    return LaunchDescription([
        use_sim_time,
        namespace,

        # Agent Manager Node
        Node(
            package='hunav_agent_manager',
            executable='arena_hunav_agent_manager',
            namespace=namespace.substitution,
            name='hunav_agent_manager',
            output='screen',
            parameters=[
                use_sim_time.param(bool)
            ]
        )
    ])
