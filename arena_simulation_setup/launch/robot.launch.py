import launch
import launch.actions
import launch.launch_description_sources
import launch.substitutions
from arena_rclpy_mixins.launch_argument import LaunchArgument
from launch_ros.actions import PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    ss_path = FindPackageShare('arena_simulation_setup')

    ld_items = []
    LaunchArgument.auto_append(ld_items)

    use_sim_time = LaunchArgument("use_sim_time", description="Use the simulation clock.")

    arena_env_node = LaunchArgument('arena_env_node', description='Name of the arena env node of this robot.')
    namespace = LaunchArgument("namespace", description="ROS namespace of the robot.")
    robot = LaunchArgument("robot", description="Robot model name.")
    frame = LaunchArgument("frame", description="Frame prefix of the robot.")

    # launch robot control
    state_pub_launch = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            launch.substitutions.PathJoinSubstitution(
                [
                    ss_path,
                    "launch",
                    "state_publisher.launch.py",
                ]
            )),
        launch_arguments={
            **use_sim_time.dict,
            **frame.dict,
            **namespace.dict,
            **robot.dict,
        }.items(),
    )

    ld = launch.LaunchDescription([
        *ld_items,
        PushRosNamespace(namespace=namespace.substitution),
        # robot_localization_node,
        # state_pub_launch,
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
