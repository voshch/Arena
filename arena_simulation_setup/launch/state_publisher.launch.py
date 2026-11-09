
import launch
import launch_ros.actions
import launch_ros.parameter_descriptions
from arena_bringup.future import PythonExpression
from arena_rclpy_mixins.launch_argument import LaunchArgument
from arena_bringup.substitutions import YAMLFileSubstitution
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robots_root = FindPackageShare('arena_robots')

    namespace = LaunchArgument('namespace', description='ROS namespace of the robot.')
    frame = LaunchArgument('frame', description='Frame prefix of the robot.')
    robot = LaunchArgument("robot", description="Robot model name.")

    control_yaml = YAMLFileSubstitution(
        PathJoinSubstitution([
            robots_root,
            robot.substitution,
            'control.yaml',
        ])
    )

    urdf_path = PathJoinSubstitution([
        robots_root,
        robot.substitution,
        'urdf',
        PythonExpression(['"', robot.substitution, '.urdf"']),
    ])

    urdf_content = Command(['cat ', urdf_path])

    robot_state_pub_node = launch_ros.actions.Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[
            {'use_sim_time': True},
            {'robot_description': launch_ros.parameter_descriptions.ParameterValue(urdf_content, value_type=str)},
            {'frame_prefix': launch_ros.parameter_descriptions.ParameterValue(frame.substitution, value_type=str)},
        ],
    )

    joint_state_pub_node = launch_ros.actions.Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': launch_ros.parameter_descriptions.ParameterValue(urdf_content, value_type=str)},
        ],
        remappings=[('/joint_states', '/joint_states')]
    )

    ld = launch.LaunchDescription([
        frame,
        robot,
        namespace,
        robot_state_pub_node,
        joint_state_pub_node,
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
