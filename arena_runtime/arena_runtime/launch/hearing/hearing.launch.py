import launch
from arena_rclpy_mixins.launch_argument import LaunchArgument
from arena_bringup.substitutions import SelectAction
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from arena_runtime.hearing import FRONTENDS, NONE


def generate_launch_description() -> launch.LaunchDescription:

    ld = []

    LaunchArgument.auto_append(ld)

    namespace = LaunchArgument(
        name='namespace',
        description='Namespace of the hearing nodes.',
    )

    environment_namespace = LaunchArgument(
        name='environment_namespace',
        description='Namespace of the env whose robots the hearing layer serves.',
    )

    policy = LaunchArgument(
        name='policy',
        description='Hearing mask layers.',
    )

    launch_hearing = SelectAction(launch.substitutions.LaunchConfiguration('frontend'))

    launch_hearing.add(NONE, launch.actions.GroupAction([]))

    for key in FRONTENDS:
        launch_hearing.add(
            key,
            launch.actions.IncludeLaunchDescription(
                PathJoinSubstitution(
                    [
                        FindPackageShare('arena_runtime'),
                        'launch',
                        'hearing',
                        'arena',
                        'arena.launch.py',
                    ]
                ),
                launch_arguments={
                    **namespace.dict,
                    **environment_namespace.dict,
                    **policy.dict,
                    'frontend': launch.substitutions.LaunchConfiguration('frontend'),
                }.items(),
            ),
        )

    frontend = LaunchArgument(
        name='frontend',
        choices=launch_hearing.keys,
        description='Hearing front-end that produces the sound detections.',
    )

    ld = launch.LaunchDescription(
        [
            *ld,
            launch_hearing,
        ]
    )
    return ld


if __name__ == '__main__':
    generate_launch_description()
