import launch
from arena_bringup.substitutions import LaunchArgument, SelectAction
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from task_generator.simulators.hearing import FRONTENDS, NONE


def generate_launch_description() -> launch.LaunchDescription:

    ld = []

    LaunchArgument.auto_append(ld)

    namespace = LaunchArgument(
        name='namespace',
    )

    environment_namespace = LaunchArgument(
        name='environment_namespace',
    )

    policy = LaunchArgument(
        name='policy',
    )

    launch_hearing = SelectAction(launch.substitutions.LaunchConfiguration('frontend'))

    launch_hearing.add(NONE, launch.actions.GroupAction([]))

    for key in FRONTENDS:
        launch_hearing.add(
            key,
            launch.actions.IncludeLaunchDescription(
                PathJoinSubstitution(
                    [
                        FindPackageShare('task_generator'),
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
